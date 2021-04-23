/**********************************************************************
Copyright (c) 2018 Habana Labs.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#ifndef FILTER_2D_I8_W33_S11_TEST_HPP
#define FILTER_2D_I8_W33_S11_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "filter_2d_i8_w33_s11.hpp"

class Filter2DI8W33S11Test : public TestBase
{
public:
    Filter2DI8W33S11Test() {}
    ~Filter2DI8W33S11Test() {}
    int runTest();

    inline static void filter_2d_refence_implementation(
        const test::Tensor<int8_t,4>& ifm,
        const test::Tensor<int8_t,3>& filter,
        const test::Tensor<int32_t,1>& bias,
        test::Tensor<int8_t,4>& ofm,
        const SpatialReductionKernels::SpatialReduction2DDef& layer_def,
        const IndexSpace& indexSpace,
        int8_t exp);
private:
    Filter2DI8W33S11Test(const Filter2DI8W33S11Test& other) = delete;
    Filter2DI8W33S11Test& operator=(const Filter2DI8W33S11Test& other) = delete;

};


inline void Filter2DI8W33S11Test::filter_2d_refence_implementation(
        const test::Tensor<int8_t,4>& ifm,
        const test::Tensor<int8_t,3>& filter,
        const test::Tensor<int32_t,1>& bias,
        test::Tensor<int8_t,4>& ofm,
        const SpatialReductionKernels::SpatialReduction2DDef& layer_def,
        const IndexSpace& indexSpace,
        int8_t exp)
{
    int output_coords [4] = {0};
    for (int b = indexSpace.offset[3]; b < indexSpace.offset[3]+indexSpace.size[3]; b += 1)
    {
        output_coords[3] = b;
        for (int h = indexSpace.offset[2]; h < indexSpace.offset[2]+indexSpace.size[2]; h += 1)
        {
            output_coords[2] = h;
            for (int w = indexSpace.offset[1]; w < indexSpace.offset[1]+indexSpace.size[1]; w += 1)
            {
                output_coords[1] = w;
                for (int d = indexSpace.offset[0] * 256; d < indexSpace.offset[0]+indexSpace.size[0] * 256; d += 1)
                {
                   int filterCoords [] = {d,0,0};
                   output_coords[0] = d;
                   float accum = {0};
                   //this loop unroll purpose is to hide pipe latency
                   for (int kh = 0 ; kh <  layer_def.kernel_h; kh++)
                   {
                        filterCoords[2] = kh;
                        for (int kw = 0 ; kw <  layer_def.kernel_w; kw++)
                        {
                            filterCoords[1] = kw;
                            int ifmCoords []= { d,
                                (layer_def.stride_w*w) -layer_def.pad_w + (kw * layer_def.dilation_w),
                                (layer_def.stride_h*h) -layer_def.pad_h + (kh * layer_def.dilation_h),
                                 b};
                            // Load input and filter values
                            float filterValue = filter.ElementAt(filterCoords);
                            float ifmVector = ifm.ElementAt(ifmCoords);
                            // Multiply input * filter and store in accumalator
                            accum += filterValue*ifmVector;
                        }
                    }
                    // add bias
                    int mod256 = d % 256;
                    int biasCoord = (d & ~255) + ((64*(mod256%4)) + mod256/4);
                    int biasCoords [] = {biasCoord,0,0};

                    int32_t biasValue = bias.ElementAt(biasCoords);
                    accum += biasValue;

                    ofm.SetElement(output_coords,accum);
                }
            }
        }
    }
}

 inline int Filter2DI8W33S11Test::runTest()
 {
    // Initalize input size
    const int ifm_height = 10;
    const int ifm_width  = 8;
    const int ofmifm_depth = 100;
    const int batch = 1;

    SpatialReductionKernels::SpatialReduction2DDef layer_def;
    layer_def.pad_w = 1;
    layer_def.pad_h = 1;
    layer_def.kernel_h = 3;
    layer_def.kernel_w = 3;
    layer_def.stride_h = 1;
    layer_def.stride_w = 1;
    layer_def.dilation_w = 1;
    layer_def.dilation_h = 1;

    // Initalize inputs
    unsigned int ifmofmInitializer[] = {ofmifm_depth,ifm_width,ifm_height,batch};
    int8_4DTensor ifm(ifmofmInitializer);
    ifm.FillWithData(5);
    unsigned int filterInitialize[] = {(unsigned)ofmifm_depth,
                                       (unsigned)layer_def.kernel_w,
                                       (unsigned)layer_def.kernel_h};

    int8_3DTensor filter (filterInitialize);
    filter.FillWithData(2);

    // Adding filter impulse response tensor description
    int32_t biasContent[ofmifm_depth];
    std::fill_n(biasContent, ofmifm_depth, 0);
    biasContent[0] = 1;
    biasContent[1] = 2;
    biasContent[2] = 3;

    unsigned int biasInitializer[] = {ofmifm_depth};
    int32_1DTensor bias(biasInitializer, &biasContent[0]);

    int8_4DTensor ofm(ifmofmInitializer);
    int8_4DTensor ofm_ref(ifmofmInitializer);

    IndexSpace indexSpace = {{0}};
    int depthIS = (ofmifm_depth + 255) / 256 ;
    indexSpace.size[0] = depthIS;
    indexSpace.size[1] = ifm_width;
    indexSpace.size[2] = ifm_height;
    indexSpace.size[3] = batch;

    // Define input and output scale for quantization
    double inputScale = (1 << 5);
    double filtrScale = 1;
    double outputScale = (1 << 5);
    m_in_defs.inputTensors[0].quantizationParam.scale = inputScale;
    m_in_defs.inputTensors[1].quantizationParam.scale = filtrScale;
    m_in_defs.outputTensors[0].quantizationParam.scale = outputScale;
    double normalizationFactor = (inputScale * filtrScale) / outputScale;
    int8_t m_exponent = RealToFixedPointWeak(normalizationFactor);

    // execute reference implementation of the kernel.
    this->filter_2d_refence_implementation(ifm,
                                    filter,
                                    bias,
                                    ofm_ref,
                                    layer_def,
                                    indexSpace,
                                    m_exponent);

    // generate input for query call
    m_in_defs.NodeParams = &layer_def;
    m_in_defs.inputTensorNr = 3;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]),ifm );
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]),filter );
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[2]),bias );

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]),ofm );

    Filter2dI8W33S11 kernelClass;
    // make the call into the glue code.
    gcapi::GlueCodeReturn_t result = kernelClass.GetGcDefinitions(&m_in_defs,&m_out_defs);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "glue test failed!! " << result << std::endl;
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDescriptor> vec;
    vec.push_back(ifm.GetTensorDescriptor());
    vec.push_back(filter.GetTensorDescriptor());
    vec.push_back(bias.GetTensorDescriptor());
    vec.push_back(ofm.GetTensorDescriptor());

    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ofm.Print(1);
    ofm_ref.Print(1);
    for (int element = 0 ; element <  ofm_ref.ElementCount() ; element++)
    {
        if (ofm.Data()[element] != ofm_ref.Data()[element])
        {
            std::cout << "test failed!!" << element << std::endl;
            return -1;
        }
    }
    std::cout << "test pass!!" << std::endl;
    return 0;
 }

#endif /* FILTER_2D_I8_W33_S11_TEST_HPP */


