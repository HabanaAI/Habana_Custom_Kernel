/**********************************************************************
Copyright (c) 2021 Habana Labs.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENcTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include "spatial_conv_f32_test.hpp"
#include "entry_points.hpp"

 void SpatialConvF32Test::spatial_conv_reference_implementation(
        const test::Tensor<float,4>& ifm,
        const test::Tensor<float,4>& filter,
        test::Tensor<float,4>& ofm,
        const SpatialReductionKernels::SpatialReduction2DDef& layer_def,
        const IndexSpace& indexSpace)
{
    int channelSize = ifm.Size(0);
    int output_coords[4] = {0};
    for (int b = indexSpace.offset[4]; b < indexSpace.offset[4]+indexSpace.size[4]; b += 1)
    {
        output_coords[3] = b;
        for (int h = indexSpace.offset[3]; h < indexSpace.offset[3]+indexSpace.size[3]; h += 1)
        {
            output_coords[2] = h;
            for (int w = indexSpace.offset[2]; w < indexSpace.offset[2]+indexSpace.size[2]; w += 1)
            {
                output_coords[1] = w;
                for(int k = indexSpace.offset[1]; k < indexSpace.offset[1]+indexSpace.size[1]; k += 1)
                {
                    output_coords[0] = k;
                    float accum_all = {0};
                    for (int d = 0 ; d < channelSize; d += 1)
                    {
                         int filterCoords [] = {d,k,0,0};
                        float accum = {0};
                        for (int kh = 0 ; kh <  layer_def.kernel_h; kh++)
                        {
                             filterCoords[3] = kh;
                             for (int kw = 0 ; kw <  layer_def.kernel_w; kw++)
                            {
                                 filterCoords[2] = kw;
                                 int ifmCoords []= {d,
                                 (layer_def.stride_w*w) -layer_def.pad_w + (kw * layer_def.dilation_w),
                                  (layer_def.stride_h*h) -layer_def.pad_h + (kh * layer_def.dilation_h),
                                  b};
                                  float filterValue = filter.ElementAt(filterCoords);
                                  float ifmVector = ifm.ElementAt(ifmCoords);
                                  accum += filterValue*ifmVector;
                             }
                         }
                         accum_all += accum;
                     }
                     ofm.SetElement(output_coords,accum_all);
                 }
             }
         }
     }
}

 int SpatialConvF32Test::runTest()
 {
    const int fm_height = 3;
    const int fm_width  = 3;
    const int fm_depth = 2;
    const int fm_batch = 1;
    const int fm_filterK = 1; //num of filters

    SpatialReductionKernels::SpatialReduction2DDef layer_def;
    layer_def.pad_w = 0;
    layer_def.pad_h = 0;
    layer_def.kernel_h = 2;
    layer_def.kernel_w = 2;
    layer_def.stride_h = 1;
    layer_def.stride_w = 1;
    layer_def.dilation_w = 1;
    layer_def.dilation_h = 1;

    //input
    unsigned int ifmInitializer[] = {fm_depth, fm_width, fm_height, fm_batch};
    float_4DTensor ifm(ifmInitializer);
    //ifm.FillWithValue(1);
    ifm.FillWithData();

    //filter
    unsigned int filterInitialize[] = {(unsigned)fm_depth,
                                       (unsigned)fm_filterK,
                                       (unsigned)layer_def.kernel_w,
                                       (unsigned)layer_def.kernel_h};

    float_4DTensor filter (filterInitialize);
    //filter.FillWithValue(1);
    filter.FillWithData();

    //output
    const unsigned int ofm_w = ((fm_width + 2 * layer_def.pad_w - layer_def.dilation_w * (layer_def.kernel_w-1) - 1) / layer_def.stride_w) + 1;
    const unsigned int ofm_h = ((fm_height+ 2 * layer_def.pad_h - layer_def.dilation_h * (layer_def.kernel_h-1) - 1) / layer_def.stride_h) + 1;
    unsigned int ofmInitializer[] = {(unsigned)fm_filterK, ofm_w, ofm_h, fm_batch};
    float_4DTensor ofm(ofmInitializer);
    float_4DTensor ofm_ref(ofmInitializer);

    IndexSpace indexSpace = {{0}};
    indexSpace.size[0] = 1;
    indexSpace.size[1] = fm_filterK;
    indexSpace.size[2] = ofm_w;
    indexSpace.size[3] = ofm_h;
    indexSpace.size[4] = fm_batch;

    // execute reference implementation of the kernel.
    spatial_conv_reference_implementation(ifm,
                                    filter,
                                    ofm_ref,
                                    layer_def,
                                    indexSpace);

    // generate input for query call
    m_in_defs.NodeParams = &layer_def;
    m_in_defs.inputTensorNr = 2;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]),ifm );
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]),filter );

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]),ofm );

    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI;

    char**   kernelNames = nullptr;
    unsigned kernelCount = 0;
    gcapi::GlueCodeReturn_t result = GetKernelNames(kernelNames, &kernelCount, gcapi::DEVICE_ID_GAUDI);
    kernelNames = new char*[kernelCount];
    for (unsigned i = 0; i < kernelCount; i++)
    {
        kernelNames[i] = new char[gcapi::MAX_NODE_NAME];
    }    
    result = GetKernelNames(kernelNames, &kernelCount, gcapi::DEVICE_ID_GAUDI);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Can't get kernel name!! " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI_KERNEL_SPATIAL_CONV_F32]);
    result  = HabanaKernel(&m_in_defs,&m_out_defs);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc> vec;
    vec.push_back(ifm.GetTensorDescriptor());
    vec.push_back(filter.GetTensorDescriptor());
    vec.push_back(ofm.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount);
    std::cout << std::endl;
    std::cout << "ofm data shown below " << std::endl;
    ofm.Print(0);
    std::cout << std::endl;
    std::cout << "ofm_ref data shown below " << std::endl;
    ofm_ref.Print(0);
    for (int element = 0 ; element <  ofm_ref.ElementCount() ; element++)
    {
        if (ofm.Data()[element] != ofm_ref.Data()[element])
        {
            std::cout << "SpatialConvF32Test failed!!" << std::endl;
            return -1;
        }
    }

    std::cout << "SpatialConvF32Test pass!!" << std::endl;
    std::cout << std::endl;
    return 0;
 }
