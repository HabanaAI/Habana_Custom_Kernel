/**********************************************************************
Copyright (c) 2021 Habana Labs.

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

#include "avg_pool_2d_fwd_f32_test.hpp"
#include "entry_points.hpp"

void AvgPool2DFwdF32Test::avg_pool_2d_fwd_reference_implementation(
        const test::Tensor<float,4>& ifm,
        test::Tensor<float,4>& ofm,
        const AvgPool2dFwdF32::AvgPool2DFwdParam& def,
        const IndexSpace& indexSpace)
{

    const int batchStart = indexSpace.offset[3];
    int       batchEnd   = indexSpace.offset[3] + indexSpace.size[3];

    const int heightStart = indexSpace.offset[2];
    const int heightEnd   = indexSpace.offset[2] + indexSpace.size[2];

    const int widthStart = indexSpace.offset[1];
    const int widthEnd   = indexSpace.offset[1] + indexSpace.size[1];

    const int channelStart = indexSpace.offset[0];
    const int channelEnd   = indexSpace.offset[0] + indexSpace.size[0];

    const int vectorSize = 256 / sizeof(float);

    int pixelsInArea = def.srdef.kernel_h * def.srdef.kernel_w;

    // Iterate over OFM
    for (int b = batchStart; b < batchEnd; b += 1)
    {
        for (int c = channelStart * vectorSize; c < channelEnd * vectorSize; c += 1)
        {
            for (int h = heightStart; h < heightEnd; h += 1)
            {
                for (int w = widthStart; w < widthEnd; w += 1)
                {
                    coord_t outCoord = {c, w, h, b};
                    int    intospacePixelsInArea = 0;
                    float  accum                 = 0;

                    for (int kh = 0; kh < def.srdef.kernel_h; kh++)
                    {
                        for (int kw = 0; kw < def.srdef.kernel_w; kw++)
                        {
                            coord_t in_coord;
                            in_coord.w = (outCoord.w * def.srdef.stride_w) + (kw * def.srdef.dilation_w) -
                                         def.srdef.pad_w;
                            in_coord.h = (outCoord.h * def.srdef.stride_h) + (kh * def.srdef.dilation_h) -
                                         def.srdef.pad_h;
                            in_coord.c = outCoord.c;
                            in_coord.b = outCoord.b;
                            bool intospace =
                                ((in_coord.w >= 0 && in_coord.w < (int)ifm.Size(1)) &&
                                 (in_coord.h >= 0 && in_coord.h < (int)ifm.Size(2)));

                            if (intospace || def.include_pads)
                            {
                                float ifm_data = ifm.ElementAt((int*)&in_coord);
                                accum += ifm_data;
                            }

                            if (intospace)
                            {
                                intospacePixelsInArea++;
                            }
                        }
                    }

                    // Store the number of pixels that inside the index space of the current
                    // area for future use when back propagating
                    //coord_t numOfSource_coord = {w, h, 0, 0};
                    //numOfSourcefm.SetElement((int*)&numOfSource_coord, intospacePixelsInArea);

                    // Divide the sum of the input pixels by the number of samples
                    int divider = def.include_pads ? (pixelsInArea) : (intospacePixelsInArea);

                    accum = (divider) ? (accum / (float)divider) : (0.0);
                    ofm.SetElement((int*)&outCoord, accum);
                }
            }
        }
    }
}

int AvgPool2DFwdF32Test::runTest()
{
    const int ifm_height = 5;
    const int ifm_width  = 5;
    const int ifm_depth = 100;
    const int ifm_batch = 1;

    AvgPool2dFwdF32::AvgPool2DFwdParam def;
    def.srdef.pad_w = 1;
    def.srdef.pad_h = 1;
    def.srdef.kernel_h = 3;
    def.srdef.kernel_w = 3;
    def.srdef.stride_h = 1;
    def.srdef.stride_w = 1;
    def.srdef.dilation_w = 1;
    def.srdef.dilation_h = 1;
    def.include_pads = 1;


    unsigned int ifmInitializer[] = {ifm_depth, ifm_width, ifm_height, ifm_batch};
    float_4DTensor ifm(ifmInitializer);
    ifm.FillWithData();

    int ofm_depth = ifm_depth;
    const int ofm_width = (ifm_width + def.srdef.pad_w - def.srdef.kernel_w*def.srdef.dilation_w)/def.srdef.stride_w;
    const int ofm_height = (ifm_height + def.srdef.pad_h - def.srdef.kernel_h*def.srdef.dilation_h)/def.srdef.stride_h;
    int ofm_batch = ifm_batch;
    if(ofm_width <= 0 || ofm_height <= 0)
    {
        std::cout << "Can't do average pooling, pooling size is too big !! " << std::endl;
        return -1;
    }

    unsigned int ofmInitializer[] = {ifm_depth, (unsigned int)ofm_width, (unsigned int)ofm_height, ifm_batch};
    
    float_4DTensor ofm(ofmInitializer);
    float_4DTensor ofm_ref(ofmInitializer);

    IndexSpace indexSpace = {{0}};
    int depthIS = (ofm_depth + 63) / 64 ;
    indexSpace.size[0] = depthIS;
    indexSpace.size[1] = ofm_width;
    indexSpace.size[2] = ofm_height;
    indexSpace.size[3] = ofm_batch;

    // execute reference implementation of the kernel.
    avg_pool_2d_fwd_reference_implementation(ifm, ofm_ref, def, indexSpace);

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI;
    m_in_defs.NodeParams = &def;
    m_in_defs.inputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), ifm);

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), ofm);

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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI_KERNEL_AVG_POOL_2D_FWD_F32]);
    result  = HabanaKernel(&m_in_defs,&m_out_defs);

    // Declaration of auxiliary tensor
    float_1DTensor aux_tensor({100});
    // Allocate memory for aux tensor if not allocated
    if (result == gcapi::GLUE_INSUFICIENT_AUX_BUFFER_SIZE)
    {
        if (m_out_defs.auxiliaryTensors[0].pData)
        {
            delete [] (int8_t*)m_out_defs.auxiliaryTensors[0].pData;
            m_out_defs.auxiliaryTensors[0].pData = NULL;
        }

        m_out_defs.auxiliaryTensors[0].pData =
                                    new float[m_out_defs.auxiliaryTensors[0].bufferSize / sizeof(float)];
        // second call of glue-code to load Auxiliary data.
        result  = HabanaKernel(&m_in_defs,&m_out_defs);
        // AUXILIARY TENSOR init based on parameters got from glue code
        aux_tensor.Init(m_out_defs.auxiliaryTensors[0].geometry.sizes,
                                    (float*)m_out_defs.auxiliaryTensors[0].pData);
    }

    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDescriptor> vec;
    vec.push_back(ifm.GetTensorDescriptor());
    vec.push_back(ofm.GetTensorDescriptor());
    vec.push_back(aux_tensor.GetTensorDescriptor());
    
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
        if (abs(ofm.Data()[element] - ofm_ref.Data()[element]) > 1e-6)
        {
            std::cout << "AvgPool2DFwdF32Test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "AvgPool2DFwdF32Test pass!!" << std::endl;
    return 0;
}


