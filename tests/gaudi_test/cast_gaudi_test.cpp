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

#include "cast_gaudi_test.hpp"
#include "entry_points.hpp"

void CastGaudiTest::cast_bf16_to_f32_ref(
         const test::Tensor<bfloat16,4>& input,
         test::Tensor<float,4>& output,
         const IndexSpace& indexSpace)
{
    int eig =128;

    int coords[4] = { 0 };
    for (int d = indexSpace.offset[0]; d < (indexSpace.offset[0] + indexSpace.size[0]) * eig; d += 1)
    {
        coords[0] = d;
        for (int b = indexSpace.offset[3]; b < indexSpace.offset[3] + indexSpace.size[3]; b += 1)
        {
            coords[3] = b;
            for (int h = indexSpace.offset[2]; h < indexSpace.offset[2] + indexSpace.size[2]; h += 1)
            {
                coords[2] = h;
                for (int w = indexSpace.offset[1]; w < indexSpace.offset[1] + indexSpace.size[1]; w += 1)
                {
                    coords[1] = w;
                    float ofmVal;
                    bfloat16 ifmVal = input.ElementAt(coords);
                    ofmVal = bf16ToFloat(ifmVal.val);
                    output.SetElement(coords, ofmVal);
                }
            }
        }
    }
}

void CastGaudiTest::cast_f32_to_bf16_ref(
    const test::Tensor<float,4>& ifm,
    test::Tensor<bfloat16,4>& ofm,
    const IndexSpace& indexSpace)
{
    int eig = 128;

    int coords[4] = { 0 };
    for (int d = indexSpace.offset[0]; d < (indexSpace.offset[0] + indexSpace.size[0]) * eig; d += 1)
    {
        coords[0] = d;
        for (int b = indexSpace.offset[3]; b < indexSpace.offset[3] + indexSpace.size[3]; b += 1)
        {
            coords[3] = b;
            for (int h = indexSpace.offset[2]; h < indexSpace.offset[2] + indexSpace.size[2]; h += 1)
            {
                coords[2] = h;
                for (int w = indexSpace.offset[1]; w < indexSpace.offset[1] + indexSpace.size[1]; w += 1)
                {
                    coords[1] = w;
                    float ifmVal = ifm.ElementAt(coords);

                    bfloat16 ofmVal = floatTobf16ToFloat(ifmVal);
                    ofm.SetElement(coords, ofmVal);
                }
            }
        }
    }
}

 int CastGaudiTest::runTest()
 {

    /**********************Test for cast bf16 to f32************************/
    // Initalize input size
    const int ifm_height = 10;
    const int ifm_width  = 8;
    const int ofmifm_depth = 300;
    const int batch = 1;


    // Initalize inputs
    unsigned int ifmofmInitializer[] = {ofmifm_depth,ifm_width,ifm_height,batch};
    bfloat16_4DTensor ifm(ifmofmInitializer);
    ifm.FillWithData();
    float_4DTensor ofm(ifmofmInitializer);
    float_4DTensor ofm_ref(ifmofmInitializer);

    IndexSpace indexSpace = {{0}};
    int depthIS = (ofmifm_depth + 127) / 128 ;
    indexSpace.size[0] = depthIS;
    indexSpace.size[1] = ifm_width;
    indexSpace.size[2] = ifm_height;
    indexSpace.size[3] = batch;

    // Define input and output scale for quantization
    float scale = m_in_defs.inputTensors[0].quantizationParam.scale = 1.0;
    CastGaudi::CastParams def;
    def.scale = scale;

    // execute reference implementation of the kernel.
    this->cast_bf16_to_f32_ref(ifm,
                               ofm_ref,
                               indexSpace);

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI;
    m_in_defs.NodeParams = &def;
    m_in_defs.inputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]),ifm );

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]),ofm );

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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI_KERNEL_CAST_BF16_F32]);
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
    vec.push_back(ofm.GetTensorDescriptor());

    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ofm.Print(0);
    ofm_ref.Print(0);
    for (int element = 0 ; element <  ofm_ref.ElementCount() ; element++)
    {
        if (ofm.Data()[element] != ofm_ref.Data()[element])
        {
            std::cout << "Cast BF16_2_F32 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "Cast BF16_2_F32 test pass!!" << std::endl;


    /**********************Test for cast f32 to bf16************************/
    float_4DTensor input(ifmofmInitializer);
    input.FillWithData();

    bfloat16_4DTensor out(ifmofmInitializer);
    bfloat16_4DTensor out_ref(ifmofmInitializer);

    depthIS = (ofmifm_depth + 63) / 64 ;
    indexSpace.size[0] = depthIS;

    // execute reference implementation of the kernel.
    this->cast_f32_to_bf16_ref(input,
                              out_ref,
                              indexSpace);

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI;
    m_in_defs.NodeParams = &def;
    m_in_defs.inputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]),input );

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]),out );

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI_KERNEL_CAST_F32_BF16]);
    result  = HabanaKernel(&m_in_defs,&m_out_defs);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }    

    // generate and load tensor descriptors
    std::vector<TensorDesc> vec1;
    vec1.push_back(input.GetTensorDescriptor());
    vec1.push_back(out.GetTensorDescriptor());

    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec1, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount);
    out.Print(0);
    out_ref.Print(0);
    for (int element = 0 ; element <  out_ref.ElementCount() ; element++)
    {
        if (out.Data()[element] != out_ref.Data()[element])
        {
            std::cout << "Cast F32_2_BF16 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "Cast F32_2_BF16 test pass!!" << std::endl;
    return 0;
 }

