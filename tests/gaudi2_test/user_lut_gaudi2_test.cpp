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

#include "user_lut_gaudi2_test.hpp"
#include "entry_points.hpp"

void UserLutGaudi2Test::user_lut_reference_implementation(
        const float_1DTensor& input0,
        const float_1DTensor& input1,
        float_1DTensor& output)
{
    float c[] {0.500000, 1.500000, 2.500000, 3.500000, 4.500000, 5.500000, 6.500000, 7.500000, 8.500000, 9.500000, 10.500000, 11.500000, 12.500000, 13.500000, 14.500000, 15.500000, 16.500000, 17.500000, 18.500000, 19.500000, 20.500000, 21.500000, 22.500000, 23.500000, 24.500000, 25.500000, 26.500000, 27.500000, 28.500000, 29.500000, 30.500000, 31.500000, 0.500000, 1.500000, 2.500000, 3.500000, 4.500000, 5.500000, 6.500000, 7.500000, 8.500000, 9.500000, 10.500000, 11.500000, 12.500000, 13.500000, 14.500000, 15.500000, 16.500000, 17.500000, 18.500000, 19.500000, 20.500000, 21.500000, 22.500000, 23.500000, 24.500000, 25.500000, 26.500000, 27.500000, 28.500000, 29.500000, 30.500000, 31.500000, };
    float c0[] {0.100000, 1.100000, 2.100000, 3.100000, 4.100000, 5.100000, 6.100000, 7.100000, 8.100000, 9.100000, 10.100000, 11.100000, 12.100000, 13.100000, 14.100000, 15.100000, 16.100000, 17.100000, 18.100000, 19.100000, 20.100000, 21.100000, 22.100000, 23.100000, 24.100000, 25.100000, 26.100000, 27.100000, 28.100000, 29.100000, 30.100000, 31.100000, 0.100000, 1.100000, 2.100000, 3.100000, 4.100000, 5.100000, 6.100000, 7.100000, 8.100000, 9.100000, 10.100000, 11.100000, 12.100000, 13.100000, 14.100000, 15.100000, 16.100000, 17.100000, 18.100000, 19.100000, 20.100000, 21.100000, 22.100000, 23.100000, 24.100000, 25.100000, 26.100000, 27.100000, 28.100000, 29.100000, 30.100000, 31.100000}; 
    float c1[] {0.200000, 1.200000, 2.200000, 3.200000, 4.200000, 5.200000, 6.200000, 7.200000, 8.200000, 9.200000, 10.200000, 11.200000, 12.200000, 13.200000, 14.200000, 15.200000, 16.200001, 17.200001, 18.200001, 19.200001, 20.200001, 21.200001, 22.200001, 23.200001, 24.200001, 25.200001, 26.200001, 27.200001, 28.200001, 29.200001, 30.200001, 31.200001, 0.200000, 1.200000, 2.200000, 3.200000, 4.200000, 5.200000, 6.200000, 7.200000, 8.200000, 9.200000, 10.200000, 11.200000, 12.200000, 13.200000, 14.200000, 15.200000, 16.200001, 17.200001, 18.200001, 19.200001, 20.200001, 21.200001, 22.200001, 23.200001, 24.200001, 25.200001, 26.200001, 27.200001, 28.200001, 29.200001, 30.200001, 31.200001};
    float c2[] {0.300000, 1.300000, 2.300000, 3.300000, 4.300000, 5.300000, 6.300000, 7.300000, 8.300000, 9.300000, 10.300000, 11.300000, 12.300000, 13.300000, 14.300000, 15.300000, 16.299999, 17.299999, 18.299999, 19.299999, 20.299999, 21.299999, 22.299999, 23.299999, 24.299999, 25.299999, 26.299999, 27.299999, 28.299999, 29.299999, 30.299999, 31.299999, 0.300000, 1.300000, 2.300000, 3.300000, 4.300000, 5.300000, 6.300000, 7.300000, 8.300000, 9.300000, 10.300000, 11.300000, 12.300000, 13.300000, 14.300000, 15.300000, 16.299999, 17.299999, 18.299999, 19.299999, 20.299999, 21.299999, 22.299999, 23.299999, 24.299999, 25.299999, 26.299999, 27.299999, 28.299999, 29.299999, 30.299999, 31.299999};

    int coords[5] = {0};
    for (unsigned d = 0; d < 64; d += 1) 
    {
        coords[0] = d;
        float x0 = input0.ElementAt(coords);
        float x1 = input1.ElementAt(coords);
        float res = c0[d] + x0 * c1[d] + x1 * c2[d];
        res *= c[d];
        output.SetElement(coords, res);
    }
}


int UserLutGaudi2Test::runTest()
{
    uint64_t fmInitializer[] = {64, 1, 1, 1, 1};

    float_1DTensor input0(fmInitializer);
    input0.InitRand(-10.0f, 10.0f);

    float_1DTensor input1(fmInitializer);
    input1.InitRand(-10.0f, 10.0f);

    float_1DTensor output(fmInitializer);
    float_1DTensor output_ref(fmInitializer);

    // execute reference implementation of the kernel.
    user_lut_reference_implementation(input0, input1, output_ref);

    // generate input for query call
    m_in_defs.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;

    m_in_defs.inputTensorNr = 2;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), input0);
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), input1);

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), output);

    tpc_lib_api::GuidInfo *guids = nullptr;
    unsigned kernelCount = 0;
    tpc_lib_api::GlueCodeReturn result = GetKernelGuids(tpc_lib_api::DEVICE_ID_GAUDI2, &kernelCount, guids);
    guids = new tpc_lib_api::GuidInfo[kernelCount];
    result = GetKernelGuids(tpc_lib_api::DEVICE_ID_GAUDI2, &kernelCount, guids);
    if (result != tpc_lib_api::GLUE_SUCCESS)
    {
        std::cout << "Can't get kernel name!! " << result << std::endl;
        ReleaseKernelNames(guids, kernelCount);
        return -1;
    }

    strcpy(m_in_defs.guid.name, guids[GAUDI2_KERNEL_USER_LUT].name);
    result  = InstantiateTpcKernel(&m_in_defs,&m_out_defs);

    // Declaration of auxiliary tensor
    float_1DTensor aux_tensor({64});
    // Allocate memory for aux tensor if not allocated
    if (result == tpc_lib_api::GLUE_INSUFFICIENT_AUX_BUFFER_SIZE)
    {
        if (m_out_defs.auxiliaryTensors[0].pData)
        {
            delete [] (int8_t*)m_out_defs.auxiliaryTensors[0].pData;
            m_out_defs.auxiliaryTensors[0].pData = NULL;
        }

        // allocate memory for aux tensor
        m_out_defs.auxiliaryTensors[0].pData = new float[m_out_defs.auxiliaryTensors[0].bufferSize/ sizeof(float)];

        // second call of glue-code to load Auxiliary data.
        result  = InstantiateTpcKernel(&m_in_defs,&m_out_defs);

        // reinit with data from glue code
        aux_tensor.Init(m_out_defs.auxiliaryTensors[0].geometry.maxSizes,
                                    (float*)m_out_defs.auxiliaryTensors[0].pData);
    }

    if (result != tpc_lib_api::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(guids, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc2> vec;
    vec.push_back(input0.GetTensorDescriptor());
    vec.push_back(input1.GetTensorDescriptor());
    vec.push_back(output.GetTensorDescriptor());
    vec.push_back(aux_tensor.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(guids, kernelCount);

    std::cout << std::endl << "tensors content:" << std::endl;
    std::cout << "aux buffer: ";
    aux_tensor.Print();
    std::cout << "input0: ";
    input0.Print(0);
    std::cout << "output: ";
    output.Print(0);
    std::cout << "reference: ";
    output_ref.Print(0);
    for (int element = 0 ; element <  output_ref.ElementCount() ; element++)
    {
        if (abs(output.Data()[element] - output_ref.Data()[element]) > 1e-6)
        {
            std::cout << "UserLut F32 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "user lut test pass!!" << std::endl;
    return 0;
}
