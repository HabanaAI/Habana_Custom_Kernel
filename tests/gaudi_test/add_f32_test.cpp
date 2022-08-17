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

#include "add_f32_test.hpp"
#include "entry_points.hpp"

void AddF32Test::addf32_reference_implementation(
        const float_5DTensor& input0,
        const float_5DTensor& input1,
        float_5DTensor& output)
{
   int coords[5] = {0};
    for (unsigned r4 = 0; r4 < input0.Size(4); r4 += 1)
    {
        coords[4] = r4;
        for (unsigned b = 0; b < input0.Size(3); b += 1)
        {
            coords[3] = b;
            for (unsigned h = 0; h < input0.Size(2); h += 1)
            {
                coords[2] = h;
                for (unsigned w = 0; w < input0.Size(1); w += 1)
                {
                    coords[1] = w;
                    for (unsigned d = 0; d < input0.Size(0); d += 1)
                    {
                        coords[0] = d;
                        float x0 = input0.ElementAt(coords);
                        float x1 = input1.ElementAt(coords);
                        float y = x0 + x1;
                        output.SetElement(coords, y);
                    }
                }
            }
        }
    }
}

int AddF32Test::runTest()
{
    const int height = 5;
    const int width  = 5;
    const int depth  = 100;
    const int batch  = 1;
    const int rank4  = 2;

    unsigned int fmInitializer[] = {depth, width, height, batch, rank4};

    float_5DTensor input0(fmInitializer);
    input0.InitRand(-10.0f, 10.0f);

    float_5DTensor input1(fmInitializer);
    input1.InitRand(-10.0f, 10.0f);

    float_5DTensor output(fmInitializer);
    float_5DTensor output_ref(fmInitializer);

    // execute reference implementation of the kernel.
    addf32_reference_implementation(input0, input1, output_ref);

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI;

    m_in_defs.inputTensorNr = 2;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), input0);
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), input1);

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), output);

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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI_KERNEL_ADD_F32]);
    result  = HabanaKernel(&m_in_defs,&m_out_defs);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc> vec;
    vec.push_back(input0.GetTensorDescriptor());
    vec.push_back(input1.GetTensorDescriptor());
    vec.push_back(output.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount);
    output.Print(0);
    output_ref.Print(0);
    for (int element = 0 ; element <  output_ref.ElementCount() ; element++)
    {
        if (abs(output.Data()[element] - output_ref.Data()[element]) > 1e-6)
        {
            std::cout << "Add F32 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "Add F32 test pass!!" << std::endl;
    return 0;
}

