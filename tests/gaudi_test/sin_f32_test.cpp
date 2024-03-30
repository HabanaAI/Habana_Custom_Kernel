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

#include "sin_f32_test.hpp"
#include "entry_points.hpp"

void SinF32Test::sinf32_reference_implementation(
        const float_5DTensor& input,
        float_5DTensor& output)
{
    int coords[5] = {0};
    for (unsigned r4 = 0; r4 < input.Size(4); r4 += 1)
    {
        coords[4] = r4;
        for (unsigned b = 0; b < input.Size(3); b += 1)
        {
            coords[3] = b;
            for (unsigned h = 0; h < input.Size(2); h += 1)
            {
                coords[2] = h;
                for (unsigned w = 0; w < input.Size(1); w += 1)
                {
                    coords[1] = w;
                    for (unsigned d = 0; d < input.Size(0); d += 1)
                    {
                        coords[0] = d;
                        float x = input.ElementAt(coords);
                        float y = sin(x);
                        output.SetElement(coords, y);
                    }
                }
            }
        }
    }
}

int SinF32Test::runTest()
{
    const int height = 5;
    const int width  = 5;
    const int depth  = 100;
    const int batch  = 1;
    const int rank4  = 2;

    uint64_t fmInitializer[] = {depth, width, height, batch, rank4};

    float_5DTensor input(fmInitializer);
    input.InitRand(-10.0f, 10.0f);

    float_5DTensor output(fmInitializer);
    float_5DTensor output_ref(fmInitializer);

    // execute reference implementation of the kernel.
    sinf32_reference_implementation(input, output_ref);

    // generate input for query call
    m_in_defs.deviceId = tpc_lib_api::DEVICE_ID_GAUDI;

    m_in_defs.inputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), input);

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), output);

    tpc_lib_api::GuidInfo *guids = nullptr;
    unsigned kernelCount = 0;
    tpc_lib_api::GlueCodeReturn result = GetKernelGuids(tpc_lib_api::DEVICE_ID_GAUDI, &kernelCount, guids);
    guids = new tpc_lib_api::GuidInfo[kernelCount];
    result = GetKernelGuids(tpc_lib_api::DEVICE_ID_GAUDI, &kernelCount, guids);
    if (result != tpc_lib_api::GLUE_SUCCESS)
    {
        std::cout << "Can't get kernel name!! " << result << std::endl;
        ReleaseKernelNames(guids, kernelCount);
        return -1;
    }

    strcpy(m_in_defs.guid.name, guids[GAUDI_KERNEL_SIN_F32].name);
    result  = InstantiateTpcKernel(&m_in_defs,&m_out_defs);
    if (result != tpc_lib_api::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(guids, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc2> vec;
    vec.push_back(input.GetTensorDescriptor());
    vec.push_back(output.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(guids, kernelCount);
    output.Print(0);
    output_ref.Print(0);
    for (int element = 0 ; element <  output_ref.ElementCount() ; element++)
    {
        if (abs(output.Data()[element] - output_ref.Data()[element]) > 1e-6)
        {
            std::cout << "Sin F32 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "Sin F32 test pass!!" << std::endl;
    return 0;
}

