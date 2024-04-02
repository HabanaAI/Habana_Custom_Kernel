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

#include "matrix_mul_fwd_f32_test.hpp"
#include "entry_points.hpp"

void MatrixMulFwdF32Test::matrix_mul_reference_implementation(
        const float_3DTensor& input0,
        const float_3DTensor& input1,
        float_3DTensor& output)
{
    uint32_t batch_size = std::max(output.Size(2), 1u);

    for (int32_t batch = 0; batch < (int32_t)batch_size; batch++)
    {
        for (int32_t row = 0; row < (int32_t)output.Size(1); row++)
        {
            for (int32_t col = 0; col < (int32_t)output.Size(0); col++)
            {
                float accum = 0.0f;

                for (int32_t common = 0; common < (int32_t)input0.Size(0); common++)
                {
                    int32_t a_coord[] = {common, row, batch};
                    int32_t b_coord[] = {col, common, batch};

                    float a_val = input0.ElementAt(a_coord);
                    float b_val = input1.ElementAt(b_coord);

                    // Call templated mac to match the precision of TPC
                    //Ops::Mac<float, float>(&a_val, &b_val, &accum, &accum, NULL);
                    accum = std::fmaf(a_val, b_val, accum);
                }
                int32_t c_coord[] = {col, row, batch};

                output.SetElement(c_coord, accum);
            }
        }
    }
}

int MatrixMulFwdF32Test::runTest()
{
    const int col = 65;
    const int row  = 6;
    const int common = 4;
    const int batch  = 1;

    uint64_t fmInitializer_a[] = {common, row, batch};
    uint64_t fmInitializer_b[] = {col, common, batch};
    uint64_t fmInitializer_c[] = {col, row, batch};

    float_3DTensor a_matrix(fmInitializer_a);
    a_matrix.InitRand(1.0f, 10.0f);

    float_3DTensor b_matrix(fmInitializer_b);
    b_matrix.InitRand(1.0f, 10.0f);

    float_3DTensor c_matrix(fmInitializer_c);
    float_3DTensor c_matrix_ref(fmInitializer_c);

    // execute reference implementation of the kernel.
    matrix_mul_reference_implementation(a_matrix, b_matrix, c_matrix_ref);

    // generate input for query call
    m_in_defs.deviceId = tpc_lib_api::DEVICE_ID_GAUDI;

    m_in_defs.inputTensorNr = 2;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), a_matrix);
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), b_matrix);

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), c_matrix);

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

    strcpy(m_in_defs.guid.name, guids[GAUDI_KERNEL_MATRIXMUL_FWD_F32].name);
    result  = InstantiateTpcKernel(&m_in_defs,&m_out_defs);
    if (result != tpc_lib_api::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(guids, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc2> vec;
    vec.push_back(a_matrix.GetTensorDescriptor());
    vec.push_back(b_matrix.GetTensorDescriptor());
    vec.push_back(c_matrix.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(guids, kernelCount);
    c_matrix.Print(0);
    c_matrix.Print(1);
    c_matrix.Print(2);
    c_matrix_ref.Print(0);
    for (int element = 0 ; element <  c_matrix_ref.ElementCount() ; element++)
    {
        if (abs(c_matrix.Data()[element] - c_matrix_ref.Data()[element]) > 1e-6)
        {
            std::cout << "Matrix multiply FWD F32 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "Matrix multiply FWD F32 test pass!!" << std::endl;
    return 0;
}

