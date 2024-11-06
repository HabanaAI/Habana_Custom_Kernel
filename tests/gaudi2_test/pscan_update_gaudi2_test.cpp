/**********************************************************************
Copyright (c) 2024 Habana Labs.

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

#include "pscan_update_gaudi2_test.hpp"
#include <type_traits>

void PscanUpdateGaudi2Test::pscan_update_fp32_ref(
         const test::Tensor<float,4>& state_M,
         const test::Tensor<float,4>& x_M,
         const test::Tensor<float,4>& C_M,
         const test::Tensor<float,4>& D_M,
         const test::Tensor<float,4>& z_M,
         test::Tensor<float,4>& output,
         const IndexSpace& indexSpace)
{
    int eig = 64;

    int coords_state[5] = { 0 };
    int coords_x[5] = { 0 };
    int coords_C[5] = { 0 };
    int coords_D[5] = { 0 };
    int coords_z[5] = { 0 };
    int coords_out[5] = { 0 };
    for (int b = indexSpace.offset[3]; b < (indexSpace.offset[3] + indexSpace.size[3]); b += 1)
    {
        coords_state[3] = b;
        coords_x[3] = b;
        coords_C[3] = b;
        coords_D[3] = 0;
        coords_z[3] = b;
        coords_out[3] = b;
        for (int h = indexSpace.offset[2]; h < indexSpace.offset[2] + indexSpace.size[2]; h += 1)
        {
            coords_state[2] = h;
            coords_x[2] = h;
            coords_C[2] = h;
            coords_D[2] = 0;
            coords_z[2] = h;
            coords_out[2] = h;

            for (int d = indexSpace.offset[0]; d < (indexSpace.offset[0] + indexSpace.size[0]) * eig; d += 1)
            {
                coords_state[0] = d;
                coords_x[0] = d;
                coords_C[0] = 0;
                coords_D[0] = d;
                coords_z[0] = d;
                coords_out[0] = d;

                float ofmVal = 0;
                float ifmVal_x = x_M.ElementAt(coords_x);
                for (int n = indexSpace.offset[1]; n < indexSpace.offset[1] + indexSpace.size[1]; n += 1)
                {
                    coords_state[1] = n;
                    coords_x[1] = 0;
                    coords_C[1] = n;
                    coords_D[1] = 0;
                    coords_z[1] = 0;
                    coords_out[1] = 0;
                    
                    float ifmVal_state = state_M.ElementAt(coords_state);
                    float ifmVal_C = C_M.ElementAt(coords_C);

                    float tmp_out = ifmVal_state * ifmVal_C;
                    ofmVal += tmp_out;

                }
                float ifmVal_D = D_M.ElementAt(coords_D);
                ofmVal += ifmVal_D * ifmVal_x;

                float ifmVal_z = z_M.ElementAt(coords_z);
                float z_temp = ifmVal_z / (1 + expf(-ifmVal_z));
                ofmVal = ofmVal * z_temp;

                output.SetElement(coords_out, ofmVal);
            }
        }
    }
}

void PscanUpdateGaudi2Test::pscan_update_bf16_ref(
         const test::Tensor<bfloat16,4>& state_M,
         const test::Tensor<bfloat16,4>& x_M,
         const test::Tensor<bfloat16,4>& C_M,
         const test::Tensor<bfloat16,4>& D_M,
         const test::Tensor<bfloat16,4>& z_M,
         test::Tensor<bfloat16,4>& output,
         const IndexSpace& indexSpace)
{
    int eig = 128;
    
    int coords_state[5] = { 0 };
    int coords_x[5] = { 0 };
    int coords_C[5] = { 0 };
    int coords_D[5] = { 0 };
    int coords_z[5] = { 0 };
    int coords_out[5] = { 0 };
    for (int b = indexSpace.offset[3]; b < (indexSpace.offset[3] + indexSpace.size[3]); b += 1)
    {
        coords_state[3] = b;
        coords_x[3] = b;
        coords_C[3] = b;
        coords_D[3] = 0;
        coords_z[3] = b;
        coords_out[3] = b;
        for (int h = indexSpace.offset[2]; h < indexSpace.offset[2] + indexSpace.size[2]; h += 1)
        {
            coords_state[2] = h;
            coords_x[2] = h;
            coords_C[2] = h;
            coords_D[2] = 0;
            coords_z[2] = h;
            coords_out[2] = h;

            for (int d = indexSpace.offset[0]; d < (indexSpace.offset[0] + indexSpace.size[0]) * eig; d += 1)
            {
                coords_state[0] = d;
                coords_x[0] = d;
                coords_C[0] = 0;
                coords_D[0] = d;
                coords_z[0] = d;
                coords_out[0] = d;

                float ofmVal = 0;
                float ifmVal_x = (float)x_M.ElementAt(coords_x);
                for (int n = indexSpace.offset[1]; n < indexSpace.offset[1] + indexSpace.size[1]; n += 1)
                {
                    coords_state[1] = n;
                    coords_x[1] = 0;
                    coords_C[1] = n;
                    coords_D[1] = 0;
                    coords_z[1] = 0;
                    coords_out[1] = 0;
                    
                    float ifmVal_state = (float)state_M.ElementAt(coords_state);
                    float ifmVal_C = (float)C_M.ElementAt(coords_C);

                    float tmp_out = ifmVal_state * ifmVal_C;
                    ofmVal += tmp_out;

                }
                float ifmVal_D = (float)D_M.ElementAt(coords_D);
                ofmVal += ifmVal_D * ifmVal_x;
                float ifmVal_z = (float)z_M.ElementAt(coords_z);
                    
                float ttemp1 = floatTobf16ToFloat(ifmVal_z);
                float z_temp = ttemp1 / (1 + expf(-ttemp1));

                ofmVal = ofmVal * z_temp;

                output.SetElement(coords_out, ofmVal);
            }
        }
    }
}

int PscanUpdateGaudi2Test::runTest(Gaudi2_Kernel_Name_e NameofKernel)
 {

    // Initalize input size
    const int ofmifm_dim = 12;
    const int ifm_dstate = 10;
    const int ifm_seq  = 8;
    const int batch = 2;

    // Initalize inputs
    uint64_t ifm_state_Initializer[] = {ofmifm_dim, ifm_dstate, ifm_seq, batch};
    uint64_t ifm_x_dt_z_Initializer[] = {ofmifm_dim, 1, ifm_seq, batch};
    uint64_t ifm_B_C_Initializer[] = {1, ifm_dstate, ifm_seq, batch};
    uint64_t ifm_D_dtbias_Initializer[] = {ofmifm_dim, 1, 1, 1};
    uint64_t ofm_out_Initializer[] = {ofmifm_dim, 1, ifm_seq, batch};
    
    if(NameofKernel == GAUDI2_KERNEL_PSCAN_UPDATE_F32)
    {
        float_4DTensor ifm_state(ifm_state_Initializer);
        float_4DTensor ifm_x(ifm_x_dt_z_Initializer);
        float_4DTensor ifm_C(ifm_B_C_Initializer);
        float_4DTensor ifm_D(ifm_D_dtbias_Initializer);
        float_4DTensor ifm_z(ifm_x_dt_z_Initializer);
        float_4DTensor ofm_out(ofm_out_Initializer);
        float_4DTensor ofm_out_ref(ofm_out_Initializer);
    

        ifm_state.InitRand(0.0f, 1.0f);
        ifm_x.InitRand(0.0f, 1.0f);
        ifm_C.InitRand(0.0f, 1.0f);
        ifm_D.InitRand(0.0f, 1.0f);
        ifm_z.InitRand(0.0f, 1.0f);

        IndexSpace indexSpace = {{0}};
        int depthIS = (ofmifm_dim + 63) / 64 ;
        indexSpace.size[0] = depthIS;
        indexSpace.size[1] = ifm_dstate;
        indexSpace.size[2] = ifm_seq;
        indexSpace.size[3] = batch;

        // execute reference implementation of the kernel.
        this->pscan_update_fp32_ref(ifm_state, ifm_x, ifm_C, ifm_D, ifm_z, ofm_out_ref, indexSpace);

        // generate input for query call
        m_in_defs.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
        m_in_defs.inputTensorNr = 5;
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]),ifm_state);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]),ifm_x);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[2]),ifm_C);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[3]),ifm_D);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[4]),ifm_z);

        m_in_defs.outputTensorNr = 1;
        LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]),ofm_out);

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

        strcpy(m_in_defs.guid.name, guids[NameofKernel].name);
        result  = InstantiateTpcKernel(&m_in_defs,&m_out_defs);
        if (result != tpc_lib_api::GLUE_SUCCESS)
        {
            std::cout << "Glue test failed, can't load kernel " << result << std::endl;
            ReleaseKernelNames(guids, kernelCount);
            return -1;
        }

        // generate and load tensor descriptors
        std::vector<TensorDesc2> vec;
        vec.push_back(ifm_state.GetTensorDescriptor());
        vec.push_back(ifm_x.GetTensorDescriptor());
        vec.push_back(ifm_C.GetTensorDescriptor());
        vec.push_back(ifm_D.GetTensorDescriptor());
        vec.push_back(ifm_z.GetTensorDescriptor());

        vec.push_back(ofm_out.GetTensorDescriptor());

        // execute a simulation of the kernel using TPC simulator,
        TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
        ReleaseKernelNames(guids, kernelCount);
        ofm_out.Print(0);
        ofm_out_ref.Print(0);

        for (int element = 0 ; element <  ofm_out_ref.ElementCount() ; element++)
        {
            float ofmVal = ofm_out.Data()[element];
            float ofmRefVal = ofm_out_ref.Data()[element];
            float absDiff = std::abs(ofmVal - ofmRefVal);        
            if (absDiff/ofmVal > 0.005)
            {
                std::cout << "Pscan Update output FP32 test failed!!" << std::endl;
                return -1;
            }
        }
        std::cout << "Pscan Update output FP32 pass!!" << std::endl;


    }
    else
    {
        bfloat16_4DTensor ifm_state(ifm_state_Initializer);
        bfloat16_4DTensor ifm_x(ifm_x_dt_z_Initializer);
        bfloat16_4DTensor ifm_C(ifm_B_C_Initializer);
        bfloat16_4DTensor ifm_D(ifm_D_dtbias_Initializer);
        bfloat16_4DTensor ifm_z(ifm_x_dt_z_Initializer);
        bfloat16_4DTensor ofm_out(ofm_out_Initializer);
        bfloat16_4DTensor ofm_out_ref(ofm_out_Initializer);

        ifm_state.InitRand(0.0f, 1.0f);
        ifm_x.InitRand(0.0f, 1.0f);
        ifm_C.InitRand(0.0f, 1.0f);
        ifm_D.InitRand(0.0f, 1.0f);
        ifm_z.InitRand(0.0f, 1.0f);

        IndexSpace indexSpace = {{0}};
        int depthIS = (ofmifm_dim + 63) / 64 ;
        indexSpace.size[0] = depthIS;
        indexSpace.size[1] = ifm_dstate;
        indexSpace.size[2] = ifm_seq;
        indexSpace.size[3] = batch;
    

        // execute reference implementation of the kernel.
        this->pscan_update_bf16_ref(ifm_state, ifm_x, ifm_C, ifm_D, ifm_z, ofm_out_ref, indexSpace);

        // generate input for query call
        m_in_defs.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
        m_in_defs.inputTensorNr = 5;

        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]),ifm_state);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]),ifm_x);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[2]),ifm_C);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[3]),ifm_D);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[4]),ifm_z);

        m_in_defs.outputTensorNr = 1;
        LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]),ofm_out);

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

        strcpy(m_in_defs.guid.name, guids[NameofKernel].name);
        result  = InstantiateTpcKernel(&m_in_defs,&m_out_defs);
        if (result != tpc_lib_api::GLUE_SUCCESS)
        {
            std::cout << "Glue test failed, can't load kernel " << result << std::endl;
            ReleaseKernelNames(guids, kernelCount);
            return -1;
        }

        // generate and load tensor descriptors
        std::vector<TensorDesc2> vec;
        vec.push_back(ifm_state.GetTensorDescriptor());
        vec.push_back(ifm_x.GetTensorDescriptor());
        vec.push_back(ifm_C.GetTensorDescriptor());
        vec.push_back(ifm_D.GetTensorDescriptor());
        vec.push_back(ifm_z.GetTensorDescriptor());

        vec.push_back(ofm_out.GetTensorDescriptor());

        // execute a simulation of the kernel using TPC simulator,
        TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
        ReleaseKernelNames(guids, kernelCount);
        ofm_out.Print(0);
        ofm_out_ref.Print(0);

        for (int element = 0 ; element <  ofm_out_ref.ElementCount() ; element++)
        {
            float ofmVal = ofm_out.Data()[element];
            float ofmRefVal = ofm_out_ref.Data()[element];
            float absDiff = std::abs(ofmVal - ofmRefVal);        
            if (absDiff/ofmVal > 0.05)
            {
                std::cout << "Pscan Update BF16 test failed!!" << std::endl;
                return -1;
            }
        }
        std::cout << "Pscan Updatetest BF16 pass!!" << std::endl;

    }


    return 0;
 }

