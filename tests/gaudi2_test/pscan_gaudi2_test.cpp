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

#include "pscan_gaudi2_test.hpp"
#include <type_traits>

void PscanGaudi2Test::pscan_fp32_ref(
         const test::Tensor<float,4>& state_M,
         const test::Tensor<float,4>& x_M,
         const test::Tensor<float,4>& dt_M,
         const test::Tensor<float,4>& A_M,
         const test::Tensor<float,4>& B_M,
         test::Tensor<float,4>& state_out,
         const IndexSpace& indexSpace)
{
    int eig = 64;

    int coords_state[5] = { 0 };
    int coords_x[5] = { 0 };
    int coords_dt[5] = { 0 };
    int coords_A[5] = { 0 };
    int coords_B[5] = { 0 };
    int coords_out[5] = { 0 };
    for (int h = indexSpace.offset[2]; h < indexSpace.offset[2] + indexSpace.size[2]; h += 1)
    {
            coords_state[2] = 0;
            coords_x[2] = h;
            coords_dt[2] = h;
            coords_A[2] = 0;
            coords_B[2] = h;
            coords_out[2] = h;

        for (int n = indexSpace.offset[1]; n < indexSpace.offset[1] + indexSpace.size[1]; n += 1)
        {
            coords_state[1] = n;
            coords_x[1] = 0;
            coords_dt[1] = 0;
            coords_A[1] = n;
            coords_B[1] = n;
            coords_out[1] = n;

            for (int d = indexSpace.offset[0]; d < (indexSpace.offset[0] + indexSpace.size[0]) * eig; d += 1)
            {
                coords_state[0] = d;
                coords_x[0] = d;
                coords_dt[0] = d;
                coords_A[0] = d;
                coords_B[0] = 0;
                coords_out[0] = d;

                for (int b = indexSpace.offset[3]; b < (indexSpace.offset[3] + indexSpace.size[3]); b += 1)
                {
                    coords_state[3] = b;
                    coords_x[3] = b;
                    coords_dt[3] = b;
                    coords_A[3] = 0;
                    coords_B[3] = b;
                    coords_out[3] = b;

                    float ifmVal_state = state_M.ElementAt(coords_state);
                    float ifmVal_A = A_M.ElementAt(coords_A);
                    float ofmVal = 0;

                    
                    float ifmVal_x = x_M.ElementAt(coords_x);
                    
                    float ifmVal_dt = dt_M.ElementAt(coords_dt);   
                    ifmVal_dt = logf(1 + expf(ifmVal_dt));

                    //float ifmVal_state = state_M.ElementAt(coords_state);
                    
                    //float ifmVal_A = A_M.ElementAt(coords_A);
                    float ifmVal_B = B_M.ElementAt(coords_B);
                    float dA = expf(ifmVal_dt * ifmVal_A);

                    float dB = ifmVal_dt * ifmVal_B;
                    ofmVal = (ifmVal_state * dA) + (dB * ifmVal_x);
                    state_out.SetElement(coords_out, ofmVal);

                }
            }
        }
    }
}

void PscanGaudi2Test::pscan_bf16_ref(
         const test::Tensor<bfloat16,4>& state_M,
         const test::Tensor<bfloat16,4>& x_M,
         const test::Tensor<bfloat16,4>& dt_M,
         const test::Tensor<bfloat16,4>& A_M,
         const test::Tensor<bfloat16,4>& B_M,
         test::Tensor<bfloat16,4>& state_out,
         const IndexSpace& indexSpace)
{
    int eig = 128;
    
    int coords_state[5] = { 0 };
    int coords_x[5] = { 0 };
    int coords_dt[5] = { 0 };
    int coords_A[5] = { 0 };
    int coords_B[5] = { 0 };
    int coords_out[5] = { 0 };
    for (int b = indexSpace.offset[3]; b < (indexSpace.offset[3] + indexSpace.size[3]); b += 1)
    {
        coords_state[3] = b;
        coords_x[3] = b;
        coords_dt[3] = b;
        coords_A[3] = 0;
        coords_B[3] = b;
        coords_out[3] = b;
        for (int h = indexSpace.offset[2]; h < indexSpace.offset[2] + indexSpace.size[2]; h += 1)
        {
            coords_state[2] = 0;
            coords_x[2] = h;
            coords_dt[2] = h;
            coords_A[2] = 0;
            coords_B[2] = h;
            coords_out[2] = h;

            for (int d = indexSpace.offset[0]; d < (indexSpace.offset[0] + indexSpace.size[0]) * eig; d += 1)
            {
                coords_state[0] = d;
                coords_x[0] = d;
                coords_dt[0] = d;
                coords_A[0] = d;
                coords_B[0] = 0;
                coords_out[0] = d;

                float ofmVal = 0;
                float ifmVal_x = (float)x_M.ElementAt(coords_x);
                for (int n = indexSpace.offset[1]; n < indexSpace.offset[1] + indexSpace.size[1]; n += 1)
                {
                    coords_state[1] = n;
                    coords_x[1] = 0;
                    coords_dt[1] = 0;
                    coords_A[1] = n;
                    coords_B[1] = n;
                    coords_out[1] = n;
                    
                    float ifmVal_dt = (float)dt_M.ElementAt(coords_dt);   
                    float ttemp = floatTobf16ToFloat(ifmVal_dt);
                    ifmVal_dt = logf(1 + expf(ttemp));

                    float ifmVal_state = (float)state_M.ElementAt(coords_state);
                   
                    float ifmVal_A = (float)A_M.ElementAt(coords_A);
                    float ifmVal_B = (float)B_M.ElementAt(coords_B);

                    float dA;
                    float ttemp1 = floatTobf16ToFloat(ifmVal_dt);
                    float ttemp2 = floatTobf16ToFloat(ifmVal_A);
                    dA = expf(ttemp1*ttemp2);

                    float dB = ifmVal_dt * ifmVal_B;
                    ofmVal = (ifmVal_state * dA) + (dB * ifmVal_x);
                    state_out.SetElement(coords_out, ofmVal);

                }
            }
        }
    }
}

int PscanGaudi2Test::runTest(Gaudi2_Kernel_Name_e NameofKernel)
 {

    // Initalize input size
    const int ofmifm_dim = 12;
    const int ifm_dstate = 16;
    const int ifm_seq  = 10;
    const int batch = 1;

    // Initalize inputs
    uint64_t ifm_state_Initializer[] = {ofmifm_dim, ifm_dstate, 1, batch};
    uint64_t ifm_x_dt_z_Initializer[] = {ofmifm_dim, 1, ifm_seq, batch};
    uint64_t ifm_A_Initializer[] = {ofmifm_dim, ifm_dstate, 1, 1};
    uint64_t ifm_B_C_Initializer[] = {1, ifm_dstate, ifm_seq, batch};
    uint64_t ofm_out_Initializer[] = {ofmifm_dim, ifm_dstate, ifm_seq, batch};
    
    if(NameofKernel == GAUDI2_KERNEL_PSCAN_F32)
    {
        float_4DTensor ifm_state(ifm_state_Initializer);
        float_4DTensor ifm_x(ifm_x_dt_z_Initializer);
        float_4DTensor ifm_dt(ifm_x_dt_z_Initializer);
        float_4DTensor ifm_A(ifm_A_Initializer);
        float_4DTensor ifm_B(ifm_B_C_Initializer);
        float_4DTensor ofm_state_out(ofm_out_Initializer);
        float_4DTensor ofm_state_out_ref(ofm_out_Initializer);
    

        ifm_state.InitRand(0.0f, 1.0f);
        ifm_x.InitRand(0.0f, 1.0f);
        ifm_dt.InitRand(0.0f, 1.0f);
        ifm_A.InitRand(0.0f, 1.0f);
        ifm_B.InitRand(0.0f, 1.0f);

        IndexSpace indexSpace = {{0}};
        int depthIS = (ofmifm_dim + 63) / 64 ;
        indexSpace.size[0] = depthIS;
        indexSpace.size[1] = ifm_dstate;
        indexSpace.size[2] = ifm_seq;
        indexSpace.size[3] = batch;

        // execute reference implementation of the kernel.
        this->pscan_fp32_ref(ifm_state, ifm_x, ifm_dt, ifm_A, ifm_B, ofm_state_out_ref, indexSpace);

        // generate input for query call
        m_in_defs.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
        //m_in_defs.nodeParams.nodeParams = &sdef;
        m_in_defs.inputTensorNr = 5;

        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]),ifm_state);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]),ifm_x);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[2]),ifm_dt);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[3]),ifm_A);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[4]),ifm_B);
        

        m_in_defs.outputTensorNr = 1;
        LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]),ofm_state_out);

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
        vec.push_back(ifm_dt.GetTensorDescriptor());
        vec.push_back(ifm_A.GetTensorDescriptor());
        vec.push_back(ifm_B.GetTensorDescriptor());

        vec.push_back(ofm_state_out.GetTensorDescriptor());

        // execute a simulation of the kernel using TPC simulator,
        TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
        ReleaseKernelNames(guids, kernelCount);

        ofm_state_out.Print(0);
        ofm_state_out_ref.Print(0);

        for (int element = 0 ; element <  ofm_state_out_ref.ElementCount() ; element++)
        {
            float ofmVal = ofm_state_out.Data()[element];
            float ofmRefVal = ofm_state_out_ref.Data()[element];
            float absDiff = std::abs(ofmVal - ofmRefVal);        
            if (absDiff/ofmVal > 0.005)
            {
                std::cout << "Pscan FP32 test failed!!" << std::endl;
                return -1;
            }
        }
        std::cout << "Pscan FP32 pass!!" << std::endl;

    }
    else
    {
        bfloat16_4DTensor ifm_state(ifm_state_Initializer);
        bfloat16_4DTensor ifm_x(ifm_x_dt_z_Initializer);
        bfloat16_4DTensor ifm_dt(ifm_x_dt_z_Initializer);
        bfloat16_4DTensor ifm_A(ifm_A_Initializer);
        bfloat16_4DTensor ifm_B(ifm_B_C_Initializer);
        bfloat16_4DTensor ofm_state_out(ofm_out_Initializer);
        bfloat16_4DTensor ofm_state_out_ref(ofm_out_Initializer);

        ifm_state.InitRand(0.0f, 1.0f);
        ifm_x.InitRand(0.0f, 1.0f);
        ifm_dt.InitRand(0.0f, 1.0f);
        ifm_A.InitRand(0.0f, 1.0f);
        ifm_B.InitRand(0.0f, 1.0f);

        IndexSpace indexSpace = {{0}};
        int depthIS = (ofmifm_dim + 63) / 64 ;
        indexSpace.size[0] = depthIS;
        indexSpace.size[1] = ifm_dstate;
        indexSpace.size[2] = ifm_seq;
        indexSpace.size[3] = batch;
    
        // execute reference implementation of the kernel.
        this->pscan_bf16_ref(ifm_state, ifm_x, ifm_dt, ifm_A, ifm_B, ofm_state_out_ref, indexSpace);

        // generate input for query call
        m_in_defs.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
        m_in_defs.inputTensorNr = 5;

        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]),ifm_state);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]),ifm_x);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[2]),ifm_dt);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[3]),ifm_A);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[4]),ifm_B);
        
        m_in_defs.outputTensorNr = 1;
        LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]),ofm_state_out);

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
        vec.push_back(ifm_dt.GetTensorDescriptor());
        vec.push_back(ifm_A.GetTensorDescriptor());
        vec.push_back(ifm_B.GetTensorDescriptor());

        vec.push_back(ofm_state_out.GetTensorDescriptor());

        // execute a simulation of the kernel using TPC simulator,
        TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
        ReleaseKernelNames(guids, kernelCount);
        ofm_state_out.Print(0);
        ofm_state_out_ref.Print(0);

        for (int element = 0 ; element <  ofm_state_out_ref.ElementCount() ; element++)
        {
            float ofmVal = ofm_state_out.Data()[element];
            float ofmRefVal = ofm_state_out_ref.Data()[element];
            float absDiff = std::abs(ofmVal - ofmRefVal);        
            if (absDiff/ofmVal > 0.05)
            {
                std::cout << "Pscan BF16 test failed!!" << std::endl;
                return -1;
            }
        }
        std::cout << "Pscan BF16 pass!!" << std::endl;

    }


    return 0;
 }

