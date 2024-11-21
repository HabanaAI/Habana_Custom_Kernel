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

#include "mygather_gaudi2_test.hpp"
#include <type_traits>

void MygatherGaudi2Test::mygather_fp32_ref(
         const test::Tensor<float,4>& in_M,
         const test::Tensor<int,4>& start_M,
         test::Tensor<float,4>& output,
         const IndexSpace& indexSpace,
         MygatherGaudi2::MygatherParam def)
{
    //int eig = 64;

    int coords_in[5] = { 0 };
    int coords_start[5] = { 0 };
    int coords_out[5] = { 0 };
    for (int b = indexSpace.offset[3]; b < (indexSpace.offset[3] + indexSpace.size[3]); b += 1)
    {
        coords_in[3] = 0;
        coords_start[3] = b;
        coords_out[3] = b;
        for (int h = indexSpace.offset[2]; h < indexSpace.offset[2] + indexSpace.size[2]; h += 1)
        {
            coords_in[2] = h;
            coords_start[2] = 0;
            coords_out[2] = h;

            for (int d = indexSpace.offset[1]; d < (indexSpace.offset[1] + indexSpace.size[1]); d += 1)
            {
                coords_in[1] = d;
                coords_start[1] = 0;
                coords_out[1] = d;

                for (int m = 0; m < def.max_ctx_len; m += 1)
                {
                    coords_in[0] = m;
                    coords_start[0] = 0;
                    coords_out[0] = m;
                    int ifmVal_start = start_M.ElementAt(coords_start);
                    coords_in[0] += ifmVal_start;
                    float ifmVal_in = in_M.ElementAt(coords_in);   
                    
                    output.SetElement(coords_out, ifmVal_in);
                }
            }
        }
    }
}

void MygatherGaudi2Test::mygather_bf16_ref(
         const test::Tensor<bfloat16,4>& in_M,
         const test::Tensor<int,4>& start_M,
         test::Tensor<bfloat16,4>& output,
         const IndexSpace& indexSpace, 
         MygatherGaudi2::MygatherParam def)
{
    //int eig = 128;
    
    int coords_in[5] = { 0 };
    int coords_start[5] = { 0 };
    int coords_out[5] = { 0 };
    for (int b = indexSpace.offset[3]; b < (indexSpace.offset[3] + indexSpace.size[3]); b += 1)
    {
        coords_in[3] = b;
        coords_start[3] = b;
        coords_out[3] = b;
        for (int h = indexSpace.offset[2]; h < indexSpace.offset[2] + indexSpace.size[2]; h += 1)
        {
            coords_in[2] = h;
            coords_start[2] = h;
            coords_out[2] = h;

            for (int d = indexSpace.offset[1]; d < (indexSpace.offset[1] + indexSpace.size[1]) ; d += 1)
            {
                coords_in[0] = d;
                coords_start[0] = d;
                coords_out[0] = d;

                for (int m = 0; m < def.max_ctx_len; m += 1)
                {
                    coords_in[0] = m;
                    coords_start[0] = 0;
                    coords_out[0] = m;
                    
                    int ifmVal_start = start_M.ElementAt(coords_start);
                    coords_in[0] += ifmVal_start;
                    float ifmVal_in = (float)in_M.ElementAt(coords_in);
                    
                    output.SetElement(coords_out, ifmVal_in);
                }
            }
        }
    }
}

int MygatherGaudi2Test::runTest(Gaudi2_Kernel_Name_e NameofKernel)
 {

    // Initalize input size
    const int ifm_mem = 128;
    const int ifm_dim = 10;
    const int ifm_nhead  = 8;
    const int batch = 2;

    // Initalize inputs
    uint64_t ifm_in_Initializer[] = {ifm_mem, ifm_dim, ifm_nhead, 1};
    uint64_t ifm_start_Initializer[] = {1, 1, 1, batch};
    uint64_t ofm_out_Initializer[] = {ifm_mem, ifm_dim, ifm_nhead, batch};
    
    if((NameofKernel == GAUDI2_KERNEL_MYGATHER_F32))
    {
        float_4DTensor ifm_in(ifm_in_Initializer);
        int32_4DTensor ifm_start(ifm_start_Initializer);
        float_4DTensor ofm_out(ofm_out_Initializer);
        float_4DTensor ofm_out_ref(ofm_out_Initializer);
    

        ifm_in.FillWithData();
        ifm_start.FillWithData();

        IndexSpace indexSpace = {{0}};
        indexSpace.size[0] = ifm_mem;
        indexSpace.size[1] = ifm_dim;
        indexSpace.size[2] = ifm_nhead;
        indexSpace.size[3] = batch;

        // Define the two flags
        MygatherGaudi2::MygatherParam sdef;
        sdef.max_ctx_len = ifm_mem/2;

        // execute reference implementation of the kernel.
        this->mygather_fp32_ref(ifm_in, ifm_start, ofm_out_ref, indexSpace, sdef);

        // generate input for query call
        m_in_defs.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
        m_in_defs.nodeParams.nodeParams = &sdef;
        m_in_defs.inputTensorNr = 2;

        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]),ifm_in);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]),ifm_start);
        
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
        vec.push_back(ifm_in.GetTensorDescriptor());
        vec.push_back(ifm_start.GetTensorDescriptor());
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
                std::cout << "Mygather output FP32 test failed!!" << std::endl;
                return -1;
            }
        }
        std::cout << "Mygather output FP32 pass!!" << std::endl;

    }
    else
    {
        bfloat16_4DTensor ifm_in(ifm_in_Initializer);
        int32_4DTensor ifm_start(ifm_start_Initializer);
        bfloat16_4DTensor ofm_out(ofm_out_Initializer);
        bfloat16_4DTensor ofm_out_ref(ofm_out_Initializer);

        ifm_in.FillWithData();
        ifm_start.FillWithData();

        IndexSpace indexSpace = {{0}};
        indexSpace.size[0] = ifm_mem;
        indexSpace.size[1] = ifm_dim;
        indexSpace.size[2] = ifm_nhead;
        indexSpace.size[3] = batch;
    
        // Define the two flags
        MygatherGaudi2::MygatherParam sdef;
        sdef.max_ctx_len = ifm_mem/2;

        // execute reference implementation of the kernel.
        this->mygather_bf16_ref(ifm_in, ifm_start, ofm_out_ref, indexSpace, sdef);

        // generate input for query call
        m_in_defs.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
        m_in_defs.nodeParams.nodeParams = &sdef;
        m_in_defs.inputTensorNr = 2;
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]),ifm_in);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]),ifm_start);

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
        vec.push_back(ifm_in.GetTensorDescriptor());
        vec.push_back(ifm_start.GetTensorDescriptor());
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
                std::cout << "Mygather BF16 test failed!!" << std::endl;
                return -1;
            }
        }
        std::cout << "Mygather test BF16 pass!!" << std::endl;

    }


    return 0;
 }