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

#include "kl_div_all_test.hpp"

void KLDivAllTest::kldiv_f32_reference_implementation(
        const float_4DTensor& gradIn,
        const float_4DTensor& inputX,
        const float_4DTensor& inputY,
        float_4DTensor& output,
        const float invLen, KLDivAll::KLDiv_mode_t mode)
{
    int coords[4] = {0};
    int coords0[4] = {0};
    float loss = 0.0f;
    for (unsigned d = 0; d < inputX.Size(0); d += 1) {
        coords[0] = d;
        for (unsigned b = 0; b < inputX.Size(3); b += 1) {
            coords[3] = b;
            for (unsigned h = 0; h < inputX.Size(2); h += 1) {
                coords[2] = h;
                for (unsigned w = 0; w < inputX.Size(1); w += 1) {
                    coords[1] = w;
                    if(mode == KLDivAll::fwd_f32) {
                        float x = inputX.ElementAt(coords);
                        float y = inputY.ElementAt(coords);
                        float out = invLen * (y * (log(y) - x));
                        loss += out;
                        output.SetElement(coords0, loss);
                    }
                    else if (mode == KLDivAll::bwd_f32) {
                        float grad = gradIn.ElementAt(coords);
                        float y = inputY.ElementAt(coords);
                        float out = grad * invLen * (0 - y);
                        output.SetElement(coords, out);
                    }
                }
            }
        }
    }
}

void KLDivAllTest::kldiv_bf16_reference_implementation(
        const bfloat16_4DTensor& gradIn,
        const bfloat16_4DTensor& inputX,
        const bfloat16_4DTensor& inputY,
        bfloat16_4DTensor& output,
        const float invLen, KLDivAll::KLDiv_mode_t mode)
{
    int coords[4] = {0};
    int coords0[4] = {0};
    float loss = 0.0f;
    for (unsigned d = 0; d < inputX.Size(0); d += 1) {
        coords[0] = d;
        for (unsigned b = 0; b < inputX.Size(3); b += 1) {
            coords[3] = b;
            for (unsigned h = 0; h < inputX.Size(2); h += 1) {
                coords[2] = h;
                for (unsigned w = 0; w < inputX.Size(1); w += 1) {
                    coords[1] = w;
                    if(mode == KLDivAll::fwd_bf16) {
                        float x = (float)inputX.ElementAt(coords);
                        float tmp_x = floatTobf16ToFloat(x);
                        float y = (float)inputY.ElementAt(coords);
                        float tmp_y = floatTobf16ToFloat(y);
                        y = floatTobf16ToFloat(log(tmp_y)) - tmp_x;
                        x = floatTobf16ToFloat(tmp_y * y);
                        float out = floatTobf16ToFloat(invLen * x);
                        loss += out;
                        output.SetElement(coords0, loss);
                    }
                    else if (mode == KLDivAll::bwd_bf16) {
                        float grad = gradIn.ElementAt(coords);
                        float tmp_grad = floatTobf16ToFloat(grad);
                        float y = inputY.ElementAt(coords);
                        float tmp_y = floatTobf16ToFloat(y);
                        y = floatTobf16ToFloat(tmp_grad * (0 - tmp_y));
                        float out = floatTobf16ToFloat(invLen * y);
                        output.SetElement(coords, out);
                    }
                }
            }
        }
    }
}

int KLDivAllTest::runTest(Gaudi_Kernel_Name_e NameofKernel)
{
    const int height = 8;
    const int width  = 8;
    const int depth  = 60;
    const int batch  = 2;

    unsigned int fmInitializer[] = {depth, width, height, batch};

    if((NameofKernel == GAUDI_KERNEL_KL_DIV_FWD_F32) || (NameofKernel == GAUDI_KERNEL_KL_DIV_BWD_F32))
    {    
        float_4DTensor gradIn(fmInitializer);
        gradIn.InitRand(-2.0f, 2.0f);
        float_4DTensor inputX(fmInitializer);
        inputX.InitRand(0.0f, 1.0f);
        float_4DTensor inputY(fmInitializer);
        inputY.InitRand(0.0f, 1.0f);
        float_4DTensor output(fmInitializer);
        float_4DTensor output_ref(fmInitializer);

        KLDivAll::KLDivAllParams param;
        // mean
        param.invLen = (float) (1.0/(height*width*depth*batch));

        // generate input for query call
        m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI;
        m_in_defs.NodeParams = &param;

        // execute reference implementation of the kernel.
       if(NameofKernel == GAUDI_KERNEL_KL_DIV_FWD_F32)
        {
            m_in_defs.inputTensorNr = 2;        
            kldiv_f32_reference_implementation(gradIn, inputX, inputY, output_ref, param.invLen, KLDivAll::fwd_f32);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), inputX);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), inputY);
        }
        else {
            m_in_defs.inputTensorNr = 3;        
            kldiv_f32_reference_implementation(gradIn, inputX, inputY, output_ref, param.invLen, KLDivAll::bwd_f32);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), gradIn);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), inputX);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[2]), inputY);            
        }

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

        strcpy(m_in_defs.nodeName, kernelNames[NameofKernel]);
        result  = HabanaKernel(&m_in_defs,&m_out_defs);
        if (result != gcapi::GLUE_SUCCESS)
        {
            std::cout << "Glue test failed, can't load kernel " << result << std::endl;
            ReleaseKernelNames(kernelNames, kernelCount);
            return -1;
        }

        // generate and load tensor descriptors
        std::vector<TensorDescriptor> vec;
        if(NameofKernel == GAUDI_KERNEL_KL_DIV_BWD_F32)
            vec.push_back(gradIn.GetTensorDescriptor());        
        vec.push_back(inputX.GetTensorDescriptor());
        vec.push_back(inputY.GetTensorDescriptor());
        vec.push_back(output.GetTensorDescriptor());
        // execute a simulation of the kernel using TPC simulator,
        TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
        ReleaseKernelNames(kernelNames, kernelCount);
        output.Print(0);
        output_ref.Print(0);        

        if(NameofKernel == GAUDI_KERNEL_KL_DIV_FWD_F32)
        {
            // scalar output, only check first element
            if (abs(output.Data()[0] - output_ref.Data()[0]) > 1e-2)
            {
                std::cout << "KL_Div FWD F32 test failed!!" << std::endl;
                return -1;
            }
        }
        else{
            for (int element = 0 ; element <  output_ref.ElementCount() ; element++)
            {
                if (abs(output.Data()[element] - output_ref.Data()[element]) > 1e-1)
                {
                     std::cout << "KL_Div BWD F32 test failed!!" << std::endl;
                    return -1;
                }
            }

        }
        if(NameofKernel == GAUDI_KERNEL_KL_DIV_FWD_F32)
            std::cout << "KL_Div FWD F32 test pass!!" << std::endl;
        else
            std::cout << "KL_Div BWD F32 test pass!!" << std::endl;
        return 0;
    }
    else {

        bfloat16_4DTensor gradIn(fmInitializer);
        gradIn.InitRand(-1.0f, 1.0f);
        bfloat16_4DTensor inputX(fmInitializer);
        inputX.InitRand(0.0f, 1.0f);
        bfloat16_4DTensor inputY(fmInitializer);
        inputY.InitRand(0.0f, 1.0f);
        bfloat16_4DTensor output(fmInitializer);
        bfloat16_4DTensor output_ref(fmInitializer);

        KLDivAll::KLDivAllParams param;
        // sum
        param.invLen = 1;

        // generate input for query call
        m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI;
        m_in_defs.NodeParams = &param;

        // execute reference implementation of the kernel.
       if(NameofKernel == GAUDI_KERNEL_KL_DIV_FWD_BF16) {
            m_in_defs.inputTensorNr = 2;        
            kldiv_bf16_reference_implementation(gradIn, inputX, inputY, output_ref, param.invLen, KLDivAll::fwd_bf16);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), inputX);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), inputY);
        }
        else {
            m_in_defs.inputTensorNr = 3;        
            kldiv_bf16_reference_implementation(gradIn, inputX, inputY, output_ref, param.invLen, KLDivAll::bwd_bf16);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), gradIn);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), inputX);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[2]), inputY);            
        }

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

        strcpy(m_in_defs.nodeName, kernelNames[NameofKernel]);
        result  = HabanaKernel(&m_in_defs,&m_out_defs);
        if (result != gcapi::GLUE_SUCCESS)
        {
            std::cout << "Glue test failed, can't load kernel " << result << std::endl;
            ReleaseKernelNames(kernelNames, kernelCount);
            return -1;
        }

        // generate and load tensor descriptors
        std::vector<TensorDescriptor> vec;
        if(NameofKernel == GAUDI_KERNEL_KL_DIV_BWD_BF16)
            vec.push_back(gradIn.GetTensorDescriptor());        
        vec.push_back(inputX.GetTensorDescriptor());
        vec.push_back(inputY.GetTensorDescriptor());
        vec.push_back(output.GetTensorDescriptor());
        // execute a simulation of the kernel using TPC simulator,
        TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
        ReleaseKernelNames(kernelNames, kernelCount);
        output.Print(0);
        output_ref.Print(0);
        bfloat16 tmp;
        if(NameofKernel == GAUDI_KERNEL_KL_DIV_FWD_BF16)
        {
            // scalar output, only check first element
            if (tmp.abs(output.Data()[0] - output_ref.Data()[0]) > 1e-2)
            {
                std::cout << "KL_Div FWD BF16 test failed!!" << std::endl;
                return -1;
            }
        }
        else{
            for (int element = 0 ; element <  output_ref.ElementCount() ; element++)
            {
                if (tmp.abs(output.Data()[element] - output_ref.Data()[element]) > 1e-1)
                {
                     std::cout << "KL_Div BWD BF16 test failed!!" << std::endl;
                    return -1;
                }
            }

        }

        if(NameofKernel == GAUDI_KERNEL_KL_DIV_FWD_BF16)
            std::cout << "KL_Div FWD BF16 test pass!!" << std::endl;
        else
            std::cout << "KL_Div BWD BF16 test pass!!" << std::endl;
        return 0;

    }
}

