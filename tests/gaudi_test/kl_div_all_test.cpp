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

void KLDivAllTest::kldiv_f32_fwd_reference_implementation(
        const float_4DTensor& inputX,
        const float_4DTensor& inputY,
        float_1DTensor& output,
        const float invLen)
{
    int coords[4] = {0};
    int coords0[1] = {0};
    float loss = 0.0f;
    for (unsigned d = 0; d < inputX.Size(0); d += 1) {
        coords[0] = d;
        for (unsigned b = 0; b < inputX.Size(3); b += 1) {
            coords[3] = b;
            for (unsigned h = 0; h < inputX.Size(2); h += 1) {
                coords[2] = h;
                for (unsigned w = 0; w < inputX.Size(1); w += 1) {
                    coords[1] = w;
                    float x = inputX.ElementAt(coords);
                    float y = inputY.ElementAt(coords);
                    float out = invLen * (y * (log(y) - x));
                    loss += out;
                    output.SetElement(coords0, loss);
                }
            }
        }
    }
}

void KLDivAllTest::kldiv_f32_bwd_reference_implementation(
        const float_1DTensor& gradIn,
        const float_4DTensor& inputX,
        const float_4DTensor& inputY,
        float_4DTensor& output,
        const float invLen)
{
    int coords[4] = {0};
    int coords0[1] = {0};
    //float loss = 0.0f;
    for (unsigned d = 0; d < inputX.Size(0); d += 1) {
        coords[0] = d;
        for (unsigned b = 0; b < inputX.Size(3); b += 1) {
            coords[3] = b;
            for (unsigned h = 0; h < inputX.Size(2); h += 1) {
                coords[2] = h;
                for (unsigned w = 0; w < inputX.Size(1); w += 1) {
                    coords[1] = w;
                    float grad = gradIn.ElementAt(coords0);
                    float y = inputY.ElementAt(coords);
                    float out = grad * invLen * (0 - y);
                    output.SetElement(coords, out);
                }
            }
        }
    }
}

void KLDivAllTest::kldiv_bf16_fwd_reference_implementation(
        const bfloat16_4DTensor& inputX,
        const bfloat16_4DTensor& inputY,
        bfloat16_1DTensor& output,
        const float invLen)
{
    int coords[4] = {0};
    int coords0[1] = {0};
    float loss = 0.0f;
    for (unsigned d = 0; d < inputX.Size(0); d += 1) {
        coords[0] = d;
        for (unsigned b = 0; b < inputX.Size(3); b += 1) {
            coords[3] = b;
            for (unsigned h = 0; h < inputX.Size(2); h += 1) {
                coords[2] = h;
                for (unsigned w = 0; w < inputX.Size(1); w += 1) {
                    coords[1] = w;
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
            }
        }
    }
}

void KLDivAllTest::kldiv_bf16_bwd_reference_implementation(
        const bfloat16_1DTensor& gradIn,
        const bfloat16_4DTensor& inputX,
        const bfloat16_4DTensor& inputY,
        bfloat16_4DTensor& output,
        const float invLen)
{
    int coords[4] = {0};
    int coords0[1] = {0};
    //float loss = 0.0f;
    for (unsigned d = 0; d < inputX.Size(0); d += 1) {
        coords[0] = d;
        for (unsigned b = 0; b < inputX.Size(3); b += 1) {
            coords[3] = b;
            for (unsigned h = 0; h < inputX.Size(2); h += 1) {
                coords[2] = h;
                for (unsigned w = 0; w < inputX.Size(1); w += 1) {
                    coords[1] = w;
                    float grad = gradIn.ElementAt(coords0);
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

int KLDivAllTest::runTest(Gaudi_Kernel_Name_e NameofKernel)
{
    const int height = 5;
    const int width  = 5;
    const int depth  = 60;
    const int batch  = 2;

    unsigned int fmInitializer[] = {depth, width, height, batch};
    unsigned int ofmInitializer[] = {1};

    if((NameofKernel == GAUDI_KERNEL_KL_DIV_FWD_F32) || (NameofKernel == GAUDI_KERNEL_KL_DIV_BWD_F32))
    {    
        float_1DTensor gradIn(ofmInitializer);
        gradIn.InitRand(-2.0f, 2.0f);

        float_4DTensor inputX(fmInitializer);
        inputX.InitRand(0.0f, 1.0f);
        float_4DTensor inputY(fmInitializer);
        inputY.InitRand(0.0f, 1.0f);

        float_1DTensor output1D(ofmInitializer);
        float_1DTensor output1D_ref(ofmInitializer);

        float_4DTensor output4D(fmInitializer);
        float_4DTensor output4D_ref(fmInitializer);

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
            kldiv_f32_fwd_reference_implementation(inputX, inputY, output1D_ref, param.invLen);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), inputX);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), inputY);
        }
        else {
            m_in_defs.inputTensorNr = 3;        
            kldiv_f32_bwd_reference_implementation(gradIn, inputX, inputY, output4D_ref, param.invLen);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), inputX);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), inputY);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[2]), gradIn);            
        }

        m_in_defs.outputTensorNr = 1;
        if(NameofKernel == GAUDI_KERNEL_KL_DIV_FWD_F32)
            LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), output1D);
        else
            LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), output4D);

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
        vec.push_back(inputX.GetTensorDescriptor());
        vec.push_back(inputY.GetTensorDescriptor());
        if(NameofKernel == GAUDI_KERNEL_KL_DIV_BWD_F32)
        {
            vec.push_back(gradIn.GetTensorDescriptor());        
            vec.push_back(output4D.GetTensorDescriptor());
        }
        if(NameofKernel == GAUDI_KERNEL_KL_DIV_FWD_F32)
            vec.push_back(output1D.GetTensorDescriptor());
                    
        // execute a simulation of the kernel using TPC simulator,
        TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
        ReleaseKernelNames(kernelNames, kernelCount);

        if(NameofKernel == GAUDI_KERNEL_KL_DIV_FWD_F32)
        {
            output1D.Print(0);
            output1D_ref.Print(0);             
            // scalar output, only check first element
            if (abs(output1D.Data()[0] - output1D_ref.Data()[0]) > 1e-2)
            {
                std::cout << "KL_Div FWD F32 test failed!!" << std::endl;
                return -1;
            }
        }
        else{
            output4D.Print(0);
            output4D_ref.Print(0);             
            for (int element = 0 ; element <  output4D_ref.ElementCount() ; element++)
            {
                if (abs(output4D.Data()[element] - output4D_ref.Data()[element]) > 1e-2)
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

        bfloat16_1DTensor gradIn(ofmInitializer);
        gradIn.InitRand(-1.0f, 1.0f);

        bfloat16_4DTensor inputX(fmInitializer);
        inputX.InitRand(0.0f, 1.0f);
        bfloat16_4DTensor inputY(fmInitializer);
        inputY.InitRand(0.0f, 1.0f);

        bfloat16_1DTensor output1D(ofmInitializer);
        bfloat16_1DTensor output1D_ref(ofmInitializer);
        bfloat16_4DTensor output4D(fmInitializer);
        bfloat16_4DTensor output4D_ref(fmInitializer);

        KLDivAll::KLDivAllParams param;
        // sum
        param.invLen = 1;

        // generate input for query call
        m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI;
        m_in_defs.NodeParams = &param;

        // execute reference implementation of the kernel.
       if(NameofKernel == GAUDI_KERNEL_KL_DIV_FWD_BF16) {
            m_in_defs.inputTensorNr = 2;        
            kldiv_bf16_fwd_reference_implementation(inputX, inputY, output1D_ref, param.invLen);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), inputX);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), inputY);
        }
        else {
            m_in_defs.inputTensorNr = 3;        
            kldiv_bf16_bwd_reference_implementation(gradIn, inputX, inputY, output4D_ref, param.invLen);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), inputX);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), inputY);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[2]), gradIn);            
        }

        m_in_defs.outputTensorNr = 1;
        if(NameofKernel == GAUDI_KERNEL_KL_DIV_FWD_BF16) 
            LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), output1D);
        else
            LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), output4D);

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
        vec.push_back(inputX.GetTensorDescriptor());
        vec.push_back(inputY.GetTensorDescriptor());
        if(NameofKernel == GAUDI_KERNEL_KL_DIV_BWD_BF16)
        {
            vec.push_back(gradIn.GetTensorDescriptor());        
            vec.push_back(output4D.GetTensorDescriptor());
        }
        if(NameofKernel == GAUDI_KERNEL_KL_DIV_FWD_BF16) 
            vec.push_back(output1D.GetTensorDescriptor());
            
        // execute a simulation of the kernel using TPC simulator,
        TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
        ReleaseKernelNames(kernelNames, kernelCount);

        bfloat16 tmp;
        if(NameofKernel == GAUDI_KERNEL_KL_DIV_FWD_BF16)
        {
            output1D.Print(0);
            output1D_ref.Print(0);            
            // scalar output, only check first element
            if (tmp.abs(output1D.Data()[0] - output1D_ref.Data()[0]) > 1e-2)
            {
                std::cout << "KL_Div FWD BF16 test failed!!" << std::endl;
                return -1;
            }
        }
        else{
            output4D.Print(0);
            output4D_ref.Print(0);
            for (int element = 0 ; element <  output4D_ref.ElementCount() ; element++)
            {
                if (tmp.abs(output4D.Data()[element] - output4D_ref.Data()[element]) > 1e-2)
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

