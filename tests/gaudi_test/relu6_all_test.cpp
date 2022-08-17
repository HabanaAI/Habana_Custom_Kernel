/**********************************************************************
Copyright (c) 2021 Habana Labs. All rights reserved.

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

#include "relu6_all_test.hpp"

void Relu6AllTest::relu6_f32_reference_implementation(
        const float_5DTensor& gradin,    
        const float_5DTensor& input,
        float_5DTensor& output, Gaudi_Kernel_Name_e mode)
{
    int coords[5] = {0};
    for (unsigned f = 0; f < input.Size(4); f += 1)
    {
        coords[4] = f;
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
                        if(mode == GAUDI_KERNEL_RELU6_FWD_F32)
                        {
                            float x = input.ElementAt(coords);
                            float y = (x < 0.0f) ? 0 : x;
                            float z = (y > 6.0f) ? 6.0 : y;
                            output.SetElement(coords, z);
                        }
                        else if (mode == GAUDI_KERNEL_RELU6_BWD_F32)
                        {
                            float g = gradin.ElementAt(coords);
                            float x = input.ElementAt(coords);
                            float y = (x < 0.0f) ? 0 : x;
                            x = (y >= 6.0f) ? 0.0 : y;
                            y = (x > 0.0f) ? 1 : 0;
                            y = y * g;
                            output.SetElement(coords, y);
                        }
                        else if(mode == GAUDI_KERNEL_RELU_FWD_F32)
                        {
                            float x = input.ElementAt(coords);
                            float y = (x < 0.0f) ? 0 : x;
                            output.SetElement(coords, y);
                        }
                        else if (mode == GAUDI_KERNEL_RELU_BWD_F32)
                        {
                            float g = gradin.ElementAt(coords);
                            float x = input.ElementAt(coords);
                            float y = (x < 0.0f) ? 0 : x;
                            x = (y > 0.0f) ? 1 : 0;
                            x = x * g;
                            output.SetElement(coords, x);
                        }
                    }
                }
            }
        }
    }
}

void Relu6AllTest::relu6_bf16_reference_implementation(
        const bfloat16_5DTensor& gradin,
        const bfloat16_5DTensor& input,
        bfloat16_5DTensor& output, Gaudi_Kernel_Name_e mode)
{
    int coords[5] = {0};
    for (unsigned f = 0; f < input.Size(4); f += 1)
    {
        coords[4] = f;
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
                        if(mode == GAUDI_KERNEL_RELU6_FWD_BF16)
                        {
                            float x = (float)input.ElementAt(coords);
                            float tmp_x = floatTobf16ToFloat(x);
                            float y = (tmp_x < 0.0f) ? 0 : tmp_x;
                            float z = (y > 6.0f) ? 6.0 : y;
                            output.SetElement(coords, z);
                        }
                        else if (mode == GAUDI_KERNEL_RELU6_BWD_BF16)
                        {

                            float g = (float)gradin.ElementAt(coords);
                            float tmp_g = floatTobf16ToFloat(g);
                            float x = input.ElementAt(coords);
                            float tmp_x = floatTobf16ToFloat(x);
                            float y = (tmp_x < 0.0f) ? 0 : tmp_x;
                            x = (y >= 6.0f) ? 0.0 : y;
                            y = (x > 0.0f) ? 1 : 0;
                            y = y * tmp_g;
                            float tmp_y = floatTobf16ToFloat(y);
                            output.SetElement(coords, tmp_y);
                        }
                        else if(mode == GAUDI_KERNEL_RELU_FWD_BF16)
                        {
                            float x = (float)input.ElementAt(coords);
                            float tmp_x = floatTobf16ToFloat(x);
                            float y = (tmp_x < 0.0f) ? 0 : tmp_x;
                            output.SetElement(coords, y);
                        }
                        else if (mode == GAUDI_KERNEL_RELU_BWD_BF16)
                        {

                            float g = (float)gradin.ElementAt(coords);
                            float tmp_g = floatTobf16ToFloat(g);
                            float x = input.ElementAt(coords);
                            float tmp_x = floatTobf16ToFloat(x);
                            float y = (tmp_x < 0.0f) ? 0 : tmp_x;
                            x = (y > 0.0f) ? 1 : 0;
                            x = x * tmp_g;
                            float tmp_y = floatTobf16ToFloat(x);
                            output.SetElement(coords, tmp_y);
                        }
                    }
                }
            }
        }
    }
}

int Relu6AllTest::runTest(Gaudi_Kernel_Name_e NameofKernel)
{
    const int height = 5;
    const int width  = 5;
    const int depth  = 100;
    const int batch  = 2;
    const int fifthdim  = 1;

    unsigned int fmInitializer[] = {depth, width, height, batch, fifthdim};
    unsigned kernelCount;
    gcapi::GlueCodeReturn_t result;
    char**   kernelNames = nullptr;

    if((NameofKernel == GAUDI_KERNEL_RELU6_FWD_F32) || (NameofKernel == GAUDI_KERNEL_RELU6_BWD_F32)
      || (NameofKernel == GAUDI_KERNEL_RELU_FWD_F32) || (NameofKernel == GAUDI_KERNEL_RELU_BWD_F32))
    {
        float_5DTensor gradin(fmInitializer);
        gradin.InitRand(-10.0f, 10.0f);    
        float_5DTensor input(fmInitializer);
        input.InitRand(-10.0f, 10.0f);
        float_5DTensor output(fmInitializer);
        float_5DTensor output_ref(fmInitializer);

        // generate input for query call
        m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI;

        if(NameofKernel == GAUDI_KERNEL_RELU6_FWD_F32 || NameofKernel == GAUDI_KERNEL_RELU_FWD_F32)
        {
            // execute reference implementation of the kernel.
            m_in_defs.inputTensorNr = 1;
            relu6_f32_reference_implementation(gradin, input, output_ref, NameofKernel);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), input);
        }
        else
        {
            // execute reference implementation of the kernel.
            m_in_defs.inputTensorNr = 2;
            relu6_f32_reference_implementation(gradin, input, output_ref, NameofKernel);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), gradin);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), input);
        }

        m_in_defs.outputTensorNr = 1;
        LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), output);

        kernelCount = 0;
        result = GetKernelNames(kernelNames, &kernelCount, gcapi::DEVICE_ID_GAUDI);
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
        std::vector<TensorDesc> vec;
        if(NameofKernel == GAUDI_KERNEL_RELU6_BWD_F32 || NameofKernel == GAUDI_KERNEL_RELU_BWD_F32)
            vec.push_back(gradin.GetTensorDescriptor());
        vec.push_back(input.GetTensorDescriptor());
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
                if(NameofKernel == GAUDI_KERNEL_RELU6_FWD_F32)
                    std::cout << "Relu6 FWD F32 test failed!!" << std::endl;
                else if(NameofKernel == GAUDI_KERNEL_RELU6_BWD_F32)
                    std::cout << "Relu6 BWD F32 test failed!!" << std::endl;
                else if(NameofKernel == GAUDI_KERNEL_RELU_FWD_F32)
                    std::cout << "Relu FWD F32 test failed!!" << std::endl;
                else
                    std::cout << "Relu BWD F32 test failed!!" << std::endl;
                return -1;
            }
        }
        if(NameofKernel == GAUDI_KERNEL_RELU6_FWD_F32)
            std::cout << "Relu6 FWD F32 test pass!!" << std::endl;
        else if(NameofKernel == GAUDI_KERNEL_RELU6_BWD_F32)
            std::cout << "Relu6 BWD F32 test pass!!" << std::endl;
        else if(NameofKernel == GAUDI_KERNEL_RELU_FWD_F32)
            std::cout << "Relu FWD F32 test pass!!" << std::endl;
        else
            std::cout << "Relu BWD F32 test pass!!" << std::endl;

    }
    else if ((NameofKernel == GAUDI_KERNEL_RELU6_FWD_BF16) || (NameofKernel == GAUDI_KERNEL_RELU6_BWD_BF16)
           || (NameofKernel == GAUDI_KERNEL_RELU_FWD_BF16) || (NameofKernel == GAUDI_KERNEL_RELU_BWD_BF16))
    {
        bfloat16_5DTensor gradin(fmInitializer);
        gradin.InitRand(-10.0f, 10.0f);    
        bfloat16_5DTensor input(fmInitializer);
        input.InitRand(-10.0f, 10.0f);
        bfloat16_5DTensor output(fmInitializer);
        bfloat16_5DTensor output_ref(fmInitializer);

        // generate input for query call
        m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI;

        if(NameofKernel == GAUDI_KERNEL_RELU6_FWD_BF16 || NameofKernel == GAUDI_KERNEL_RELU_FWD_BF16)
        {
            m_in_defs.inputTensorNr = 1;
            relu6_bf16_reference_implementation(gradin, input, output_ref, NameofKernel);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), input);
        }
        else
        {
            m_in_defs.inputTensorNr = 2;
            relu6_bf16_reference_implementation(gradin, input, output_ref, NameofKernel);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), gradin);
            LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), input);
        }

        m_in_defs.outputTensorNr = 1;
        LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), output);

        kernelCount = 0;
        result = GetKernelNames(kernelNames, &kernelCount, gcapi::DEVICE_ID_GAUDI);
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
        std::vector<TensorDesc> vec;
        if(NameofKernel == GAUDI_KERNEL_RELU6_BWD_BF16 || NameofKernel == GAUDI_KERNEL_RELU_BWD_BF16)
            vec.push_back(gradin.GetTensorDescriptor());
        vec.push_back(input.GetTensorDescriptor());
        vec.push_back(output.GetTensorDescriptor());
        // execute a simulation of the kernel using TPC simulator,
        TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
        ReleaseKernelNames(kernelNames, kernelCount);
        output.Print(0);
        output_ref.Print(0);
        bfloat16 tmp;
        for (int element = 0 ; element <  output_ref.ElementCount() ; element++)
        {
            if (tmp.abs(output.Data()[element] - output_ref.Data()[element]) > 1e-6)
            {
                if(NameofKernel == GAUDI_KERNEL_RELU6_FWD_BF16)
                    std::cout << "Relu6 FWD BF16 test failed!!" << std::endl;
                else if(NameofKernel == GAUDI_KERNEL_RELU6_BWD_BF16)
                    std::cout << "Relu6 BWD BF16 test failed!!" << std::endl;
                else if(NameofKernel == GAUDI_KERNEL_RELU_FWD_BF16)
                    std::cout << "Relu FWD BF16 test failed!!" << std::endl;
                else
                    std::cout << "Relu BWD BF16 test failed!!" << std::endl;
                return -1;
            }
        }
        if(NameofKernel == GAUDI_KERNEL_RELU6_FWD_BF16)
            std::cout << "Relu6 FWD BF16 test pass!!" << std::endl;
        else if(NameofKernel == GAUDI_KERNEL_RELU6_BWD_BF16)
            std::cout << "Relu6 BWD BF16 test pass!!" << std::endl;
        if(NameofKernel == GAUDI_KERNEL_RELU_FWD_BF16)
            std::cout << "Relu FWD BF16 test pass!!" << std::endl;
        else
            std::cout << "Relu BWD BF16 test pass!!" << std::endl;

    }

    return 0;
}

