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

#include "tensor.h"
#include "softmax_bf16_gaudi2_test.hpp"
#include "entry_points.hpp"

inline float zeroSubNormals2(const float fp)
{
    if (std::fpclassify(fp) == FP_SUBNORMAL)
    {
        return 0.f;
    }
    return fp;
}

template<typename T, typename T_res>
void Mul2(T* op1, T* op2, T_res* res)
{
    *res = (T_res)(*op1) * (T_res)(*op2);
}

void SoftMaxBF16Gaudi2Test::softmax_reference_implementation(
        test::Tensor<bfloat16,2>& input,
        test::Tensor<bfloat16,2>& output,
        int axis)
{

    // Inner most loop contains the axis
    const int idx_lut[2][2] = {{0, 1},     // reduction dimension 0
                               {1, 0}};    // reduction dimension 1
    const int * pIdx = idx_lut[axis];

    const int depth  = pIdx[0];
    const int width  = pIdx[1];

    // depth
    const int depthStep  = 1;
    const int depthStart = 0;
    const int depthEnd   = (int) input.Size(depth);

    // width
    const int widthStep  = 1;
    const int widthStart = 0;
    const int widthEnd   = (int) input.Size(width);

    int inputCoords[gcapi::MAX_TENSOR_DIM] = { 0 };
    int ofmCoords[gcapi::MAX_TENSOR_DIM] = { 0 };

    for (int w = widthStart; w < widthEnd; w += widthStep)
    {
        inputCoords[width] = w;   ofmCoords[width] = w;

        float sumExp = 0.f;
        float temp_sumExp = 0.f;
        // Calculate and find the maximum value
        bfloat16 Xmax=std::numeric_limits<bfloat16>::infinity();
        Xmax = ((bfloat16)0.0)-Xmax;
        for (int d = depthStart; d < depthEnd; d += depthStep)
        {
            inputCoords[depth] = d;
            bfloat16 x =  input.ElementAt( inputCoords );
            Xmax = x>Xmax ? x :  Xmax  ;
        }
        // Calculate sum of exponents along the axis and convert to bf16
        for (int d = depthStart; d < depthEnd; d += depthStep)
        {
            inputCoords[depth] = d;

            float temp_expX = expf( input.ElementAt( inputCoords ) - Xmax );
            temp_expX = zeroSubNormals2(temp_expX);
            float expX = floatTobf16ToFloat(temp_expX);
            temp_sumExp += expX;
            sumExp = floatTobf16ToFloat(temp_sumExp);
        }

        float temp_recip = zeroSubNormals2( 1.f / sumExp);
        float recip = floatTobf16ToFloat(temp_recip);

        float res = 0;
        float temp_res = 0;
        // x = exponent(x) / sum_of_exponents
        for (int d = depthStart; d < depthEnd; d += depthStep)
        {
            inputCoords[depth] = d;   ofmCoords[depth] = d;

            float temp_expX = expf( input.ElementAt( inputCoords ) - Xmax);
            temp_expX = zeroSubNormals2(temp_expX);
            float expX = floatTobf16ToFloat(temp_expX);
            Mul2<float, float>(&expX, &recip, &temp_res);
            temp_res = zeroSubNormals2(temp_res);
            res = floatTobf16ToFloat(temp_res);
            output.SetElement(ofmCoords, res);
         }
     }
}

 int SoftMaxBF16Gaudi2Test::runTest()
 {
    // Initialize input data
    const int fm_dim1 = 9;
    const int fm_dim2  = 4;
    bfloat16 tmp;

    unsigned int fmInitializer[] = {fm_dim1, fm_dim2};
    bfloat16_2DTensor input(fmInitializer);
    input.FillWithData();

    bfloat16_2DTensor ofm(fmInitializer);
    bfloat16_2DTensor ofm_ref(fmInitializer);

    SoftMaxBF16Gaudi2::SoftMaxParam def;

    // Test for axis 0 softmax kernel
    def.axis = 0;

    // execute reference implementation of the kernel.
    softmax_reference_implementation(input,
                                     ofm_ref,
                                     def.axis);

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI2;
    m_in_defs.inputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]),input );

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]),ofm );

    m_in_defs.NodeParams = &def;

    char**   kernelNames = nullptr;
    unsigned kernelCount = 0;
    gcapi::GlueCodeReturn_t result = GetKernelNames(kernelNames, &kernelCount, gcapi::DEVICE_ID_GAUDI2);
    kernelNames = new char*[kernelCount];
    for (unsigned i = 0; i < kernelCount; i++)
    {
        kernelNames[i] = new char[gcapi::MAX_NODE_NAME];
    }    
    result = GetKernelNames(kernelNames, &kernelCount, gcapi::DEVICE_ID_GAUDI2);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Can't get kernel name!! " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_SOFTMAX_FCD_BF16]);
    result  = HabanaKernel(&m_in_defs,&m_out_defs);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "glue test failed!!" << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc> vec;
    vec.push_back(input.GetTensorDescriptor());
    vec.push_back(ofm.GetTensorDescriptor());

    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ofm.Print(0);
    ofm_ref.Print(0);
    for (int element = 0 ; element <  ofm_ref.ElementCount() ; element++)
    {
        if (tmp.abs(ofm.Data()[element] - ofm_ref.Data()[element])  > 1e-8)
        {
            std::cout << "Softmax BF16 FCD Gaudi 2 test failed!!" << std::endl;
            ReleaseKernelNames(kernelNames, kernelCount);
            return -1;
        }
    }
    if (m_out_defs.auxiliaryTensors[0].pData)
    {
        delete [] (int8_t*)m_out_defs.auxiliaryTensors[0].pData;
        m_out_defs.auxiliaryTensors[0].pData = NULL;
    }

    std::cout << "Softmax BF16 FCD Gaudi 2 test pass!!" << std::endl;

    // Test for axis 1 softmax kernel
    def.axis = 1;

    // execute reference implementation of the kernel.
    softmax_reference_implementation(input,
                                     ofm_ref,
                                     def.axis);

    m_in_defs.NodeParams = &def;

    // make the call into the glue code.
    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_SOFTMAX_NONFCD_BF16]);
    result  = HabanaKernel(&m_in_defs,&m_out_defs);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc> vec2;
    vec2.push_back(input.GetTensorDescriptor());
    vec2.push_back(ofm.GetTensorDescriptor());

    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec2, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount);
    ofm.Print(0);
    ofm_ref.Print(0);
    for (int element = 0 ; element <  ofm_ref.ElementCount() ; element++)
    {
        if (tmp.abs(ofm.Data()[element] - ofm_ref.Data()[element])  > 1e-7)
        {
            std::cout << "Softmax BF 16 Non FCD Gaudi 2 test failed!!" << std::endl;
            return -1;
        }
    }

    std::cout << "Softmax BF 16 axis Non FCD Gaudi 2 test pass!!" << std::endl;
    return 0;
 }



