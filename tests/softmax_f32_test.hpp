/**********************************************************************
Copyright (c) 2018 Habana Labs.

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

#ifndef SOFTMAX_F32_TEST_HPP
#define SOFTMAX_F32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "softmax_f32.hpp"

class SoftMaxF32Test : public TestBase
{
public:
    SoftMaxF32Test() {}
    ~SoftMaxF32Test() {}
    int runTest();

    inline static void softmax_reference_implementation(
         test::Tensor<float,2>& input,
         test::Tensor<float,2>& output,
         int axis);
private:
    SoftMaxF32Test(const SoftMaxF32Test& other) = delete;
    SoftMaxF32Test& operator=(const SoftMaxF32Test& other) = delete;

};

inline float zeroSubNormals(const float fp)
{
    if (std::fpclassify(fp) == FP_SUBNORMAL)
    {
        return 0.f;
    }
    return fp;
}

template<typename T, typename T_res>
void Mul(T* op1, T* op2, T_res* res1, T_res* res2)
{
    *res1 = (T_res)(*op1) * (T_res)(*op2);
}

inline void SoftMaxF32Test::softmax_reference_implementation(
        test::Tensor<float,2>& input,
        test::Tensor<float,2>& output,
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
        // Calculate sum of exponents along the axis
        for (int d = depthStart; d < depthEnd; d += depthStep)
        {
            inputCoords[depth] = d;

            float expX = expf( input.ElementAt( inputCoords ) );
            expX = zeroSubNormals(expX);
            sumExp += expX;
        }

        float recip = zeroSubNormals( 1.f / sumExp);

        float res = 0;
        // x = exponent(x) / sum_of_exponents
        for (int d = depthStart; d < depthEnd; d += depthStep)
        {
            inputCoords[depth] = d;   ofmCoords[depth] = d;

            float expX = expf( input.ElementAt( inputCoords ) );
            expX = zeroSubNormals(expX);
            Mul<float, float>(&expX, &recip, &res, NULL );
            res = zeroSubNormals(res);
            output.SetElement(ofmCoords, res);
         }
     }
}

 inline int SoftMaxF32Test::runTest()
 {
    // Initialize input data
    const int fm_dim1 = 9;
    const int fm_dim2  = 4;

    unsigned int fmInitializer[] = {fm_dim1, fm_dim2};
    float_2DTensor input(fmInitializer);
    input.FillWithData();

    float_2DTensor ofm(fmInitializer);
    float_2DTensor ofm_ref(fmInitializer);

    SoftMaxF32::SoftMaxParam def;

    // Test for axis 0 softmax kernel
    def.axis = 0;

    // execute reference implementation of the kernel.
    softmax_reference_implementation(input,
                                     ofm_ref,
                                     def.axis);

    // generate input for query call
    m_in_defs.inputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]),input );

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]),ofm );

    m_in_defs.NodeParams = &def;

    SoftMaxF32 kernelClass;
    // make the call into the glue code.
    gcapi::GlueCodeReturn_t result = kernelClass.GetGcDefinitions(&m_in_defs,&m_out_defs);
    // Declaration of auxiliary tensor
    int8_1DTensor aux_tensor({100});
    // Allocate memory for aux tensor if not allocated
    if (result == gcapi::GLUE_INSUFICIENT_AUX_BUFFER_SIZE)
    {
        if (m_out_defs.auxiliaryTensors[0].pData)
        {
            delete [] (int8_t*)m_out_defs.auxiliaryTensors[0].pData;
            m_out_defs.auxiliaryTensors[0].pData = NULL;
        }

        m_out_defs.auxiliaryTensors[0].pData =
                                    new int8_t[m_out_defs.auxiliaryTensors[0].bufferSize / sizeof(int8_t)];
        // second call of glue-code to load Auxiliary data.
        result = kernelClass.GetGcDefinitions(&m_in_defs,&m_out_defs);
        // AUXILIARY TENSOR init based on parameters got from glue code
        aux_tensor.Init(m_out_defs.auxiliaryTensors[0].geometry.sizes,
                                    (int8_t*)m_out_defs.auxiliaryTensors[0].pData);
    }

    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "glue test failed!!" << result << std::endl;
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDescriptor> vec;
    vec.push_back(input.GetTensorDescriptor());
    vec.push_back(ofm.GetTensorDescriptor());
    vec.push_back(aux_tensor.GetTensorDescriptor());

    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ofm.Print(0);
    ofm_ref.Print(0);
    for (int element = 0 ; element <  ofm_ref.ElementCount() ; element++)
    {
        if (std::abs(ofm.Data()[element] - ofm_ref.Data()[element])  > 1e-8)
        {
            std::cout << "test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "test pass!!" << std::endl;

    // Test for axis 1 softmax kernel
    def.axis = 1;

    // execute reference implementation of the kernel.
    softmax_reference_implementation(input,
                                     ofm_ref,
                                     def.axis);

    m_in_defs.NodeParams = &def;

    // make the call into the glue code.
    result = kernelClass.GetGcDefinitions(&m_in_defs,&m_out_defs);

    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "glue test failed!! " << result << std::endl;
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDescriptor> vec2;
    vec2.push_back(input.GetTensorDescriptor());
    vec2.push_back(ofm.GetTensorDescriptor());

    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec2, m_in_defs, m_out_defs);
    ofm.Print(0);
    ofm_ref.Print(0);
    for (int element = 0 ; element <  ofm_ref.ElementCount() ; element++)
    {
        if (std::abs(ofm.Data()[element] - ofm_ref.Data()[element])  > 1e-7)
        {
            std::cout << "test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "test pass!!" << std::endl;
    return 0;
 }

#endif /* SOFTMAX_F32_TEST_HPP */

