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

#ifndef LEAKYRELU_F32_TEST_HPP
#define LEAKYRELU_F32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "leakyrelu_f32.hpp"

class LeakyReluF32Test : public TestBase
{
public:
    LeakyReluF32Test() {}
    ~LeakyReluF32Test() {}
    int runTest();

    inline static void leakyrelu_reference_implementation(
            const float_4DTensor& input,
            float_4DTensor& output,
            const float alpha);
private:
    LeakyReluF32Test(const LeakyReluF32Test& other) = delete;
    LeakyReluF32Test& operator=(const LeakyReluF32Test& other) = delete;

};


inline void LeakyReluF32Test::leakyrelu_reference_implementation(
        const float_4DTensor& input,
        float_4DTensor& output,
        const float alpha)
{
    int coords[4] = {0};
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
                    float y = (x < 0.0f) ? (x * alpha) : x;
                    output.SetElement(coords, y);
                }
            }
        }
    }
}

inline int LeakyReluF32Test::runTest()
{
    const int height = 5;
    const int width  = 5;
    const int depth  = 100;
    const int batch  = 1;

    unsigned int fmInitializer[] = {depth, width, height, batch};

    float_4DTensor input(fmInitializer);
    input.InitRand(-10.0f, 10.0f);

    float_4DTensor output(fmInitializer);
    float_4DTensor output_ref(fmInitializer);

    LeakyReluF32::LeakyReluParam param;
    param.alpha = 0.00034;

    // execute reference implementation of the kernel.
    leakyrelu_reference_implementation(input, output_ref, param.alpha);

    // generate input for query call
    m_in_defs.NodeParams = &param;

    m_in_defs.inputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), input);

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), output);

    LeakyReluF32 kernelClass;
    // make the call into the glue code.
    gcapi::GlueCodeReturn_t result = kernelClass.GetGcDefinitions(&m_in_defs, &m_out_defs);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "glue test failed!! " << result << std::endl;
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDescriptor> vec;
    vec.push_back(input.GetTensorDescriptor());
    vec.push_back(output.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    output.Print(0);
    output_ref.Print(0);
    for (int element = 0 ; element <  output_ref.ElementCount() ; element++)
    {
        if (output.Data()[element] != output_ref.Data()[element])
        {
            std::cout << "test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "test pass!!" << std::endl;
    return 0;
}

#endif /* LEAKYRELU_F32_TEST_HPP */

