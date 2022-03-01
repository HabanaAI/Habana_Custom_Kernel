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

#ifndef RELU6_ALL_TEST_HPP
#define RELU6_ALL_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "relu6_all.hpp"
#include "entry_points.hpp"

class Relu6AllTest : public TestBase
{
public:
    Relu6AllTest() {}
    ~Relu6AllTest() {}
    int runTest(Gaudi_Kernel_Name_e NameofKernel);

    static void relu6_f32_reference_implementation(
            const float_5DTensor& gradin,  
            const float_5DTensor& input,
            float_5DTensor& output, Gaudi_Kernel_Name_e mode);

    static void relu6_bf16_reference_implementation(
            const bfloat16_5DTensor& gradin,        
            const bfloat16_5DTensor& input,
            bfloat16_5DTensor& output, Gaudi_Kernel_Name_e mode);
            
private:
    Relu6AllTest(const Relu6AllTest& other) = delete;
    Relu6AllTest& operator=(const Relu6AllTest& other) = delete;

};


#endif /* RELU6_ALL_TEST_HPP */

