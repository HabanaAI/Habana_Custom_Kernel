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

#ifndef SOFTMAX_BF16_GAUDI2_TEST_HPP
#define SOFTMAX_BF16_GAUDI2_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "softmax_bf16_gaudi2.hpp"

class SoftMaxBF16Gaudi2Test : public TestBase
{
public:
    SoftMaxBF16Gaudi2Test() {}
    ~SoftMaxBF16Gaudi2Test() {}
    int runTest();

    static void softmax_reference_implementation(
         test::Tensor<bfloat16,2>& input,
         test::Tensor<bfloat16,2>& output,
         int axis);
private:
    SoftMaxBF16Gaudi2Test(const SoftMaxBF16Gaudi2Test& other) = delete;
    SoftMaxBF16Gaudi2Test& operator=(const SoftMaxBF16Gaudi2Test& other) = delete;

};


#endif /* SOFTMAX_BF16_TEST_HPP */

