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

#ifndef MYGATHER_GAUDI2_TEST_HPP
#define MYGATHER_GAUDI2_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "mygather_gaudi2.hpp"
#include "entry_points.hpp"

class MygatherGaudi2Test : public TestBase
{
public:
    MygatherGaudi2Test() {}
    ~MygatherGaudi2Test() {}
    int runTest(Gaudi2_Kernel_Name_e NameofKernel);

    static void mygather_fp32_ref(
         const test::Tensor<float,4>& in_M,
         const test::Tensor<float,4>& start_M,
         test::Tensor<float,4>& output,
         const IndexSpace& indexSpace, 
         MygatherGaudi2::MygatherParam def);

    static void mygather_bf16_ref(
         const test::Tensor<bfloat16,4>& in_M,
         const test::Tensor<bfloat16,4>& start_M,
         test::Tensor<bfloat16,4>& output,
         const IndexSpace& indexSpace,
         MygatherGaudi2::MygatherParam def);
private:
    MygatherGaudi2Test(const MygatherGaudi2Test& other) = delete;
    MygatherGaudi2Test& operator=(const MygatherGaudi2Test& other) = delete;

};

#endif /* MYGATHER_GAUDI2_TEST_HPP */