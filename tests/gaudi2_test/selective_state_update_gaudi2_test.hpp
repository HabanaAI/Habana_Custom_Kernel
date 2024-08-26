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

#ifndef SELECTIVE_STATE_UPDATE_GAUDI2_TEST_HPP
#define ELECTIVE_STATE_UPDATE_GAUDI2_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "selective_state_update_gaudi2.hpp"
#include "entry_points.hpp"

class SelectiveStateUpdateGaudi2Test : public TestBase
{
public:
    SelectiveStateUpdateGaudi2Test() {}
    ~SelectiveStateUpdateGaudi2Test() {}
    int runTest(Gaudi2_Kernel_Name_e NameofKernel);

    static void selective_state_update_fp32_ref(
         const test::Tensor<float,4>& state_M,
         const test::Tensor<float,4>& x_M,
         const test::Tensor<float,4>& dt_M,
         const test::Tensor<float,4>& A_M,
         const test::Tensor<float,4>& B_M,
         const test::Tensor<float,4>& C_M,
         const test::Tensor<float,4>& D_M,
         const test::Tensor<float,4>& dt_bias_M,
         const test::Tensor<float,4>& z_M,
         test::Tensor<float,4>& output,
         const IndexSpace& indexSpace, const int head_group,
         SelectiveStateUpdateGaudi2::SSUParam def, Gaudi2_Kernel_Name_e NameofKernel);

    static void selective_state_update_bf16_ref(
         const test::Tensor<bfloat16,4>& state_M,
         const test::Tensor<bfloat16,4>& x_M,
         const test::Tensor<bfloat16,4>& dt_M,
         const test::Tensor<bfloat16,4>& A_M,
         const test::Tensor<bfloat16,4>& B_M,
         const test::Tensor<bfloat16,4>& C_M,
         const test::Tensor<bfloat16,4>& D_M,
         const test::Tensor<bfloat16,4>& dt_bias_M,
         const test::Tensor<bfloat16,4>& z_M,
         test::Tensor<bfloat16,4>& output,
         const IndexSpace& indexSpace, const int head_group,
         SelectiveStateUpdateGaudi2::SSUParam def,
         Gaudi2_Kernel_Name_e NameofKernel);
private:
    SelectiveStateUpdateGaudi2Test(const SelectiveStateUpdateGaudi2Test& other) = delete;
    SelectiveStateUpdateGaudi2Test& operator=(const SelectiveStateUpdateGaudi2Test& other) = delete;

};

#endif /* ELECTIVE_STATE_UPDATE_GAUDI2_TEST_HPP */
