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
#ifndef SPARSE_LENGTH_SUM_BF16_TEST_HPP
#define SPARSE_LENGTH_SUM_BF16_TEST_HPP

#include "tensor.h"
#include "test_base.hpp"
#include "sparse_lengths_sum_bf16.hpp"

class SparseLengthsSumBF16Test : public TestBase {
public:
    SparseLengthsSumBF16Test() {}
    ~SparseLengthsSumBF16Test() {}

    int runTest();

    void SparseLengthsSumRefImplementation(
            bfloat16_2DTensor &input_tensor,
            int32_1DTensor &indices_tensor,
            int32_1DTensor &lengths_tensor,
            float_2DTensor &output_tensor);

private:
    SparseLengthsSumBF16Test(const SparseLengthsSumBF16Test& other) = delete;
    SparseLengthsSumBF16Test& operator=(const SparseLengthsSumBF16Test& other) = delete;
};

#endif /*SPARSE_LENGTH_SUM_BF16_TEST_HPP*/
