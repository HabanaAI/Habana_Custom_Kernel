/**********************************************************************
Copyright (c) 2022 Habana Labs.

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

#ifndef CAST_F16_TO_I16_GAUDI2_TEST_HPP
#define CAST_F16_TO_I16_GAUDI2_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "cast_f16_to_i16_gaudi2.hpp"

class CastF16toI16Gaudi2Test : public TestBase
{
public:
    CastF16toI16Gaudi2Test() {}
    ~CastF16toI16Gaudi2Test() {}
    int runTest();

    static void cast_f16_to_i16_ref(
         const test::Tensor<float16,5>& input,
         test::Tensor<int16_t,5>& output,
         const IndexSpace& indexSpace, unsigned int rounding);

private:
    CastF16toI16Gaudi2Test(const CastF16toI16Gaudi2Test& other) = delete;
    CastF16toI16Gaudi2Test& operator=(const CastF16toI16Gaudi2Test& other) = delete;

};

#endif /* CAST_F16_TO_I16_TEST_HPP */
