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

#ifndef CAST_GAUDI_TEST_HPP
#define CAST_GAUDI_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "cast_gaudi.hpp"
#include "ConvInterface.h"

class CastGaudiTest : public TestBase
{
public:
    CastGaudiTest() {}
    ~CastGaudiTest() {}
    int runTest();

    static void cast_bf16_to_f32_ref(
         const test::Tensor<bfloat16,4>& input,
         test::Tensor<float,4>& output,
         const IndexSpace& indexSpace);

    static void cast_f32_to_bf16_ref(
        const test::Tensor<float,4>& ifm,
        test::Tensor<bfloat16,4>& ofm,
        const IndexSpace& indexSpace);
private:
    CastGaudiTest(const CastGaudiTest& other) = delete;
    CastGaudiTest& operator=(const CastGaudiTest& other) = delete;

};

#endif /* CAST_TEST_HPP */
