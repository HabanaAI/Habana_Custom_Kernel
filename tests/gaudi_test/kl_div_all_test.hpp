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

#ifndef KL_DIV_ALL_TEST_HPP
#define KL_DIV_ALL_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "kl_div_all.hpp"
#include "entry_points.hpp"

class KLDivAllTest : public TestBase
{
public:
    KLDivAllTest() {}
    ~KLDivAllTest() {}
    int runTest(Gaudi_Kernel_Name_e NameofKernel);

    inline static void kldiv_f32_fwd_reference_implementation(
            const float_5DTensor& inputX,
            const float_5DTensor& inputY,
            float_1DTensor& output,
            const float invLen, const int log_target);

    inline static void kldiv_f32_bwd_reference_implementation(
            const float_1DTensor& gradIn,
            const float_5DTensor& inputX,
            const float_5DTensor& inputY,
            float_5DTensor& output,
            const float invLen, const int log_target);


private:
    KLDivAllTest(const KLDivAllTest& other) = delete;
    KLDivAllTest& operator=(const KLDivAllTest& other) = delete;

};


#endif /* KL_DIV_ALL_TEST_HPP */
