/**********************************************************************
Copyright (c) 2020 Habana Labs.

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

#ifndef FILTER_FWD_2D_BF16_TEST_HPP
#define FILTER_FWD_2D_BF16_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "filter_fwd_2d_bf16.hpp"

class FilterFwd2DBF16Test : public TestBase
{
public:
    FilterFwd2DBF16Test() {}
    ~FilterFwd2DBF16Test() {}
    int runTest();

    static void filter_2d_reference_implementation(
        const test::Tensor<bfloat16,4>& ifm,
        const test::Tensor<bfloat16,3>& filter,
        test::Tensor<bfloat16,4>& ofm,
        const SpatialReductionKernels::SpatialReduction2DDef& layer_def,
        const IndexSpace& indexSpace);
private:
    FilterFwd2DBF16Test(const FilterFwd2DBF16Test& other) = delete;
    FilterFwd2DBF16Test& operator=(const FilterFwd2DBF16Test& other) = delete;

};

#endif /* FILTER_FWD_2D_BF16_TEST_HPP */

