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

#ifndef AVG_POOL_2D_F32_TEST_HPP
#define AVG_POOL_2D_F32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "avg_pool_2d_f32.hpp"
#include "entry_points.hpp"

class AvgPool2DF32Test : public TestBase
{
public:
    AvgPool2DF32Test() {}
    ~AvgPool2DF32Test() {}
    int runTest(Gaudi_Kernel_Name_e NameofKernel);

    static void avg_pool_2d_fwd_reference_implementation(
        const test::Tensor<float,4>& ifm,
        test::Tensor<float,4>& ofm,
        const AvgPool2dF32::AvgPool2DParam& def,
        const IndexSpace& indexSpace);

    static void avg_pool_2d_get_intospacePixelsInArea(
        const test::Tensor<float,4>& ifm,
        test::Tensor<int32_t,2>& numOfSourcefm,
        const AvgPool2dF32::AvgPool2DParam& def,
        const IndexSpace& indexSpace);

    static void avg_pool_2d_bwd_reference_implementation(
        const test::Tensor<float,4>& ifm,
        test::Tensor<float,4>& ofm,
        test::Tensor<int32_t,2>& numOfSourcefm,
        const AvgPool2dF32::AvgPool2DParam& def,
        const IndexSpace& indexSpace);

private:
    AvgPool2DF32Test(const AvgPool2DF32Test& other) = delete;
    AvgPool2DF32Test& operator=(const AvgPool2DF32Test& other) = delete;
    typedef struct coord_t
    {
        int c;
        int w;
        int h;
        int b;
    } coord_t;    

};

#endif /* AVG_POOL_2D_F32_TEST_HPP */

