/**********************************************************************
Copyright (c) 2018 Habana Labs.

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

#ifndef BATCHNORM_F32_TEST_HPP
#define BATCHNORM_F32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "batch_norm_f32.hpp"

class BatchNormF32Test : public TestBase
{
public:
    BatchNormF32Test() {}
    ~BatchNormF32Test() {}
    int runTest();

    static void batchnorm_fwd_reference_implementation(
          test::Tensor<float, 4> &ofm,
          test::Tensor<float, 1> &mean,
          test::Tensor<float, 1> &istd,
          const test::Tensor<float, 4> &ifm,
          const test::Tensor<float, 1> &beta,
          const test::Tensor<float, 1> &gamma,
          const float momentum);
private:
    BatchNormF32Test(const BatchNormF32Test& other) = delete;
    BatchNormF32Test& operator=(const BatchNormF32Test& other) = delete;

};

#endif /* BATCHNORM_F32_TEST_HPP */

