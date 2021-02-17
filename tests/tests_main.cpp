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


#include "filter_2d_f32_test.hpp"
#include "filter_fwd_2d_bf16_test.hpp"
#include "filter_2d_i8_w33_s11_test.hpp"
#include "printf_tests.hpp"
#include "sparse_lengths_sum_test.hpp"
#include "softmax_f32_test.hpp"
#include "softmax_bf16_test.hpp"
#include "cast_test.hpp"
#include "cast_gaudi_test.hpp"
#include "leakyrelu_f32_test.hpp"
#include "batchnorm_f32_test.hpp"
#include "leakyrelu_f32_gaudi_test.hpp"
#include "sparse_lengths_sum_bf16_test.hpp"
#include "customdiv_fwd_f32_test.hpp"


int main(int argc, char** argv)
{
    int result = 0;

    Filter2DF32Test testFilter;
    testFilter.SetUp();
    result = testFilter.runTest();
    testFilter.TearDown();
    if (result != 0)
    {
        return result;
    }

    FilterFwd2DBF16Test test_bf16;
    test_bf16.SetUp();
    result = test_bf16.runTest();
    test_bf16.TearDown();
    if (result != 0)
    {
        return result;
    }

    PrintfTest testPrint;
    testPrint.SetUp();
    result = testPrint.runTest();
    testPrint.TearDown();
    if (result != 0)
    {
        return result;
    }

    SparseLengthsSumTest testSparseLenGoya;
    testSparseLenGoya.SetUp();
    result = testSparseLenGoya.runTest();
    testSparseLenGoya.TearDown();
    if (result != 0)
    {
        return result;
    }

    Filter2DI8W33S11Test testFilter_i8;
    testFilter_i8.SetUp();
    result = testFilter_i8.runTest();
    testFilter_i8.TearDown();
    if (result != 0)
    {
        return result;
    }

    SoftMaxF32Test testSoftMax;
    testSoftMax.SetUp();
    result = testSoftMax.runTest();
    testSoftMax.TearDown();
    if (result != 0)
    {
        return result;
    }

    SoftMaxBF16Test testSoftMaxBF16;
    testSoftMaxBF16.SetUp();
    result = testSoftMaxBF16.runTest();
    testSoftMaxBF16.TearDown();
    if (result != 0)
    {
        return result;
    }

    CastTest testCastGoya;
    testCastGoya.SetUp();
    result = testCastGoya.runTest();
    testCastGoya.TearDown();
    if (result != 0)
    {
        return result;
    }

    CastGaudiTest testCaseGaudi;
    testCaseGaudi.SetUp();
    result = testCaseGaudi.runTest();
    testCaseGaudi.TearDown();
    if (result != 0)
    {
        return result;
    }

    LeakyReluF32Test testLeakyRelu;
    testLeakyRelu.SetUp();
    result = testLeakyRelu.runTest();
    testLeakyRelu.TearDown();
    if (result != 0)
    {
        return result;
    }

    BatchNormF32Test testBatchNorm;
    testBatchNorm.SetUp();
    result = testBatchNorm.runTest();
    testBatchNorm.TearDown();
    if (result != 0)
    {
        return result;
    }

    LeakyReluF32GaudiTest testLeakyReluGaudi;
    testLeakyReluGaudi.SetUp();
    result = testLeakyReluGaudi.runTest();
    testLeakyReluGaudi.TearDown();
    if (result != 0)
    {
        return result;
    }

    SparseLengthsSumBF16Test testSparseLenGaudi;
    testSparseLenGaudi.SetUp();
    result = testSparseLenGaudi.runTest();
    testSparseLenGaudi.TearDown();
    if (result != 0)
    {
        return result;
    }

    CustomdivFwdF32Test testCustomDivFwdF32;
    testCustomDivFwdF32.SetUp();
    result = testCustomDivFwdF32.runTest();
    testCustomDivFwdF32.TearDown();
    if (result != 0)
    {
        return result;
    }

    std::cout << "All tests passs!" <<std::endl;
    return 0;
}
