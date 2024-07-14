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

#include <iostream>
#include "filter_fwd_2d_bf16_test.hpp"
#include "softmax_bf16_test.hpp"
#include "softmax_bf16_gaudi2_test.hpp"
#include "cast_gaudi_test.hpp"
#include "batchnorm_f32_test.hpp"
#include "leakyrelu_f32_gaudi_test.hpp"
#include "sparse_lengths_sum_bf16_test.hpp"
#include "customdiv_fwd_f32_test.hpp"
#include "relu6_all_test.hpp"
#include "matrix_mul_fwd_f32_test.hpp"
#include "spatial_conv_f32_test.hpp"
#include "sin_f32_test.hpp"
#include "add_f32_test.hpp"
#include "avg_pool_2d_f32_test.hpp"
#include "avg_pool_2d_f32_gaudi2_test.hpp"
#include "cast_f16_to_i16_gaudi2_test.hpp"
#include "searchsorted_f32_test.hpp"
#include "gather_fwd_i32_test.hpp"
#include "kl_div_all_test.hpp"
#include "user_lut_gaudi2_test.hpp"

int check_arg(int argc, char** argv, const char* device, const char* test)
{
    if( argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2], device) ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2], test) ==0))) ||
        (argc == 5 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2], device) ==0))  
        && (((strcmp(argv[3], "--test") ==0) || (strcmp(argv[3], "-t") ==0))
        && (strcmp(argv[4], test) ==0))) ||
        (argc == 5 && (((strcmp(argv[3], "--device") ==0) || (strcmp(argv[3], "-d") ==0))
        && (strcmp(argv[4], device) ==0))  
        && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2], test) ==0))))
        return 1;
    else
        return 0;
}
int main(int argc, char** argv)
{
    int result = 0;
    static int testCount = 0;

    if(argc == 2 && ((strcmp(argv[1], "--help") ==0) || (strcmp(argv[1],"-h") ==0)))
    {
        std::cout << argv[0] << " " << "[options]" << std::endl <<
            "Options:" << std::endl <<
            "N/A                        Run all test cases" << std::endl <<
            "-h | --help                Print this help" << std::endl <<
            "-d | --device <DeviceName> Run only kernels for the DeviceName" << std::endl <<
            "-t | --test  <TestName>    Run <TestName>> only   " << std::endl <<
            "DeviceName:" << std::endl <<
            "Gaudi                      Run all Gaudi kernels only   " << std::endl <<
            "Gaudi2                     Run all Gaudi2 kernels only   " << std::endl <<            
            "TestName:" << std::endl <<
            "FilterFwd2DBF16Test        Run FilterFwd2DBF16Test only   " << std::endl <<
            "SoftMaxBF16Test            Run SoftMaxBF16Test only   " << std::endl <<
            "CastGaudiTest              Run CastGaudiTest only   " << std::endl <<
            "BatchNormF32Test           Run BatchNormF32Test only   " << std::endl <<
            "LeakyReluF32GaudiTest      Run LeakyReluF32GaudiTest only   " << std::endl <<
            "SparseLengthsBF16Test      Run SparseLengthsBF16Test only   " << std::endl <<
            "CustomdivFwdF32Test        Run CustomdivFwdF32Test only   " << std::endl <<
            "Relu6FwdF32                Run Relu6FwdF32 only   " << std::endl <<
            "Relu6BwdF32                Run Relu6BwdF32 only   " << std::endl <<
            "Relu6FwdBF16               Run Relu6FwdBF16 only   " << std::endl <<
            "Relu6BwdBF16               Run Relu6BwdBF16 only   " << std::endl <<
            "ReluFwdF32                 Run ReluFwdF32 only   " << std::endl <<
            "ReluBwdF32                 Run ReluBwdF32 only   " << std::endl <<
            "ReluFwdBF16                Run ReluFwdBF16 only   " << std::endl <<
            "ReluBwdBF16                Run ReluBwdBF16 only   " << std::endl <<
            "MatrixMulFwdF32Test        Run MatrixMulFwdF32Test only   " << std::endl <<
            "SpatialConvF32Test         Run SpatialConvF32Test only   " << std::endl <<
            "SinF32Test                 Run SinF32Test only   " << std::endl <<
            "AddF32Test                 Run AddF32Test only   " << std::endl <<
            "AvgPool2DFwdF32Test        Run AvgPool2DFwdF32Test only   " << std::endl <<
            "AvgPool2DBwdF32Test        Run AvgPool2DBwdF32Test only   " << std::endl <<
            "SearchSortedFwdF32Test     Run SearchSortedFwdF32Test only   " << std::endl <<
            "GatherFwdDim0I32Test       Run GatherFwdDim0I32Test only   " << std::endl <<
            "KLDivFwdF32                Run KLDivFwdF32 only   "          << std::endl <<

            "AvgPool2DFwdF32Gaudi2Test  Run AvgPool2DFwdF32Gaudi2Test only   " << std::endl <<
            "AvgPool2DBwdF32Gaudi2Test  Run AvgPool2DBwdF32Gaudi2Test only   " << std::endl <<
            "CastF16toI16Gaudi2Test     Run CastF16toI16Gaudi2Test only   " << std::endl <<
            "SoftMaxBF16Gaudi2Test      Run SoftMaxBF16Gaudi2Test only   " << std::endl <<
            "UserLutGaudi2Test          Run UserLutGaudi2Test only   " << std::endl;

        exit(0);
    }
    else if(argc == 2) 
    {
        std::cout << "Please use --help or -h for more infomation" << std::endl;
        exit(0);
    }

    if(check_arg(argc, argv, "Gaudi", "FilterFwd2DBF16Test"))
    {
        FilterFwd2DBF16Test test_bf16;
        test_bf16.SetUp();
        result = test_bf16.runTest();
        test_bf16.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    
    if(check_arg(argc, argv, "Gaudi", "SoftMaxBF16Test"))
    {
        SoftMaxBF16Test testSoftMaxBF16;
        testSoftMaxBF16.SetUp();
        result = testSoftMaxBF16.runTest();
        testSoftMaxBF16.TearDown();
        testCount += 2;
        if (result != 0)
        {
            return result;
        }
    }

    if(check_arg(argc, argv, "Gaudi", "CastGaudiTest"))
    {
        CastGaudiTest testCaseGaudi;
        testCaseGaudi.SetUp();
        result = testCaseGaudi.runTest();
        testCaseGaudi.TearDown();
        testCount += 2;
        if (result != 0)
        {
            return result;
        }
    }

    if(check_arg(argc, argv, "Gaudi", "BatchNormF32Test"))
    {
        BatchNormF32Test testBatchNorm;
        testBatchNorm.SetUp();
        result = testBatchNorm.runTest();
        testBatchNorm.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(check_arg(argc, argv, "Gaudi", "LeakyReluF32GaudiTest"))
    {
        LeakyReluF32GaudiTest testLeakyReluGaudi;
        testLeakyReluGaudi.SetUp();
        result = testLeakyReluGaudi.runTest();
        testLeakyReluGaudi.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(check_arg(argc, argv, "Gaudi", "SparseLengthsBF16Test"))
    {
        SparseLengthsSumBF16Test testSparseLenGaudi;
        testSparseLenGaudi.SetUp();
        result = testSparseLenGaudi.runTest();
        testSparseLenGaudi.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(check_arg(argc, argv, "Gaudi", "CustomdivFwdF32Test"))
    {
        CustomdivFwdF32Test testCustomDivFwdF32;
        testCustomDivFwdF32.SetUp();
        result = testCustomDivFwdF32.runTest();
        testCustomDivFwdF32.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    Relu6AllTest testRelu6;
    if(check_arg(argc, argv, "Gaudi", "Relu6FwdF32"))
    {
        testRelu6.SetUp();
        result = testRelu6.runTest(GAUDI_KERNEL_RELU6_FWD_F32);
        testRelu6.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(check_arg(argc, argv, "Gaudi", "Relu6BwdF32"))
    {
        testRelu6.SetUp();
        result = testRelu6.runTest(GAUDI_KERNEL_RELU6_BWD_F32);
        testRelu6.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(check_arg(argc, argv, "Gaudi", "Relu6FwdBF16"))
    {
        testRelu6.SetUp();
        result = testRelu6.runTest(GAUDI_KERNEL_RELU6_FWD_BF16);
        testRelu6.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(check_arg(argc, argv, "Gaudi", "Relu6BwdBF16"))
    {
        testRelu6.SetUp();
        result = testRelu6.runTest(GAUDI_KERNEL_RELU6_BWD_BF16);
        testRelu6.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    Relu6AllTest testRelu;
    if(check_arg(argc, argv, "Gaudi", "ReluFwdF32"))
    {
        testRelu.SetUp();
        result = testRelu.runTest(GAUDI_KERNEL_RELU_FWD_F32);
        testRelu.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(check_arg(argc, argv, "Gaudi", "ReluBwdF32"))
    {
        testRelu.SetUp();
        result = testRelu.runTest(GAUDI_KERNEL_RELU_BWD_F32);
        testRelu.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(check_arg(argc, argv, "Gaudi", "ReluFwdBF16"))
    {
        testRelu.SetUp();
        result = testRelu.runTest(GAUDI_KERNEL_RELU_FWD_BF16);
        testRelu.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(check_arg(argc, argv, "Gaudi", "ReluBwdBF16"))
    {
        testRelu.SetUp();
        result = testRelu.runTest(GAUDI_KERNEL_RELU_BWD_BF16);
        testRelu.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(check_arg(argc, argv, "Gaudi", "MatrixMulFwdF32Test"))
    {
        MatrixMulFwdF32Test testMatrixMulFwdF32;
        testMatrixMulFwdF32.SetUp();
        result = testMatrixMulFwdF32.runTest();
        testMatrixMulFwdF32.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(check_arg(argc, argv, "Gaudi", "SpatialConvF32Test"))
    {
        SpatialConvF32Test spatialConv;
        spatialConv.SetUp();
        result = spatialConv.runTest();
        spatialConv.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(check_arg(argc, argv, "Gaudi", "SinF32Test"))
    {
        SinF32Test sinf32ins;
        sinf32ins.SetUp();
        result = sinf32ins.runTest();
        sinf32ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(check_arg(argc, argv, "Gaudi", "AddF32Test"))
    {
        AddF32Test addf32ins;
        addf32ins.SetUp();
        result = addf32ins.runTest();
        addf32ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    AvgPool2DF32Test avgpool2df32ins;
    if(check_arg(argc, argv, "Gaudi", "AvgPool2DFwdF32Test"))
    {
        avgpool2df32ins.SetUp();
        result = avgpool2df32ins.runTest(GAUDI_KERNEL_AVG_POOL_2D_FWD_F32);
        avgpool2df32ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(check_arg(argc, argv, "Gaudi", "AvgPool2DBwdF32Test"))
    {
        avgpool2df32ins.SetUp();
        result = avgpool2df32ins.runTest(GAUDI_KERNEL_AVG_POOL_2D_BWD_F32);
        avgpool2df32ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(check_arg(argc, argv, "Gaudi", "SearchSortedFwdF32Test"))
    {
        SearchSortedF32Test searchsortedf32ins;
        searchsortedf32ins.SetUp();
        result = searchsortedf32ins.runTest();
        searchsortedf32ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(check_arg(argc, argv, "Gaudi", "GatherFwdDim0I32Test"))
    {
        GatherFwdI32Test gatheri32ins;
        gatheri32ins.SetUp();
        result = gatheri32ins.runTest(GAUDI_KERNEL_GATHER_FWD_DIM0_I32);
        gatheri32ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(check_arg(argc, argv, "Gaudi", "KLDivFwdF32"))
    {
        KLDivAllTest testKLDiv;
        testKLDiv.SetUp();
        result = testKLDiv.runTest(GAUDI_KERNEL_KL_DIV_FWD_F32);
        testKLDiv.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    // The following ones are for Gaudi2
    AvgPool2DF32Gaudi2Test avgpool2df32Gaudi2ins;
    if(check_arg(argc, argv, "Gaudi2", "AvgPool2DFwdF32Gaudi2Test"))
    {
        avgpool2df32Gaudi2ins.SetUp();
        result = avgpool2df32Gaudi2ins.runTest(GAUDI2_KERNEL_AVG_POOL_2D_FWD_F32);
        avgpool2df32Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    AvgPool2DF32Gaudi2Test avgpool2dbf32Gaudi2ins;
    if(check_arg(argc, argv, "Gaudi2", "AvgPool2DBwdF32Gaudi2Test"))
    {
        avgpool2dbf32Gaudi2ins.SetUp();
        result = avgpool2dbf32Gaudi2ins.runTest(GAUDI2_KERNEL_AVG_POOL_2D_BWD_F32);
        avgpool2dbf32Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    
    if(check_arg(argc, argv, "Gaudi2", "CastF16toI16Gaudi2Test"))
    {
        CastF16toI16Gaudi2Test castf16tpi16Gaudi2ins;
        castf16tpi16Gaudi2ins.SetUp();
        result = castf16tpi16Gaudi2ins.runTest();
        castf16tpi16Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }
    
    if(check_arg(argc, argv, "Gaudi2", "SoftMaxBF16Gaudi2Test"))
    {
        SoftMaxBF16Gaudi2Test testSoftMaxBF16Gaudi2;
        testSoftMaxBF16Gaudi2.SetUp();
        result = testSoftMaxBF16Gaudi2.runTest();
        testSoftMaxBF16Gaudi2.TearDown();
        testCount += 2;
        if (result != 0)
        {
            return result;
        }
    }

    if(check_arg(argc, argv, "Gaudi2", "UserLutGaudi2Test"))
    {
        UserLutGaudi2Test userLutTest;
        userLutTest.SetUp();
        result = userLutTest.runTest();
        userLutTest.TearDown();
        testCount++;
        if (result != 0)
        {
            return result;
        }
    }

    if(testCount > 0)
        std::cout << "All " << testCount  <<" tests pass!" <<std::endl;
    else
        std::cout << "Please use --help or -h for more infomation" << std::endl;
    return 0;
}
