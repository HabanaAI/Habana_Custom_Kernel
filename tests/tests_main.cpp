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

#include <iostream>
#include "filter_2d_f32_test.hpp"
#include "filter_fwd_2d_bf16_test.hpp"
#include "filter_2d_i8_w33_s11_test.hpp"
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
#include "relu6_all_test.hpp"
#include "matrix_mul_fwd_f32_test.hpp"


int main(int argc, char** argv)
{
    int result = 0;

    if(argc == 2 && ((strcmp(argv[1], "--help") ==0) || (strcmp(argv[1],"-h") ==0)))
    {
        std::cout << argv[0] << " " << "[options]" << std::endl <<
            "Options:" << std::endl <<
            "N/A                        Run all test cases" << std::endl <<
            "-h | --help                Print this help" << std::endl <<
            "-d | --device <DeviceName> Run only kernels for the DeviceName" << std::endl <<
            "-t | --test  <TestName>    Run <TestName>> only   " << std::endl <<
            "DeviceName:" << std::endl <<
            "Goya                       Run all Goya kernels only   " << std::endl <<
            "Gaudi                      Run all Gaudi kernels only   " << std::endl <<
            "TestName:" << std::endl <<
            "Filter2DF32Test            Run Filter2DF32Test only   " << std::endl <<
            "FilterFwd2DBF16Test        Run FilterFwd2DBF16Test only   " << std::endl <<
            "Filter2DI8W33S11Test       Run Filter2DI8W33S11Test only   " << std::endl <<
            "SparseLengthsSumTest       Run SparseLengthsSumTest only   " << std::endl <<
            "SoftMaxF32Test             Run SoftMaxF32Test only   " << std::endl <<
            "SoftMaxBF16Test            Run SoftMaxBF16Test only   " << std::endl <<
            "CastTest                   Run CastTest only   " << std::endl <<
            "CastGaudiTest              Run CastGaudiTest only   " << std::endl <<
            "LeakyReluF32Test           Run LeakyReluF32Test only   " << std::endl <<
            "BatchNormF32Test           Run BatchNormF32Test only   " << std::endl <<                                 
            "LeakyReluF32GaudiTest      Run LeakyReluF32GaudiTest only   " << std::endl <<
            "SparseLengthsBF16Test      Run SparseLengthsBF16Test only   " << std::endl <<
            "CustomdivFwdF32Test        Run CustomdivFwdF32Test only   " << std::endl <<
            "Relu6FwdF32                Run Relu6FwdF32 only   " << std::endl <<
            "Relu6BwdF32                Run Relu6BwdF32 only   " << std::endl <<
            "Relu6FwdBF16               Run Relu6FwdBF16 only   " << std::endl <<
            "Relu6BwdBF16               Run Relu6BwdBF16 only   " << std::endl <<
            "MatrixMulFwdF32Test        Run MatrixMulFwdF32Test only   " << std::endl;

        exit(0);
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0)) 
        && (strcmp(argv[2],"Goya") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0)) 
        && (strcmp(argv[2],"Filter2DF32Test") ==0))))
    {
        Filter2DF32Test testFilter;
        testFilter.SetUp();
        result = testFilter.runTest();
        testFilter.TearDown();
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0)) 
        && (strcmp(argv[2],"Gaudi") ==0)))  ||    
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0)) 
        && (strcmp(argv[2],"FilterFwd2DBF16Test") ==0))))
    {
        FilterFwd2DBF16Test test_bf16;
        test_bf16.SetUp();
        result = test_bf16.runTest();
        test_bf16.TearDown();
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0)) 
        && (strcmp(argv[2],"Goya") ==0)))  ||    
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0)) 
        && (strcmp(argv[2],"SparseLengthsSumTest") ==0))))
    {
        SparseLengthsSumTest testSparseLenGoya;
        testSparseLenGoya.SetUp();
        result = testSparseLenGoya.runTest();
        testSparseLenGoya.TearDown();
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0)) 
        && (strcmp(argv[2],"Goya") ==0)))  ||    
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0)) 
        && (strcmp(argv[2],"Filter2DI8W33S11Test") ==0))))
    {
        Filter2DI8W33S11Test testFilter_i8;
        testFilter_i8.SetUp();
        result = testFilter_i8.runTest();
        testFilter_i8.TearDown();
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0)) 
        && (strcmp(argv[2],"Goya") ==0)))  ||    
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0)) 
        && (strcmp(argv[2],"SoftMaxF32Test") ==0))))
    {
        SoftMaxF32Test testSoftMax;
        testSoftMax.SetUp();
        result = testSoftMax.runTest();
        testSoftMax.TearDown();
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0)) 
        && (strcmp(argv[2],"Gaudi") ==0)))  ||    
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0)) 
        && (strcmp(argv[2],"SoftMaxBF16Test") ==0)))) 
    {   
        SoftMaxBF16Test testSoftMaxBF16;
        testSoftMaxBF16.SetUp();
        result = testSoftMaxBF16.runTest();
        testSoftMaxBF16.TearDown();
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0)) 
        && (strcmp(argv[2],"Goya") ==0)))  ||    
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0)) 
        && (strcmp(argv[2],"CastTest") ==0))))
    {
        CastTest testCastGoya;
        testCastGoya.SetUp();
        result = testCastGoya.runTest();
        testCastGoya.TearDown();
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0)) 
        && (strcmp(argv[2],"Gaudi") ==0)))  ||    
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0)) 
        && (strcmp(argv[2],"CastGaudiTest") ==0))))
    {
        CastGaudiTest testCaseGaudi;
        testCaseGaudi.SetUp();
        result = testCaseGaudi.runTest();
        testCaseGaudi.TearDown();
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0)) 
        && (strcmp(argv[2],"Goya") ==0)))  ||    
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0)) 
        && (strcmp(argv[2],"LeakyReluF32Test") ==0))))
    {
        LeakyReluF32Test testLeakyRelu;
        testLeakyRelu.SetUp();
        result = testLeakyRelu.runTest();
        testLeakyRelu.TearDown();
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0)) 
        && (strcmp(argv[2],"Gaudi") ==0)))  ||    
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0)) 
        && (strcmp(argv[2],"BatchNormF32Test") ==0))))
    {
        BatchNormF32Test testBatchNorm;
        testBatchNorm.SetUp();
        result = testBatchNorm.runTest();
        testBatchNorm.TearDown();
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0)) 
        && (strcmp(argv[2],"Gaudi") ==0)))  ||    
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0)) 
        && (strcmp(argv[2],"LeakyReluF32GaudiTest") ==0))))
    {
        LeakyReluF32GaudiTest testLeakyReluGaudi;
        testLeakyReluGaudi.SetUp();
        result = testLeakyReluGaudi.runTest();
        testLeakyReluGaudi.TearDown();
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0)) 
        && (strcmp(argv[2],"Gaudi") ==0)))  ||    
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0)) 
        && (strcmp(argv[2],"SparseLengthsBF16Test") ==0))))
    {
        SparseLengthsSumBF16Test testSparseLenGaudi;
        testSparseLenGaudi.SetUp();
        result = testSparseLenGaudi.runTest();
        testSparseLenGaudi.TearDown();
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0)) 
        && (strcmp(argv[2],"Gaudi") ==0)))  ||    
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0)) 
        && (strcmp(argv[2],"CustomdivFwdF32Test") ==0))))
    {
        CustomdivFwdF32Test testCustomDivFwdF32;
        testCustomDivFwdF32.SetUp();
        result = testCustomDivFwdF32.runTest();
        testCustomDivFwdF32.TearDown();
        if (result != 0)
        {
            return result;
        }
    }

    Relu6AllTest testRelu6;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0)) 
        && (strcmp(argv[2],"Gaudi") ==0)))  ||    
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0)) 
        && (strcmp(argv[2],"Relu6FwdF32") ==0))))
    {
        testRelu6.SetUp();
        result = testRelu6.runTest(GAUDI_KERNEL_RELU6_FWD_F32);
        testRelu6.TearDown();
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0)) 
        && (strcmp(argv[2],"Gaudi") ==0)))  ||    
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0)) 
        && (strcmp(argv[2],"Relu6BwdF32") ==0))))
    {
        testRelu6.SetUp();
        result = testRelu6.runTest(GAUDI_KERNEL_RELU6_BWD_F32);
        testRelu6.TearDown();
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0)) 
        && (strcmp(argv[2],"Gaudi") ==0)))  ||    
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0)) 
        && (strcmp(argv[2],"Relu6FwdBF16") ==0))))
    {
        testRelu6.SetUp();
        result = testRelu6.runTest(GAUDI_KERNEL_RELU6_FWD_BF16);
        testRelu6.TearDown();
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0)) 
        && (strcmp(argv[2],"Gaudi") ==0)))  ||    
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0)) 
        && (strcmp(argv[2],"Relu6BwdBF16") ==0))))
    {
        testRelu6.SetUp();
        result = testRelu6.runTest(GAUDI_KERNEL_RELU6_BWD_BF16);
        testRelu6.TearDown();
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0)) 
        && (strcmp(argv[2],"MatrixMulFwdF32Test") ==0))))
    {
        MatrixMulFwdF32Test testMatrixMulFwdF32;
        testMatrixMulFwdF32.SetUp();
        result = testMatrixMulFwdF32.runTest();
        testMatrixMulFwdF32.TearDown();
        if (result != 0)
        {
            return result;
        }
    }

    std::cout << "All tests pass!" <<std::endl;
    return 0;
}
