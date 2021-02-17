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

#include "tensor.h"
#include "test_base.hpp"
#include "printf_test.hpp"


class PrintfTest : public TestBase
{
public:
    PrintfTest() {}
    ~PrintfTest() {}
    int runTest();

private:
    PrintfTest(const PrintfTest& other) = delete;
    PrintfTest& operator=(const PrintfTest& other) = delete;
};


 inline int PrintfTest::runTest()
 {

    // Initialize tensor inputs with random values
    unsigned sizesInput [] = {  1024*6};
    test::Tensor<float,1> input(sizesInput);

    // Initialize scalar inputs with random values
    PrintfTestKernel::PrintfTestParams inputValues = {0};
    inputValues.int_val = -2388;
    inputValues.float_val = 0.5698;
    inputValues.pos = 5;

    // fill input tensor with reference values
    input.FillWithData(-2);
    input.Data()[inputValues.pos] = 23.5f;

    // generate input for query call
    m_in_defs.NodeParams = &inputValues;
    m_in_defs.inputTensorNr = 1;
    m_in_defs.outputTensorNr = 0;

    // generate and load tensor descriptors
    std::vector<TensorDescriptor> vec;
    vec.push_back(input.GetTensorDescriptor());

    // Call glue code for validation of input and output
    PrintfTestKernel kernelClass;
    gcapi::GlueCodeReturn_t retVal = kernelClass.GetGcDefinitions(&m_in_defs, &m_out_defs);
    if (retVal != gcapi::GLUE_SUCCESS)
    {
        std::cout << "glue test failed!! " << retVal << std::endl;
        return -1;
    }

    // Executes a simulation of kernel using TPC simulator
    RunSimulation(vec, m_in_defs, m_out_defs);

    // Get the scalar values from kernel
    uint32_t * pScalarParams  = m_out_defs.kernel.scalarParams;

    // Check scalar values that was printed using %f or %d
    for(int n = 0; n < TestBase::s_ptr_printfTensor->GetNumMessages() - 1; n++)
    {
        if(TestBase::s_ptr_printfTensor->ValueIs(n))  // if the message has a value (is not sole string)
        {
            // Verify if the message contains the correct scalar data
            if (*pScalarParams != TestBase::s_ptr_printfTensor->GetImprint(n))
            {
                std::cout << "test failed!!" << std::endl;
                return -1;
            }
            pScalarParams++;
        }
    }

    // Check message number 3
    char buf[256];
    int n = sprintf(buf, "value in vector is %f\n", input.Data()[inputValues.pos]);
    std::string ref_message = std::string(buf, n);
    if(ref_message.compare(TestBase::s_ptr_printfTensor->GetMessageWithNumber(3)))
    {
        std::cout << "test failed!!" << std::endl;
        return -1;
    }

    // Check the number of messages
    if (TestBase::s_ptr_printfTensor->GetNumMessages() != 4)
    {
        std::cout << "test failed!!" << std::endl;
        return -1;
    }

    std::cout << "test pass!!" << std::endl;
    return 0;
}
