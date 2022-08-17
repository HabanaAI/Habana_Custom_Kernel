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
#ifndef _TEST_BASE_HPP
#define _TEST_BASE_HPP

#include <vector>
#include <memory>

#include "tensor.h"
#include "gc_interface.h"

class TestBase
{
public:
    TestBase() {}
    ~TestBase() {}
    static const int c_default_isa_buffer_size = 1024 * 1024;

    typedef enum _IndexSpaceMappingTest_t
    {
        e_defaultMode           = 0,
        e_ignoreMode            = 1,
        e_partialReadMode       = 2,
        e_partialWriteMode      = 3,
        e_partialReadWriteMode  = 4,
    } IndexSpaceMappingTest_t;

    unsigned int RunSimulation( std::vector<TensorDesc>& descriptors,
                                const gcapi::HabanaKernelParams_t& gc_input,
                                const gcapi::HabanaKernelInstantiation_t& gc_output,
                                IndexSpaceMappingTest_t testMode = e_defaultMode);

    virtual void SetUp();

    virtual void TearDown();

    template <class T, int DIM>
    static void LoadTensorToGcDescriptor(gcapi::Tensor_t* pTargetTensor,
                                 const test::Tensor<T,DIM>& inputTensor)
    {
        pTargetTensor->dataType = test::getGcDataType(inputTensor);
        pTargetTensor->geometry.dims = DIM;
        for (int i = 0 ; i < DIM ; i++)
        {
            pTargetTensor->geometry.sizes[i] = inputTensor.Size(i);
        }
    }

       
    void ReleaseKernelNames(char** kernelNames, unsigned kernelCount)
    {
        // release memory
        for (size_t i = 0; i < kernelCount; i++)
        {
            delete[] kernelNames[i];
        }
        delete[] kernelNames;
    }    

    gcapi::HabanaKernelParams_t         m_in_defs;
    gcapi::HabanaKernelInstantiation_t  m_out_defs;
private:
    // this is a debug helper function to print glue code outputs.
    void PrintKernelOutputParams(const gcapi::HabanaKernelParams_t* gc_input,
                                 const gcapi::HabanaKernelInstantiation_t*gc_output);
    // this is a debug helper function to print glue code inputs.
    void PrintKernelInputParams(const gcapi::HabanaKernelParams_t* gc_input);
    TestBase(const TestBase& other) = delete;
    TestBase& operator=(const TestBase& other) = delete;
};


#endif  // _TEST_BASE_HPP
