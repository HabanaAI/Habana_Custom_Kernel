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

#ifndef TEST_BASE_TPC_CALLBACK_HPP
#define TEST_BASE_TPC_CALLBACK_HPP

#include <string.h> //required for memcpy

#include "TPCCallbackInterface.h"
#include "gc_interface.h"
#include "test_base.hpp"

#define ADRF_NUM 8

class TestBaseTPCCallback : public TPCCallbackInterfaceImpl
{
public:

    TestBaseTPCCallback(const gcapi::HabanaKernelParams_t* gc_input,
                        const gcapi::HabanaKernelInstantiation_t* gc_output,
                        int offsets[gcapi::MAX_TENSOR_DIM]) :
                        m_pGcInput(gc_input),
                        m_pGcOutput(gc_output)
                        {
                            memcpy(m_offsets, offsets,
                                    gcapi::MAX_TENSOR_DIM * sizeof(int));
                            InitMinMaxIndices();
                            InitMinMaxIndices();
                        }

    virtual void LoadStoreAccessCallback(   uint32_t operation,
                                            uint32_t tensorID,
                                            int32_t* pIdxReg,
                                            uint32_t dst);

    virtual void LoadStoreGlobalCallback(   uint32_t operation,
                                            uint32_t src);

    void ValidateAccessPattern(TestBase::IndexSpaceMappingTest_t testMode);

private:
    const unsigned int c_f32ElementsInVector = 64;
    const unsigned int c_i16ElementsInVector = 128;
    const unsigned int c_i8ElementsInVector  = 256;
    typedef struct _IndexRange_t
    {
        int minIdx;
        int maxIdx;
    } IndexRange_t;

    typedef struct _TensorIndexRange_t
    {
        IndexRange_t dim[gcapi::MAX_TENSOR_DIM];
    } TensorIndexRange_t;

    typedef struct _InOutIndexRange_t
    {
        TensorIndexRange_t inTensors[gcapi::MAX_TENSOR_NR];
        TensorIndexRange_t outTensors[gcapi::MAX_TENSOR_NR];
    } InOutIndexRange_t;

    typedef struct _GenAddrData_t
    {
        uint32_t tensorID;
        int32_t idxReg[gcapi::MAX_TENSOR_DIM];
    } GenAddrData_t;

    typedef enum _IndexSpaceMappingReturn_t
    {
        e_validIndexSpaceMapping            = 0,
        e_kernelExceededIndexSpaceMapping   = 1,
        e_kernelIndexSpaceMappingNotCovered = 2,
    } IndexSpaceMappingReturn_t;

    int32_t GetVectorLength(gcapi::TensorDataType_t scalarType) const
    {
        switch(scalarType)
        {
            case gcapi::DATA_I8:
            case gcapi::DATA_U8:
                return c_i8ElementsInVector;
                break;
            case gcapi::DATA_I16:
            case gcapi::DATA_BF16:
                return c_i16ElementsInVector;
                break;
            case gcapi::DATA_I32:
            case gcapi::DATA_F32:
                return c_f32ElementsInVector;
                break;
            default:
                assert(0);
                break;
        }
        return 0;
    }

    void InitMinMaxIndices();
    void UpdateMinMaxIndices(   uint32_t operation,
                                uint32_t tensorID,
                                int32_t* pIdxReg);
    IndexSpaceMappingReturn_t TestMinMaxAccess(
                            const gcapi::TensorGeometry_t&      tensorGeometry,
                            const TensorIndexRange_t&           tensorIdxRange,
                            const gcapi::TensorAccessPattern_t& accessPattern,
                            std::stringstream&  ss,
                            bool partialAccess);

    const gcapi::HabanaKernelParams_t* m_pGcInput;
    const gcapi::HabanaKernelInstantiation_t* m_pGcOutput;
    int m_offsets[gcapi::MAX_TENSOR_DIM];
    InOutIndexRange_t m_idxRange;
    GenAddrData_t m_ADRFMemorySt1[ADRF_NUM]; // In use in perform operation stage.
    GenAddrData_t m_ADRFMemorySt2[ADRF_NUM]; // In use in commit result stage.
};

#endif /* TEST_BASE_TPC_CALLBACK_HPP */

