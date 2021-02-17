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
#include <sstream>
#include <limits>

#include "test_base_tpc_callback.hpp"

#define LOAD_OP_LD_TNSR             17
#define LOAD_OP_LD_TNSR_LOW         18
#define LOAD_OP_LD_TNSR_HIGH        19
#define STORE_OP_ST_TNSR            11
#define STORE_OP_ST_TNSR_LOW        12
#define STORE_OP_ST_TNSR_HIGH       13
#define LOAD_STORE_OP_GEN_ADDR      0
#define LOAD_OP_LD_G                12
#define STORE_OP_ST_G               6

#define VPE_SRC_DST_ADRF_FIRST    160
#define VPE_SRC_DST_ADRF_LAST     167

void TestBaseTPCCallback::LoadStoreAccessCallback(  uint32_t operation,
                                                    uint32_t tensorID,
                                                    int32_t* pIdxReg,
                                                    uint32_t dst)
{
    switch(operation)
    {
        case LOAD_OP_LD_TNSR:
        case STORE_OP_ST_TNSR:
        case LOAD_OP_LD_TNSR_LOW:
        case STORE_OP_ST_TNSR_LOW:
        case LOAD_OP_LD_TNSR_HIGH:
        case STORE_OP_ST_TNSR_HIGH:
        {
            UpdateMinMaxIndices(operation, tensorID, pIdxReg);
        }
        break;
        case LOAD_STORE_OP_GEN_ADDR:
        {
            assert((dst >= VPE_SRC_DST_ADRF_FIRST) &&
                    (dst <= VPE_SRC_DST_ADRF_LAST) &&
                    "Invalid ADRF address");

            int adrfIdx = dst - VPE_SRC_DST_ADRF_FIRST;
            m_ADRFMemorySt1[adrfIdx].tensorID = tensorID;
            memcpy(m_ADRFMemorySt1[adrfIdx].idxReg, pIdxReg, gcapi::MAX_TENSOR_DIM * sizeof(int));
        }
        break;
        default:
            assert(0 && "Wrong operation");
    }
}

void TestBaseTPCCallback::LoadStoreGlobalCallback(  uint32_t operation,
                                                    uint32_t src)
{
    assert((src >= VPE_SRC_DST_ADRF_FIRST) &&
                    (src <= VPE_SRC_DST_ADRF_LAST) &&
                    "Invalid ADRF address");

    int adrfIdx = src - VPE_SRC_DST_ADRF_FIRST;

    switch(operation)
    {
        case LOAD_STORE_OP_GEN_ADDR:
        {
            m_ADRFMemorySt2[adrfIdx].tensorID = m_ADRFMemorySt1[adrfIdx].tensorID;
            memcpy(m_ADRFMemorySt2[adrfIdx].idxReg, m_ADRFMemorySt1[adrfIdx].idxReg, gcapi::MAX_TENSOR_DIM * sizeof(int));
        }
        break;
        case LOAD_OP_LD_G:
        case STORE_OP_ST_G:
        {
            UpdateMinMaxIndices(LOAD_STORE_OP_GEN_ADDR, m_ADRFMemorySt2[adrfIdx].tensorID, m_ADRFMemorySt2[adrfIdx].idxReg);
        }
        break;
        default:
            assert(0 && "Wrong operation");
    }
}

TestBaseTPCCallback::IndexSpaceMappingReturn_t TestBaseTPCCallback::TestMinMaxAccess(
                                            const gcapi::TensorGeometry_t&      tensorGeometry,
                                            const TensorIndexRange_t&           tensorIdxRange,
                                            const gcapi::TensorAccessPattern_t& accessPattern,
                                            std::stringstream& ss,
                                            bool partialAccess)
{
    if (!accessPattern.allRequired)
    {
        for (unsigned j = 0; j < tensorGeometry.dims; j++)
        {
            const gcapi::DimTransform_t& tran = accessPattern.dim[j];
            const unsigned indexSpaceDim = tran.dim;
            const unsigned indexSpaceStart = m_offsets[indexSpaceDim];
            const unsigned indexSpaceEnd = m_offsets[indexSpaceDim] + m_pGcOutput->indexSpaceGeometry.sizes[indexSpaceDim] - 1;
            const int indexStart = indexSpaceStart * tran.start_a + tran.start_b;
            const int indexEnd = indexSpaceEnd * tran.end_a + tran.end_b;
            const int indexMin = tensorIdxRange.dim[j].minIdx;
            const int indexMax = tensorIdxRange.dim[j].maxIdx;

            ss << "Dimenstion = "    << j          << ", ";
            ss << "mapping start = " << indexStart << ", ";
            ss << "mapping end = "   << indexEnd   << ", ";
            ss << "actual start = "  << indexMin   << ", ";
            ss << "actual end = "    << indexMax   << std::endl;
            if (indexMin < indexStart || indexMax > indexEnd)
            {
                return e_kernelExceededIndexSpaceMapping;
            }
            if ((!partialAccess) &&
                    (indexMin > indexStart || (indexMax < indexEnd &&
                    indexMax != ((int)tensorGeometry.sizes[j] - 1))))
            {
                return e_kernelIndexSpaceMappingNotCovered;
            }
        }
    }
    return e_validIndexSpaceMapping;
}

void TestBaseTPCCallback::ValidateAccessPattern(TestBase::IndexSpaceMappingTest_t testMode)
{
    if (testMode != TestBase::e_ignoreMode)
    {
        for (unsigned i = 0; i < m_pGcInput->inputTensorNr; i++)
        {
            const TensorIndexRange_t& tensorIdxRange = m_idxRange.inTensors[i];
            const gcapi::TensorGeometry_t& tensorGeometry = m_pGcInput->inputTensors[i].geometry;
            const gcapi::TensorAccessPattern_t& accessPattern = m_pGcOutput->inputTensorAccessPattern[i];

            std::stringstream ss;
            ss << std::endl;
            bool partialAccess = (testMode == TestBase::e_partialReadMode ||
                                    testMode == TestBase::e_partialReadWriteMode);
            IndexSpaceMappingReturn_t retVal = TestMinMaxAccess(tensorGeometry,
                                                                tensorIdxRange,
                                                                accessPattern,
                                                                ss,
                                                                partialAccess);
            switch(retVal)
            {
                case e_kernelExceededIndexSpaceMapping:
                    std::cout << std::endl << "Input tensor " << i << " has invalid access pattern -> kernel exceeded index space mapping.\n";
                    std::cout << ss.str();
                    break;
                case e_kernelIndexSpaceMappingNotCovered:
                    std::cout << std::endl << "Input tensor " << i << " has sub-optimal access pattern.\n";
                    std::cout << ss.str();
                    break;
                case e_validIndexSpaceMapping:
                default:
                    break;
            }
        }
        for (unsigned i = 0; i < m_pGcInput->outputTensorNr; i++)
        {
            const TensorIndexRange_t& tensorIdxRange = m_idxRange.outTensors[i];
            const gcapi::TensorGeometry_t& tensorGeometry = m_pGcInput->outputTensors[i].geometry;
            const gcapi::TensorAccessPattern_t& accessPattern = m_pGcOutput->outputTensorAccessPattern[i];

            std::stringstream ss;
            ss << std::endl;
            bool partialAccess = (testMode == TestBase::e_partialWriteMode ||
                                    testMode == TestBase::e_partialReadWriteMode);
            IndexSpaceMappingReturn_t retVal = TestMinMaxAccess(tensorGeometry,
                                                                tensorIdxRange,
                                                                accessPattern,
                                                                ss,
                                                                partialAccess);
            switch(retVal)
            {
                case e_kernelExceededIndexSpaceMapping:
                    std::cout << std::endl << "Output tensor " << i << " has invalid access pattern -> kernel exceeded index space mapping.\n";
                    std::cout << ss.str();
                    break;
                case e_kernelIndexSpaceMappingNotCovered:
                    std::cout << std::endl << "Output tensor " << i << " has sub-optimal access pattern.\n";
                    std::cout << ss.str();
                    break;
                case e_validIndexSpaceMapping:
                default:
                    break;
            }
        }
    }
}

void TestBaseTPCCallback::InitMinMaxIndices()
{
    for (unsigned i = 0; i < m_pGcInput->inputTensorNr; i++)
    {
        for (unsigned j = 0; j < m_pGcInput->inputTensors[i].geometry.dims; j++)
        {
            m_idxRange.inTensors[i].dim[j].minIdx = std::numeric_limits<int>::max();
            m_idxRange.inTensors[i].dim[j].maxIdx = std::numeric_limits<int>::min();
        }
    }
    for (unsigned i = 0; i < m_pGcInput->outputTensorNr; i++)
    {
        for (unsigned j = 0; j < m_pGcInput->outputTensors[i].geometry.dims; j++)
        {
            m_idxRange.outTensors[i].dim[j].minIdx = std::numeric_limits<int>::max();
            m_idxRange.outTensors[i].dim[j].maxIdx = std::numeric_limits<int>::min();
        }
    }
}

void TestBaseTPCCallback::UpdateMinMaxIndices(  uint32_t operation,
                                                uint32_t tensorID,
                                                int32_t* pIdxReg)
{
    const gcapi::Tensor_t* currTensor;
    TensorIndexRange_t* currIndexRange;

    if (tensorID < m_pGcInput->inputTensorNr)
    {
        currTensor = &m_pGcInput->inputTensors[tensorID];
        currIndexRange = &m_idxRange.inTensors[tensorID];
    }
    else if (tensorID < (m_pGcInput->inputTensorNr + m_pGcInput->outputTensorNr))
    {
        currTensor = &m_pGcInput->outputTensors[tensorID - m_pGcInput->inputTensorNr];
        currIndexRange = &m_idxRange.outTensors[tensorID - m_pGcInput->inputTensorNr];
    }
    // If tensorID points to aux-tensor no need to check load / store access.
    else
    {
        return;
    }

    // In the FCD accessing certain index in load/store ops maps to a single
    // scalar, vector or partial vector according to the operation.
    // Here we update the min/max indices according to the operation.
    int minIdx0, maxIdx0;
    const int index0 = *(pIdxReg);
    const unsigned eig = GetVectorLength(currTensor->dataType);
    switch(operation)
    {
        case LOAD_OP_LD_TNSR:
        case STORE_OP_ST_TNSR:
        {
            minIdx0 = index0;
            maxIdx0 = index0 + eig - 1;
        }
        break;
        case LOAD_OP_LD_TNSR_LOW:
        case STORE_OP_ST_TNSR_LOW:
        {
            minIdx0 = index0;
            maxIdx0 = index0 + (eig / 2) - 1;
        }
        break;
        case LOAD_OP_LD_TNSR_HIGH:
        case STORE_OP_ST_TNSR_HIGH:
        {
            minIdx0 = index0 + (eig / 2);
            maxIdx0 = index0 + eig - 1;
        }
        break;
        case LOAD_STORE_OP_GEN_ADDR:
        {
            minIdx0 = index0;
            maxIdx0 = index0;
        }
        break;
        default:
            assert(0 && "Wrong operation");
    }
    currIndexRange->dim[0].minIdx = std::min(minIdx0, currIndexRange->dim[0].minIdx);
    currIndexRange->dim[0].maxIdx = std::max(maxIdx0, currIndexRange->dim[0].maxIdx);

    for (unsigned i = 1; i < currTensor->geometry.dims; i++)
    {
        const int index = *(pIdxReg + i);
        currIndexRange->dim[i].minIdx = std::min(index, currIndexRange->dim[i].minIdx);
        currIndexRange->dim[i].maxIdx = std::max(index, currIndexRange->dim[i].maxIdx);
    }
}
