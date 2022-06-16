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

#include <cstring>
#include <math.h>
#include <stdio.h>
#include "bitonic_sort_f32.hpp"


extern unsigned char _binary___bitonic_sort_datain_dataout_o_start;
extern unsigned char _binary___bitonic_sort_datain_dataout_o_end;

gcapi::GlueCodeReturn_t BitonicSortF32::GetKernelName(
        char kernelName [gcapi::MAX_NODE_NAME])
{
    strcpy(kernelName,"custom_bitonic_sort_f32");
    return gcapi::GLUE_SUCCESS;
}

gcapi::GlueCodeReturn_t
BitonicSortF32::AuxiliaryTensorSettings(gcapi::HabanaKernelParams_t*      pInDefs,
                                     gcapi::HabanaKernelInstantiation_t* pOutDefs, int vecsize, int chunkSize, SortType dir) const
{
    /* Auxiliary tensor is required to store two tables:
    1. shuffTable - To store 3 shuffle patterns required in kernel
    2. predTable  - To store all required predicate patterns - 21 patterns in total
    */
    pOutDefs->auxiliaryTensorCount                  = 1;
    pOutDefs->auxiliaryTensors[0].dataType          = gcapi::DATA_U8;
    pOutDefs->auxiliaryTensors[0].geometry.dims     = 1;
    pOutDefs->auxiliaryTensors[0].geometry.sizes[1] = 0;
    pOutDefs->auxiliaryTensors[0].geometry.sizes[2] = 0;
    pOutDefs->auxiliaryTensors[0].geometry.sizes[3] = 0;
    pOutDefs->auxiliaryTensors[0].geometry.sizes[4] = 0;
    const int shuffTabSize                          = sizeof(shuffle_adj_ele_tab_f32) +
                             sizeof(shuffle_adj_ele_pair_tab_f32) +
                             sizeof(shuffle_adj_ele_quad_tab_f32);
    const int numPredPatterns                       = 21;
    const int predTabSize                           = 256 * numPredPatterns;
    pOutDefs->auxiliaryTensors[0].geometry.sizes[0] = shuffTabSize + predTabSize;

    unsigned requiredSize = pOutDefs->auxiliaryTensors[0].geometry.sizes[0] * sizeof(uint8_t);
    if (requiredSize > pOutDefs->auxiliaryTensors[0].bufferSize)
    {
        pOutDefs->auxiliaryTensors[0].bufferSize = requiredSize;
        return gcapi::GLUE_INSUFICIENT_AUX_BUFFER_SIZE;
    }

    // Copy shuffle tables to aux-0
    size_t tabSize0 = sizeof(shuffle_adj_ele_tab_f32);
    size_t tabSize1 = sizeof(shuffle_adj_ele_pair_tab_f32);
    size_t tabSize2 = sizeof(shuffle_adj_ele_quad_tab_f32);
    std::memcpy(pOutDefs->auxiliaryTensors[0].pData, shuffle_adj_ele_tab_f32, tabSize0);
    std::memcpy(
        static_cast<void*>(static_cast<uint8_t*>(pOutDefs->auxiliaryTensors[0].pData) + tabSize0),
        shuffle_adj_ele_pair_tab_f32, tabSize1);
    std::memcpy(static_cast<void*>(static_cast<uint8_t*>(pOutDefs->auxiliaryTensors[0].pData) +
                                   tabSize0 + tabSize1),
                shuffle_adj_ele_quad_tab_f32, tabSize2);

    // Generate predicate pattern and copy to aux -1
    bool sign = true;
    if (chunkSize == 64)
    {
        // Pattern should correspond to original sort direction for chunk size of 64 because,
        // there is no bitonic merge phase after initial vector sort.
        // For all larger sizes, these predicates can be set for ascending direction. The required
        // sort direction will be set during the subsequent bitonic merge phases
        sign = (dir == SORT_ASCENDING);
    }
    int nStages = log2(vecsize);
    int predInd = 0;
    // Iterate over each stage
    for (int i = 1; i <= nStages; i++)
    {
        // Iterate over each sub-stage
        for (int j = 1; j <= i; j++)
        {
            std::vector<uint8_t> pred = GeneratePredPattern(vecsize, sign, i, j);
            std::copy(pred.begin(), pred.end(),
                      static_cast<uint8_t*>(pOutDefs->auxiliaryTensors[0].pData) + shuffTabSize +
                          predInd * pred.size());
            predInd++;
        }
    }

    return gcapi::GLUE_SUCCESS;
}

std::vector<uint8_t> BitonicSortF32::GeneratePredPattern(const int32_t vecSize, const bool sign,
                                                      const int32_t stageNo,
                                                      const int32_t subStageNo) const
{
    std::vector<uint8_t> pred;
    int                  curBlockSize  = pow(2, stageNo);
    int                  curGroupSize  = curBlockSize / pow(2, subStageNo - 1);
    int                  nBlocks       = vecSize / curBlockSize;
    int                  nGroupsPerBlk = curBlockSize / curGroupSize;
    int                  curSign       = sign;

    // Iterate over all blocks in vector
    for (int b = 0; b < nBlocks; b++)
    {
        // Iterate over all groups within the block
        for (int m = 0; m < nGroupsPerBlk; m++)
        {
            // Fill the first half of group
            for (int j = 0; j < curGroupSize / 2; j++)
            {
                // Only 64 values in an F32 vector. Hence each element of bool256 should be
                // repeated 4 times
                for (int k = 0; k < 4; k++)
                {
                    pred.push_back(curSign == true ? 0 : 1);
                }
            }
            // Fill the second half of group
            for (int j = 0; j < curGroupSize / 2; j++)
            {
                // Only 64 values in an F32 vector. Hence each element of bool256 should be
                // repeated 4 times
                for (int k = 0; k < 4; k++)
                {
                    pred.push_back(curSign == true ? 1 : 0);
                }
            }
        }
        // Alternate sign between blocks
        curSign = !curSign;
    }
    return pred;
}

gcapi::GlueCodeReturn_t BitonicSortF32::GetGcDefinitions(
        gcapi::HabanaKernelParams_t* params,
        gcapi::HabanaKernelInstantiation_t* kernel)
{
    gcapi::GlueCodeReturn_t retVal;
    BitonicSortF32Param* def = static_cast<BitonicSortF32Param*>(params->NodeParams);
    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (params->inputTensorNr != 1)
    {
        params->inputTensorNr  = 1;
        return gcapi::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    //validate correct amount of output tensors
    if (params->outputTensorNr != 2)
    {
        params->outputTensorNr  = 2;
        return gcapi::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    // validate input and output data type
    if (params->inputTensors[0].dataType != gcapi::DATA_F32 ||
        params->outputTensors[0].dataType != gcapi::DATA_F32 ||
        params->outputTensors[1].dataType != gcapi::DATA_I32) 
    {
        params->inputTensors[0].dataType = gcapi::DATA_F32;
        params->outputTensors[0].dataType = gcapi::DATA_F32;
        params->outputTensors[1].dataType = gcapi::DATA_I32;
        return gcapi::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    // Tensor 0 should be input feature map.
    // The semantics of the input tensors and their order is a convention
    // between TPC kernel writer and the write of the layer at the
    // framework level.
    unsigned int outputSizes[gcapi::MAX_TENSOR_DIM] = {0};

    memcpy(outputSizes, params->inputTensors[0].geometry.sizes, sizeof(outputSizes));

    // verify that output feature map dimension are correct
    if (memcmp(params->outputTensors[0].geometry.sizes, outputSizes,
               params->outputTensors[0].geometry.dims * sizeof(unsigned) ) != 0)
    {
        memcpy(params->outputTensors[0].geometry.sizes, params->inputTensors[0].geometry.sizes, sizeof(outputSizes));
        return gcapi::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    int elementsInVec = 64;

    //round up to elementsInVec and divide by elementsInVec.
    unsigned depthIndex = (outputSizes[0] + (elementsInVec - 1)) / elementsInVec;
    kernel->indexSpaceGeometry.dims = 5;
    kernel->indexSpaceGeometry.sizes[0] = depthIndex;
	//reduce index space due to unroll.
    kernel->indexSpaceGeometry.sizes[1] = outputSizes[1]; 
    kernel->indexSpaceGeometry.sizes[2] = outputSizes[2];
    kernel->indexSpaceGeometry.sizes[3] = outputSizes[3];
    kernel->indexSpaceGeometry.sizes[4] = outputSizes[4];

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    // f_start f(i) = elementsInVec*i + 0;
    // f_end   f(i) = elementsInVec*i + (elementsInVec - 1);
    // Resource 0 (IFM) dim 0
    for (unsigned i = 0; i < params->inputTensorNr; i++)
    {
        kernel->inputTensorAccessPattern[i].dim[0].dim      = 0;
        kernel->inputTensorAccessPattern[i].dim[0].start_a  = elementsInVec;
        kernel->inputTensorAccessPattern[i].dim[0].end_a    = elementsInVec;
        kernel->inputTensorAccessPattern[i].dim[0].start_b  = 0;
        kernel->inputTensorAccessPattern[i].dim[0].end_b    = elementsInVec - 1;

        // f_start f(i) = 1*i + 0;
        // f_end   f(i) = 1*i + 0;
        // Resource 0 (IFM) dim 1-4
        for (unsigned int dims = 1; dims < kernel->indexSpaceGeometry.dims; dims++)
        {
            kernel->inputTensorAccessPattern[i].dim[dims].dim      = dims;
            kernel->inputTensorAccessPattern[i].dim[dims].start_a  = 1;
            kernel->inputTensorAccessPattern[i].dim[dims].end_a    = 1;
            kernel->inputTensorAccessPattern[i].dim[dims].start_b  = 0;
            kernel->inputTensorAccessPattern[i].dim[dims].end_b    = 1 - 1;
        }        
    }    

    // f_start f(i) = elementsInVec*i + 0;
    // f_end   f(i) = elementsInVec*i + (elementsInVec - 1);
    // Resource 0 (OFM) dim 0
    for (unsigned i = 0; i < params->outputTensorNr; i++)    
    {
        kernel->outputTensorAccessPattern[i].dim[0].dim      = 0;
        kernel->outputTensorAccessPattern[i].dim[0].start_a  = elementsInVec;
        kernel->outputTensorAccessPattern[i].dim[0].end_a    = elementsInVec;
        kernel->outputTensorAccessPattern[i].dim[0].start_b  = 0;
        kernel->outputTensorAccessPattern[i].dim[0].end_b    = elementsInVec - 1;
        
        // f_start f(i) = 1*i + 0;
        // f_end   f(i) = 1*i + 0;
        // Resource 0 (OFM) dim 1-4
        for (unsigned int dims = 1; dims < kernel->indexSpaceGeometry.dims; dims++)
        {
            kernel->outputTensorAccessPattern[i].dim[dims].dim      = dims;
            kernel->outputTensorAccessPattern[i].dim[dims].start_a  = 1;
            kernel->outputTensorAccessPattern[i].dim[dims].end_a    = 1;
            kernel->outputTensorAccessPattern[i].dim[dims].start_b  = 0;
            kernel->outputTensorAccessPattern[i].dim[dims].end_b    = 1 - 1;
        }
    }


    /*************************************************************************************
    *    Stage IV -  define scalar parameters and aux tensor
    **************************************************************************************/
    kernel->kernel.paramsNr = sizeof(*def)/ sizeof(int);
    memcpy(&( kernel->kernel.scalarParams[0]), def, sizeof(*def));

    gcapi::GlueCodeReturn_t ret = AuxiliaryTensorSettings(params, kernel, elementsInVec, def->chunkSize, (SortType)def->sortDir);
    if(ret == gcapi::GLUE_INSUFICIENT_AUX_BUFFER_SIZE)
        return ret;

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___bitonic_sort_datain_dataout_o_end - &_binary___bitonic_sort_datain_dataout_o_start);
    unsigned givenBinarySize = kernel->elfSize;
    kernel->elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        memcpy (kernel->kernelElf ,
                    &_binary___bitonic_sort_datain_dataout_o_start,
                    IsaSize);
    }
    else
    {
        retVal = gcapi::GLUE_INSUFICIENT_ELF_BUFFER;
        return retVal;
    }

    return gcapi::GLUE_SUCCESS;
}

