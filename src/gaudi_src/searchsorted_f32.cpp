/**********************************************************************
Copyright (c) 2023 Habana Labs.

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

#include "searchsorted_f32.hpp"
#include <stdio.h>

extern unsigned char _binary___searchsorted_fwd_f32_o_start;
extern unsigned char _binary___searchsorted_fwd_f32_o_end;

 tpc_lib_api::GlueCodeReturn SearchSortedF32::GetKernelName(
             char kernelName [tpc_lib_api::MAX_NODE_NAME])
 {
    strcpy(kernelName,"searchsorted_fwd_f32");
    return tpc_lib_api::GLUE_SUCCESS;
 }

tpc_lib_api::GlueCodeReturn SearchSortedF32::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* params,
            tpc_lib_api::HabanaKernelInstantiation* kernel)
{
    tpc_lib_api::GlueCodeReturn retVal;
    SearchSortedParam* def = static_cast<SearchSortedParam*>(params->nodeParams.nodeParams);

    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (params->inputTensorNr != 2)
    {
        params->inputTensorNr  = 2;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }

    //validate correct amount of output tensors
    if (params->outputTensorNr != 1)
    {
        params->outputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
 
    // validate input and output data type
    if (params->inputTensors[0].geometry.maxSizes[0] != params->inputTensors[1].geometry.maxSizes[0])
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }        
    if (params->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32 ||
        params->inputTensors[1].geometry.dataType != tpc_lib_api::DATA_F32 ||
        params->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_I32)
    {
        params->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
        params->inputTensors[1].geometry.dataType = tpc_lib_api::DATA_F32;
        params->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_I32;
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    int elementsInVec = 64;
    const int c_unrollCount = 1;
    uint64_t outputSizes[gcapi::MAX_TENSOR_DIM] = {0};
    memcpy(outputSizes, params->inputTensors[1].geometry.maxSizes, sizeof(outputSizes));

    //round up to elementsInVec and divide by elementsInVec.
    unsigned depthIndex = (outputSizes[0] + (elementsInVec - 1)) / elementsInVec;
    kernel->indexSpaceRank= 5;
    kernel->indexSpaceGeometry[0] = depthIndex;
	//reduce index space due to unroll.
    kernel->indexSpaceGeometry[1] = (outputSizes[1] +(c_unrollCount-1)) / c_unrollCount; 
    kernel->indexSpaceGeometry[2] = outputSizes[2];
    kernel->indexSpaceGeometry[3] = outputSizes[3];
    kernel->indexSpaceGeometry[4] = outputSizes[4];

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/

    // Index space mapping is calculated using f(i) = Ai + B
    // 'i' is the index space member and A/B constants to be defined.

    for (unsigned i = 0; i < params->inputTensorNr; i++)
    {
        kernel->inputTensorAccessPattern[i].mapping[0].indexSpaceDim  = 0;
        kernel->inputTensorAccessPattern[i].mapping[0].a        = elementsInVec;
        kernel->inputTensorAccessPattern[i].mapping[0].start_b  = 0;
        kernel->inputTensorAccessPattern[i].mapping[0].end_b    = elementsInVec - 1;

        for (int dims = 1; dims < (int)kernel->indexSpaceRank; dims++)
        {
            kernel->inputTensorAccessPattern[i].mapping[dims].indexSpaceDim  = dims;
            kernel->inputTensorAccessPattern[i].mapping[dims].a        = 1;
            kernel->inputTensorAccessPattern[i].mapping[dims].start_b  = 0;
            kernel->inputTensorAccessPattern[i].mapping[dims].end_b    = 1 - 1;
        }        
    }    

    kernel->outputTensorAccessPattern[0].mapping[0].indexSpaceDim  = 0;
    kernel->outputTensorAccessPattern[0].mapping[0].a        = elementsInVec;
    kernel->outputTensorAccessPattern[0].mapping[0].start_b  = 0;
    kernel->outputTensorAccessPattern[0].mapping[0].end_b    = elementsInVec - 1;
	
    for (int dims = 1; dims < (int)kernel->indexSpaceRank; dims++)
    {
        kernel->outputTensorAccessPattern[0].mapping[dims].indexSpaceDim  = dims;
        kernel->outputTensorAccessPattern[0].mapping[dims].a        = 1;
        kernel->outputTensorAccessPattern[0].mapping[dims].start_b  = 0;
        kernel->outputTensorAccessPattern[0].mapping[dims].end_b    = 1 - 1;
    }

    /*************************************************************************************
    *    Stage IV -  Set Auxiliary Tensor
    **************************************************************************************/
    kernel->kernel.paramsNr = sizeof(*def)/ sizeof(int);
    memcpy(&( kernel->kernel.scalarParams[0]), def, sizeof(*def));

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = &_binary___searchsorted_fwd_f32_o_end - &_binary___searchsorted_fwd_f32_o_start;
    unsigned givenBinarySize = kernel->kernel.elfSize;
    kernel->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        memcpy (kernel->kernel.kernelElf ,
                    &_binary___searchsorted_fwd_f32_o_start,
                    IsaSize);
    }
    else
    {
        retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
        return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}


