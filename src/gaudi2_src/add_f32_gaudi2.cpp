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

#include <vector>
#include <cstring>
#include <iostream>
#include "add_f32_gaudi2.hpp"


extern unsigned char _binary___add_f32_gaudi2_o_start;
extern unsigned char _binary___add_f32_gaudi2_o_end;

 tpc_lib_api::GlueCodeReturn AddF32Gaudi2::GetKernelName(
             char kernelName [tpc_lib_api::MAX_NODE_NAME])
 {
     strcpy(kernelName,"custom_add_f32_gaudi2");
     return tpc_lib_api::GLUE_SUCCESS;
 }


tpc_lib_api::GlueCodeReturn AddF32Gaudi2::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* in_defs,
            tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    tpc_lib_api::GlueCodeReturn retVal;
    const int c_unrollCount = 4;

    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (in_defs->inputTensorNr != 2)
    {
        in_defs->inputTensorNr  = 2;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    //validate correct amount of output tensors
    if (in_defs->outputTensorNr !=1)
    {
        in_defs->outputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    //validate matrix dimensions
    if (in_defs->inputTensors[0].geometry.maxSizes[0] != in_defs->inputTensors[1].geometry.maxSizes[0])
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    // validate input and output data type
    if (in_defs->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32 ||
        in_defs->inputTensors[1].geometry.dataType != tpc_lib_api::DATA_F32 ||
        in_defs->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32)
    {
        in_defs->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
        in_defs->inputTensors[1].geometry.dataType = tpc_lib_api::DATA_F32;
        in_defs->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    int elementsInVec = 64;
    uint64_t outputSizes[gcapi::MAX_TENSOR_DIM] = {0};
    memcpy(outputSizes, in_defs->inputTensors[0].geometry.maxSizes, sizeof(outputSizes));

    //round up to elementsInVec and divide by elementsInVec.
    unsigned depthIndex = (outputSizes[0] + (elementsInVec - 1)) / elementsInVec;
    out_defs->indexSpaceRank = 5;
    out_defs->indexSpaceGeometry[0] = depthIndex;
	//reduce index space due to unroll.
    out_defs->indexSpaceGeometry[1] = (outputSizes[1] +(c_unrollCount-1)) / c_unrollCount; 
    out_defs->indexSpaceGeometry[2] = outputSizes[2];
    out_defs->indexSpaceGeometry[3] = outputSizes[3];
    out_defs->indexSpaceGeometry[4] = outputSizes[4];

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/

    // Index space mapping is calculated using f(i) = Ai + B
    // 'i' is the index space member and A/B constants to be defined.
    out_defs->inputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].a        = elementsInVec;
    out_defs->inputTensorAccessPattern[0].mapping[0].start_b  = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].end_b    = elementsInVec - 1;

	out_defs->inputTensorAccessPattern[0].mapping[1].indexSpaceDim      = 1;
    out_defs->inputTensorAccessPattern[0].mapping[1].a        = c_unrollCount;
    out_defs->inputTensorAccessPattern[0].mapping[1].start_b  = 0;
    out_defs->inputTensorAccessPattern[0].mapping[1].end_b    = c_unrollCount - 1;

	out_defs->inputTensorAccessPattern[0].mapping[2].indexSpaceDim      = 2;
    out_defs->inputTensorAccessPattern[0].mapping[2].a        = c_unrollCount;
    out_defs->inputTensorAccessPattern[0].mapping[2].start_b  = 0;
    out_defs->inputTensorAccessPattern[0].mapping[2].end_b    = c_unrollCount - 1;

    // f_start f(i) = 1*i + 0;
    // f_end   f(i) = 1*i + 0;
    // Resource 0 (IFM) dim 1-4
    for (int dims = 3; dims < 5; dims++)
    {
        out_defs->inputTensorAccessPattern[0].mapping[dims].indexSpaceDim      = dims;
        out_defs->inputTensorAccessPattern[0].mapping[dims].a        = 1;
        out_defs->inputTensorAccessPattern[0].mapping[dims].start_b  = 0;
        out_defs->inputTensorAccessPattern[0].mapping[dims].end_b    = 1 - 1;
    }

    // f_start f(i) = elementsInVec*i + 0;
    // f_end   f(i) = elementsInVec*i + (elementsInVec - 1);
    // Resource 0 (OFM) dim 0
    out_defs->outputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].a        = elementsInVec;
    out_defs->outputTensorAccessPattern[0].mapping[0].start_b  = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].end_b    = elementsInVec - 1;
	
	out_defs->outputTensorAccessPattern[0].mapping[1].indexSpaceDim      = 1;
    out_defs->outputTensorAccessPattern[0].mapping[1].a        = c_unrollCount;
    out_defs->outputTensorAccessPattern[0].mapping[1].start_b  = 0;
    out_defs->outputTensorAccessPattern[0].mapping[1].end_b    = c_unrollCount - 1;

	out_defs->outputTensorAccessPattern[0].mapping[2].indexSpaceDim      = 2;
    out_defs->outputTensorAccessPattern[0].mapping[2].a        = c_unrollCount;
    out_defs->outputTensorAccessPattern[0].mapping[2].start_b  = 0;
    out_defs->outputTensorAccessPattern[0].mapping[2].end_b    = c_unrollCount - 1;

    // f_start f(i) = 1*i + 0;
    // f_end   f(i) = 1*i + 0;
    // Resource 0 (OFM) dim 1-4
    for (int dims = 3; dims < 5; dims++)
    {
        out_defs->outputTensorAccessPattern[0].mapping[dims].indexSpaceDim      = dims;
        out_defs->outputTensorAccessPattern[0].mapping[dims].a        = 1;
        out_defs->outputTensorAccessPattern[0].mapping[dims].start_b  = 0;
        out_defs->outputTensorAccessPattern[0].mapping[dims].end_b    = 1 - 1;
    }

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___add_f32_gaudi2_o_end - &_binary___add_f32_gaudi2_o_start);
    unsigned givenBinarySize = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (out_defs->kernel.kernelElf,
                &_binary___add_f32_gaudi2_o_start,
                IsaSize);
    }
    else
    {
       retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
       return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

