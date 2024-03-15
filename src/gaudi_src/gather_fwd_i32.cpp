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

#include "gather_fwd_i32.hpp"

extern unsigned char _binary___gather_fwd_dim0_i32_o_start;
extern unsigned char _binary___gather_fwd_dim0_i32_o_end;
extern unsigned char _binary___gather_fwd_dim1_i32_o_start;
extern unsigned char _binary___gather_fwd_dim1_i32_o_end;

tpc_lib_api::GlueCodeReturn GatherFwdI32::GetKernelName(
        char kernelName [tpc_lib_api::MAX_NODE_NAME])
{
    if(m_mode == gather_fwd_dim0)
        strcpy(kernelName,"custom_gather_fwd_dim0_i32");
    else if (m_mode == gather_fwd_dim1)
        strcpy(kernelName,"custom_gather_fwd_dim1_i32");
    else
        return tpc_lib_api::GLUE_NODE_NOT_FOUND;    
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn GatherFwdI32::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* params,
            tpc_lib_api::HabanaKernelInstantiation* kernel)
{
    tpc_lib_api::GlueCodeReturn retVal;
    GatherFwdParam* gwParam = static_cast<GatherFwdParam*>(params->nodeParams.nodeParams);
    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    // validate correct amount of input tensors
    if (params->inputTensorNr != 2)
    {
        params->inputTensorNr  = 2;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }

    // validate correct amount of output tensors
    if (params->outputTensorNr != 1)
    {
        params->outputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    // validate input data type
    if (params->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_I32 ||
        params->inputTensors[1].geometry.dataType != tpc_lib_api::DATA_I32 ||
        params->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_I32)
    {
        params->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_I32;
        params->inputTensors[1].geometry.dataType = tpc_lib_api::DATA_I32;
        params->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_I32;
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor
    **************************************************************************************/
    kernel->indexSpaceRank = 5;

    unsigned elementsInVec = 64;
    unsigned unrollAlongWidth = 4;

    // round up to 64 and divide by 64 (int32 vec size).
    unsigned depthIndex = (params->outputTensors[0].geometry.maxSizes[0] + elementsInVec - 1) / elementsInVec;
    kernel->indexSpaceGeometry[0] = depthIndex;
    // round up to 2 and divide by 2 (2 is the unroll count on dim 1).
    unsigned widthIndex =
            (params->outputTensors[0].geometry.maxSizes[1] + unrollAlongWidth - 1) / unrollAlongWidth;
    kernel->indexSpaceGeometry[1] = widthIndex;
    kernel->indexSpaceGeometry[2] = params->outputTensors[0].geometry.maxSizes[2];
    kernel->indexSpaceGeometry[3] = params->outputTensors[0].geometry.maxSizes[3];
    kernel->indexSpaceGeometry[4] = params->outputTensors[0].geometry.maxSizes[4];

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    //Stating Access Patterns

   // Index space mapping is calculated using f(i) = Ai + B
    // 'i' is the index space member and A/B constants to be defined.
    kernel->inputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
    kernel->inputTensorAccessPattern[0].mapping[0].a        = elementsInVec;
    kernel->inputTensorAccessPattern[0].mapping[0].start_b  = 0;
    kernel->inputTensorAccessPattern[0].mapping[0].end_b    = elementsInVec - 1;

    kernel->inputTensorAccessPattern[1].mapping[0].indexSpaceDim      = 0;
    kernel->inputTensorAccessPattern[1].mapping[0].a        = elementsInVec;
    kernel->inputTensorAccessPattern[1].mapping[0].start_b  = 0;
    kernel->inputTensorAccessPattern[1].mapping[0].end_b    = elementsInVec - 1;

    // f_start f(i) = 1*i + 0;
    // f_end   f(i) = 1*i + 0;
    // Resource 0 (IFM) dim 1-4
    for (int dims = 1; dims < 5; dims++)
    {

        kernel->inputTensorAccessPattern[0].mapping[dims].indexSpaceDim      = dims;
        kernel->inputTensorAccessPattern[0].mapping[dims].a        = 1;
        kernel->inputTensorAccessPattern[0].mapping[dims].start_b  = 0;
        kernel->inputTensorAccessPattern[0].mapping[dims].end_b    = 1 - 1;
    }


    // f_start f(i) = elementsInVec*i + 0;
    // f_end   f(i) = elementsInVec*i + (elementsInVec - 1);
    // Resource 0 (OFM) dim 0
    kernel->outputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
    kernel->outputTensorAccessPattern[0].mapping[0].a        = elementsInVec;
    kernel->outputTensorAccessPattern[0].mapping[0].start_b  = 0;
    kernel->outputTensorAccessPattern[0].mapping[0].end_b    = elementsInVec - 1;
	
    // f_start f(i) = 1*i + 0;
    // f_end   f(i) = 1*i + 0;
    // Resource 0 (OFM) dim 1-4
    for (int dims = 1; dims < 5; dims++)
    {
        kernel->outputTensorAccessPattern[0].mapping[dims].indexSpaceDim      = dims;
        kernel->outputTensorAccessPattern[0].mapping[dims].a        = 1;
        kernel->outputTensorAccessPattern[0].mapping[dims].start_b  = 0;
        kernel->outputTensorAccessPattern[0].mapping[dims].end_b    = 1 - 1;
    }

    kernel->inputTensorAccessPattern[0].mapping[gwParam->axis].a = 0;
    kernel->inputTensorAccessPattern[0].mapping[gwParam->axis].start_b = 0;
    kernel->inputTensorAccessPattern[0].mapping[gwParam->axis].end_b =
        params->inputTensors[0].geometry.maxSizes[gwParam->axis] - 1;

    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
    
    kernel->kernel.paramsNr = sizeof(*gwParam) / sizeof(int);
    memcpy(&(kernel->kernel.scalarParams[0]), gwParam, sizeof(*gwParam));

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___gather_fwd_dim0_i32_o_end - &_binary___gather_fwd_dim0_i32_o_start);
    unsigned char *binary_kernel =  &_binary___gather_fwd_dim0_i32_o_start;

    switch (m_mode)
    {
        case gather_fwd_dim0:
            IsaSize = (&_binary___gather_fwd_dim0_i32_o_end - &_binary___gather_fwd_dim0_i32_o_start);
            binary_kernel = &_binary___gather_fwd_dim0_i32_o_start;
            break;
        case gather_fwd_dim1:
            IsaSize = (&_binary___gather_fwd_dim1_i32_o_end - &_binary___gather_fwd_dim1_i32_o_start);
            binary_kernel = &_binary___gather_fwd_dim1_i32_o_start;
            break;

        default:
            break;
    
    }
        
    unsigned givenBinarySize = kernel->kernel.elfSize;
    kernel->kernel.elfSize = IsaSize;
    if (givenBinarySize >= IsaSize)
    {
        memcpy (kernel->kernel.kernelElf, binary_kernel, IsaSize);
    }
    else
    {
        retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
        return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}