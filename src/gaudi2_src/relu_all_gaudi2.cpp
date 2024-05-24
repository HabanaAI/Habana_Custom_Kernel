/**********************************************************************
Copyright (c) 2024 Habana Labs. All rights reserved.

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
#include "relu_all_gaudi2.hpp"

extern unsigned char _binary___relu_fwd_f32_gaudi2_o_start;
extern unsigned char _binary___relu_fwd_f32_gaudi2_o_end;
extern unsigned char _binary___relu_bwd_f32_gaudi2_o_start;
extern unsigned char _binary___relu_bwd_f32_gaudi2_o_end;
extern unsigned char _binary___relu_fwd_bf16_gaudi2_o_start;
extern unsigned char _binary___relu_fwd_bf16_gaudi2_o_end;
extern unsigned char _binary___relu_bwd_bf16_gaudi2_o_start;
extern unsigned char _binary___relu_bwd_bf16_gaudi2_o_end;


tpc_lib_api::GlueCodeReturn ReluAllGaudi2::GetKernelName(
        char kernelName [tpc_lib_api::MAX_NODE_NAME], Relu_mode_t mode)
{
    if(mode == relu_fwd_f32)
        strcpy(kernelName,"custom_relu_fwd_f32_gaudi2");
    else if(mode == relu_bwd_f32)
        strcpy(kernelName,"custom_relu_bwd_f32_gaudi2");
    else if(mode == relu_fwd_bf16)
        strcpy(kernelName,"custom_relu_fwd_bf16_gaudi2");
    else if(mode == relu_bwd_bf16)
        strcpy(kernelName,"custom_relu_bwd_bf16_gaudi2");
    else
        return tpc_lib_api::GLUE_NODE_NOT_FOUND;
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn ReluAllGaudi2::GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* in_defs,
        tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
	const int c_unrollCount = 4;
    tpc_lib_api::GlueCodeReturn retVal;
    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if(m_mode == relu_fwd_f32 || m_mode == relu_fwd_bf16)
    {
        if (in_defs->inputTensorNr != 1)
        {
            in_defs->inputTensorNr  = 1;
            return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
        }
    }
    else{
        if (in_defs->inputTensorNr != 2)
        {
            in_defs->inputTensorNr  = 2;
            return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
        }

    }
    //validate correct amount of output tensors
    if (in_defs->outputTensorNr != 1)
    {
        in_defs->outputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    // validate input and output data type
    if(m_mode == relu_fwd_f32 || m_mode == relu_bwd_f32)
    {
        if (in_defs->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32 ||
            in_defs->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32)
        {
            in_defs->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
            in_defs->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    else{
        if (in_defs->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_BF16 ||
            in_defs->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_BF16)
        {
            in_defs->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_BF16;
            in_defs->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_BF16;
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    // Tensor 0 should be input feature map.
    // The semantics of the input tensors and their order is a convention
    // between TPC kernel writer and the write of the layer at the
    // framework level.
    uint64_t outputSizes[gcapi::MAX_TENSOR_DIM] = {0};

    memcpy(outputSizes, in_defs->inputTensors[0].geometry.maxSizes, sizeof(outputSizes));

    // verify that output feature map dimension are correct
    if (memcmp(in_defs->outputTensors[0].geometry.maxSizes, outputSizes,
               in_defs->outputTensors[0].geometry.dims * sizeof(uint64_t) ) != 0)
    {
        memcpy(in_defs->outputTensors[0].geometry.maxSizes, in_defs->inputTensors[0].geometry.maxSizes, sizeof(outputSizes));
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    int elementsInVec;
    if(m_mode == relu_fwd_f32 || m_mode == relu_bwd_f32)
        elementsInVec = 64;
    else
        elementsInVec = 128;

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
    // f_start f(i) = elementsInVec*i + 0;
    // f_end   f(i) = elementsInVec*i + (elementsInVec - 1);
    // Resource 0 (IFM) dim 0
    out_defs->inputTensorAccessPattern[0].allRequired = true;
    out_defs->inputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].a        = elementsInVec;
    out_defs->inputTensorAccessPattern[0].mapping[0].start_b  = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].end_b    = elementsInVec - 1;

	out_defs->inputTensorAccessPattern[0].mapping[1].indexSpaceDim      = 1;
    out_defs->inputTensorAccessPattern[0].mapping[1].a        = c_unrollCount;
    out_defs->inputTensorAccessPattern[0].mapping[1].start_b  = 0;
    out_defs->inputTensorAccessPattern[0].mapping[1].end_b    = c_unrollCount - 1;
	
    // f_start f(i) = 1*i + 0;
    // f_end   f(i) = 1*i + 0;
    // Resource 0 (IFM) dim 1-4
    for (unsigned int dims = 2; dims < out_defs->indexSpaceRank; dims++)
    {
        out_defs->inputTensorAccessPattern[0].mapping[dims].indexSpaceDim      = dims;
        out_defs->inputTensorAccessPattern[0].mapping[dims].a        = 1;
        out_defs->inputTensorAccessPattern[0].mapping[dims].start_b  = 0;
        out_defs->inputTensorAccessPattern[0].mapping[dims].end_b    = 1 - 1;
    }

    //out_defs->inputTensorAccessPattern[1].allRequired = true;
    out_defs->inputTensorAccessPattern[1].mapping[0].indexSpaceDim      = 0;
    out_defs->inputTensorAccessPattern[1].mapping[0].a        = elementsInVec;
    out_defs->inputTensorAccessPattern[1].mapping[0].start_b  = 0;
    out_defs->inputTensorAccessPattern[1].mapping[0].end_b    = elementsInVec - 1;

	out_defs->inputTensorAccessPattern[1].mapping[1].indexSpaceDim      = 1;
    out_defs->inputTensorAccessPattern[1].mapping[1].a        = c_unrollCount;
    out_defs->inputTensorAccessPattern[1].mapping[1].start_b  = 0;
    out_defs->inputTensorAccessPattern[1].mapping[1].end_b    = c_unrollCount - 1;
	
    // f_start f(i) = 1*i + 0;
    // f_end   f(i) = 1*i + 0;
    // Resource 0 (IFM) dim 1-4
    for (unsigned int dims = 2; dims < out_defs->indexSpaceRank; dims++)
    {
        out_defs->inputTensorAccessPattern[1].mapping[dims].indexSpaceDim      = dims;
        out_defs->inputTensorAccessPattern[1].mapping[dims].a        = 1;
        out_defs->inputTensorAccessPattern[1].mapping[dims].start_b  = 0;
        out_defs->inputTensorAccessPattern[1].mapping[dims].end_b    = 1 - 1;
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

    // f_start f(i) = 1*i + 0;
    // f_end   f(i) = 1*i + 0;
    // Resource 0 (OFM) dim 1-4
    for (unsigned int dims = 2; dims < out_defs->indexSpaceRank; dims++)
    {
        out_defs->outputTensorAccessPattern[0].mapping[dims].indexSpaceDim      = dims;
        out_defs->outputTensorAccessPattern[0].mapping[dims].a        = 1;
        out_defs->outputTensorAccessPattern[0].mapping[dims].start_b  = 0;
        out_defs->outputTensorAccessPattern[0].mapping[dims].end_b    = 1 - 1;
    }


    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
    out_defs->kernel.paramsNr =0;

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___relu_fwd_f32_gaudi2_o_end - &_binary___relu_fwd_f32_gaudi2_o_start);
    unsigned char *binary_kernel =  &_binary___relu_fwd_f32_gaudi2_o_start;
    switch (m_mode)
    {
        case relu_fwd_f32:
            IsaSize = (&_binary___relu_fwd_f32_gaudi2_o_end - &_binary___relu_fwd_f32_gaudi2_o_start);
            binary_kernel = &_binary___relu_fwd_f32_gaudi2_o_start;
            break;
        case relu_bwd_f32:
            IsaSize = (&_binary___relu_bwd_f32_gaudi2_o_end - &_binary___relu_bwd_f32_gaudi2_o_start);
            binary_kernel = &_binary___relu_bwd_f32_gaudi2_o_start;
            break;
        case relu_fwd_bf16:
            IsaSize = (&_binary___relu_fwd_bf16_gaudi2_o_end - &_binary___relu_fwd_bf16_gaudi2_o_start);
            binary_kernel = &_binary___relu_fwd_bf16_gaudi2_o_start;
            break;
        case relu_bwd_bf16:
            IsaSize = (&_binary___relu_bwd_bf16_gaudi2_o_end - &_binary___relu_bwd_bf16_gaudi2_o_start);
            binary_kernel = &_binary___relu_bwd_bf16_gaudi2_o_start;
            break;

        default:
            break;
    
    }
        
    unsigned givenBinarySize = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = IsaSize;
    if (givenBinarySize >= IsaSize)
    {
        memcpy (out_defs->kernel.kernelElf, binary_kernel, IsaSize);
    }
    else
    {
        retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
        return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}


