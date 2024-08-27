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
#include "selective_state_update_gaudi2.hpp"
#include <stdio.h>

extern unsigned char _binary___sel_state_update_f32_gaudi2_o_start;
extern unsigned char _binary___sel_state_update_f32_gaudi2_o_end;
extern unsigned char _binary___sel_state_update_nosp_f32_gaudi2_o_start;
extern unsigned char _binary___sel_state_update_nosp_f32_gaudi2_o_end;
extern unsigned char _binary___sel_state_update_noz_f32_gaudi2_o_start;
extern unsigned char _binary___sel_state_update_noz_f32_gaudi2_o_end;
extern unsigned char _binary___sel_state_update_nosp_noz_f32_gaudi2_o_start;
extern unsigned char _binary___sel_state_update_nosp_noz_f32_gaudi2_o_end;

extern unsigned char _binary___sel_state_update_bf16_gaudi2_o_start;
extern unsigned char _binary___sel_state_update_bf16_gaudi2_o_end;
extern unsigned char _binary___sel_state_update_nosp_bf16_gaudi2_o_start;
extern unsigned char _binary___sel_state_update_nosp_bf16_gaudi2_o_end;
extern unsigned char _binary___sel_state_update_noz_bf16_gaudi2_o_start;
extern unsigned char _binary___sel_state_update_noz_bf16_gaudi2_o_end;
extern unsigned char _binary___sel_state_update_nosp_noz_bf16_gaudi2_o_start;
extern unsigned char _binary___sel_state_update_nosp_noz_bf16_gaudi2_o_end;


tpc_lib_api::GlueCodeReturn SelectiveStateUpdateGaudi2::GetKernelName(
        char kernelName [tpc_lib_api::MAX_NODE_NAME], SSU_mode_t mode)
{
    switch(mode)
    {
        case sel_state_update_f32:
            strcpy(kernelName,"selective_state_update_f32_gaudi2");
            break;
        case sel_state_update_nosp_f32:
            strcpy(kernelName,"selective_state_update_nosp_f32_gaudi2");
            break;
        case sel_state_update_noz_f32:
            strcpy(kernelName,"selective_state_update_noz_f32_gaudi2");
            break;
        case sel_state_update_nosp_noz_f32:
            strcpy(kernelName,"selective_state_update_nosp_noz_f32_gaudi2");
            break;
        case sel_state_update_bf16:
            strcpy(kernelName,"selective_state_update_bf16_gaudi2");
            break;
        case sel_state_update_nosp_bf16:
            strcpy(kernelName,"selective_state_update_nosp_bf16_gaudi2");
            break;
        case sel_state_update_noz_bf16:
            strcpy(kernelName,"selective_state_update_noz_bf16_gaudi2");
            break;
        case sel_state_update_nosp_noz_bf16:
            strcpy(kernelName,"selective_state_update_nosp_noz_bf16_gaudi2");
            break;
        default:
            strcpy(kernelName,"selective_state_update_f32_gaudi2");
            break;

    }

    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn SelectiveStateUpdateGaudi2::GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* in_defs,
        tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    tpc_lib_api::GlueCodeReturn retVal;
    SSUParam* def = static_cast<SSUParam*>(in_defs->nodeParams.nodeParams);
    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if(m_mode == sel_state_update_f32 || m_mode == sel_state_update_nosp_f32 || m_mode == sel_state_update_bf16 || m_mode == sel_state_update_nosp_bf16)
    {
        if (in_defs->inputTensorNr != 9)
        {
            in_defs->inputTensorNr  = 9;
            return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
        }
    }
    else{
        if (in_defs->inputTensorNr != 8)
        {
            in_defs->inputTensorNr  = 8;
            return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
        }

    }
    //validate correct amount of output tensors
    if (in_defs->outputTensorNr != 2)
    {
        in_defs->outputTensorNr  = 2;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    // validate input and output data type
    if(m_mode == sel_state_update_f32 || m_mode == sel_state_update_nosp_f32 || m_mode == sel_state_update_noz_f32 || m_mode == sel_state_update_nosp_noz_f32)
    {
        for(unsigned int i = 0; i < in_defs->inputTensorNr; i++)
        {
            if (in_defs->inputTensors[i].geometry.dataType != tpc_lib_api::DATA_F32)
            {
                in_defs->inputTensors[i].geometry.dataType = tpc_lib_api::DATA_F32;
                return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
            }
        }
        if (in_defs->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32)
        {
            in_defs->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
        
    }
    else{
        for(unsigned int i = 0; i < in_defs->inputTensorNr; i++)
        {
            if (in_defs->inputTensors[i].geometry.dataType != tpc_lib_api::DATA_BF16)
            {
                in_defs->inputTensors[i].geometry.dataType = tpc_lib_api::DATA_BF16;
                return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
            }
        }
        if (in_defs->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_BF16)
        {
            in_defs->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_BF16;
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    // Tensor 0 should be input feature map.
    // The semantics of the input tensors and their order is a convention
    // between TPC kernel writer and the write of the layer at the
    // framework level.
    uint64_t inputSizes[gcapi::MAX_TENSOR_DIM] = {0};

    memcpy(inputSizes, in_defs->inputTensors[0].geometry.maxSizes, sizeof(inputSizes));
 
    // verify that output feature map dimension are correct
    for(uint64_t i=0;i<in_defs->outputTensors[0].geometry.dims;i++)
    {
        if (i == 1) // the output tensor dimension 1 is fixed 1, not the same as first input tensor
            continue;
        if (memcmp(&(in_defs->outputTensors[0].geometry.maxSizes[i]), &(inputSizes[i]),
                    sizeof(uint64_t) ) != 0)
        {
            memcpy(in_defs->outputTensors[0].geometry.maxSizes, in_defs->inputTensors[0].geometry.maxSizes, sizeof(inputSizes));

            return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        }
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    int elementsInVec;
    if(m_mode < sel_state_update_bf16 )
        elementsInVec = 64;
    else
        elementsInVec = 128;

    //round up to elementsInVec and divide by elementsInVec.
    unsigned depthIndex = (inputSizes[0] + (elementsInVec - 1)) / elementsInVec;
    out_defs->indexSpaceRank = 4;
    out_defs->indexSpaceGeometry[0] = depthIndex;
    out_defs->indexSpaceGeometry[1] = 1;
    out_defs->indexSpaceGeometry[2] = inputSizes[2];
    out_defs->indexSpaceGeometry[3] = inputSizes[3];

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    // f_start f(i) = elementsInVec*i + 0;
    // f_end   f(i) = elementsInVec*i + (elementsInVec - 1);
    // Resource 0 (IFM) dim 0
    int dim_a;
    int dim_end_b;
    int dstate_a;
    int dstate_end_b;
    int batch_a;
    for (unsigned i = 0; i < in_defs->inputTensorNr; i++)
    {
        dim_a = elementsInVec;
        dim_end_b = elementsInVec - 1;
        dstate_a = 0;
        dstate_end_b = 0;

        batch_a = 1;

        if((i == 1) || (i == 2) || (i == 8))
        {
            dstate_a = 0;
            dstate_end_b = 0;
        }
        if(i == 3)
            batch_a = 0;
        if((i== 4) || (i == 5))
        {
            dim_a = 0;
            dim_end_b = 0;
        }
        /*if((i==6) || (i ==7))
        {
            dstate_a = batch_a = 0;
            dstate_end_b = 0;
        }*/
        out_defs->inputTensorAccessPattern[i].allRequired = true;
        out_defs->inputTensorAccessPattern[i].mapping[0].indexSpaceDim      = 0;
        out_defs->inputTensorAccessPattern[i].mapping[0].a        = dim_a;
        out_defs->inputTensorAccessPattern[i].mapping[0].start_b  = 0;
        out_defs->inputTensorAccessPattern[i].mapping[0].end_b    = dim_end_b;

        out_defs->inputTensorAccessPattern[i].mapping[1].indexSpaceDim      = 1;
        out_defs->inputTensorAccessPattern[i].mapping[1].a        = dstate_a;
        out_defs->inputTensorAccessPattern[i].mapping[1].start_b  = 0;
        out_defs->inputTensorAccessPattern[i].mapping[1].end_b    = dstate_end_b;

        out_defs->inputTensorAccessPattern[i].mapping[2].indexSpaceDim      = 2;
        out_defs->inputTensorAccessPattern[i].mapping[2].a        = 1;
        out_defs->inputTensorAccessPattern[i].mapping[2].start_b  = 0;
        out_defs->inputTensorAccessPattern[i].mapping[2].end_b    = 0;

        out_defs->inputTensorAccessPattern[i].mapping[3].indexSpaceDim      = 3;
        out_defs->inputTensorAccessPattern[i].mapping[3].a        = batch_a;
        out_defs->inputTensorAccessPattern[i].mapping[3].start_b  = 0;
        out_defs->inputTensorAccessPattern[i].mapping[3].end_b    = 0;

    }

    out_defs->outputTensorAccessPattern[0].memsetBeforeExecution = 1;
    // f_start f(i) = elementsInVec*i + 0;
    // f_end   f(i) = elementsInVec*i + (elementsInVec - 1);
    // Resource 0 (OFM) dim 0
    out_defs->outputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].a        = elementsInVec;
    out_defs->outputTensorAccessPattern[0].mapping[0].start_b  = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].end_b    = elementsInVec - 1;
	
    // one 1 for this dstate
	out_defs->outputTensorAccessPattern[0].mapping[1].indexSpaceDim      = 1;
    out_defs->outputTensorAccessPattern[0].mapping[1].a        = 0;
    out_defs->outputTensorAccessPattern[0].mapping[1].start_b  = 0;
    out_defs->outputTensorAccessPattern[0].mapping[1].end_b    = 0;

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

    out_defs->outputTensorAccessPattern[1].mapping[0].indexSpaceDim      = 0;
    out_defs->outputTensorAccessPattern[1].mapping[0].a        = elementsInVec;
    out_defs->outputTensorAccessPattern[1].mapping[0].start_b  = 0;
    out_defs->outputTensorAccessPattern[1].mapping[0].end_b    = elementsInVec - 1;
	
    // one 1 for this dstate
	out_defs->outputTensorAccessPattern[1].mapping[1].indexSpaceDim      = 1;
    out_defs->outputTensorAccessPattern[1].mapping[1].a        = 0;
    out_defs->outputTensorAccessPattern[1].mapping[1].start_b  = 0;
    out_defs->outputTensorAccessPattern[1].mapping[1].end_b    = 0;

    // f_start f(i) = 1*i + 0;
    // f_end   f(i) = 1*i + 0;
    // Resource 0 (OFM) dim 1-4
    for (unsigned int dims = 2; dims < out_defs->indexSpaceRank; dims++)
    {
        out_defs->outputTensorAccessPattern[1].mapping[dims].indexSpaceDim      = dims;
        out_defs->outputTensorAccessPattern[1].mapping[dims].a        = 1;
        out_defs->outputTensorAccessPattern[1].mapping[dims].start_b  = 0;
        out_defs->outputTensorAccessPattern[1].mapping[dims].end_b    = 1 - 1;
    }

    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
    out_defs->kernel.paramsNr = sizeof(*def)/ sizeof(int);
    memcpy(&( out_defs->kernel.scalarParams[0]),def, sizeof(*def));

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___sel_state_update_f32_gaudi2_o_end - &_binary___sel_state_update_f32_gaudi2_o_start);
    unsigned char *binary_kernel =  &_binary___sel_state_update_f32_gaudi2_o_start;
    switch (m_mode)
    {
        case sel_state_update_f32:
            IsaSize = (&_binary___sel_state_update_f32_gaudi2_o_end - &_binary___sel_state_update_f32_gaudi2_o_start);
            binary_kernel = &_binary___sel_state_update_f32_gaudi2_o_start;
            break;
        case sel_state_update_nosp_f32:
            IsaSize = (&_binary___sel_state_update_nosp_f32_gaudi2_o_end - &_binary___sel_state_update_nosp_f32_gaudi2_o_start);
            binary_kernel = &_binary___sel_state_update_nosp_f32_gaudi2_o_start;
            break;
        case sel_state_update_noz_f32:
            IsaSize = (&_binary___sel_state_update_noz_f32_gaudi2_o_end - &_binary___sel_state_update_noz_f32_gaudi2_o_start);
            binary_kernel = &_binary___sel_state_update_noz_f32_gaudi2_o_start;
            break;
        case sel_state_update_nosp_noz_f32:
            IsaSize = (&_binary___sel_state_update_nosp_noz_f32_gaudi2_o_end - &_binary___sel_state_update_nosp_noz_f32_gaudi2_o_start);
            binary_kernel = &_binary___sel_state_update_nosp_noz_f32_gaudi2_o_start;
            break;
        case sel_state_update_bf16:
            IsaSize = (&_binary___sel_state_update_bf16_gaudi2_o_end - &_binary___sel_state_update_bf16_gaudi2_o_start);
            binary_kernel = &_binary___sel_state_update_bf16_gaudi2_o_start;
            break;
        case sel_state_update_nosp_bf16:
            IsaSize = (&_binary___sel_state_update_nosp_bf16_gaudi2_o_end - &_binary___sel_state_update_nosp_bf16_gaudi2_o_start);
            binary_kernel = &_binary___sel_state_update_nosp_bf16_gaudi2_o_start;
            break;
        case sel_state_update_noz_bf16:
            IsaSize = (&_binary___sel_state_update_noz_bf16_gaudi2_o_end - &_binary___sel_state_update_noz_bf16_gaudi2_o_start);
            binary_kernel = &_binary___sel_state_update_noz_bf16_gaudi2_o_start;
            break;
        case sel_state_update_nosp_noz_bf16:
            IsaSize = (&_binary___sel_state_update_nosp_noz_bf16_gaudi2_o_end - &_binary___sel_state_update_nosp_noz_bf16_gaudi2_o_start);
            binary_kernel = &_binary___sel_state_update_nosp_noz_bf16_gaudi2_o_start;
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


