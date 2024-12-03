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
#include "mygather_where_gaudi3.hpp"
#include <stdio.h>
#include <iostream>

extern unsigned char _binary___mygather_where_f32_gaudi3_o_start;
extern unsigned char _binary___mygather_where_f32_gaudi3_o_end;

extern unsigned char _binary___mygather_where_bf16_gaudi3_o_start;
extern unsigned char _binary___mygather_where_bf16_gaudi3_o_end;


tpc_lib_api::GlueCodeReturn MygatherwhereGaudi3::GetKernelName(
        char kernelName [tpc_lib_api::MAX_NODE_NAME], mygather_where_mode_t mode)
{
    switch(mode)
    {
        case mygatherw_f32:
            strcpy(kernelName,"custom_mygather_where_f32_gaudi3");
            break;
        case mygatherw_bf16:
            strcpy(kernelName,"custom_mygather_where_bf16_gaudi3");
            break;
        default:
            strcpy(kernelName,"custom_mygather_where_f32_gaudi3");
            break;

    }

    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn MygatherwhereGaudi3::GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* in_defs,
        tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    tpc_lib_api::GlueCodeReturn retVal;
    MygatherwhereParam* def = static_cast<MygatherwhereParam*>(in_defs->nodeParams.nodeParams);
    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (in_defs->inputTensorNr != 4)
    {
        in_defs->inputTensorNr  = 4;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    //validate correct amount of output tensors
    if (in_defs->outputTensorNr != 1)
    {
        in_defs->outputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    // validate input and output data type
    if(m_mode == mygatherw_f32)
    {
        if (in_defs->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32)
        {
            in_defs->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
        if (in_defs->inputTensors[1].geometry.dataType != tpc_lib_api::DATA_I32)
        {
            in_defs->inputTensors[1].geometry.dataType = tpc_lib_api::DATA_I32;
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
        if (in_defs->inputTensors[2].geometry.dataType != tpc_lib_api::DATA_I32)
        {
            in_defs->inputTensors[2].geometry.dataType = tpc_lib_api::DATA_I32;
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
        if (in_defs->inputTensors[3].geometry.dataType != tpc_lib_api::DATA_F32)
        {
            in_defs->inputTensors[3].geometry.dataType = tpc_lib_api::DATA_F32;
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
        if (in_defs->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32)
        {
            in_defs->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
        
    }
    else{
        if (in_defs->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_BF16)
        {
            in_defs->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_BF16;
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
        if (in_defs->inputTensors[1].geometry.dataType != tpc_lib_api::DATA_I32)
        {
            in_defs->inputTensors[1].geometry.dataType = tpc_lib_api::DATA_I32;
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
        if (in_defs->inputTensors[2].geometry.dataType != tpc_lib_api::DATA_I16)
        {
            in_defs->inputTensors[2].geometry.dataType = tpc_lib_api::DATA_I16;
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
        if (in_defs->inputTensors[3].geometry.dataType != tpc_lib_api::DATA_BF16)
        {
            in_defs->inputTensors[3].geometry.dataType = tpc_lib_api::DATA_BF16;
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
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
    uint64_t inputSizes_1[gcapi::MAX_TENSOR_DIM] = {0};

    // copy input state tensor size
    memcpy(inputSizes, in_defs->inputTensors[0].geometry.maxSizes, sizeof(inputSizes));
    memcpy(inputSizes_1, in_defs->inputTensors[1].geometry.maxSizes, sizeof(inputSizes_1));
 
    // verify that output feature map dimension are correct
    for(uint64_t i=0;i<in_defs->outputTensors[0].geometry.dims;i++)
    {
        if (i == 0 || i == 3) // the output tensor dimension 3 is fixed 1, not the same as first input tensor
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
    out_defs->indexSpaceRank = 4;
    out_defs->indexSpaceGeometry[0] = 1; //depthIndex;
    out_defs->indexSpaceGeometry[1] = inputSizes[1];
    out_defs->indexSpaceGeometry[2] = inputSizes[2];
    out_defs->indexSpaceGeometry[3] = inputSizes_1[3];

    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
    out_defs->kernel.paramsNr = sizeof(*def) / sizeof(int);
    memcpy(&(out_defs->kernel.scalarParams[0]), def, sizeof(*def));

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    // f_start f(i) = elementsInVec*i + 0;
    // f_end   f(i) = elementsInVec*i + (elementsInVec - 1);
    // Resource 0 (IFM) dim 0
    int dim_a = 1;
    int dim_end_b = 0;
    // seq_len dimension is doing the pscan
    int head_a = 1;
    int head_end_b = 0;
    int batch_a = 1;
    for (unsigned i = 0; i < in_defs->inputTensorNr; i++)
    {
        dim_a = 1;
        head_a = 1;
        batch_a = 0;
        if(i == 1 || i == 2)
        {
            dim_a = 0;
            head_a = 0;
            batch_a = 1;
        }
        if(i == 3)
        {
            dim_a = 1;
            head_a = 1;
            batch_a = 0;
        }

        out_defs->inputTensorAccessPattern[i].allRequired = true;
        out_defs->inputTensorAccessPattern[i].mapping[0].indexSpaceDim      = 0;
        out_defs->inputTensorAccessPattern[i].mapping[0].a        = 0;
        out_defs->inputTensorAccessPattern[i].mapping[0].start_b  = 0;
        out_defs->inputTensorAccessPattern[i].mapping[0].end_b    = 0;

        out_defs->inputTensorAccessPattern[i].mapping[1].indexSpaceDim      = 1;
        out_defs->inputTensorAccessPattern[i].mapping[1].a        = dim_a;
        out_defs->inputTensorAccessPattern[i].mapping[1].start_b  = 0;
        out_defs->inputTensorAccessPattern[i].mapping[1].end_b    = dim_end_b;

        out_defs->inputTensorAccessPattern[i].mapping[2].indexSpaceDim      = 2;
        out_defs->inputTensorAccessPattern[i].mapping[2].a        = head_a;
        out_defs->inputTensorAccessPattern[i].mapping[2].start_b  = 0;
        out_defs->inputTensorAccessPattern[i].mapping[2].end_b    = head_end_b;

        out_defs->inputTensorAccessPattern[i].mapping[3].indexSpaceDim      = 3;
        out_defs->inputTensorAccessPattern[i].mapping[3].a        = batch_a;
        out_defs->inputTensorAccessPattern[i].mapping[3].start_b  = 0;
        out_defs->inputTensorAccessPattern[i].mapping[3].end_b    = 0;

    }

    // This is to memset the output tensor memory location
    out_defs->outputTensorAccessPattern[0].memsetBeforeExecution = 1;
    // f_start f(i) = elementsInVec*i + 0;
    // f_end   f(i) = elementsInVec*i + (elementsInVec - 1);
    // Resource 0 (OFM) dim 0
    out_defs->outputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].a        = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].start_b  = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].end_b    = 0;
	
    // one 1 for this dstate
	out_defs->outputTensorAccessPattern[0].mapping[1].indexSpaceDim      = 1;
    out_defs->outputTensorAccessPattern[0].mapping[1].a        = 1;
    out_defs->outputTensorAccessPattern[0].mapping[1].start_b  = 0;
    out_defs->outputTensorAccessPattern[0].mapping[1].end_b    = 0;

    out_defs->outputTensorAccessPattern[0].mapping[2].indexSpaceDim      = 2;
    out_defs->outputTensorAccessPattern[0].mapping[2].a        = 1;
    out_defs->outputTensorAccessPattern[0].mapping[2].start_b  = 0;
    out_defs->outputTensorAccessPattern[0].mapping[2].end_b    = 0;

    // f_start f(i) = 1*i + 0;
    // f_end   f(i) = 1*i + 0;
    out_defs->outputTensorAccessPattern[0].mapping[3].indexSpaceDim      = 3;
    out_defs->outputTensorAccessPattern[0].mapping[3].a        = 1;
    out_defs->outputTensorAccessPattern[0].mapping[3].start_b  = 0;
    out_defs->outputTensorAccessPattern[0].mapping[3].end_b    = 1 - 1;



    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___mygather_where_f32_gaudi3_o_end - &_binary___mygather_where_f32_gaudi3_o_start);
    unsigned char *binary_kernel =  &_binary___mygather_where_f32_gaudi3_o_start;
    switch (m_mode)
    {
        case mygatherw_f32:
            IsaSize = (&_binary___mygather_where_f32_gaudi3_o_end - &_binary___mygather_where_f32_gaudi3_o_start);
            binary_kernel = &_binary___mygather_where_f32_gaudi3_o_start;
            break;
        case mygatherw_bf16:
            IsaSize = (&_binary___mygather_where_bf16_gaudi3_o_end - &_binary___mygather_where_bf16_gaudi3_o_start);
            binary_kernel = &_binary___mygather_where_bf16_gaudi3_o_start;
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