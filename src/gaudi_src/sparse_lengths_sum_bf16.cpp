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

#include "sparse_lengths_sum_bf16.hpp"

extern unsigned char _binary___sparse_lengths_sum_bf16_2D_f32_embed_o_start;
extern unsigned char _binary___sparse_lengths_sum_bf16_2D_f32_embed_o_end;

tpc_lib_api::GlueCodeReturn SparseLengthsSumBF16::GetKernelName(
        char kernelName [tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName,"custom_sparse_lengths_sum_bf16_2D_embed_f32");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn SparseLengthsSumBF16::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* params,
            tpc_lib_api::HabanaKernelInstantiation* kernel)
{
    tpc_lib_api::GlueCodeReturn retVal;
    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    // validate correct amount of input tensors
    if (params->inputTensorNr != 3)
    {
        params->inputTensorNr  = 3;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }

    // validate correct amount of output tensors
    if (params->outputTensorNr != 1)
    {
        params->outputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    // validate tensor dimensions
    if (params->inputTensors[0].geometry.dims  != 2 ||
        params->outputTensors[0].geometry.dims != 2)
    {
        params->inputTensors[0].geometry.dims   = 2;
        params->outputTensors[0].geometry.dims  = 2;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    // length tensor and index tensor are 1D
    if (params->inputTensors[1].geometry.dims != 1 ||
        params->inputTensors[2].geometry.dims != 1)
    {
        params->inputTensors[1].geometry.dims  = 1;
        params->inputTensors[2].geometry.dims  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    // output's dimension 0 size is equal to input's 0 dimension size,
    // minus the embed scale and zero-point size.
    if (params->outputTensors[0].geometry.maxSizes[0] !=
        (params->inputTensors[0].geometry.maxSizes[0] -
        (2 * sizeof(float) / sizeof(int8_t))))
    {
        params->outputTensors[0].geometry.maxSizes[0] =
                (params->inputTensors[0].geometry.maxSizes[0]
                - (2 * sizeof(float) / sizeof(int8_t)));
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    //  output's dimension 1 size is equal to length tensor size
    if (params->outputTensors[0].geometry.maxSizes[1] !=
        params->inputTensors[2].geometry.maxSizes[0])
    {
        params->outputTensors[0].geometry.maxSizes[1] =
                params->inputTensors[2].geometry.maxSizes[0];
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    // validate input data type
    if (params->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_BF16 ||
        params->inputTensors[1].geometry.dataType != tpc_lib_api::DATA_I32 ||
        params->inputTensors[2].geometry.dataType != tpc_lib_api::DATA_I32 ||
        params->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32)
    {
        params->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_BF16;
        params->inputTensors[1].geometry.dataType = tpc_lib_api::DATA_I32;
        params->inputTensors[2].geometry.dataType = tpc_lib_api::DATA_I32;
        params->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor
    **************************************************************************************/
    kernel->indexSpaceRank = 2;

    unsigned eig = 128;
    unsigned unrollCount = 2;

    // round up to 256 and divide by 256 (int8 vec size).
    uint64_t depthIndex = (params->outputTensors[0].geometry.maxSizes[0] + eig - 1) / eig;
    kernel->indexSpaceGeometry[0] = depthIndex;
    // round up to 2 and divide by 2 (2 is the unroll count on dim 1).
    uint64_t widthIndex =
            (params->outputTensors[0].geometry.maxSizes[1] + unrollCount - 1) / unrollCount;
    kernel->indexSpaceGeometry[1] = widthIndex;

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    //Stating Access Patterns

    //InputTensor
    kernel->inputTensorAccessPattern[0].allRequired = true;
    //Index tensor
    kernel->inputTensorAccessPattern[1].allRequired = true;
    //length tensor
    kernel->inputTensorAccessPattern[2].mapping[0].indexSpaceDim        = 1;
    kernel->inputTensorAccessPattern[2].mapping[0].a          = 0;
    kernel->inputTensorAccessPattern[2].mapping[0].start_b    = 0;
    kernel->inputTensorAccessPattern[2].mapping[0].end_b      = unrollCount - 1;
    //we need all elements from length tensor including and before the current lengths being used

    //OutputTensor
    kernel->outputTensorAccessPattern[0].mapping[0].indexSpaceDim       = 0;
    kernel->outputTensorAccessPattern[0].mapping[0].a         = eig;
    kernel->outputTensorAccessPattern[0].mapping[0].start_b   = 0;
    kernel->outputTensorAccessPattern[0].mapping[0].end_b     = eig - 1;

    kernel->outputTensorAccessPattern[0].mapping[1].indexSpaceDim       = 1;
    kernel->outputTensorAccessPattern[0].mapping[1].a         = unrollCount;
    kernel->outputTensorAccessPattern[0].mapping[1].start_b   = 0;
    kernel->outputTensorAccessPattern[0].mapping[1].end_b     = unrollCount - 1;

    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
    kernel->kernel.paramsNr = 0;

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___sparse_lengths_sum_bf16_2D_f32_embed_o_end - &_binary___sparse_lengths_sum_bf16_2D_f32_embed_o_start);
    unsigned givenBinarySize = kernel->kernel.elfSize;
    kernel->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (kernel->kernel.kernelElf ,
                &_binary___sparse_lengths_sum_bf16_2D_f32_embed_o_start,
                IsaSize);
    }
    else
    {
        retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
        return retVal;
    }
    return tpc_lib_api::GLUE_SUCCESS;
}