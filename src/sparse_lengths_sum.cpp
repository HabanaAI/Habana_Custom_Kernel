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

#include "sparse_lengths_sum.hpp"

extern unsigned char _binary___sparse_lengths_sum_i8_2D_f32_embed_o_start;
extern unsigned char _binary___sparse_lengths_sum_i8_2D_f32_embed_o_end;

gcapi::GlueCodeReturn_t SparseLengthsSum::GetKernelName(
        char kernelName [gcapi::MAX_NODE_NAME])
{
    strcpy(kernelName,"sparse_lengths_sum_i8_2D_embed_f32");
    return gcapi::GLUE_SUCCESS;
}

gcapi::GlueCodeReturn_t SparseLengthsSum::GetGcDefinitions(
            gcapi::HabanaKernelParams_t* in_defs,
            gcapi::HabanaKernelInstantiation_t* out_defs)
{
    gcapi::GlueCodeReturn_t retVal;
    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    // validate correct amount of input tensors
    if (in_defs->inputTensorNr != 3)
    {
        in_defs->inputTensorNr  = 3;
        return gcapi::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }

    // validate correct amount of output tensors
    if (in_defs->outputTensorNr != 1)
    {
        in_defs->outputTensorNr  = 1;
        return gcapi::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    // validate tensor dimensions
    if (in_defs->inputTensors[0].geometry.dims  != 2 ||
        in_defs->outputTensors[0].geometry.dims != 2)
    {
        in_defs->inputTensors[0].geometry.dims   = 2;
        in_defs->outputTensors[0].geometry.dims  = 2;
        return gcapi::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    // length tensor and index tensor are 1D
    if (in_defs->inputTensors[1].geometry.dims != 1 ||
        in_defs->inputTensors[2].geometry.dims != 1)
    {
        in_defs->inputTensors[1].geometry.dims  = 1;
        in_defs->inputTensors[2].geometry.dims  = 1;
        return gcapi::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    // output's dimension 0 size is equal to input's 0 dimension size,
    // minus the embed scale and zero-point size.
    if (in_defs->outputTensors[0].geometry.sizes[0] !=
        (in_defs->inputTensors[0].geometry.sizes[0] -
        (2 * sizeof(float) / sizeof(int8_t))))
    {
        in_defs->outputTensors[0].geometry.sizes[0] =
                (in_defs->inputTensors[0].geometry.sizes[0]
                - (2 * sizeof(float) / sizeof(int8_t)));
        return gcapi::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    //  output's dimension 1 size is equal to length tensor size
    if (in_defs->outputTensors[0].geometry.sizes[1] !=
        in_defs->inputTensors[2].geometry.sizes[0])
    {
        in_defs->outputTensors[0].geometry.sizes[1] =
                in_defs->inputTensors[2].geometry.sizes[0];
        return gcapi::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    // validate input data type
    if (in_defs->inputTensors[0].dataType != gcapi::DATA_I8 ||
        in_defs->inputTensors[1].dataType != gcapi::DATA_I32 ||
        in_defs->inputTensors[2].dataType != gcapi::DATA_I32 ||
        in_defs->outputTensors[0].dataType != gcapi::DATA_F32)
    {
        in_defs->inputTensors[0].dataType = gcapi::DATA_I8;
        in_defs->inputTensors[1].dataType = gcapi::DATA_I32;
        in_defs->inputTensors[2].dataType = gcapi::DATA_I32;
        in_defs->outputTensors[0].dataType = gcapi::DATA_F32;
        return gcapi::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor
    **************************************************************************************/
    out_defs->indexSpaceGeometry.dims = 2;

    unsigned eig = 256;
    unsigned unrollCount = 2;

    // round up to 256 and divide by 256 (int8 vec size).
    unsigned depthIndex = (in_defs->outputTensors[0].geometry.sizes[0] + eig - 1) / eig;
    out_defs->indexSpaceGeometry.sizes[0] = depthIndex;
    // round up to 2 and divide by 2 (2 is the unroll count on dim 1).
    unsigned widthIndex =
            (in_defs->outputTensors[0].geometry.sizes[1] + unrollCount - 1) / unrollCount;
    out_defs->indexSpaceGeometry.sizes[1] = widthIndex;

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    //Stating Access Patterns

    //InputTensor
    out_defs->inputTensorAccessPattern[0].allRequired = true;
    //Index tensor
    out_defs->inputTensorAccessPattern[1].allRequired = true;
    //length tensor
    out_defs->inputTensorAccessPattern[2].dim[0].dim        = 1;
    out_defs->inputTensorAccessPattern[2].dim[0].start_a    = 0;
    out_defs->inputTensorAccessPattern[2].dim[0].end_a      = unrollCount;
    out_defs->inputTensorAccessPattern[2].dim[0].start_b    = 0;
    out_defs->inputTensorAccessPattern[2].dim[0].end_b      = unrollCount - 1;
    //we need all elements from length tensor including and before the current lengths being used

    //OutputTensor
    out_defs->outputTensorAccessPattern[0].dim[0].dim       = 0;
    out_defs->outputTensorAccessPattern[0].dim[0].start_a   = eig;
    out_defs->outputTensorAccessPattern[0].dim[0].end_a     = eig;
    out_defs->outputTensorAccessPattern[0].dim[0].start_b   = 0;
    out_defs->outputTensorAccessPattern[0].dim[0].end_b     = eig - 1;

    out_defs->outputTensorAccessPattern[0].dim[1].dim       = 1;
    out_defs->outputTensorAccessPattern[0].dim[1].start_a   = unrollCount;
    out_defs->outputTensorAccessPattern[0].dim[1].end_a     = unrollCount;
    out_defs->outputTensorAccessPattern[0].dim[1].start_b   = 0;
    out_defs->outputTensorAccessPattern[0].dim[1].end_b     = unrollCount - 1;

    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
    out_defs->kernel.paramsNr = 0;

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___sparse_lengths_sum_i8_2D_f32_embed_o_end - &_binary___sparse_lengths_sum_i8_2D_f32_embed_o_start);
    unsigned givenBinarySize = out_defs->elfSize;
    out_defs->elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (out_defs->kernelElf ,
                &_binary___sparse_lengths_sum_i8_2D_f32_embed_o_start,
                IsaSize);
    }
    else
    {
        retVal = gcapi::GLUE_INSUFICIENT_ELF_BUFFER;
        return retVal;
    }
    return gcapi::GLUE_SUCCESS;
}