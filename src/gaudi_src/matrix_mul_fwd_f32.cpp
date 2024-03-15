/**********************************************************************
Copyright (c) 2020 Habana Labs.

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
#include "matrix_mul_fwd_f32.hpp"


extern unsigned char _binary___matrix_mul_fwd_f32_o_start;
extern unsigned char _binary___matrix_mul_fwd_f32_o_end;

 tpc_lib_api::GlueCodeReturn MatrixMulFwdF32::GetKernelName(
             char kernelName [tpc_lib_api::MAX_NODE_NAME])
 {
     strcpy(kernelName,"custom_matrix_multiply_fwd_f32");
     return tpc_lib_api::GLUE_SUCCESS;
 }


tpc_lib_api::GlueCodeReturn MatrixMulFwdF32::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* in_defs,
            tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    tpc_lib_api::GlueCodeReturn retVal;

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
    if ((in_defs->inputTensors[0].geometry.dims != 2 &&
         in_defs->inputTensors[0].geometry.dims != 3) ||
        (in_defs->inputTensors[1].geometry.dims != 2 &&
         in_defs->inputTensors[1].geometry.dims != 3) ||
        in_defs->inputTensors[0].geometry.maxSizes[0] != in_defs->inputTensors[1].geometry.maxSizes[1])
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if ((in_defs->outputTensors[0].geometry.dims != 2 &&
         in_defs->outputTensors[0].geometry.dims != 3) ||
        in_defs->outputTensors[0].geometry.maxSizes[0] != in_defs->inputTensors[1].geometry.maxSizes[0] ||
        in_defs->outputTensors[0].geometry.maxSizes[1] != in_defs->inputTensors[0].geometry.maxSizes[1])
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
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
    *    Stage II-IV -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    int32_t c_vlen             = 64;
    int32_t c_height_step_size = 6;   

    out_defs->indexSpaceRank = 3;
    out_defs->indexSpaceGeometry[0] =
        (in_defs->outputTensors[0].geometry.maxSizes[0] + c_vlen - 1) / c_vlen;
    out_defs->indexSpaceGeometry[1] =
        (in_defs->outputTensors[0].geometry.maxSizes[1] + c_height_step_size - 1) / c_height_step_size;
    out_defs->indexSpaceGeometry[2] =
        std::max(in_defs->outputTensors[0].geometry.maxSizes[2], (uint64_t)1);

    // Matrix C - Tensor Access Pattern
    out_defs->outputTensorAccessPattern[0].mapping[0].indexSpaceDim     = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].a = c_vlen;
    out_defs->outputTensorAccessPattern[0].mapping[0].start_b = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].end_b   = c_vlen - 1;

    out_defs->outputTensorAccessPattern[0].mapping[1].indexSpaceDim     = 1;
    out_defs->outputTensorAccessPattern[0].mapping[1].a = c_height_step_size;
    out_defs->outputTensorAccessPattern[0].mapping[1].start_b = 0;
    out_defs->outputTensorAccessPattern[0].mapping[1].end_b   = c_height_step_size - 1;

    out_defs->outputTensorAccessPattern[0].mapping[2].indexSpaceDim     = 2;
    out_defs->outputTensorAccessPattern[0].mapping[2].a = 1;
    out_defs->outputTensorAccessPattern[0].mapping[2].start_b = 0;
    out_defs->outputTensorAccessPattern[0].mapping[2].end_b   = 1 - 1;

    // Matrix A - Tensor Access Pattern
    out_defs->inputTensorAccessPattern[0].mapping[0].indexSpaceDim     = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].a = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].start_b = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].end_b =
        in_defs->inputTensors[0].geometry.maxSizes[0] - 1;

    out_defs->inputTensorAccessPattern[0].mapping[1].indexSpaceDim     = 1;
    out_defs->inputTensorAccessPattern[0].mapping[1].a = c_height_step_size;
    out_defs->inputTensorAccessPattern[0].mapping[1].start_b = 0;
    out_defs->inputTensorAccessPattern[0].mapping[1].end_b   = c_height_step_size - 1;

    out_defs->inputTensorAccessPattern[0].mapping[2].indexSpaceDim     = 2;
    out_defs->inputTensorAccessPattern[0].mapping[2].a = 1;
    out_defs->inputTensorAccessPattern[0].mapping[2].start_b = 0;
    out_defs->inputTensorAccessPattern[0].mapping[2].end_b   = 1 - 1;

    // Matrix B - Tensor Access Pattern
    out_defs->inputTensorAccessPattern[1].mapping[0].indexSpaceDim     = 0;
    out_defs->inputTensorAccessPattern[1].mapping[0].a = c_vlen;
    out_defs->inputTensorAccessPattern[1].mapping[0].start_b = 0;
    out_defs->inputTensorAccessPattern[1].mapping[0].end_b   = c_vlen - 1;

    out_defs->inputTensorAccessPattern[1].mapping[1].indexSpaceDim     = 1;
    out_defs->inputTensorAccessPattern[1].mapping[1].a = 0;
    out_defs->inputTensorAccessPattern[1].mapping[1].start_b = 0;
    out_defs->inputTensorAccessPattern[1].mapping[1].end_b =
        in_defs->inputTensors[1].geometry.maxSizes[1] - 1;

    out_defs->inputTensorAccessPattern[1].mapping[2].indexSpaceDim     = 2;
    out_defs->inputTensorAccessPattern[1].mapping[2].a = 1;
    out_defs->inputTensorAccessPattern[1].mapping[2].start_b = 0;
    out_defs->inputTensorAccessPattern[1].mapping[2].end_b   = 1 - 1;


    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___matrix_mul_fwd_f32_o_end - &_binary___matrix_mul_fwd_f32_o_start);
    unsigned givenBinarySize = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (out_defs->kernel.kernelElf,
                &_binary___matrix_mul_fwd_f32_o_start,
                IsaSize);
    }
    else
    {
       retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
       return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

