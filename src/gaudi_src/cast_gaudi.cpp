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
#include <cmath>
#include <limits>
#include <iostream>
#include "cast_gaudi.hpp"


extern unsigned char _binary___cast_bf16_to_f32_o_start;
extern unsigned char _binary___cast_bf16_to_f32_o_end;
extern unsigned char _binary___cast_f32_to_bf16_o_start;
extern unsigned char _binary___cast_f32_to_bf16_o_end;

 tpc_lib_api::GlueCodeReturn CastGaudi::GetKernelName(
             char kernelName [tpc_lib_api::MAX_NODE_NAME],
             CastGaudi::CastDataType_t mode)
 {
     strcpy(kernelName, "custom_cast_");
     strcat(kernelName, castDataType[mode]);
     return tpc_lib_api::GLUE_SUCCESS;
 }

tpc_lib_api::GlueCodeReturn CastGaudi::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* in_defs,
            tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    tpc_lib_api::GlueCodeReturn retVal;
    CastParams* def = static_cast<CastParams*>(in_defs->nodeParams.nodeParams);
    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (in_defs->inputTensorNr != 1)
    {
        in_defs->inputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    //validate correct amount of output tensors
    if (in_defs->outputTensorNr != 1)
    {
        in_defs->outputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    // validate input and output data type
    if (m_mode == 0)
    {
        if (in_defs->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_BF16 ||
            in_defs->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32)
        {
            in_defs->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_BF16;
            in_defs->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    else if (m_mode == 1)
    {
        if (in_defs->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32 ||
            in_defs->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_BF16)
        {
            in_defs->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
            in_defs->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_BF16;
            return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    else
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
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
    out_defs->indexSpaceRank = 4;
    int elementsInVec =  128; // for both modes

    //round up to elementsInVec and divide by elementsInVec.
    unsigned depthIndex = (outputSizes[0] + (elementsInVec - 1)) / elementsInVec;
    out_defs->indexSpaceGeometry[0] = depthIndex;
    out_defs->indexSpaceGeometry[1] = outputSizes[1] ;
    out_defs->indexSpaceGeometry[2] = outputSizes[2];
    out_defs->indexSpaceGeometry[3] = outputSizes[3];

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    // f_start(i) = elementsInVec*i + 0;
    // f_end f(i) = elementsInVec*i + (elementsInVec - 1);
    // Resource 0 (IFM) dim 0
    out_defs->inputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].a        = elementsInVec;
    out_defs->inputTensorAccessPattern[0].mapping[0].start_b  = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].end_b    = elementsInVec - 1;

    // f_start(i) = 1*i + 0;
    // f_end f(i) = 1*i + 0;
    // Resource 0 (IFM) dim 1-4
    for (int dims= 1; dims < 4; dims++)
    {
        out_defs->inputTensorAccessPattern[0].mapping[dims].indexSpaceDim      = dims;
        out_defs->inputTensorAccessPattern[0].mapping[dims].a        = 1;
        out_defs->inputTensorAccessPattern[0].mapping[dims].start_b  = 0;
        out_defs->inputTensorAccessPattern[0].mapping[dims].end_b    = 1 - 1;
    }

    // f_start(i) = elementsInVec*i + 0;
    // f_end f(i) = elementsInVec*i + (elementsInVec - 1);
    // Resource 0 (OFM) dim 0
    out_defs->outputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].a        = elementsInVec;
    out_defs->outputTensorAccessPattern[0].mapping[0].start_b  = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].end_b    = elementsInVec - 1;

    // f_start(i) = 1*i + 0;
    // f_end f(i) = 1*i + 0;
    // Resource 0 (OFM) dim 1-4
    for (int dims= 1; dims < 4; dims++)
    {
        out_defs->outputTensorAccessPattern[0].mapping[dims].indexSpaceDim      = dims;
        out_defs->outputTensorAccessPattern[0].mapping[dims].a        = 1;
        out_defs->outputTensorAccessPattern[0].mapping[dims].start_b  = 0;
        out_defs->outputTensorAccessPattern[0].mapping[dims].end_b    = 1 - 1;
    }


    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
    out_defs->kernel.paramsNr = sizeof(*def)/ sizeof(int);
    memcpy(&( out_defs->kernel.scalarParams[0]),def, sizeof(*def));

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___cast_bf16_to_f32_o_end - &_binary___cast_bf16_to_f32_o_start);
    if (m_mode == 1)
    {
        IsaSize = (&_binary___cast_f32_to_bf16_o_end - &_binary___cast_f32_to_bf16_o_start);
    }
    unsigned givenBinarySize = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        if (m_mode == 0)
        {
            // copy binary out
            memcpy (out_defs->kernel.kernelElf ,
                    &_binary___cast_bf16_to_f32_o_start,
                    IsaSize);
        }
        else
        {
            // copy binary out
            memcpy (out_defs->kernel.kernelElf ,
                    &_binary___cast_f32_to_bf16_o_start,
                    IsaSize);
        }
    }
    else
    {
       retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
       return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}


