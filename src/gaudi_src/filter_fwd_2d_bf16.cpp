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
#include "filter_fwd_2d_bf16.hpp"


extern unsigned char _binary___filter_fwd_2d_bf16_o_start;
extern unsigned char _binary___filter_fwd_2d_bf16_o_end;

 tpc_lib_api::GlueCodeReturn FilterFwd2dBF16::GetKernelName(
             char kernelName [tpc_lib_api::MAX_NODE_NAME])
 {
     strcpy(kernelName,"custom_filter_fwd_2d_bf16");
     return tpc_lib_api::GLUE_SUCCESS;
 }


tpc_lib_api::GlueCodeReturn FilterFwd2dBF16::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* in_defs,
            tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    tpc_lib_api::GlueCodeReturn retVal;
    SpatialReduction2DDef* def = static_cast<SpatialReduction2DDef*>(in_defs->nodeParams.nodeParams);
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
    //check that filter depth match IFM
    if (in_defs->inputTensors[1].geometry.maxSizes[0] !=
        in_defs->inputTensors[0].geometry.maxSizes[0])
    {
        in_defs->inputTensors[1].geometry.maxSizes[0] =
                in_defs->inputTensors[0].geometry.maxSizes[0];
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    retVal = ValidateTensorsDataType(in_defs->inputTensors,
                                        in_defs->inputTensorNr,
                                        tpc_lib_api::DATA_BF16);
    if (retVal != tpc_lib_api::GLUE_SUCCESS)
    {
        return retVal;
    }

    retVal = ValidateTensorsDataType(in_defs->outputTensors,
                                        in_defs->outputTensorNr,
                                        tpc_lib_api::DATA_BF16);
    if (retVal != tpc_lib_api::GLUE_SUCCESS)
    {
        return retVal;
    }

    // Tensor 0 should be input feature map.
    // The semantics of the input tensors and their order is a convention
    // between TPC kernel writer and the write of the layer at the
    // framework level.
    uint64_t outputSizes[gcapi::MAX_TENSOR_DIM];

    if (!GetOfmSize(in_defs->inputTensors[0].geometry.maxSizes,def,outputSizes))
    {
        return tpc_lib_api::GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    }

    // verify that output feature map dimension are correct
    if (memcmp(in_defs->outputTensors[0].geometry.maxSizes,outputSizes,
               in_defs->outputTensors[0].geometry.dims * sizeof(unsigned) ) != 0)
    {
        memcpy(in_defs->outputTensors[0].geometry.maxSizes,outputSizes,sizeof(outputSizes));
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    out_defs->indexSpaceRank = 4;
    //round up to 128 and divide by 128.
    unsigned depthIndex = (outputSizes[0] + 127) / 128;
    out_defs->indexSpaceGeometry[0] = depthIndex;
    out_defs->indexSpaceGeometry[1] = outputSizes[1];
    out_defs->indexSpaceGeometry[2] = outputSizes[2];
    out_defs->indexSpaceGeometry[3] = outputSizes[3];

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    GetAccessPatterns(out_defs,def,c_bf16ElementsInVector);

    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
    out_defs->kernel.paramsNr = sizeof(*def)/ sizeof(int);
    memcpy(&( out_defs->kernel.scalarParams[0]),def, sizeof(*def));

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___filter_fwd_2d_bf16_o_end - &_binary___filter_fwd_2d_bf16_o_start);
    unsigned givenBinarySize = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (out_defs->kernel.kernelElf,
                &_binary___filter_fwd_2d_bf16_o_start,
                IsaSize);
    }
    else
    {
       retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
       return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

