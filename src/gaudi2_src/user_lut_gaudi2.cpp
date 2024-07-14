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

#include <cmath>
#include <cstring>
#include "user_lut_gaudi2.hpp"

extern unsigned char _binary___user_lut_f32_gaudi2_o_start;
extern unsigned char _binary___user_lut_f32_gaudi2_o_end;


const float LutTable[] 
{
// function 0 - 32b
 0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5,
 NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,

// function 1 - 32bx3
 0.1,  0.2,  0.3,  1.1,  1.2,  1.3,  2.1,  2.2,  2.3,  3.1,  3.2,  3.3,  4.1,  4.2,  4.3,  5.1,  5.2,  5.3,  6.1,  6.2,  6.3,  7.1,  7.2,  7.3,  8.1,  8.2,  8.3,  9.1,  9.2,  9.3,  NAN,  NAN,
10.1, 10.2, 10.3, 11.1, 11.2, 11.3, 12.1, 12.2, 12.3, 13.1, 13.2, 13.3, 14.1, 14.2, 14.3, 15.1, 15.2, 15.3, 16.1, 16.2, 16.3, 17.1, 17.2, 17.3, 18.1, 18.2, 18.3, 19.1, 19.2, 19.3,  NAN,  NAN,
20.1, 20.2, 20.3, 21.1, 21.2, 21.3, 22.1, 22.2, 22.3, 23.1, 23.2, 23.3, 24.1, 24.2, 24.3, 25.1, 25.2, 25.3, 26.1, 26.2, 26.3, 27.1, 27.2, 27.3, 28.1, 28.2, 28.3, 29.1, 29.2, 29.3,  NAN,  NAN,
30.1, 30.2, 30.3, 31.1, 31.2, 31.3,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,  NAN,

};

tpc_lib_api::GlueCodeReturn UserLutGaudi2::GetKernelName(
        char kernelName [tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName,"user_lut_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn UserLutGaudi2::GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* in_defs,
        tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    tpc_lib_api::GlueCodeReturn retVal;

    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    if (in_defs->inputTensorNr != 2)
    {
        in_defs->inputTensorNr  = 2;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }

    if (in_defs->outputTensorNr != 1)
    {
        in_defs->outputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    if (in_defs->inputTensors[0].geometry.maxSizes[0] != 64 || 
        in_defs->inputTensors[1].geometry.maxSizes[0] != 64)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    if (in_defs->outputTensors[0].geometry.maxSizes[0] != 64)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    out_defs->indexSpaceRank = 1;
    out_defs->indexSpaceGeometry[0] = 1;

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    int elementsInVec = 64;

    // Index space mapping is calculated using f(i) = Ai + B
    // 'i' is the index space member and A/B constants to be defined.
    out_defs->inputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].a        = elementsInVec;
    out_defs->inputTensorAccessPattern[0].mapping[0].start_b  = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].end_b    = elementsInVec - 1;

    // f_start f(i) = elementsInVec*i + 0;
    // f_end   f(i) = elementsInVec*i + (elementsInVec - 1);
    // Resource 0 (OFM) dim 0
    out_defs->outputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].a        = elementsInVec;
    out_defs->outputTensorAccessPattern[0].mapping[0].start_b  = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].end_b    = elementsInVec - 1;
	
    /*************************************************************************************
    *    Stage IV -  define scalar parameters/Set Auxiliary Tensor
    **************************************************************************************/
    unsigned tableSize = sizeof(LutTable)/sizeof(LutTable[0]);

    out_defs->auxiliaryTensorNr = 1;
    out_defs->auxiliaryTensors[0].geometry.dims = 1;
    out_defs->auxiliaryTensors[0].geometry.maxSizes[0] = tableSize;
    out_defs->auxiliaryTensors[0].geometry.maxSizes[1] = 0;
    out_defs->auxiliaryTensors[0].geometry.maxSizes[2] = 0;
    out_defs->auxiliaryTensors[0].geometry.maxSizes[3] = 0;
    out_defs->auxiliaryTensors[0].geometry.maxSizes[4] = 0;

    out_defs->auxiliaryTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
    
    unsigned required_size = out_defs->auxiliaryTensors[0].geometry.maxSizes[0] * sizeof(float);
    // Check whether required memory is allocated for auxiliary tensor
    if (required_size > out_defs->auxiliaryTensors[0].bufferSize)
    {
        out_defs->auxiliaryTensors[0].bufferSize = required_size;
        return tpc_lib_api::GLUE_INSUFFICIENT_AUX_BUFFER_SIZE;
    }

    // fill aux 0 with data
    memcpy(out_defs->auxiliaryTensors[0].pData, LutTable, required_size);

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___user_lut_f32_gaudi2_o_end - &_binary___user_lut_f32_gaudi2_o_start);
    unsigned givenBinarySize = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (out_defs->kernel.kernelElf,
                &_binary___user_lut_f32_gaudi2_o_start,
                IsaSize);
    }
    else
    {
       retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
       return retVal;
    }
    return tpc_lib_api::GLUE_SUCCESS;
}
