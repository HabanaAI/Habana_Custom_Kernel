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

#include <cstring>
#include "customdiv_fwd_f32.hpp"


extern unsigned char _binary___customdiv_fwd_f32_o_start;
extern unsigned char _binary___customdiv_fwd_f32_o_end;

tpc_lib_api::GlueCodeReturn CustomdivFwdF32::GetKernelName(
        char kernelName [tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName,"customdiv_fwd_f32");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn CustomdivFwdF32::GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* params,
        tpc_lib_api::HabanaKernelInstantiation* kernel)
{
    tpc_lib_api::GlueCodeReturn retVal;
    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (params->inputTensorNr != 2)
    {
        params->inputTensorNr  = 2;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    //validate correct amount of output tensors
    if (params->outputTensorNr != 1)
    {
        params->outputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    // validate input and output data type
    if (params->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32 ||
        params->inputTensors[1].geometry.dataType != tpc_lib_api::DATA_F32 ||
        params->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32)
    {
        params->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
        params->inputTensors[1].geometry.dataType = tpc_lib_api::DATA_F32;
        params->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    // Tensor 0 should be input feature map.
    // The semantics of the input tensors and their order is a convention
    // between TPC kernel writer and the write of the layer at the
    // framework level.
    unsigned int outputSizes[gcapi::MAX_TENSOR_DIM] = {0};

    memcpy(outputSizes, params->inputTensors[0].geometry.maxSizes, sizeof(outputSizes));

    // verify that output feature map dimension are correct
    if (memcmp(params->outputTensors[0].geometry.maxSizes, outputSizes,
               params->outputTensors[0].geometry.dims * sizeof(unsigned) ) != 0)
    {
        memcpy(params->outputTensors[0].geometry.maxSizes, params->inputTensors[0].geometry.maxSizes, sizeof(outputSizes));
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    int elementsInVec = 64;

    //round up to elementsInVec and divide by elementsInVec.
    unsigned depthIndex = (outputSizes[0] + (elementsInVec - 1)) / elementsInVec;
    kernel->indexSpaceRank = 5;
    kernel->indexSpaceGeometry[0] = depthIndex;
	//reduce index space due to unroll.
    kernel->indexSpaceGeometry[1] = outputSizes[1]; 
    kernel->indexSpaceGeometry[2] = outputSizes[2];
    kernel->indexSpaceGeometry[3] = outputSizes[3];
    kernel->indexSpaceGeometry[4] = outputSizes[4];

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    // f_start f(i) = elementsInVec*i + 0;
    // f_end   f(i) = elementsInVec*i + (elementsInVec - 1);
    // Resource 0 (IFM) dim 0
    for (unsigned i = 0; i < params->inputTensorNr; i++)
    {
        kernel->inputTensorAccessPattern[i].mapping[0].indexSpaceDim      = 0;
        kernel->inputTensorAccessPattern[i].mapping[0].a        = elementsInVec;
        kernel->inputTensorAccessPattern[i].mapping[0].start_b  = 0;
        kernel->inputTensorAccessPattern[i].mapping[0].end_b    = elementsInVec - 1;

        // f_start f(i) = 1*i + 0;
        // f_end   f(i) = 1*i + 0;
        // Resource 0 (IFM) dim 1-4
        for (int dims = 1; dims < (int)kernel->indexSpaceRank; dims++)
        {
            kernel->inputTensorAccessPattern[i].mapping[dims].indexSpaceDim      = dims;
            kernel->inputTensorAccessPattern[i].mapping[dims].a        = 1;
            kernel->inputTensorAccessPattern[i].mapping[dims].start_b  = 0;
            kernel->inputTensorAccessPattern[i].mapping[dims].end_b    = 1 - 1;
        }        
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
    for (int dims = 1; dims < (int)kernel->indexSpaceRank; dims++)
    {
        kernel->outputTensorAccessPattern[0].mapping[dims].indexSpaceDim      = dims;
        kernel->outputTensorAccessPattern[0].mapping[dims].a        = 1;
        kernel->outputTensorAccessPattern[0].mapping[dims].start_b  = 0;
        kernel->outputTensorAccessPattern[0].mapping[dims].end_b    = 1 - 1;
    }


    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
    kernel->kernel.paramsNr = 0;

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___customdiv_fwd_f32_o_end - &_binary___customdiv_fwd_f32_o_start);
    unsigned givenBinarySize = kernel->kernel.elfSize;
    kernel->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        memcpy (kernel->kernel.kernelElf ,
                    &_binary___customdiv_fwd_f32_o_start,
                    IsaSize);
    }
    else
    {
        retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
        return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}


