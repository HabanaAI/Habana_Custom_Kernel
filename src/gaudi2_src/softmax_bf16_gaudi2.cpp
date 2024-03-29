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

#include "softmax_bf16_gaudi2.hpp"

extern unsigned char _binary___softmax_fcd_bf16_gaudi2_o_start;
extern unsigned char _binary___softmax_fcd_bf16_gaudi2_o_end;
extern unsigned char _binary___softmax_non_fcd_bf16_gaudi2_o_start;
extern unsigned char _binary___softmax_non_fcd_bf16_gaudi2_o_end;

 tpc_lib_api::GlueCodeReturn SoftMaxBF16Gaudi2::GetKernelNameFcd(
             char kernelName [tpc_lib_api::MAX_NODE_NAME])
 {
     strcpy(kernelName,"custom_softmax_fcd_bf16_gaudi2");
     return tpc_lib_api::GLUE_SUCCESS;
 }

 tpc_lib_api::GlueCodeReturn SoftMaxBF16Gaudi2::GetKernelNameNonFcd(
             char kernelName [tpc_lib_api::MAX_NODE_NAME])
 {
     strcpy(kernelName,"custom_softmax_non_fcd_bf16_gaudi2");
     return tpc_lib_api::GLUE_SUCCESS;
 }

tpc_lib_api::GlueCodeReturn SoftMaxBF16Gaudi2::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* params,
            tpc_lib_api::HabanaKernelInstantiation* kernel)
{
    tpc_lib_api::GlueCodeReturn retVal;
    SoftMaxParam* def = static_cast<SoftMaxParam*>(params->nodeParams.nodeParams);

    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (params->inputTensorNr != 1)
    {
        params->inputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    //validate correct amount of output tensors
    if (params->outputTensorNr != 1)
    {
        params->outputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    // Validate number of dimensions for input and output tensors
    if (params->inputTensors[0].geometry.dims != 2 ||
        params->outputTensors[0].geometry.dims != 2)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    // validate input data type
    if (params->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_BF16 ||
        params->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_BF16)
    {
        params->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_BF16;
        params->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_BF16;
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    // Tensor 0 should be input feature map.
    // The semantics of the input tensors and their order is a convention
    // between TPC kernel writer and the write of the layer at the
    // framework level.
    uint64_t outputSizes[gcapi::MAX_TENSOR_DIM] = {0};

    std::copy(params->inputTensors[0].geometry.maxSizes,
              params->inputTensors[0].geometry.maxSizes + gcapi::MAX_TENSOR_DIM,
            outputSizes);

    uint64_t * inputSizes = params->inputTensors[0].geometry.maxSizes;

    // verify that output feature map dimension are correct
    if (memcmp(params->outputTensors[0].geometry.maxSizes,inputSizes,
               params->outputTensors[0].geometry.dims * sizeof(uint64_t) ) != 0)
    {
        memcpy(params->outputTensors[0].geometry.maxSizes,inputSizes,sizeof(outputSizes));
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    kernel->indexSpaceRank = 4;
    //round up to 128 and divide by 128.
    unsigned depthIndex = (outputSizes[0] + 127) / 128 * 128;
    kernel->indexSpaceGeometry[0] = outputSizes[0];
    kernel->indexSpaceGeometry[1] = outputSizes[1];
    kernel->indexSpaceGeometry[2] = 1;
    kernel->indexSpaceGeometry[3] = 1;

    // Single index space along axis of softmax calculation
    // Single index space is used when there is data dependency among index spaces
    kernel->indexSpaceGeometry[def->axis] = 1;

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/

    // Index space mapping is calculated using f(i) = Ai + B
    // 'i' is the index space member and A/B constants to be defined.
    if (def->axis == 0)
    {

        // f_start(i) = 0;
        // f_end f(i) = size[0] - 1;
        // Resource 0 (IFM) dim 0 (depth)
        // Access is given to all the elements since single indexspace is used
        kernel->inputTensorAccessPattern[0].mapping[0].indexSpaceDim        = 0;
        kernel->inputTensorAccessPattern[0].mapping[0].a          = 0;
        kernel->inputTensorAccessPattern[0].mapping[0].start_b    = 0;
        kernel->inputTensorAccessPattern[0].mapping[0].end_b      = depthIndex - 1;

        // f_start(i) = 1*i + 0;
        // f_end f(i) = 1*i + 0;
        // Resource 0 (IFM) dim 1 (width)
        kernel->inputTensorAccessPattern[0].mapping[1].indexSpaceDim        = 1;
        kernel->inputTensorAccessPattern[0].mapping[1].a          = 1;
        kernel->inputTensorAccessPattern[0].mapping[1].start_b    = 0;
        kernel->inputTensorAccessPattern[0].mapping[1].end_b      = 1 - 1;

        // f_start(i) = 0;
        // f_end f(i) = size[0] - 1;
        // Resource 0 (OFM) dim 0 (depth)
        // Access is given to all the elements since single indexspace is used
        kernel->outputTensorAccessPattern[0].mapping[0].indexSpaceDim        = 0;
        kernel->outputTensorAccessPattern[0].mapping[0].a          = 0;
        kernel->outputTensorAccessPattern[0].mapping[0].start_b    = 0;
        kernel->outputTensorAccessPattern[0].mapping[0].end_b      = depthIndex - 1;

        // f_start(i) = 1*i + 0;
        // f_end f(i) = 1*i + 0;
        // Resource 0 (OFM) dim 1 (width)
        kernel->outputTensorAccessPattern[0].mapping[1].indexSpaceDim        = 1;
        kernel->outputTensorAccessPattern[0].mapping[1].a          = 1;
        kernel->outputTensorAccessPattern[0].mapping[1].start_b    = 0;
        kernel->outputTensorAccessPattern[0].mapping[1].end_b      = 1 - 1;
    }
    else
    {
        // f_start(i) = 128*i + 0;
        // f_end f(i) = 128*i + 127;
        // Resource 0 (IFM) dim 0 (depth)
        kernel->inputTensorAccessPattern[0].mapping[0].indexSpaceDim        = 0;
        kernel->inputTensorAccessPattern[0].mapping[0].a         = 128;
        kernel->inputTensorAccessPattern[0].mapping[0].start_b    = 0;
        kernel->inputTensorAccessPattern[0].mapping[0].end_b      = 127;

        // f_start(i) = 0;
        // f_end f(i) = size[1] - 1;
        // Resource 0 (IFM) dim 1 (width)
        // Access is given to all the elements since single indexspace is used
        kernel->inputTensorAccessPattern[0].mapping[1].indexSpaceDim        = 1;
        kernel->inputTensorAccessPattern[0].mapping[1].a          = 0;
        kernel->inputTensorAccessPattern[0].mapping[1].start_b    = 0;
        kernel->inputTensorAccessPattern[0].mapping[1].end_b      = outputSizes[1] - 1;

        // f_start(i) = 128*i + 0;
        // f_end f(i) = 128*i + 127;
        // Resource 0 (OFM) dim 0 (depth)
        kernel->outputTensorAccessPattern[0].mapping[0].indexSpaceDim        = 0;
        kernel->outputTensorAccessPattern[0].mapping[0].a          = 128;
        kernel->outputTensorAccessPattern[0].mapping[0].start_b    = 0;
        kernel->outputTensorAccessPattern[0].mapping[0].end_b      = 127;

        // f_start(i) = 0;
        // f_end f(i) = size[1] - 1;
        // Resource 0 (OFM) dim 1 (width)
        kernel->outputTensorAccessPattern[0].mapping[1].indexSpaceDim        = 1;
        kernel->outputTensorAccessPattern[0].mapping[1].a          = 0;
        kernel->outputTensorAccessPattern[0].mapping[1].start_b    = 0;
        kernel->outputTensorAccessPattern[0].mapping[1].end_b      = outputSizes[1] - 1;
    }

    /*************************************************************************************
    *    Stage IV -  Set Auxiliary Tensor
    **************************************************************************************/
 
    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize1 = (&_binary___softmax_fcd_bf16_gaudi2_o_end - &_binary___softmax_fcd_bf16_gaudi2_o_start);
    unsigned IsaSize2 = (&_binary___softmax_non_fcd_bf16_gaudi2_o_end - &_binary___softmax_non_fcd_bf16_gaudi2_o_start);
    unsigned givenBinarySize = kernel->kernel.elfSize;
    unsigned IsaSize = def->axis==0? IsaSize1:IsaSize2;

    kernel->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        if(def->axis == 0)
        {
            // copy binary out
            memcpy (kernel->kernel.kernelElf ,
                    &_binary___softmax_fcd_bf16_gaudi2_o_start,
                    IsaSize);
        }
        else
        {
            // copy binary out
            memcpy (kernel->kernel.kernelElf ,
                    &_binary___softmax_non_fcd_bf16_gaudi2_o_start,
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


