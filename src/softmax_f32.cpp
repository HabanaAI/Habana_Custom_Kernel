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

#include "softmax_f32.hpp"

extern unsigned char _binary___softmax_fcd_f32_o_start;
extern unsigned char _binary___softmax_fcd_f32_o_end;
extern unsigned char _binary___softmax_non_fcd_f32_o_start;
extern unsigned char _binary___softmax_non_fcd_f32_o_end;

 gcapi::GlueCodeReturn_t SoftMaxF32::GetKernelName1(
             char kernelName [gcapi::MAX_NODE_NAME])
 {
     strcpy(kernelName,"softmax_fcd_f32");
     return gcapi::GLUE_SUCCESS;
 }

 gcapi::GlueCodeReturn_t SoftMaxF32::GetKernelName2(
             char kernelName [gcapi::MAX_NODE_NAME])
 {
     strcpy(kernelName,"softmax_non_fcd_f32");
     return gcapi::GLUE_SUCCESS;
 }

gcapi::GlueCodeReturn_t SoftMaxF32::GetGcDefinitions(
            gcapi::HabanaKernelParams_t* in_defs,
            gcapi::HabanaKernelInstantiation_t* out_defs)
{
    gcapi::GlueCodeReturn_t retVal;
    SoftMaxParam* def = static_cast<SoftMaxParam*>(in_defs->NodeParams);

    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (in_defs->inputTensorNr != 1)
    {
        in_defs->inputTensorNr  = 1;
        return gcapi::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    //validate correct amount of output tensors
    if (in_defs->outputTensorNr != 1)
    {
        in_defs->outputTensorNr  = 1;
        return gcapi::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    // Validate number of dimensions for input and output tensors
    if (in_defs->inputTensors[0].geometry.dims != 2 ||
        in_defs->outputTensors[0].geometry.dims != 2)
    {
        return gcapi::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    // validate input data type
    if (in_defs->inputTensors[0].dataType != gcapi::DATA_F32 ||
        in_defs->outputTensors[0].dataType != gcapi::DATA_F32)
    {
        in_defs->inputTensors[0].dataType = gcapi::DATA_F32;
        in_defs->outputTensors[0].dataType = gcapi::DATA_F32;
        return gcapi::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    // Tensor 0 should be input feature map.
    // The semantics of the input tensors and their order is a convention
    // between TPC kernel writer and the write of the layer at the
    // framework level.
    unsigned int outputSizes[gcapi::MAX_TENSOR_DIM] = {0};

    std::copy(in_defs->inputTensors[0].geometry.sizes,
              in_defs->inputTensors[0].geometry.sizes + gcapi::MAX_TENSOR_DIM,
            outputSizes);

    unsigned * inputSizes = in_defs->inputTensors[0].geometry.sizes;

    // verify that output feature map dimension are correct
    if (memcmp(in_defs->outputTensors[0].geometry.sizes,inputSizes,
               in_defs->outputTensors[0].geometry.dims * sizeof(unsigned) ) != 0)
    {
        memcpy(in_defs->outputTensors[0].geometry.sizes,inputSizes,sizeof(outputSizes));
        return gcapi::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    out_defs->indexSpaceGeometry.dims = 4;
    //round up to 64 and divide by 64.
    unsigned depthIndex = (outputSizes[0] + 63) / 64 * 64;
    out_defs->indexSpaceGeometry.sizes[0] = outputSizes[0];
    out_defs->indexSpaceGeometry.sizes[1] = outputSizes[1];
    out_defs->indexSpaceGeometry.sizes[2] = 1;
    out_defs->indexSpaceGeometry.sizes[3] = 1;

    // Single index space along axis of softmax calculation
    // Single index space is used when there is data dependency among index spaces
    out_defs->indexSpaceGeometry.sizes[def->axis] = 1;

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
        out_defs->inputTensorAccessPattern[0].dim[0].dim        = 0;
        out_defs->inputTensorAccessPattern[0].dim[0].start_a    = 0;
        out_defs->inputTensorAccessPattern[0].dim[0].end_a      = 0;
        out_defs->inputTensorAccessPattern[0].dim[0].start_b    = 0;
        out_defs->inputTensorAccessPattern[0].dim[0].end_b      = depthIndex - 1;

        // f_start(i) = 1*i + 0;
        // f_end f(i) = 1*i + 0;
        // Resource 0 (IFM) dim 1 (width)
        out_defs->inputTensorAccessPattern[0].dim[1].dim        = 1;
        out_defs->inputTensorAccessPattern[0].dim[1].start_a    = 1;
        out_defs->inputTensorAccessPattern[0].dim[1].end_a      = 1;
        out_defs->inputTensorAccessPattern[0].dim[1].start_b    = 0;
        out_defs->inputTensorAccessPattern[0].dim[1].end_b      = 1 - 1;

        // f_start(i) = 0;
        // f_end f(i) = size[0] - 1;
        // Resource 0 (OFM) dim 0 (depth)
        // Access is given to all the elements since single indexspace is used
        out_defs->outputTensorAccessPattern[0].dim[0].dim        = 0;
        out_defs->outputTensorAccessPattern[0].dim[0].start_a    = 0;
        out_defs->outputTensorAccessPattern[0].dim[0].end_a      = 0;
        out_defs->outputTensorAccessPattern[0].dim[0].start_b    = 0;
        out_defs->outputTensorAccessPattern[0].dim[0].end_b      = depthIndex - 1;

        // f_start(i) = 1*i + 0;
        // f_end f(i) = 1*i + 0;
        // Resource 0 (OFM) dim 1 (width)
        out_defs->outputTensorAccessPattern[0].dim[1].dim        = 1;
        out_defs->outputTensorAccessPattern[0].dim[1].start_a    = 1;
        out_defs->outputTensorAccessPattern[0].dim[1].end_a      = 1;
        out_defs->outputTensorAccessPattern[0].dim[1].start_b    = 0;
        out_defs->outputTensorAccessPattern[0].dim[1].end_b      = 1 - 1;
    }
    else
    {
        // f_start(i) = 64*i + 0;
        // f_end f(i) = 64*i + 63;
        // Resource 0 (IFM) dim 0 (depth)
        out_defs->inputTensorAccessPattern[0].dim[0].dim        = 0;
        out_defs->inputTensorAccessPattern[0].dim[0].start_a    = 64;
        out_defs->inputTensorAccessPattern[0].dim[0].end_a      = 64;
        out_defs->inputTensorAccessPattern[0].dim[0].start_b    = 0;
        out_defs->inputTensorAccessPattern[0].dim[0].end_b      = 63;

        // f_start(i) = 0;
        // f_end f(i) = size[1] - 1;
        // Resource 0 (IFM) dim 1 (width)
        // Access is given to all the elements since single indexspace is used
        out_defs->inputTensorAccessPattern[0].dim[1].dim        = 1;
        out_defs->inputTensorAccessPattern[0].dim[1].start_a    = 0;
        out_defs->inputTensorAccessPattern[0].dim[1].end_a      = 0;
        out_defs->inputTensorAccessPattern[0].dim[1].start_b    = 0;
        out_defs->inputTensorAccessPattern[0].dim[1].end_b      = outputSizes[1] - 1;

        // f_start(i) = 64*i + 0;
        // f_end f(i) = 64*i + 63;
        // Resource 0 (OFM) dim 0 (depth)
        out_defs->outputTensorAccessPattern[0].dim[0].dim        = 0;
        out_defs->outputTensorAccessPattern[0].dim[0].start_a    = 64;
        out_defs->outputTensorAccessPattern[0].dim[0].end_a      = 64;
        out_defs->outputTensorAccessPattern[0].dim[0].start_b    = 0;
        out_defs->outputTensorAccessPattern[0].dim[0].end_b      = 63;

        // f_start(i) = 0;
        // f_end f(i) = size[1] - 1;
        // Resource 0 (OFM) dim 1 (width)
        out_defs->outputTensorAccessPattern[0].dim[1].dim        = 1;
        out_defs->outputTensorAccessPattern[0].dim[1].start_a    = 0;
        out_defs->outputTensorAccessPattern[0].dim[1].end_a      = 0;
        out_defs->outputTensorAccessPattern[0].dim[1].start_b    = 0;
        out_defs->outputTensorAccessPattern[0].dim[1].end_b      = outputSizes[1] - 1;
    }

    /*************************************************************************************
    *    Stage IV -  Set Auxiliary Tensor
    **************************************************************************************/

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize1 = (&_binary___softmax_fcd_f32_o_end - &_binary___softmax_fcd_f32_o_start);
    unsigned IsaSize2 = (&_binary___softmax_non_fcd_f32_o_end - &_binary___softmax_non_fcd_f32_o_start);
    unsigned givenBinarySize = out_defs->elfSize;
    unsigned IsaSize = def->axis==0? IsaSize1:IsaSize2;

    out_defs->elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        if(def->axis == 0)
        {
            // copy binary out
            memcpy (out_defs->kernelElf ,
                    &_binary___softmax_fcd_f32_o_start,
                    IsaSize);
        }
        else
        {
            // copy binary out
            memcpy (out_defs->kernelElf ,
                    &_binary___softmax_non_fcd_f32_o_start,
                    IsaSize);
        }
    }
    else
    {
       retVal = gcapi::GLUE_INSUFICIENT_ELF_BUFFER;
       return retVal;
    }

    return gcapi::GLUE_SUCCESS;
}


