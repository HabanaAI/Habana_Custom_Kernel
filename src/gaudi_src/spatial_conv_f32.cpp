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

#include <vector>
#include <cstring>
#include <iostream>
#include "spatial_conv_f32.hpp"


extern unsigned char _binary___spatial_conv_f32_o_start;
extern unsigned char _binary___spatial_conv_f32_o_end;

 tpc_lib_api::GlueCodeReturn SpatialConvF32::GetKernelName(
             char kernelName [tpc_lib_api::MAX_NODE_NAME])
 {
     strcpy(kernelName,"custom_spatial_conv_f32");
     return tpc_lib_api::GLUE_SUCCESS;
 }

tpc_lib_api::GlueCodeReturn SpatialConvF32::GetGcDefinitions(
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
                                        tpc_lib_api::DATA_F32);
    if (retVal != tpc_lib_api::GLUE_SUCCESS)
    {
        return retVal;
    }

    retVal = ValidateTensorsDataType(in_defs->outputTensors,
                                        in_defs->outputTensorNr,
                                        tpc_lib_api::DATA_F32);
    if (retVal != tpc_lib_api::GLUE_SUCCESS)
    {
        return retVal;
    }

    // Tensor 0 should be input feature map.
    // The semantics of the input tensors and their order is a convention
    // between TPC kernel writer and the write of the layer at the
    // framework level.

    uint64_t outputSizes[gcapi::MAX_TENSOR_DIM];

    if (!GetSpatialConvOfmSize(in_defs->inputTensors[0].geometry.maxSizes,
                                 in_defs->inputTensors[1].geometry.maxSizes,
                                 def,
                                 outputSizes))
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
    *************************************************************************************/
    out_defs->indexSpaceRank = 5;
    out_defs->indexSpaceGeometry[0] = 1; //all channels are summed together - can't split
    out_defs->indexSpaceGeometry[1] = outputSizes[0]; //num of filters
    out_defs->indexSpaceGeometry[2] = outputSizes[1]; //width
    out_defs->indexSpaceGeometry[3] = outputSizes[2]; //height
    out_defs->indexSpaceGeometry[4] = outputSizes[3]; //batch

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    GetSpatialConvAccessPatterns(out_defs, def, in_defs->inputTensors[0].geometry.maxSizes[0]);

    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
    out_defs->kernel.paramsNr = sizeof(*def)/ sizeof(int);
    memcpy(&( out_defs->kernel.scalarParams[0]),def, sizeof(*def));

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___spatial_conv_f32_o_end - &_binary___spatial_conv_f32_o_start);
    unsigned givenBinarySize = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (out_defs->kernel.kernelElf,
                &_binary___spatial_conv_f32_o_start,
                IsaSize);
    }
    else
    {
       retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
       return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

 /*************************************************************************************
 *    GetSpatialConvOfmSize - similar to SpatialReductionKernels.GetOfmSize
 *    only suitable for spatial conv
 **************************************************************************************/
bool SpatialConvF32::GetSpatialConvOfmSize(
                                   uint64_t IfmSize [gcapi::MAX_TENSOR_DIM],
                                   uint64_t FilterSize [gcapi::MAX_TENSOR_DIM],
                                   const SpatialReduction2DDef* def,
                                   uint64_t OfmSize [gcapi::MAX_TENSOR_DIM])
{
    if ((2 * def->pad_w + ((int)IfmSize[1])) <  def->dilation_w * (def->kernel_w-1) + 1)
    { // IFM is smaller than window size in width
        return false;
    }
    if ((2 * def->pad_h + ((int)IfmSize[2])) <  def->dilation_h * (def->kernel_h-1) + 1)
    { // IFM is smaller than window size in height
        return false;
    }

    OfmSize[0] = FilterSize[1];
    OfmSize[1] = ((IfmSize[1] + 2 * def->pad_w - def->dilation_w * (def->kernel_w-1) - 1) / def->stride_w) + 1;
    OfmSize[2] = ((IfmSize[2] + 2 * def->pad_h - def->dilation_h * (def->kernel_h-1) - 1) / def->stride_h) + 1;
    OfmSize[3] = IfmSize[3];
    OfmSize[4] = 1;
    return true;
}

 /*************************************************************************************
 *    GetSpatialConvAccessPatterns - similar to SpatialReductionKernels.GetAccessPatterns
 *    only suitable for spatial conv
 **************************************************************************************/
void SpatialConvF32::GetSpatialConvAccessPatterns(
                     tpc_lib_api::HabanaKernelInstantiation* out_defs,
                     const SpatialReduction2DDef* def,
                     unsigned int channelSize)
{
    // now define how the index space maps to IFM using f(i) = Ai+B .
    // transformation. 'i' is the index space member and A/B constants to be defined.

    // start f(i) = 0*i + 0;
    // end f(i) = 0*i + (channelSize - 1);
    // Resource 0 (IFM) dim 0 (depth).
    out_defs->inputTensorAccessPattern[0].mapping[0].indexSpaceDim = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].a       = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].start_b = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].end_b   = channelSize - 1;

    // start f(i) = stride_w*i + (-pad_w);
    // end f(i) = stride_w*i + ((kernel_w-1)*dilation_w - padw);
    // Resource 0 (IFM) dim 1 (width).
    out_defs->inputTensorAccessPattern[0].mapping[1].indexSpaceDim = 2;
    out_defs->inputTensorAccessPattern[0].mapping[1].a       = def->stride_w;
    out_defs->inputTensorAccessPattern[0].mapping[1].start_b = -def->pad_w;
    out_defs->inputTensorAccessPattern[0].mapping[1].end_b   = -def->pad_w + (def->kernel_w - 1) * def->dilation_w;

    // start f(i) = stride_h*i + (-pad_h);
    // end f(i) = stride_h*i + ((kernel_h-1)*dilation_h - pad_h);
    // Resource 0 (IFM) dim 2 (height).
    out_defs->inputTensorAccessPattern[0].mapping[2].indexSpaceDim = 3;
    out_defs->inputTensorAccessPattern[0].mapping[2].a    = def->stride_h;
    out_defs->inputTensorAccessPattern[0].mapping[2].start_b =  -def->pad_h;
    out_defs->inputTensorAccessPattern[0].mapping[2].end_b   = -def->pad_h + (def->kernel_h - 1) * def->dilation_h;

    // start f(i) = 1*i + 0;
    // end f(i) = 1*i + 0;
    // Resource 0 (IFM) dim 3 (batch).
    out_defs->inputTensorAccessPattern[0].mapping[3].indexSpaceDim = 4;
    out_defs->inputTensorAccessPattern[0].mapping[3].a    = 1;
    out_defs->inputTensorAccessPattern[0].mapping[3].start_b =  0;
    out_defs->inputTensorAccessPattern[0].mapping[3].end_b   = 0;

    ////////////////////////////////////////////////////////////////////////////
    // define how the index space maps to the filter tensor

    // start f(i) = 0*i + 0;
    // end f(i) = 0*i + (channelSize - 1);
    // Resource 1 (FILTER) dim 0 (depth).
    out_defs->inputTensorAccessPattern[1].mapping[0].indexSpaceDim = 0;
    out_defs->inputTensorAccessPattern[1].mapping[0].a       = 0;
    out_defs->inputTensorAccessPattern[1].mapping[0].start_b = 0;
    out_defs->inputTensorAccessPattern[1].mapping[0].end_b   = channelSize - 1;

    // start f(i) = 1*i + 0;
    // end f(i) = 1*i + 0;
    // Resource 0 (FILTER) dim 1 (K - num of filters).
    out_defs->inputTensorAccessPattern[1].mapping[1].indexSpaceDim = 1;
    out_defs->inputTensorAccessPattern[1].mapping[1].a       = 1;
    out_defs->inputTensorAccessPattern[1].mapping[1].start_b =  0;
    out_defs->inputTensorAccessPattern[1].mapping[1].end_b   = 0;

    // start f(i) = 0*i + 0;
    // end f(i) = 0*i + kernel_w;
    // Resource 0 (FILTER) dim 2 (width).
    out_defs->inputTensorAccessPattern[1].mapping[2].indexSpaceDim = 2;
    out_defs->inputTensorAccessPattern[1].mapping[2].a       = 0;
    out_defs->inputTensorAccessPattern[1].mapping[2].start_b =  0;
    out_defs->inputTensorAccessPattern[1].mapping[2].end_b   = def->kernel_w-1;

    // start f(i) = 0*i + 0;
    // end f(i) = 0*i + kernel_h;
    // Resource 0 (FILTER) dim 3 (height).
    out_defs->inputTensorAccessPattern[1].mapping[3].indexSpaceDim = 3;
    out_defs->inputTensorAccessPattern[1].mapping[3].a       = 0;
    out_defs->inputTensorAccessPattern[1].mapping[3].start_b =  0;
    out_defs->inputTensorAccessPattern[1].mapping[3].end_b   = def->kernel_h-1;

    ////////////////////////////////////////////////////////////////////////////
    // define how the index space maps to the output tensor

    // start f(i) = 1*i + 0;
    // end f(i) = 1*i + 0;
    // Resource 0 (OFM) dim 0 (K - num of filters).
    for (unsigned int dims = 0; dims < 4; dims++)
    {
        out_defs->outputTensorAccessPattern[0].mapping[dims].indexSpaceDim = dims+1;
        out_defs->outputTensorAccessPattern[0].mapping[dims].a       = 1;
        out_defs->outputTensorAccessPattern[0].mapping[dims].start_b =  0;
        out_defs->outputTensorAccessPattern[0].mapping[dims].end_b   = 0;
    }

}