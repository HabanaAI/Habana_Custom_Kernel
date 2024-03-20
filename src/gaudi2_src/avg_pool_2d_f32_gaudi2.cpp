/**********************************************************************
Copyright (c) 2022 Habana Labs.

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
#include "avg_pool_2d_f32_gaudi2.hpp"

extern unsigned char _binary___avg_pool_2d_fwd_f32_gaudi2_o_start;
extern unsigned char _binary___avg_pool_2d_fwd_f32_gaudi2_o_end;
extern unsigned char _binary___avg_pool_2d_bwd_f32_gaudi2_o_start;
extern unsigned char _binary___avg_pool_2d_bwd_f32_gaudi2_o_end;

 tpc_lib_api::GlueCodeReturn AvgPool2dF32Gaudi2::GetKernelName(
             char kernelName [tpc_lib_api::MAX_NODE_NAME])
 {

    if(m_mode == fwd)
        strcpy(kernelName,"custom_avg_pool_2d_fwd_f32_gaudi2");
    else if(m_mode == bwd)
        strcpy(kernelName,"custom_avg_pool_2d_bwd_f32_gaudi2");
    else
        return tpc_lib_api::GLUE_NODE_NOT_FOUND;
     return tpc_lib_api::GLUE_SUCCESS;
 }


tpc_lib_api::GlueCodeReturn AvgPool2dF32Gaudi2::fill_reciprocal_table(float* reciprocal_table, int num_elements) const
{
    reciprocal_table[0] = 0;
    for (int i = 1; i < num_elements; i++)
    {
        reciprocal_table[i] = (float)((double)1.0 / (double)i);
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn AvgPool2dF32Gaudi2::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* in_defs,
            tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    
    tpc_lib_api::GlueCodeReturn retVal;
    AvgPool2DParam* def = static_cast<AvgPool2DParam*>(in_defs->nodeParams.nodeParams);
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

    
    retVal = ValidateTensorsDataType(in_defs->inputTensors,
                                        1,
                                        tpc_lib_api::DATA_F32);
    if (retVal != tpc_lib_api::GLUE_SUCCESS)
    {
        return retVal;
    }

    retVal = ValidateTensorsDataType(&(in_defs->inputTensors[1]),
                                        1,
                                        tpc_lib_api::DATA_I32);
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
    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    uint64_t outputSizes[gcapi::MAX_TENSOR_DIM];

    out_defs->indexSpaceRank = 4;
    uint64_t* indexSpaceSizes         = in_defs->inputTensors[0].geometry.maxSizes;
    outputSizes[0] = indexSpaceSizes[0];
    if(m_mode == fwd)
    {
        outputSizes[1] = (indexSpaceSizes[1] + def->srdef.pad_w - def->srdef.kernel_w * def->srdef.dilation_w) / def->srdef.stride_w;
        outputSizes[2] = (indexSpaceSizes[2] + def->srdef.pad_h - def->srdef.kernel_h * def->srdef.dilation_h) / def->srdef.stride_h;
    }
    else
    {
        outputSizes[1] = (indexSpaceSizes[1] * def->srdef.stride_w) - def->srdef.pad_w + (def->srdef.kernel_w * def->srdef.dilation_w);
        outputSizes[2] = (indexSpaceSizes[2] * def->srdef.stride_h) - def->srdef.pad_h + (def->srdef.kernel_h * def->srdef.dilation_h);
    }
    outputSizes[3] = indexSpaceSizes[3];
    // verify that output feature map dimension are correct
    if (memcmp(in_defs->outputTensors[0].geometry.maxSizes,outputSizes,
               in_defs->outputTensors[0].geometry.dims * sizeof(uint64_t) ) != 0)
    {
        memcpy(in_defs->outputTensors[0].geometry.maxSizes,outputSizes,sizeof(outputSizes));
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    //round up to 64 and divide by 64.
    out_defs->indexSpaceGeometry[0] = (outputSizes[0] + 63) /64;
    out_defs->indexSpaceGeometry[1] = outputSizes[1];
    out_defs->indexSpaceGeometry[2] = outputSizes[2];
    out_defs->indexSpaceGeometry[3] = outputSizes[3] ? outputSizes[3] : 1;

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    for (unsigned i = 0; i < in_defs->inputTensorNr; i++)
    {
        for (unsigned j = 0; j < out_defs->indexSpaceRank; j++)
        {
            out_defs->inputTensorAccessPattern[i].mapping[j].indexSpaceDim     = 0;
            out_defs->inputTensorAccessPattern[i].mapping[j].a = 0;
            out_defs->inputTensorAccessPattern[i].mapping[j].start_b = 0;
            out_defs->inputTensorAccessPattern[i].mapping[j].end_b   = 0;
        }
    }
    for (unsigned int i = 0; i < out_defs->indexSpaceRank; i++)
    {
        out_defs->outputTensorAccessPattern[0].mapping[i].indexSpaceDim     = i;
        out_defs->outputTensorAccessPattern[0].mapping[i].a = 0;
        out_defs->outputTensorAccessPattern[0].mapping[i].start_b = 0;
        out_defs->outputTensorAccessPattern[0].mapping[i].end_b   = 0;
    }

    GetAccessPatterns(out_defs,&(def->srdef),64);
    if(m_mode == bwd)
    {
        out_defs->inputTensorAccessPattern[0].mapping[1].indexSpaceDim = 1;
        out_defs->inputTensorAccessPattern[0].mapping[1].a = (1.0/def->srdef.stride_w);
        out_defs->inputTensorAccessPattern[0].mapping[1].start_b = -((def->srdef.kernel_w - 1) + (def->srdef.stride_w - 1)) * def->srdef.dilation_w / def->srdef.stride_w;;
        out_defs->inputTensorAccessPattern[0].mapping[1].end_b = (def->srdef.pad_w / (float)def->srdef.stride_w);

        // start f(i) = stride*i + (-padh);
        // end f(i) = stride*i + (kernelh*dilationh - padh );
        // Resource 0 (IFM) dim 2 (height).
        out_defs->inputTensorAccessPattern[0].mapping[2].indexSpaceDim = 2;
        out_defs->inputTensorAccessPattern[0].mapping[2].a = (1.0/def->srdef.stride_h);
        out_defs->inputTensorAccessPattern[0].mapping[2].start_b =  -((def->srdef.kernel_h - 1) + (def->srdef.stride_h - 1)) * def->srdef.dilation_h / def->srdef.stride_h;
        out_defs->inputTensorAccessPattern[0].mapping[2].end_b = (def->srdef.pad_h / (float)def->srdef.stride_h);

        out_defs->inputTensorAccessPattern[0].memsetBeforeExecution = 1;
        out_defs->outputTensorAccessPattern[0].memsetBeforeExecution = 1;
    }

    /*************************************************************************************
    *    Stage IV -  define scalar parameters/Set Auxiliary Tensor
    **************************************************************************************/
    out_defs->kernel.paramsNr = sizeof(*def)/ sizeof(int);
    memcpy(&( out_defs->kernel.scalarParams[0]), def, sizeof(*def));

    const int maxWindowSize = def->srdef.kernel_h * def->srdef.kernel_w + 1;
    out_defs->auxiliaryTensorNr = 1;
    out_defs->auxiliaryTensors[0].geometry.dims = 1;
    out_defs->auxiliaryTensors[0].geometry.maxSizes[0] = maxWindowSize;
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
    out_defs->auxiliaryTensors[0].bufferSize = required_size;
    if (out_defs->auxiliaryTensors[0].bufferSize >= required_size)
    {
        // fill aux 0 (for 1/x) with data
        float* reciprocalTable = (float*)malloc(required_size);
        fill_reciprocal_table(reciprocalTable, maxWindowSize);
        // Initialize the auxiliary tensor with reduction_fcd_tab
        out_defs->auxiliaryTensors[0].pData = reciprocalTable;
        //memcpy(out_defs->auxiliaryTensors[0].pData, reciprocalTable, required_size);
        //free(reciprocalTable);
    }
    else
    {
        return tpc_lib_api::GLUE_INSUFFICIENT_AUX_BUFFER_SIZE;
    }

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___avg_pool_2d_fwd_f32_gaudi2_o_end - &_binary___avg_pool_2d_fwd_f32_gaudi2_o_start);
    unsigned char *binary_kernel = &_binary___avg_pool_2d_fwd_f32_gaudi2_o_start;
    switch (m_mode){
        case fwd:
            IsaSize = (&_binary___avg_pool_2d_fwd_f32_gaudi2_o_end - &_binary___avg_pool_2d_fwd_f32_gaudi2_o_start);
            binary_kernel = &_binary___avg_pool_2d_fwd_f32_gaudi2_o_start;
            break;
        case bwd:
            IsaSize = (&_binary___avg_pool_2d_bwd_f32_gaudi2_o_end - &_binary___avg_pool_2d_bwd_f32_gaudi2_o_start);
            binary_kernel = &_binary___avg_pool_2d_bwd_f32_gaudi2_o_start;
            break;
        default:
            break;

    }
    unsigned givenBinarySize = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (out_defs->kernel.kernelElf,
                binary_kernel,
                IsaSize);
    }
    else
    {
       retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
       return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

