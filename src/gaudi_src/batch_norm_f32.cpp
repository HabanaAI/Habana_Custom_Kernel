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

#include "batch_norm_f32.hpp"

extern unsigned char _binary___batch_norm_fwd_f32_o_start;
extern unsigned char _binary___batch_norm_fwd_f32_o_end;

tpc_lib_api::GlueCodeReturn BatchNormF32::GetKernelName(
             char kernelName [tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, "custom_batch_norm_fwd_f32");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn  BatchNormF32::ValidateTensorsDataType(
           tpc_lib_api::Tensor* pTensors,
           int tensorCount,
           tpc_lib_api::TensorDataType expected)
{
    tpc_lib_api::GlueCodeReturn retVal = tpc_lib_api::GLUE_SUCCESS;
    for (int i = 0 ; i < tensorCount ; i++)
    {
        if (pTensors[i].geometry.dataType != expected)
        {
            retVal = tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
            pTensors[i].geometry.dataType = expected;
        }
    }
    return retVal;
}

tpc_lib_api::GlueCodeReturn BatchNormF32::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* in_defs,
            tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    tpc_lib_api::GlueCodeReturn retVal;
    BatchNormParams* def = static_cast<BatchNormParams*>(in_defs->nodeParams.nodeParams);
    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (in_defs->inputTensorNr != 3)
    {
        in_defs->inputTensorNr  = 3;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    //validate correct amount of output tensors
    if (in_defs->outputTensorNr != 3)
    {
        in_defs->outputTensorNr  = 3;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    // validate input and output data type
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

   // Validate input and output tensor sizes are same
   uint64_t * inputTensorSizes = in_defs->inputTensors[0].geometry.maxSizes;
   bool SizesAreEqual = true;

   for(unsigned int dim = 0; dim < 4; dim++)
   {
      SizesAreEqual &= (in_defs->outputTensors[0].geometry.maxSizes[dim]
              == inputTensorSizes[dim]);
   }

   if(!SizesAreEqual)
   {
      return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
   }

   // Validate dim0 of all input tensors are equal
   for(unsigned int tns = 1; tns < in_defs->inputTensorNr; tns++)
   {
      SizesAreEqual &= (in_defs->inputTensors[tns].geometry.maxSizes[0] == inputTensorSizes[0]);
   }

   if(!SizesAreEqual)
   {
      return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
   }

    // Beta and Gamma tensors are expected to be 1D
    if ((in_defs->inputTensors[1].geometry.dims != 1 || in_defs->inputTensors[2].geometry.dims != 1))
    {
           return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    if ((in_defs->outputTensors[1].geometry.dims != 1 ||
        in_defs->outputTensors[2].geometry.dims != 1))

    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }


    /*************************************************************************************
    *    Stage II -  Define index space geometry.
    **************************************************************************************/
    out_defs->indexSpaceRank = 4;

    int elementsInVec =  64;
    unsigned depthIndex = (inputTensorSizes[0] + (elementsInVec - 1)) / elementsInVec;
    out_defs->indexSpaceGeometry[0] = depthIndex;
    out_defs->indexSpaceGeometry[1] = 1;
    out_defs->indexSpaceGeometry[2] = 1;
    out_defs->indexSpaceGeometry[3] = 1;

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    // f_start(i) = elementsInVec*i + 0;
    // f_end f(i) = elementsInVec*i + (elementsInVec - 1);
    // Resource 0-4 (IFM) dim 0
    for (unsigned i = 0; i < in_defs->inputTensorNr; i++)
    {
        out_defs->inputTensorAccessPattern[i].mapping[0].indexSpaceDim      = 0;
        out_defs->inputTensorAccessPattern[i].mapping[0].a        = elementsInVec;
        out_defs->inputTensorAccessPattern[i].mapping[0].start_b  = 0;
        out_defs->inputTensorAccessPattern[i].mapping[0].end_b    = elementsInVec - 1;
    }

    float unroll = 4.0; // float is used to promote float divison
    // f_start(i) = 0;
    // f_end f(i) = size[1] - 1;
    // Resource 0 (IFM) dim 1
    // Access is given to all the elements(factor of unroll) since single indexspace is used
    out_defs->inputTensorAccessPattern[0].mapping[1].indexSpaceDim = 1;
    out_defs->inputTensorAccessPattern[0].mapping[1].a =
                ceilf(in_defs->inputTensors[0].geometry.maxSizes[1] / unroll) * unroll;
    out_defs->inputTensorAccessPattern[0].mapping[1].start_b = 0;
    out_defs->inputTensorAccessPattern[0].mapping[1].end_b =
                ceilf(in_defs->inputTensors[0].geometry.maxSizes[1] / unroll) * unroll - 1;

    // f_start(i) = 0;
    // f_end f(i) = size[2] - 1;
    // Resource 0 (IFM) dim 2
    // Access is given to all the elements(factor of unroll) since single indexspace is used
    out_defs->inputTensorAccessPattern[0].mapping[2].indexSpaceDim = 2;
    out_defs->inputTensorAccessPattern[0].mapping[2].a       = in_defs->inputTensors[0].geometry.maxSizes[2];
    out_defs->inputTensorAccessPattern[0].mapping[2].start_b = 0;
    out_defs->inputTensorAccessPattern[0].mapping[2].end_b   = in_defs->inputTensors[0].geometry.maxSizes[2] - 1;

    // f_start(i) = 0;
    // f_end f(i) = size[3] - 1;
    // Resource 0 (IFM) dim 3
    // Access is given to all the elements(factor of unroll) since single indexspace is used
    out_defs->inputTensorAccessPattern[0].mapping[3].indexSpaceDim = 3;
    out_defs->inputTensorAccessPattern[0].mapping[3].a       = in_defs->inputTensors[0].geometry.maxSizes[3];
    out_defs->inputTensorAccessPattern[0].mapping[3].start_b = 0;
    out_defs->inputTensorAccessPattern[0].mapping[3].end_b   = in_defs->inputTensors[0].geometry.maxSizes[3] - 1;

    // f_start(i) = elementsInVec*i + 0;
    // f_end f(i) = elementsInVec*i + (elementsInVec - 1);
    // Resource 0-4 (OFM) dim 0
    for (unsigned i = 0; i < in_defs->outputTensorNr; i++)
    {
        out_defs->outputTensorAccessPattern[i].mapping[0].indexSpaceDim      = 0;
        out_defs->outputTensorAccessPattern[i].mapping[0].a        = elementsInVec;
        out_defs->outputTensorAccessPattern[i].mapping[0].start_b  = 0;
        out_defs->outputTensorAccessPattern[i].mapping[0].end_b    = elementsInVec - 1;
    }

    // f_start(i) = 0;
    // f_end f(i) = size[1] - 1;
    // Resource 0 (OFM) dim 1
    // Access is given to all the elements(factor of unroll) since single indexspace is used
    out_defs->outputTensorAccessPattern[0].mapping[1].indexSpaceDim = 1;
    out_defs->outputTensorAccessPattern[0].mapping[1].a =
                ceilf(in_defs->inputTensors[0].geometry.maxSizes[1] / unroll) * unroll;
    out_defs->outputTensorAccessPattern[0].mapping[1].start_b = 0;
    out_defs->outputTensorAccessPattern[0].mapping[1].end_b =
                ceilf(in_defs->inputTensors[0].geometry.maxSizes[1] / unroll) * unroll - 1;

    // f_start(i) = 0;
    // f_end f(i) = size[2] - 1;
    // Resource 0 (OFM) dim 2
    // Access is given to all the elements(factor of unroll) since single indexspace is used
    out_defs->outputTensorAccessPattern[0].mapping[2].indexSpaceDim = 2;
    out_defs->outputTensorAccessPattern[0].mapping[2].a       = in_defs->inputTensors[0].geometry.maxSizes[2];
    out_defs->outputTensorAccessPattern[0].mapping[2].start_b = 0;
    out_defs->outputTensorAccessPattern[0].mapping[2].end_b   = in_defs->inputTensors[0].geometry.maxSizes[2] - 1;

    // f_start(i) = 0;
    // f_end f(i) = size[3] - 1;
    // Resource 0 (OFM) dim 3
    // Access is given to all the elements(factor of unroll) since single indexspace is used
    out_defs->outputTensorAccessPattern[0].mapping[3].indexSpaceDim = 3;
    out_defs->outputTensorAccessPattern[0].mapping[3].a       = in_defs->inputTensors[0].geometry.maxSizes[3];
    out_defs->outputTensorAccessPattern[0].mapping[3].start_b = 0;
    out_defs->outputTensorAccessPattern[0].mapping[3].end_b   = in_defs->inputTensors[0].geometry.maxSizes[3] - 1;
    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
    float N = in_defs->inputTensors[0].geometry.maxSizes[1] *
              in_defs->inputTensors[0].geometry.maxSizes[2] *
              in_defs->inputTensors[0].geometry.maxSizes[3];
    def->N = N;
    def->N_reciprocal = 1.0 / N;
    out_defs->kernel.paramsNr = sizeof(*def)/ sizeof(int);
    memcpy(&( out_defs->kernel.scalarParams[0]),def, sizeof(*def));

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___batch_norm_fwd_f32_o_end - &_binary___batch_norm_fwd_f32_o_start);
    unsigned givenBinarySize = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (out_defs->kernel.kernelElf ,
                &_binary___batch_norm_fwd_f32_o_start,
                IsaSize);
    }
    else
    {
       retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
       return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

