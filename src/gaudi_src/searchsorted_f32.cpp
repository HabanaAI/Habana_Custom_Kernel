/**********************************************************************
Copyright (c) 2023 Habana Labs.

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

#include "searchsorted_f32.hpp"

extern unsigned char _binary___searchsorted_fwd_f32_o_start;
extern unsigned char _binary___searchsorted_fwd_f32_o_end;
//extern unsigned char _binary___searchsorted_bwd_f32_o_start;
//extern unsigned char _binary___searchsorted_bwd_f32_o_end;

 gcapi::GlueCodeReturn_t SearchSortedF32::GetKernelName(
             char kernelName [gcapi::MAX_NODE_NAME])
 {
    if(m_mode == fwd)
        strcpy(kernelName,"searchsorted_fwd_f32");
    else
        strcpy(kernelName,"searchsorted_bwd_f32");

     return gcapi::GLUE_SUCCESS;
 }

gcapi::GlueCodeReturn_t SearchSortedF32::GetGcDefinitions(
            gcapi::HabanaKernelParams_t* params,
            gcapi::HabanaKernelInstantiation_t* kernel)
{
    gcapi::GlueCodeReturn_t retVal;
    SearchSortedParam* def = static_cast<SearchSortedParam*>(params->NodeParams);

    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (params->inputTensorNr != 2)
    {
        params->inputTensorNr  = 2;
        return gcapi::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    //validate correct amount of output tensors
    if (params->outputTensorNr != 1)
    {
        params->outputTensorNr  = 1;
        return gcapi::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
     //validate matrix dimensions
    if (params->inputTensors[0].geometry.sizes[0] != params->inputTensors[1].geometry.sizes[0])
    {
        return gcapi::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    // validate input and output data type
    if (params->inputTensors[0].dataType != gcapi::DATA_F32 ||
        params->inputTensors[1].dataType != gcapi::DATA_F32 ||
        params->outputTensors[0].dataType != gcapi::DATA_I32)
    {
        params->inputTensors[0].dataType = gcapi::DATA_F32;
        params->inputTensors[1].dataType = gcapi::DATA_F32;
        params->outputTensors[0].dataType = gcapi::DATA_I32;
        return gcapi::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    int elementsInVec = 64;
    const int c_unrollCount = 1;
    unsigned int outputSizes[gcapi::MAX_TENSOR_DIM] = {0};
    memcpy(outputSizes, params->inputTensors[1].geometry.sizes, sizeof(outputSizes));

    //round up to elementsInVec and divide by elementsInVec.
    unsigned depthIndex = (outputSizes[0] + (elementsInVec - 1)) / elementsInVec;
    kernel->indexSpaceGeometry.dims = 5;
    kernel->indexSpaceGeometry.sizes[0] = depthIndex;
	//reduce index space due to unroll.
    kernel->indexSpaceGeometry.sizes[1] = (outputSizes[1] +(c_unrollCount-1)) / c_unrollCount; 
    kernel->indexSpaceGeometry.sizes[2] = outputSizes[2];
    kernel->indexSpaceGeometry.sizes[3] = outputSizes[3];
    kernel->indexSpaceGeometry.sizes[4] = outputSizes[4];

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/

    // Index space mapping is calculated using f(i) = Ai + B
    // 'i' is the index space member and A/B constants to be defined.

    // f_start(i) = 0;
    // f_end f(i) = size[0] - 1;
    // Resource 0 (IFM) dim 0 (depth)
    // Access is given to all the elements since single indexspace is used
    for (unsigned i = 0; i < params->inputTensorNr; i++)
    {
        kernel->inputTensorAccessPattern[i].dim[0].dim      = 0;
        kernel->inputTensorAccessPattern[i].dim[0].start_a  = elementsInVec;
        kernel->inputTensorAccessPattern[i].dim[0].end_a    = elementsInVec;
        kernel->inputTensorAccessPattern[i].dim[0].start_b  = 0;
        kernel->inputTensorAccessPattern[i].dim[0].end_b    = elementsInVec - 1;

        // f_start f(i) = 1*i + 0;
        // f_end   f(i) = 1*i + 0;
        // Resource 0 (IFM) dim 1-4
        for (int dims = 1; dims < 5; dims++)
        {
            kernel->inputTensorAccessPattern[i].dim[dims].dim      = dims;
            kernel->inputTensorAccessPattern[i].dim[dims].start_a  = 1;
            kernel->inputTensorAccessPattern[i].dim[dims].end_a    = 1;
            kernel->inputTensorAccessPattern[i].dim[dims].start_b  = 0;
            kernel->inputTensorAccessPattern[i].dim[dims].end_b    = 1 - 1;
        }        
    }    

    // f_start f(i) = elementsInVec*i + 0;
    // f_end   f(i) = elementsInVec*i + (elementsInVec - 1);
    // Resource 0 (OFM) dim 0
    kernel->outputTensorAccessPattern[0].dim[0].dim      = 0;
    kernel->outputTensorAccessPattern[0].dim[0].start_a  = elementsInVec;
    kernel->outputTensorAccessPattern[0].dim[0].end_a    = elementsInVec;
    kernel->outputTensorAccessPattern[0].dim[0].start_b  = 0;
    kernel->outputTensorAccessPattern[0].dim[0].end_b    = elementsInVec - 1;
	
    // f_start f(i) = 1*i + 0;
    // f_end   f(i) = 1*i + 0;
    // Resource 0 (OFM) dim 1-4
    for (int dims = 1; dims < 5; dims++)
    {
        kernel->outputTensorAccessPattern[0].dim[dims].dim      = dims;
        kernel->outputTensorAccessPattern[0].dim[dims].start_a  = 1;
        kernel->outputTensorAccessPattern[0].dim[dims].end_a    = 1;
        kernel->outputTensorAccessPattern[0].dim[dims].start_b  = 0;
        kernel->outputTensorAccessPattern[0].dim[dims].end_b    = 1 - 1;
    }


    /*************************************************************************************
    *    Stage IV -  Set Auxiliary Tensor
    **************************************************************************************/
    kernel->kernel.paramsNr = sizeof(*def)/ sizeof(int);
    memcpy(&( kernel->kernel.scalarParams[0]), def, sizeof(*def));
    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = &_binary___searchsorted_fwd_f32_o_end - &_binary___searchsorted_fwd_f32_o_start;
    unsigned char *binary_kernel = &_binary___searchsorted_fwd_f32_o_start;
    switch (m_mode){
        case fwd:
            IsaSize = (&_binary___searchsorted_fwd_f32_o_end - &_binary___searchsorted_fwd_f32_o_start);
            binary_kernel = &_binary___searchsorted_fwd_f32_o_start;
            break;
        //case bwd:
        //    IsaSize = (&_binary___searchsorted_bwd_f32_o_end - &_binary___searchsorted_bwd_f32_o_start);
        //    binary_kernel = &_binary___searchsorted_bwd_f32_o_start;
        //    break;
        default:
            break;

    }

    unsigned givenBinarySize = kernel->elfSize;
    kernel->elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (kernel->kernelElf ,
                binary_kernel,
                IsaSize);
    }
    else
    {
       retVal = gcapi::GLUE_INSUFICIENT_ELF_BUFFER;
       return retVal;
    }

    return gcapi::GLUE_SUCCESS;
}


