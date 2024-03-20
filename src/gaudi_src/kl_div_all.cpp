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

#include "kl_div_all.hpp"
#include <iostream>

extern unsigned char _binary___kl_div_fwd_f32_o_start;
extern unsigned char _binary___kl_div_fwd_f32_o_end;
extern unsigned char _binary___kl_div_bwd_f32_o_start;
extern unsigned char _binary___kl_div_bwd_f32_o_end;
extern unsigned char _binary___kl_div_fwd_f32_gaudi2_o_start;
extern unsigned char _binary___kl_div_fwd_f32_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn KLDivAll::GetKernelName(
             char kernelName [tpc_lib_api::MAX_NODE_NAME])
{
    if(m_mode == fwd_f32)
        strcpy(kernelName,"custom_kl_div_fwd_f32");
    else if(m_mode == bwd_f32)
        strcpy(kernelName,"custom_kl_div_bwd_f32");
    else if(m_mode == fwd_f32_gaudi2)
        strcpy(kernelName,"custom_kl_div_fwd_f32_gaudi2");    
    else
        return tpc_lib_api::GLUE_NODE_NOT_FOUND;
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn  KLDivAll::ValidateTensorsDataType(
           tpc_lib_api::Tensor* pTensors,
           int tensorCount)
{
    tpc_lib_api::GlueCodeReturn retVal = tpc_lib_api::GLUE_SUCCESS;
    for (int i = 0 ; i < tensorCount ; i++)
    {
        if(m_mode == fwd_f32 || m_mode == bwd_f32) {
            if (pTensors[i].geometry.dataType != tpc_lib_api::DATA_F32)
            {
                retVal = tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
                pTensors[i].geometry.dataType = tpc_lib_api::DATA_F32;
            }
        }
    }
    return retVal;
}

void KLDivAll::SetGeometryAlongAxis(tpc_lib_api::HabanaKernelParams*      in_defs,
                                          tpc_lib_api::HabanaKernelInstantiation* out_defs, int axis,
                                          int pixels_per_loop, uint32_t inpTensorMask,
                                          uint32_t outTensorMask)
{
    if ((0 != in_defs->inputTensorNr) && (axis != 0))
    {
        out_defs->indexSpaceGeometry[axis] =
            (std::max(in_defs->inputTensors[0].geometry.maxSizes[axis], (uint64_t)1) + pixels_per_loop - 1) /
            pixels_per_loop;
    }
    else
    {
        out_defs->indexSpaceGeometry[axis] =
            (out_defs->indexSpaceGeometry[axis] + pixels_per_loop - 1) / pixels_per_loop;
    }

    if (0u != inpTensorMask)
    {
        for (unsigned k = 0; (k < in_defs->inputTensorNr); k++)
        {
            if ((inpTensorMask & (1u << k)))
            {
                out_defs->inputTensorAccessPattern[k].mapping[axis].indexSpaceDim = axis;
                out_defs->inputTensorAccessPattern[k].mapping[axis].a *= pixels_per_loop;
                out_defs->inputTensorAccessPattern[k].mapping[axis].start_b = 0;
                out_defs->inputTensorAccessPattern[k].mapping[axis].end_b =
                    (out_defs->inputTensorAccessPattern[k].mapping[axis].end_b + 1) * pixels_per_loop -
                    1;
            }
        }
    }

    if (0u != outTensorMask)
    {
        for (unsigned k = 0; (k < in_defs->outputTensorNr); k++)
        {
            if ((outTensorMask & (1u << k)))
            {
                out_defs->outputTensorAccessPattern[k].mapping[axis].indexSpaceDim = axis;
                out_defs->outputTensorAccessPattern[k].mapping[axis].a *= pixels_per_loop;
                out_defs->outputTensorAccessPattern[k].mapping[axis].start_b = 0;
                out_defs->outputTensorAccessPattern[k].mapping[axis].end_b =
                    (out_defs->outputTensorAccessPattern[k].mapping[axis].end_b + 1) * pixels_per_loop -
                    1;
            }
        }
    }
}

tpc_lib_api::GlueCodeReturn KLDivAll::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* in_defs,
            tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    tpc_lib_api::GlueCodeReturn retVal;
    const int c_unrollCount = 4;
    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if(m_mode == fwd_f32 || m_mode == fwd_f32_gaudi2) {
        if (in_defs->inputTensorNr != 2)
        {
            in_defs->inputTensorNr  = 2;
            return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
        }
    }
    else { // backward path
       if (in_defs->inputTensorNr != 3)
        {
            in_defs->inputTensorNr  = 3;
            return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
        }        
    }

    //validate correct amount of output tensors
    if (in_defs->outputTensorNr != 1) {
        in_defs->outputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    // validate input and output data type
    retVal = ValidateTensorsDataType(in_defs->inputTensors,
                                        in_defs->inputTensorNr);
    if (retVal != tpc_lib_api::GLUE_SUCCESS)
    {
        return retVal;
    }

    retVal = ValidateTensorsDataType(in_defs->outputTensors,
                                        in_defs->outputTensorNr);
    if (retVal != tpc_lib_api::GLUE_SUCCESS)
    {
        return retVal;
    }

   // Validate input and output tensor sizes are same
    uint64_t * inputTensorSizes = in_defs->inputTensors[0].geometry.maxSizes;
    bool SizesAreEqual = true;

    if(m_mode == fwd_f32 || m_mode == fwd_f32_gaudi2) 
    {
        SizesAreEqual &= in_defs->outputTensors[0].geometry.maxSizes[0] == 1;
    }
    else
    {
        SizesAreEqual &= in_defs->inputTensors[2].geometry.maxSizes[0] == 1;
        inputTensorSizes = in_defs->inputTensors[1].geometry.maxSizes;
        for(unsigned int dim = 0; dim < 2; dim++)
        {
            SizesAreEqual &= (in_defs->outputTensors[0].geometry.maxSizes[dim]
                    == inputTensorSizes[dim]);
        }
    }

    if(!SizesAreEqual)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    // Validate dim0 of all input tensors are equal
    for(unsigned int tns = 0; tns < in_defs->inputTensorNr; tns++)
    {
        if(tns != 2) // #2 tensor is 1D
            SizesAreEqual &= (in_defs->inputTensors[tns].geometry.maxSizes[0] == inputTensorSizes[0]);
    }

    if(!SizesAreEqual)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry.
    **************************************************************************************/
    int elementsInVec;
    unsigned depthIndex, elements;
    if(m_mode == fwd_f32 || m_mode == fwd_f32_gaudi2|| m_mode == bwd_f32)
        elementsInVec = 64;
    else
        elementsInVec = 128;

    depthIndex = (inputTensorSizes[0] + (elementsInVec - 1)) / elementsInVec;
    elements   = elementsInVec * depthIndex;
    if(m_mode == fwd_f32 || m_mode == fwd_f32_gaudi2) 
    {
        out_defs->indexSpaceRank = 1;
        out_defs->indexSpaceGeometry[0] = 1;
    }
    else{
        out_defs->indexSpaceGeometry[0] = depthIndex;
        out_defs->indexSpaceGeometry[1] = inputTensorSizes[1];
        out_defs->indexSpaceGeometry[2] = inputTensorSizes[2];
        out_defs->indexSpaceGeometry[3] = inputTensorSizes[3];        
    }

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    // f_start(i) = elementsInVec*i + 0;
    // f_end f(i) = elementsInVec*i + (elementsInVec - 1);
    // Resource 0-4 (IFM) dim 0
    if(m_mode == fwd_f32 || m_mode == fwd_f32_gaudi2)    
    {
        std::cout << "in_defs->inputTensorNr is " << in_defs->inputTensorNr << std::endl;
        for(unsigned int ii = 0;ii < in_defs->inputTensorNr; ii++) {
            //out_defs->inputTensorAccessPattern[ii].allRequired = true;
            out_defs->inputTensorAccessPattern[ii].mapping[0].indexSpaceDim      = 0;
            out_defs->inputTensorAccessPattern[ii].mapping[0].a        = 0;
            out_defs->inputTensorAccessPattern[ii].mapping[0].start_b  = 0;
            out_defs->inputTensorAccessPattern[ii].mapping[0].end_b    = elements - 1;

            for (int dims = 1; dims < 5; dims++)
            {
                out_defs->inputTensorAccessPattern[ii].mapping[dims].indexSpaceDim      = dims;
                out_defs->inputTensorAccessPattern[ii].mapping[dims].a        = 0;
                out_defs->inputTensorAccessPattern[ii].mapping[dims].start_b  = 0;
                out_defs->inputTensorAccessPattern[ii].mapping[dims].end_b    = inputTensorSizes[dims] - 1;
            }
        }


        out_defs->outputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
        out_defs->outputTensorAccessPattern[0].mapping[0].a        = 0;
        out_defs->outputTensorAccessPattern[0].mapping[0].start_b  = 0;
        out_defs->outputTensorAccessPattern[0].mapping[0].end_b    = elements - 1;
        
        out_defs->outputTensorAccessPattern[0].mapping[1].indexSpaceDim      = 1;
        out_defs->outputTensorAccessPattern[0].mapping[1].a        = 0;
        out_defs->outputTensorAccessPattern[0].mapping[1].start_b  = 0;
        out_defs->outputTensorAccessPattern[0].mapping[1].end_b    = 
                    ((inputTensorSizes[1] + (c_unrollCount - 1)) / c_unrollCount) *
                    c_unrollCount - 1;

        for (int dims = 2; dims < 5; dims++)
        {
            out_defs->outputTensorAccessPattern[0].mapping[dims].indexSpaceDim      = dims;
            out_defs->outputTensorAccessPattern[0].mapping[dims].a        = 0;
            out_defs->outputTensorAccessPattern[0].mapping[dims].start_b  = 0;
            out_defs->outputTensorAccessPattern[0].mapping[dims].end_b    = inputTensorSizes[dims] - 1;
        }
    }
    else 
    {
        out_defs->indexSpaceRank = 4;
        out_defs->indexSpaceGeometry[0] =
            ceilf((float)in_defs->outputTensors[0].geometry.maxSizes[0] / elementsInVec);
        out_defs->indexSpaceGeometry[1] =
            (in_defs->inputTensors[0].geometry.maxSizes[1] + (c_unrollCount - 1)) / c_unrollCount;

        for (int i = 2; i < 5; i++)
        {
            out_defs->indexSpaceGeometry[i] =
                std::max(in_defs->outputTensors[0].geometry.maxSizes[i], (uint64_t)1);
        }

        for(unsigned int ii = 0; ii < in_defs->inputTensorNr; ii++) {
            out_defs->inputTensorAccessPattern[ii].allRequired = true;
            out_defs->inputTensorAccessPattern[ii].mapping[0].indexSpaceDim      = 0;
            out_defs->inputTensorAccessPattern[ii].mapping[0].a        = elementsInVec;
            out_defs->inputTensorAccessPattern[ii].mapping[0].start_b  = 0;
            out_defs->inputTensorAccessPattern[ii].mapping[0].end_b    = elementsInVec - 1;

            out_defs->inputTensorAccessPattern[ii].mapping[1].indexSpaceDim      = 1;
            out_defs->inputTensorAccessPattern[ii].mapping[1].a        = 1;
            out_defs->inputTensorAccessPattern[ii].mapping[1].start_b  = 0;
            out_defs->inputTensorAccessPattern[ii].mapping[1].end_b    = 1 - 1;
            
       //#2 (start with #0) tensor is 1D
            if(ii ==2) {
                out_defs->inputTensorAccessPattern[2].mapping[0].a       = 0;
                out_defs->inputTensorAccessPattern[2].mapping[0].start_b = 0;
                out_defs->inputTensorAccessPattern[2].mapping[0].end_b = 0;            
                continue;
            }
            for (int dims = 2; dims < 5; dims++)
            {
                out_defs->inputTensorAccessPattern[ii].mapping[dims].indexSpaceDim      = dims;
                out_defs->inputTensorAccessPattern[ii].mapping[dims].a        = 1;
                out_defs->inputTensorAccessPattern[ii].mapping[dims].start_b  = 0;
                out_defs->inputTensorAccessPattern[ii].mapping[dims].end_b    = 1 - 1;
            }
        }

        out_defs->outputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
        out_defs->outputTensorAccessPattern[0].mapping[0].a        = elementsInVec;
        out_defs->outputTensorAccessPattern[0].mapping[0].start_b  = 0;
        out_defs->outputTensorAccessPattern[0].mapping[0].end_b    = elementsInVec - 1;
        
        out_defs->outputTensorAccessPattern[0].mapping[1].indexSpaceDim      = 1;
        out_defs->outputTensorAccessPattern[0].mapping[1].a        = 1;
        out_defs->outputTensorAccessPattern[0].mapping[1].start_b  = 0;
        out_defs->outputTensorAccessPattern[0].mapping[1].end_b    = 1 - 1;

        for (int dims = 2; dims < 5; dims++)
        {
            out_defs->outputTensorAccessPattern[0].mapping[dims].indexSpaceDim      = dims;
            out_defs->outputTensorAccessPattern[0].mapping[dims].a        = 1;
            out_defs->outputTensorAccessPattern[0].mapping[dims].start_b  = 0;
            out_defs->outputTensorAccessPattern[0].mapping[dims].end_b    = 1 - 1;
        }

        SetGeometryAlongAxis(in_defs, out_defs, 1, 1, 7, 1);

    }
    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
    KLDivAllParams* def = static_cast<KLDivAllParams*>(in_defs->nodeParams.nodeParams);
    out_defs->kernel.paramsNr = sizeof(*def)/ sizeof(float);
    memcpy(&( out_defs->kernel.scalarParams[0]),def, sizeof(*def));

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___kl_div_fwd_f32_o_end - &_binary___kl_div_fwd_f32_o_start);
    unsigned char *binary_kernel = &_binary___kl_div_bwd_f32_o_start;
    switch (m_mode)
    {
        case fwd_f32:
            IsaSize = (&_binary___kl_div_fwd_f32_o_end - &_binary___kl_div_fwd_f32_o_start);
            binary_kernel = &_binary___kl_div_fwd_f32_o_start;
            break;
        case bwd_f32:
            IsaSize = (&_binary___kl_div_bwd_f32_o_end - &_binary___kl_div_bwd_f32_o_start);
            binary_kernel = &_binary___kl_div_bwd_f32_o_start;
            break;
        case fwd_f32_gaudi2:
            IsaSize = (&_binary___kl_div_fwd_f32_gaudi2_o_end - &_binary___kl_div_fwd_f32_gaudi2_o_start);
            binary_kernel = &_binary___kl_div_fwd_f32_gaudi2_o_start;
            break;

        default:
            break;
    
    }
        
    unsigned givenBinarySize = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = IsaSize;
    if (givenBinarySize >= IsaSize)
    {
        memcpy (out_defs->kernel.kernelElf, binary_kernel, IsaSize);
    }
    else
    {
        retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
        return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}
