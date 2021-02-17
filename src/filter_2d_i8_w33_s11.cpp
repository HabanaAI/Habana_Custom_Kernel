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

#include <vector>
#include <cstring>
#include <cmath>
#include <limits>
#include <iostream>
#include "filter_2d_i8_w33_s11.hpp"


extern unsigned char _binary___filter_2d_i8_w33_s11_o_start;
extern unsigned char _binary___filter_2d_i8_w33_s11_o_end;

 gcapi::GlueCodeReturn_t Filter2dI8W33S11::GetKernelName(
             char kernelName [gcapi::MAX_NODE_NAME])
 {
     strcpy(kernelName,"filter_2d_i8_w33_s11");
     return gcapi::GLUE_SUCCESS;
 }


/*! \brief RealToFixedPointWeak calculate flexpoint exponent.
 */
int8_t RealToFixedPointWeak(double dblNum)
{
    if (dblNum <= 0)
        return -1;

    double a = dblNum;
    double b = pow(2, ceil(log2(a)));
    double diff = std::abs(a-b);
    // we round number which are close to power of two
    if (diff < 0.00001)
    {
        a = b;
    }

    uint64_t* pNum = reinterpret_cast<uint64_t*>(&a);

    // Mantissa is first 52 bits.
    uint64_t mantissa =  (*pNum) & 0xFFFFFFFFFFFFFLL;

    // if mantissa is zero the number is a power of two.
    if (mantissa != 0) // so scale = 1
        return -1;

    int _exponent;
    frexp(dblNum, &_exponent);
    int8_t exponent = -(_exponent - 1/*numBitsForScale*/);

    return exponent;
}

gcapi::GlueCodeReturn_t Filter2dI8W33S11::GetGcDefinitions(
            gcapi::HabanaKernelParams_t* in_defs,
            gcapi::HabanaKernelInstantiation_t* out_defs)
{
    gcapi::GlueCodeReturn_t retVal;
    SpatialReduction2DDef* def = static_cast<SpatialReduction2DDef*>(in_defs->NodeParams);
    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (in_defs->inputTensorNr != 3)
    {
        in_defs->inputTensorNr  = 3;
        return gcapi::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    //validate correct amount of output tensors
    if (in_defs->outputTensorNr !=1)
    {
        in_defs->outputTensorNr  = 1;
        return gcapi::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    //check that filter depth match IFM
    if (in_defs->inputTensors[1].geometry.sizes[0] !=
        in_defs->inputTensors[0].geometry.sizes[0])
    {
        in_defs->inputTensors[1].geometry.sizes[0] =
                in_defs->inputTensors[0].geometry.sizes[0];
        return gcapi::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    //check that bias depth match IFM
    if (in_defs->inputTensors[2].geometry.sizes[0] !=
        in_defs->inputTensors[0].geometry.sizes[0])
    {
        in_defs->inputTensors[2].geometry.sizes[0] =
             in_defs->inputTensors[0].geometry.sizes[0];
        return gcapi::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    // Check whether kernel size is 3 and stride is 1
    if (def->stride_h != 1 || def->dilation_h != 1 ||
        def->stride_w != 1 || def->dilation_w != 1 ||
        def->kernel_w != 3 || def->kernel_h   != 3)
    {
        return gcapi::GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    }
    // Validate datatype of input tensors
    retVal = ValidateTensorsDataType(in_defs->inputTensors,
                                        2,
                                        gcapi::DATA_I8);
    if (retVal != gcapi::GLUE_SUCCESS)
    {
        return retVal;
    }
    // Validate datatype of bias input
    retVal = ValidateTensorsDataType(in_defs->inputTensors + 2,
                                        1,
                                        gcapi::DATA_I32);
    if (retVal != gcapi::GLUE_SUCCESS)
    {
        return retVal;
    }
    // Validate datatype of output tensors
    retVal = ValidateTensorsDataType(in_defs->outputTensors,
                                        in_defs->outputTensorNr,
                                        gcapi::DATA_I8);
    if (retVal != gcapi::GLUE_SUCCESS)
    {
        return retVal;
    }
    // Tensor 0 should be input feature map.
    // The semantics of the input tensors and their order is a convention
    // between TPC kernel writer and the write of the layer at the
    // framework level.
    unsigned int outputSizes[gcapi::MAX_TENSOR_DIM];

    if (!GetOfmSize(in_defs->inputTensors[0].geometry.sizes,def,outputSizes))
    {
        return gcapi::GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    }

    // verify that output feature map dimension are correct
    if (memcmp(in_defs->outputTensors[0].geometry.sizes,outputSizes,
               in_defs->outputTensors[0].geometry.dims * sizeof(unsigned) ) != 0)
    {
        memcpy(in_defs->outputTensors[0].geometry.sizes,outputSizes,sizeof(outputSizes));
        return gcapi::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    out_defs->indexSpaceGeometry.dims = 4;
    //round up to 256 and divide by 256.
    unsigned depthIndex = (outputSizes[0] + 255) / 256;
    out_defs->indexSpaceGeometry.sizes[0] = depthIndex;
    out_defs->indexSpaceGeometry.sizes[1] = (outputSizes[1] + 3) / 4;
    out_defs->indexSpaceGeometry.sizes[2] = outputSizes[2];
    out_defs->indexSpaceGeometry.sizes[3] = outputSizes[3];

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    GetAccessPatterns(out_defs,def,c_i8ElementsInVector);
    out_defs->inputTensorAccessPattern[2].dim[0].dim      = 0;
    out_defs->inputTensorAccessPattern[2].dim[0].start_a  = 256;
    out_defs->inputTensorAccessPattern[2].dim[0].end_a    = 256;
    out_defs->inputTensorAccessPattern[2].dim[0].start_b  = 0;
    out_defs->inputTensorAccessPattern[2].dim[0].end_b    = 255;
    // Modify access pattern for dim1 with unrolling  factor 4
    OverrideAccessPatternForMultipleElements(out_defs, def, 1, 4);

    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
    // Calculate exponent (scale factor) for converting to int8_t
    double normalizationFactor = (in_defs->inputTensors[0].quantizationParam.scale *
                                  in_defs->inputTensors[1].quantizationParam.scale)
                                  / in_defs->outputTensors[0].quantizationParam.scale;
    int32_t exponent = RealToFixedPointWeak(normalizationFactor);
    Filter2dSpecDef mdef;
    mdef.pad_w = def->pad_w;
    mdef.pad_h = def->pad_h;
    mdef.scale_factor = exponent;
    out_defs->kernel.paramsNr = sizeof(mdef)/ sizeof(int);
    memcpy(&( out_defs->kernel.scalarParams[0]),&mdef, sizeof(mdef));

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___filter_2d_i8_w33_s11_o_end - &_binary___filter_2d_i8_w33_s11_o_start);
    unsigned givenBinarySize = out_defs->elfSize;
    out_defs->elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (out_defs->kernelElf ,
                &_binary___filter_2d_i8_w33_s11_o_start,
                IsaSize);
    }
    else
    {
       retVal = gcapi::GLUE_INSUFICIENT_ELF_BUFFER;
       return retVal;
    }

    return gcapi::GLUE_SUCCESS;
}


