/**********************************************************************
Copyright (c) 2024 Habana Labs.

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


#include "spatial_reduction_kernels.hpp"


tpc_lib_api::GlueCodeReturn  SpatialReductionKernels::ValidateTensorsDataType(
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


bool SpatialReductionKernels::GetOfmSize(
                                   uint64_t IfmSize [gcapi::MAX_TENSOR_DIM],
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

    OfmSize[0] = IfmSize[0];
    OfmSize[1] = ((IfmSize[1] + 2 * def->pad_w - def->dilation_w * (def->kernel_w-1) - 1) / def->stride_w) + 1;
    OfmSize[2] = ((IfmSize[2] + 2 * def->pad_h - def->dilation_h * (def->kernel_h-1) - 1) / def->stride_h) + 1;
    OfmSize[3] = IfmSize[3];
    OfmSize[4] = 1;
    return true;
}

void SpatialReductionKernels::GetAccessPatterns
            (tpc_lib_api::HabanaKernelInstantiation* out_defs,
            const SpatialReduction2DDef* def,
            unsigned int elementsInVector)
{
    // now define how the index space maps to IFM using f(i) = Ai+B .
    // transformation. 'i' is the index space member and A/B constants to be defined.
    // f_start(i) = 64*i +0;
    // f_end f(i) = 64*i + 64;
    // Resource 0 (IFM) dim 0 (depth).
    out_defs->inputTensorAccessPattern[0].mapping[0].indexSpaceDim = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].a = elementsInVector;
    out_defs->inputTensorAccessPattern[0].mapping[0].start_b = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].end_b = elementsInVector - 1;

    // start f(i) = stride*i + (-padw);
    // end f(i) = stride*i + kernelw*dilationw - padw );
    // Resource 0 (IFM) dim 1 (width).
    out_defs->inputTensorAccessPattern[0].mapping[1].indexSpaceDim = 1;
    out_defs->inputTensorAccessPattern[0].mapping[1].a = def->stride_w;
    out_defs->inputTensorAccessPattern[0].mapping[1].start_b = -def->pad_w;
    out_defs->inputTensorAccessPattern[0].mapping[1].end_b = -def->pad_w + (def->kernel_w - 1) * def->dilation_w;

    // start f(i) = stride*i + (-padh);
    // end f(i) = stride*i + (kernelh*dilationh - padh );
    // Resource 0 (IFM) dim 2 (height).
    out_defs->inputTensorAccessPattern[0].mapping[2].indexSpaceDim = 2;
    out_defs->inputTensorAccessPattern[0].mapping[2].a = def->stride_h;
    out_defs->inputTensorAccessPattern[0].mapping[2].start_b =  -def->pad_h;
    out_defs->inputTensorAccessPattern[0].mapping[2].end_b = -def->pad_h + (def->kernel_h - 1) * def->dilation_h;

    // start f(i) = 1*i + 0;
    // end f(i) = 1*i + 1;
    // Resource 0 (IFM) dim 3 (batch).
    out_defs->inputTensorAccessPattern[0].mapping[3].indexSpaceDim = 3;
    out_defs->inputTensorAccessPattern[0].mapping[3].a = 1;
    out_defs->inputTensorAccessPattern[0].mapping[3].start_b =  0;
    out_defs->inputTensorAccessPattern[0].mapping[3].end_b = 0;

    ////////////////////////////////////////////////////////////////////////////
    // define how the index space maps to the filter tensor
    out_defs->inputTensorAccessPattern[1].allRequired = true;
    ////////////////////////////////////////////////////////////////////////////
    // define how the index space maps to the output tensor
    out_defs->outputTensorAccessPattern[0].mapping[0].indexSpaceDim = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].a = elementsInVector;
    out_defs->outputTensorAccessPattern[0].mapping[0].start_b = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].end_b = elementsInVector - 1;

    // start f(i) = 1*i + 0;
    // end f(i) = 1*i + 1;
    // Resource 0 (IFM) dim 1 (width).
    out_defs->outputTensorAccessPattern[0].mapping[1].indexSpaceDim = 1;
    out_defs->outputTensorAccessPattern[0].mapping[1].a = 1;
    out_defs->outputTensorAccessPattern[0].mapping[1].start_b =  0;
    out_defs->outputTensorAccessPattern[0].mapping[1].end_b = 0;

    // start f(i) = i
    // end f(i) = i + 1
    // Resource 0 (IFM) dim 2 (height).
    out_defs->outputTensorAccessPattern[0].mapping[2].indexSpaceDim = 2;
    out_defs->outputTensorAccessPattern[0].mapping[2].a = 1;
    out_defs->outputTensorAccessPattern[0].mapping[2].start_b =  0;
    out_defs->outputTensorAccessPattern[0].mapping[2].end_b = 0;

    // start f(i) = 1*i + 0;
    // end f(i) = 1*i + 1;
    // Resource 0 (IFM) dim 3 (batch).
    out_defs->outputTensorAccessPattern[0].mapping[3].indexSpaceDim = 3;
    out_defs->outputTensorAccessPattern[0].mapping[3].a = 1;
    out_defs->outputTensorAccessPattern[0].mapping[3].start_b =  0;
    out_defs->outputTensorAccessPattern[0].mapping[3].end_b = 0;
}

void SpatialReductionKernels::OverrideAccessPatternForMultipleElements
                        (tpc_lib_api::HabanaKernelInstantiation* out_defs,
                         const SpatialReduction2DDef* def,
                         unsigned int dim,
                         unsigned int elementsNr)
{
    int stride   = def->stride_w;
    int kernel   = def->kernel_w;
    int dilation = def->dilation_w;
    int pad      = def->pad_w;
    int kernelSize  = (kernel + (kernel -1) * (dilation - 1));
    int overlapSize = (kernelSize - stride);
    int a = elementsNr * stride;
    int b = -pad + kernelSize * elementsNr - overlapSize * (elementsNr - 1) - 1;

    // Here n denotes loop unrolling factor
    // f_start(i) = n*stride*i - pad;
    // f_end f(i) = n*stride*i - pad + n*kernelSize - overlapSize*(n-1) - 1;
    // Resource 0 (IFM)
    out_defs->inputTensorAccessPattern[0].mapping[dim].indexSpaceDim = dim;
    out_defs->inputTensorAccessPattern[0].mapping[dim].a = a;
    out_defs->inputTensorAccessPattern[0].mapping[dim].start_b = -pad;
    out_defs->inputTensorAccessPattern[0].mapping[dim].end_b = b;

    // f_start(i) = n*i + 0;
    // f_end f(i) = n*i + (n-1);
    // Resource 0 (OFM)
    out_defs->outputTensorAccessPattern[0].mapping[dim].indexSpaceDim = dim;
    out_defs->outputTensorAccessPattern[0].mapping[dim].a = elementsNr;
    out_defs->outputTensorAccessPattern[0].mapping[dim].start_b =  0;
    out_defs->outputTensorAccessPattern[0].mapping[dim].end_b = elementsNr-1;
}
