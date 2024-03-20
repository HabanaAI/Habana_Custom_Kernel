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

#ifndef _SPATIAL_REDUCTION_KERNELS_HPP
#define _SPATIAL_REDUCTION_KERNELS_HPP


#include <gc_interface.h>
#include "tpc_kernel_lib_interface.h"

class SpatialReductionKernels
{
public:

    SpatialReductionKernels() {}
    virtual ~SpatialReductionKernels() {}

    // This struct is common between the TPC kernel writer and the framework
    // layer writer. The programmer who adds a new layer to the framework-backend
    // is responsible to fill the structure with valid data.
    // TPC kernel writer accepts this struct as input.
    struct SpatialReduction2DDef
    {
        int pad_w;
        int pad_h;
        int kernel_w;
        int kernel_h;
        int stride_w;
        int stride_h;
        int dilation_w;
        int dilation_h;
    };

    // function common to all host glue code
    static bool GetOfmSize(uint64_t IfmSize [gcapi::MAX_TENSOR_DIM],
                                   const SpatialReduction2DDef * def,
                                   uint64_t OfmSize [gcapi::MAX_TENSOR_DIM]);

    static void GetAccessPatterns(tpc_lib_api::HabanaKernelInstantiation* out_defs,
                           const SpatialReduction2DDef * def,
                           unsigned int elementsInVector);

    void OverrideAccessPatternForMultipleElements(tpc_lib_api::HabanaKernelInstantiation* out_defs,
                                                  const SpatialReduction2DDef* def,
                                                  unsigned int dim,
                                                  unsigned int elementsNr);

    unsigned int ElementsInVector() const;

    tpc_lib_api::GlueCodeReturn  ValidateTensorsDataType(
                                tpc_lib_api::Tensor* pTensors,
                                int tensorCount,
                                tpc_lib_api::TensorDataType expected);


protected:
     const unsigned int c_f32ElementsInVector = 64;
     const unsigned int c_bf16ElementsInVector = 128;
     const unsigned int c_i8ElementsInVector  = 256;

private:
    SpatialReductionKernels(const SpatialReductionKernels& other) = delete;
    SpatialReductionKernels& operator=(const SpatialReductionKernels& other) = delete;
};

#endif  // _SPATIAL_REDUCTION_KERNELS_HPP
