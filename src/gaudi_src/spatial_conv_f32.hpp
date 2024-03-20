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

#ifndef _SPATIAL_CONV_F32_HPP
#define _SPATIAL_CONV_F32_HPP

#include <vector>
#include <cstring>
#include "spatial_reduction_kernels.hpp"


class SpatialConvF32 : public SpatialReductionKernels
{
public:
    SpatialConvF32() {}
    virtual ~SpatialConvF32() {}

    virtual tpc_lib_api::GlueCodeReturn GetGcDefinitions(
                                  tpc_lib_api::HabanaKernelParams* in_defs,
                                  tpc_lib_api::HabanaKernelInstantiation* out_defs);

     virtual tpc_lib_api::GlueCodeReturn GetKernelName(
             char kernelName [tpc_lib_api::MAX_NODE_NAME]);

     static bool GetSpatialConvOfmSize(
             uint64_t IfmSize [gcapi::MAX_TENSOR_DIM],
             uint64_t FilterSize [gcapi::MAX_TENSOR_DIM],
             const SpatialReduction2DDef* def,
             uint64_t OfmSize [gcapi::MAX_TENSOR_DIM]);

     static void GetSpatialConvAccessPatterns(tpc_lib_api::HabanaKernelInstantiation* out_defs,
                           const SpatialReduction2DDef * def,
                           unsigned int channelSize);

private:
    SpatialConvF32(const SpatialConvF32& other) = delete;
    SpatialConvF32& operator=(const SpatialConvF32& other) = delete;
};

#endif

