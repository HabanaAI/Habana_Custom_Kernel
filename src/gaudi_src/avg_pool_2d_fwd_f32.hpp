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

#ifndef _AVG_POOL_2D_FWD_F32_HPP
#define _AVG_POOL_2D_FWD_F32_HPP

#include <vector>
#include <cstring>
#include "spatial_reduction_kernels.hpp"



class AvgPool2dFwdF32 : public SpatialReductionKernels
{
public:
    AvgPool2dFwdF32() {}
    virtual ~AvgPool2dFwdF32() {}

    virtual gcapi::GlueCodeReturn_t GetGcDefinitions(
                                  gcapi::HabanaKernelParams_t* in_defs,
                                  gcapi::HabanaKernelInstantiation_t* out_defs);

     virtual gcapi::GlueCodeReturn_t GetKernelName(
             char kernelName [gcapi::MAX_NODE_NAME]);
    gcapi::GlueCodeReturn_t fill_reciprocal_table(float* table, int num_elements) const;


    struct AvgPool2DFwdParam
    {
        SpatialReduction2DDef srdef;
        int include_pads;
    };             
private:
    AvgPool2dFwdF32(const AvgPool2dFwdF32& other) = delete;
    AvgPool2dFwdF32& operator=(const AvgPool2dFwdF32& other) = delete;
};

#endif // _AVG_POOL_2D_FWD_F32_HPP

