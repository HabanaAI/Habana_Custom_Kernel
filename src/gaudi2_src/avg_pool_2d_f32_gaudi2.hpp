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

#ifndef _AVG_POOL_2D_F32_GAUDI2_HPP
#define _AVG_POOL_2D_F32_GAUDI2_HPP

#include <vector>
#include <cstring>
#include "spatial_reduction_kernels.hpp"



class AvgPool2dF32Gaudi2 : public SpatialReductionKernels
{
public:
    typedef enum _AvgPool2D_mode_t
    {
        fwd,
        bwd
    } AvgPool2D_mode_t;

    AvgPool2dF32Gaudi2(AvgPool2D_mode_t mode = fwd) {m_mode = mode;}
    virtual ~AvgPool2dF32Gaudi2() {}

    virtual tpc_lib_api::GlueCodeReturn GetGcDefinitions(
                                  tpc_lib_api::HabanaKernelParams* in_defs,
                                  tpc_lib_api::HabanaKernelInstantiation* out_defs);

     virtual tpc_lib_api::GlueCodeReturn GetKernelName(
             char kernelName [tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn fill_reciprocal_table(float* table, int num_elements) const;


    struct AvgPool2DParam
    {
        SpatialReduction2DDef srdef;
        int include_pads;
        int numTpc;
        float invNumTpc;
    };             
private:
    AvgPool2D_mode_t m_mode;
    AvgPool2dF32Gaudi2(const AvgPool2dF32Gaudi2& other) = delete;
    AvgPool2dF32Gaudi2& operator=(const AvgPool2dF32Gaudi2& other) = delete;
};

#endif // _AVG_POOL_2D_F32_GAUDI2_HPP

