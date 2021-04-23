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

#ifndef _FILTER_2D_I8_W33_S11_HPP
#define _FILTER_2D_I8_W33_S11_HPP

#include <vector>
#include <cstring>
#include "spatial_reduction_kernels.hpp"



class Filter2dI8W33S11 : public SpatialReductionKernels
{
public:
    Filter2dI8W33S11() {}
    virtual ~Filter2dI8W33S11() {}

    virtual gcapi::GlueCodeReturn_t GetGcDefinitions(
                                  gcapi::HabanaKernelParams_t* in_defs,
                                  gcapi::HabanaKernelInstantiation_t* out_defs);

     virtual gcapi::GlueCodeReturn_t GetKernelName(
             char kernelName [gcapi::MAX_NODE_NAME]);


    // This struct is common between the TPC kernel writer and the framework
    // layer writer. The programmer who adds a new layer to the framework-backend
    // is responsible to fill the structure with valid data.
    // TPC kernel writer accepts this struct as input.
    struct Filter2dSpecDef
    {
        int32_t pad_w;
        int32_t pad_h;
        int32_t scale_factor;
    };

private:
    Filter2dI8W33S11(const Filter2dI8W33S11& other) = delete;
    Filter2dI8W33S11& operator=(const Filter2dI8W33S11& other) = delete;
};

int8_t RealToFixedPointWeak(double dblNum);

#endif


