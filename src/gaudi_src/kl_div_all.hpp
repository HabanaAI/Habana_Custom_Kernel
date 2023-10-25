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

#ifndef _KL_DIV_ALL_HPP
#define _KL_DIV_ALL_HPP

#include <vector>
#include <cstring>
#include <cmath>
#include "gc_interface.h"

class KLDivAll
{
public:
    typedef enum _KLDiv_mode_t
    {
        fwd_f32,
        bwd_f32,
        fwd_f32_gaudi2
    } KLDiv_mode_t;

    KLDivAll(KLDiv_mode_t mode=fwd_f32) {m_mode = mode;}
    virtual ~KLDivAll() {};

    virtual gcapi::GlueCodeReturn_t GetGcDefinitions(
                                 gcapi::HabanaKernelParams_t* in_defs,
                                 gcapi::HabanaKernelInstantiation_t* out_defs);

    virtual gcapi::GlueCodeReturn_t GetKernelName(
            char kernelName [gcapi::MAX_NODE_NAME]);

    void SetGeometryAlongAxis(  gcapi::HabanaKernelParams_t* in_defs,
                                gcapi::HabanaKernelInstantiation_t* out_defs, int axis,
                                int pixels_per_loop, uint32_t inpTensorMask,
                                uint32_t outTensorMask);

    gcapi::GlueCodeReturn_t  ValidateTensorsDataType(
                                gcapi::Tensor_t* pTensors,
                                int tensorCount);

    // This struct is common between the TPC kernel writer and the framework layer writer
    struct KLDivAllParams
    {
        float invLen;
        int log_target;
    };


private:
    KLDiv_mode_t m_mode;
    KLDivAll(const KLDivAll& other) = delete;
    KLDivAll& operator=(const KLDivAll& other) = delete;
};

#endif //_KL_DIV_ALL_HPP
