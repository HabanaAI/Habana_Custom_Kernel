/**********************************************************************
Copyright (c) 2021 Habana Labs. All rights reserved.

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

#ifndef _RELU6_F32_HPP
#define _RELU6_F32_HPP

#include "gc_interface.h"

class Relu6All
{
public:
    typedef enum _Relu6_mode_t
    {
        fwd_f32,
        bwd_f32,
        fwd_bf16,
        bwd_bf16
    } Relu6_mode_t;

    Relu6All(Relu6_mode_t mode=fwd_f32) {m_mode = mode;}
    virtual ~Relu6All() {}

    virtual gcapi::GlueCodeReturn_t GetGcDefinitions(
            gcapi::HabanaKernelParams_t* params,
            gcapi::HabanaKernelInstantiation_t* kernel);

    virtual gcapi::GlueCodeReturn_t GetKernelName(
            char kernelName [gcapi::MAX_NODE_NAME], Relu6_mode_t mode);

private:
    Relu6_mode_t m_mode;
    Relu6All(const Relu6All& other) = delete;
    Relu6All& operator=(const Relu6All& other) = delete;
};


#endif //_RELU6_F32_HPP
