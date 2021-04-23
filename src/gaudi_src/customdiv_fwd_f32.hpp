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

#ifndef _CUSTOMDIV_FWD_F32_HPP
#define _CUSTOMDIV_FWD_F32_HPP

#include "gc_interface.h"

class CustomdivFwdF32
{
public:
    CustomdivFwdF32() {}
    virtual ~CustomdivFwdF32() {}

    virtual gcapi::GlueCodeReturn_t GetGcDefinitions(
            gcapi::HabanaKernelParams_t* params,
            gcapi::HabanaKernelInstantiation_t* kernel);

    virtual gcapi::GlueCodeReturn_t GetKernelName(
            char kernelName [gcapi::MAX_NODE_NAME]);

private:
    CustomdivFwdF32(const CustomdivFwdF32& other) = delete;
    CustomdivFwdF32& operator=(const CustomdivFwdF32& other) = delete;
};


#endif //_CUSTOMDIV_FWD_F32_HPP
