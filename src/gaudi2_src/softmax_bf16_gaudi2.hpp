/**********************************************************************
Copyright (c) 2020 Habana Labs.

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

#ifndef _SOFTMAX_BF16_GAUDI2_HPP
#define _SOFTMAX_BF16_GAUDI2_HPP

#include <vector>
#include <cstring>
#include "gc_interface.h"



class SoftMaxBF16Gaudi2
{
public:
    SoftMaxBF16Gaudi2() {}
    virtual ~SoftMaxBF16Gaudi2() {}

    virtual gcapi::GlueCodeReturn_t GetGcDefinitions(
                                  gcapi::HabanaKernelParams_t* in_defs,
                                  gcapi::HabanaKernelInstantiation_t* out_defs);

     virtual gcapi::GlueCodeReturn_t GetKernelNameFcd(
             char kernelName [gcapi::MAX_NODE_NAME]);

     virtual gcapi::GlueCodeReturn_t GetKernelNameNonFcd(
             char kernelName [gcapi::MAX_NODE_NAME]);


    // This struct is common between the TPC kernel writer and the framework
    // layer writer. The programmer who adds a new layer to the framework-backend
    // is responsible to fill the structure with valid data.
    struct SoftMaxParam
    {
        int32_t axis;
    };


private:
    SoftMaxBF16Gaudi2(const SoftMaxBF16Gaudi2& other) = delete;
    SoftMaxBF16Gaudi2& operator=(const SoftMaxBF16Gaudi2& other) = delete;
};


#endif


