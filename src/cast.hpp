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

#ifndef _CAST_HPP
#define _CAST_HPP

#include <vector>
#include <cstring>
#include "gc_interface.h"

class Cast
{
public:
    // List of supported casting types
    typedef enum _CastF32DataType_t
    {
        i8_to_f32,
        f32_to_i16
    } CastDataType_t;

    char castDataType[2][15] = {"i8_to_f32", "f32_to_i16"};
    Cast(CastDataType_t mode = i8_to_f32)
    {
        m_mode = mode;
    }

    virtual ~Cast() {}

    virtual gcapi::GlueCodeReturn_t GetGcDefinitions(
                                 gcapi::HabanaKernelParams_t* in_defs,
                                 gcapi::HabanaKernelInstantiation_t* out_defs);

    virtual gcapi::GlueCodeReturn_t GetKernelName(
            char kernelName [gcapi::MAX_NODE_NAME],
            CastDataType_t mode);

    // This struct is common between the TPC kernel writer and the framework
    // layer writer. The programmer who adds a new layer to the framework-backend
    // is responsible to fill the structure with valid dat
    struct CastParams
    {
        float scale;
    };

private:

    CastDataType_t m_mode;
    Cast(const Cast& other) = delete;
    Cast& operator=(const Cast& other) = delete;
};

#endif


