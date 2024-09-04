/**********************************************************************
Copyright (c) 2024 Habana Labs. All rights reserved.

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

#ifndef _PSCAN_UPDATE_GAUDI2_HPP
#define _PSCAN_UPDATE_GAUDI2_HPP

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class PscanUpdateGaudi2
{
public:
    typedef enum _pscan_mode_t
    {
        pscan_update_f32,
        pscan_update_bf16,
    } pscan_update_mode_t;

    PscanUpdateGaudi2(pscan_update_mode_t mode=pscan_update_f32) {m_mode = mode;}
    virtual ~PscanUpdateGaudi2() {}

    virtual tpc_lib_api::GlueCodeReturn GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* params,
            tpc_lib_api::HabanaKernelInstantiation* kernel);

    virtual tpc_lib_api::GlueCodeReturn GetKernelName(
            char kernelName [tpc_lib_api::MAX_NODE_NAME], pscan_update_mode_t mode);
    
private:
    pscan_update_mode_t m_mode;
    PscanUpdateGaudi2(const PscanUpdateGaudi2& other) = delete;
    PscanUpdateGaudi2& operator=(const PscanUpdateGaudi2& other) = delete;
};


#endif //_PSCAN_UPDATE_GAUDI2_HPP
