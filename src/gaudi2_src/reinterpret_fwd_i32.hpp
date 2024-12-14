// reinterpret_fwd_i32.hpp
#ifndef REINTERPRET_FWD_I32_HPP
#define REINTERPRET_FWD_I32_HPP

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class ReinterpretFwdI32
{
public:
    ReinterpretFwdI32() {}
    virtual ~ReinterpretFwdI32() {}

    virtual tpc_lib_api::GlueCodeReturn
    GetGcDefinitions(tpc_lib_api::HabanaKernelParams*      in_defs,
                     tpc_lib_api::HabanaKernelInstantiation* out_defs);

    virtual tpc_lib_api::GlueCodeReturn GetKernelName(
                char kernelName [tpc_lib_api::MAX_NODE_NAME]);                            

private:
    ReinterpretFwdI32(const ReinterpretFwdI32& other) = delete;
    ReinterpretFwdI32& operator=(const ReinterpretFwdI32& other) = delete;
};

#endif // REINTERPRET_FWD_I32_HPP
