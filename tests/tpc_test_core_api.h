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

#ifndef TPC_TEST_CORE_API_H
#define TPC_TEST_CORE_API_H

#include "TensorDescriptor.h"
#include "gc_interface.h"
#include <vector>

namespace tpc_tests {

// One of RunSimulation arguments - defines testing mode for index-space mapping testing.
typedef enum _IndexSpaceMappingTest_t
{
    // Indicates that index-space mapping testing enabled.
    e_accessPatternDefaultMode = 0,
    // Indicates that index-space mapping testing disabled.
    e_accessPatternIgnoreMode = 1,
    // Indicates that input tensors will be partially read.
    // Allows that index-space mapping won't be covered.
    e_accessPatternPartialReadMode = 2,
    // Indicates that output tensors will be partially written.
    // Allows that index-space mapping won't be covered.
    e_accessPatternPartialWriteMode = 3,
    // Allows that index-space mapping won't be covered for in/out tensors.
    e_accessPatternPartialReadWriteMode = 4,
    // Use indication per tensor for the access pattern checks.
    e_accessPatternPerTensorMode = 5,
} IndexSpaceMappingTest_t;

unsigned int
RunSimulation(const gcapi::HabanaKernelParams_t&              inDefs,
              const gcapi::HabanaKernelInstantiation_t&       outDefs,
              std::vector<TensorDescriptor>&                  descriptors,
              IndexSpaceMappingTest_t              testMode = e_accessPatternDefaultMode,
              std::vector<IndexSpaceMappingTest_t> inTensorTestModeList  = {},
              std::vector<IndexSpaceMappingTest_t> outTensorTestModeList = {});

} // namespace tpc_tests

#endif /* TPC_TEST_CORE_API_H */