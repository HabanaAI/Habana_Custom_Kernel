/**********************************************************************
Copyright (c) 2023 Habana Labs.

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
#ifndef GATHER_FWD_I32_TEST_HPP
#define GATHER_FWD_I32_TEST_HPP

#include "tensor.h"
#include "test_base.hpp"
#include "gather_fwd_i32.hpp"
#include "entry_points.hpp"

class GatherFwdI32Test : public TestBase {
public:
    GatherFwdI32Test() {}
    ~GatherFwdI32Test() {}

    int runTest(Gaudi_Kernel_Name_e NameofKernel);

    void GatherFwdRefImplementation(
            int32_5DTensor &input_tensor,
            int32_5DTensor &index_tensor,
            int32_5DTensor &output_tensor,
            int inputDims, int indexDims,
            int axis);

    void GatherElementsOnnxRef(
        int32_5DTensor &ifm,
        int32_5DTensor &indices,
        int32_5DTensor &ofm,
        int axis);

private:
    GatherFwdI32Test(const GatherFwdI32Test& other) = delete;
    GatherFwdI32Test& operator=(const GatherFwdI32Test& other) = delete;
};

#endif /*GATHER_FWD_I32_TEST_HPP*/
