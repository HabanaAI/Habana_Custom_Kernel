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

#ifndef SEARCHSORTED_F32_TEST_HPP
#define SEARCHSORTED_F32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "searchsorted_f32.hpp"

class SearchSortedF32Test : public TestBase
{
public:
    SearchSortedF32Test() {}
    ~SearchSortedF32Test() {}
    int runTest();

    inline static void searchsorted_fwd_f32_reference_implementation(
            const float_5DTensor& input0,
            const float_5DTensor& input1,
            int32_5DTensor& output,
            const SearchSortedF32::SearchSortedParam& def);

private:
    SearchSortedF32Test(const SearchSortedF32Test& other) = delete;
    SearchSortedF32Test& operator=(const SearchSortedF32Test& other) = delete;

};


#endif /* SEARCHSORTED_F32_TEST_HPP */

