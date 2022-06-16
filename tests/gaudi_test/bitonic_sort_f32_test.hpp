/**********************************************************************
Copyright (c) 2022 Habana Labs.
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

#ifndef BITONIC_SORT_F32_TEST_HPP
#define BITONIC_SORT_F32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "bitonic_sort_f32.hpp"

typedef std::vector<float>          floatVec_t;
typedef std::vector<int>            intVec_t;
typedef std::pair<int, float>       indDataPair_t;
typedef std::vector<indDataPair_t>  indDataVec_t;

class BitonicSortF32Test : public TestBase
{
public:

    BitonicSortF32Test() {}
    ~BitonicSortF32Test() {}
    int m_vecSize;
    int runTest();

    void OneVectorSort(indDataVec_t& curVec, SortType curDir, bool sortDir);
    intVec_t GeneratePredPattern(int32_t vecSize, bool curSign, int32_t stageNo, int32_t subStageNo);
    intVec_t GenerateShufflePattern(int vecSize, int stageNo, int subStageNo);
    template<class vecType> void SortVec(vecType& curVec, intVec_t& lut, intVec_t& pred, bool sortDir);

    inline void bitonic_sort_reference_implementation(float_5DTensor& ifmData, float_5DTensor& ofmRef,
                       int32_5DTensor& ofmIdxRef, int chunkSize, SortType dir);
private:
    BitonicSortF32Test(const BitonicSortF32Test& other) = delete;
    BitonicSortF32Test& operator=(const BitonicSortF32Test& other) = delete;
    

};


#endif /* BITONIC_SORT_F32_TEST_HPP */
