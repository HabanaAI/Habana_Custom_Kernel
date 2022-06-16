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

#ifndef _BITONIC_SORT_F32_HPP
#define _BITONIC_SORT_F32_HPP

#include <vector>

#include "gc_interface.h"

typedef enum sort_t{
    SORT_ASCENDING,
    SORT_DESCENDING
} SortType;

constexpr uint8_t SHUFFLE_EL_F32(uint8_t n)
{
    // all double group consist 16 element
    return n | ((n > 7) << 5) | 0x80;
}

#define SH_EL(n) \
    SHUFFLE_EL_F32(n), SHUFFLE_EL_F32(n), SHUFFLE_EL_F32(n), SHUFFLE_EL_F32(n)

/* This table is used to shuffle adjacent elements in an F32 vector */
const unsigned char shuffle_adj_ele_tab_f32[256] =
{
    /* DUAL GROUP 0 */
    /*group 0 */SH_EL(1), SH_EL(0),
                SH_EL(3), SH_EL(2),
                SH_EL(5), SH_EL(4),
                SH_EL(7), SH_EL(6),

    /*group 1 */SH_EL(9), SH_EL(8),
                SH_EL(11), SH_EL(10),
                SH_EL(13), SH_EL(12),
                SH_EL(15), SH_EL(14),

    /* DUAL GROUP 1 */
    /*group 0 */SH_EL(1), SH_EL(0),
                SH_EL(3), SH_EL(2),
                SH_EL(5), SH_EL(4),
                SH_EL(7), SH_EL(6),

    /*group 1 */SH_EL(9), SH_EL(8),
                SH_EL(11), SH_EL(10),
                SH_EL(13), SH_EL(12),
                SH_EL(15), SH_EL(14),

    /* DUAL GROUP 2 */
    /*group 0 */SH_EL(1), SH_EL(0),
                SH_EL(3), SH_EL(2),
                SH_EL(5), SH_EL(4),
                SH_EL(7), SH_EL(6),

    /*group 1 */SH_EL(9), SH_EL(8),
                SH_EL(11), SH_EL(10),
                SH_EL(13), SH_EL(12),
                SH_EL(15), SH_EL(14),

    /* DUAL GROUP 3 */
    /*group 0 */SH_EL(1), SH_EL(0),
                SH_EL(3), SH_EL(2),
                SH_EL(5), SH_EL(4),
                SH_EL(7), SH_EL(6),

    /*group 1 */SH_EL(9), SH_EL(8),
                SH_EL(11), SH_EL(10),
                SH_EL(13), SH_EL(12),
                SH_EL(15), SH_EL(14),
};

/* This table is used to shuffle all adjacent pairs of elements in an F32 vector */
const unsigned char shuffle_adj_ele_pair_tab_f32[256] =
{
    /* DUAL GROUP 0 */
    /*group 0 */SH_EL(2), SH_EL(3),
                SH_EL(0), SH_EL(1),
                SH_EL(6), SH_EL(7),
                SH_EL(4), SH_EL(5),

    /*group 1 */SH_EL(10), SH_EL(11),
                SH_EL(8), SH_EL(9),
                SH_EL(14), SH_EL(15),
                SH_EL(12), SH_EL(13),

    /* DUAL GROUP 1 */
    /*group 0 */SH_EL(2), SH_EL(3),
                SH_EL(0), SH_EL(1),
                SH_EL(6), SH_EL(7),
                SH_EL(4), SH_EL(5),

    /*group 1 */SH_EL(10), SH_EL(11),
                SH_EL(8), SH_EL(9),
                SH_EL(14), SH_EL(15),
                SH_EL(12), SH_EL(13),

    /* DUAL GROUP 2 */
    /*group 0 */SH_EL(2), SH_EL(3),
                SH_EL(0), SH_EL(1),
                SH_EL(6), SH_EL(7),
                SH_EL(4), SH_EL(5),

    /*group 1 */SH_EL(10), SH_EL(11),
                SH_EL(8), SH_EL(9),
                SH_EL(14), SH_EL(15),
                SH_EL(12), SH_EL(13),

    /* DUAL GROUP 3 */
    /*group 0 */SH_EL(2), SH_EL(3),
                SH_EL(0), SH_EL(1),
                SH_EL(6), SH_EL(7),
                SH_EL(4), SH_EL(5),

    /*group 1 */SH_EL(10), SH_EL(11),
                SH_EL(8), SH_EL(9),
                SH_EL(14), SH_EL(15),
                SH_EL(12), SH_EL(13),
};

/* This table is used to shuffle all adjacent quartets of elements in an F32 vector */
const unsigned char shuffle_adj_ele_quad_tab_f32[256] =
{
    /* DUAL GROUP 0 */
    /*group 0 */SH_EL(4), SH_EL(5), SH_EL(6), SH_EL(7),
                SH_EL(0), SH_EL(1), SH_EL(2), SH_EL(3),

    /*group 1 */SH_EL(12), SH_EL(13), SH_EL(14), SH_EL(15),
                SH_EL(8), SH_EL(9), SH_EL(10), SH_EL(11),

    /* DUAL GROUP 1 */
    /*group 0 */SH_EL(4), SH_EL(5), SH_EL(6), SH_EL(7),
                SH_EL(0), SH_EL(1), SH_EL(2), SH_EL(3),

    /*group 1 */SH_EL(12), SH_EL(13), SH_EL(14), SH_EL(15),
                SH_EL(8), SH_EL(9), SH_EL(10), SH_EL(11),

    /* DUAL GROUP 2 */
    /*group 0 */SH_EL(4), SH_EL(5), SH_EL(6), SH_EL(7),
                SH_EL(0), SH_EL(1), SH_EL(2), SH_EL(3),

    /*group 1 */SH_EL(12), SH_EL(13), SH_EL(14), SH_EL(15),
                SH_EL(8), SH_EL(9), SH_EL(10), SH_EL(11),

    /* DUAL GROUP 3 */
    /*group 0 */SH_EL(4), SH_EL(5), SH_EL(6), SH_EL(7),
                SH_EL(0), SH_EL(1), SH_EL(2), SH_EL(3),

    /*group 1 */SH_EL(12), SH_EL(13), SH_EL(14), SH_EL(15),
                SH_EL(8), SH_EL(9), SH_EL(10), SH_EL(11),
};

class BitonicSortF32
{
public:
    BitonicSortF32() {}
    virtual ~BitonicSortF32() {}

    virtual gcapi::GlueCodeReturn_t GetGcDefinitions(
            gcapi::HabanaKernelParams_t* pInDefs,
            gcapi::HabanaKernelInstantiation_t* pOutDefs);

    virtual gcapi::GlueCodeReturn_t GetKernelName(
            char kernelName [gcapi::MAX_NODE_NAME]);
    std::vector<uint8_t> GeneratePredPattern(const int32_t vecSize, const bool sign,
                                                      const int32_t stageNo,
                                                      const int32_t subStageNo) const;
                                                      
    gcapi::GlueCodeReturn_t AuxiliaryTensorSettings(gcapi::HabanaKernelParams_t*      pInDefs,
                                     gcapi::HabanaKernelInstantiation_t* pOutDefs, int vecsize, int chunkSize, SortType dir) const;

    struct BitonicSortF32Param
    {
        int sortDir;
        int chunkSize;
        int expChunkSize;
    };         

private:
    BitonicSortF32(const BitonicSortF32& other) = delete;
    BitonicSortF32& operator=(const BitonicSortF32& other) = delete;
};


#endif //_BITONIC_SORT_F32_HPP
