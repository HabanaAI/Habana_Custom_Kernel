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

#include "kernel_config.h"

#define     MAX_NVEC_VLM    128 // = Maximum chunk size (8192)/VECTOR_SIZE(64)
#define     PRED_TAB_NR     15
#define     TRUE            1
#define     SORT_ASCENDING  0
#define     VECTOR_SIZE_EXP 6 // VECTOR_SIZE = 64 = 2^6
#define     ORIG_DIR        0

// The following macros are used to organize code only.
#define     BITONIC_BLOCKS_1_TO_6           // Bitonic stages 1.1 to 6.6    => BS64 module
#define     BITONIC_BLOCKS_7_TO_N_MINUS_2   // Bitonic stages 7.1 to (N-2).(N-2) => V2V module 1
#define     BITONIC_BLOCK_N_MINUS_1         // Bitonic stages (N-1).1 to (N-1).(N-1) => V2V module 2
#define     BITONIC_BLOCK_N_STAGE_1_TO_N_MINUS_6    // Bitonic stages N.1 to N.(N-6) => V2V module 3
#define     BITONIC_BLOCK_N_LAST_6_STAGES           // Bitonic Stages N.(N-5) to N.N => BM64 module

typedef struct _double_index_data_pair_t
{
    index_data_pair_t p1; // index data pair 1
    index_data_pair_t p2; // index data pair 2
} double_index_data_pair_t;

__local__ index_data_pair_t     vecListVlm[MAX_NVEC_VLM];

// Only 16 VPRFs available. Hence VLM has to be used.
/*TODO : Move this inside main and let compiler spill to VLM as required*/
__local__ bool256               predTab[PRED_TAB_NR];

double_index_data_pair_t BitonicSort64(VECTOR data1, int64 idx1, VECTOR data2, int64 idx2,
    uchar256 lutTab0, uchar256 lutTab1, uchar256 lutTab2,
    bool256 predTab6_1, bool256 predTab6_2, bool256 predTab6_3,
    bool256 predTab6_4, bool256 predTab6_5, bool256 predTab6_6, int iVec);

double_index_data_pair_t BitonicMerge64DualDir(VECTOR data1, int64 idx1, VECTOR data2, int64 idx2,
    uchar256 lutTab0, uchar256 lutTab1, uchar256 lutTab2,
    bool256 predTab6_1, bool256 predTab6_2, bool256 predTab6_3,
    bool256 predTab6_4, bool256 predTab6_5, bool256 predTab6_6);

/*
* This function performs bitonic merge of 64 elements on the index-data vector passed to it, in the
* original direction. Merge is done w.r.t. data and indices are moved along with data.
*/
index_data_pair_t BitonicMerge64OrigDir(VECTOR data, int64 idx,
    uchar256 lutTab0, uchar256 lutTab1, uchar256 lutTab2,
    bool256 predTab6_1, bool256 predTab6_2, bool256 predTab6_3,
    bool256 predTab6_4, bool256 predTab6_5, bool256 predTab6_6);

void main(tensor ifmData,
        tensor  ofmData,
        tensor  ofmIdx,        
        tensor  auxLutAndPred,
        int     sortDir,
        int     chunkSize,
        int     expChunkSize
    )
{
    const int depth  = 0;
    const int width  = 1;
    const int height = 2;
    const int batch  = 3;
    const int fifthDim = 4;

    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd   = get_index_space_size() + indexSpaceStart;

    int sortDepthEnd = indexSpaceEnd[depth] * chunkSize;

    // DEPTH
/*  Static Full Bitonic Sort (FBS) or
    Static Bitonic Sort with Scalar Merge (BS+SM) */
    const int depthStep   = chunkSize;
    const int depthStart  = indexSpaceStart[depth] * depthStep;
// i.e. Squeezed BS+SM

// FBS limit empty chunk processing

    // Static FBS
    const int ifmDepth = get_dim_size(ifmData, 0);
    const int fullDepthEnd = (((ifmDepth - 1) >> expChunkSize)
       << expChunkSize) + chunkSize;
    const int depthEndOrig = sortDepthEnd;
    sortDepthEnd = s_i32_min(sortDepthEnd, fullDepthEnd);
    sortDepthEnd = s_i32_max(depthStart, sortDepthEnd);

    // WIDTH
    const int widthStep   = 1;
    const int widthStart  = indexSpaceStart[width];
    const int widthEnd    = indexSpaceEnd[width];

    // HEIGHT
    const int heightStep  = 1;
    const int heightStart = indexSpaceStart[height];
    const int heightEnd   = indexSpaceEnd[height];

    // BATCH
    const int batchStep   = 1;
    const int batchStart  = indexSpaceStart[batch];
    const int batchEnd   = indexSpaceEnd[batch];

    // FIFTHDIM
    const int fifthDimStep   = 1;
    const int fifthDimStart  = indexSpaceStart[fifthDim];
    const int fifthDimEnd   = indexSpaceEnd[fifthDim];

    // Compute no of vectors in each chunk
    int nVecChunk = chunkSize >> VECTOR_SIZE_EXP;

    int5 fmCoords1 = { 0 };
    int5 fmCoords2 = { 0 };
    int5 fmCoords3 = { 0 };
    int5 fmCoords4 = { 0 };


    int ofmDataIdx = 1;

    int ofmIndexIdx = ofmDataIdx + 1;
    int auxLutAndPredIdx = ofmIndexIdx + 1;

    int ofmDepth = get_dim_size(ofmDataIdx, 0);
    bool isLastNode = (chunkSize == ofmDepth);
    bool isChunkVector = (chunkSize == VECTOR_SIZE) && !isLastNode;

    // Set pad value based on sort direction
    #if defined(FLOAT32)
        float padVal = s_f32_mov(1.0/0.0, 0, -1.0/0.0, sortDir, 1);
    #elif defined(INT32)
        int padVal = s_i32_mov(0x7FFFFFFF, 0, 0x80000000, sortDir, 1);
    #endif

    bool predGt64 = s_i32_cmp_grt(nVecChunk, 1);
    bool predGt128 = s_i32_cmp_grt(nVecChunk, 2);

    // Load lut tables for BS64
    int5 auxCoords = { 0 };
    uchar256 lutTab0 = v_u8_ld_tnsr_b(auxCoords, auxLutAndPredIdx);
    auxCoords[0] += 256;
    uchar256 lutTab1 = v_u8_ld_tnsr_b(auxCoords, auxLutAndPredIdx);
    auxCoords[0] += 256;
    uchar256 lutTab2 = v_u8_ld_tnsr_b(auxCoords, auxLutAndPredIdx);
    auxCoords[0] += 256;

    /* Load predicate patterns for in-vector bitonic sort of 64 elements
       Patterns specific to stages 1.1 to 5.5*/
    for (int i = 0; i < PRED_TAB_NR; i++)
    {
        uchar256 tmp = v_u8_ld_tnsr_b(auxCoords, auxLutAndPredIdx);
        predTab[i] = v_u8_cmp_eq_b(tmp, 1);
        auxCoords[0] += 256;
    }
    /* Patterns specific to stages 6.1 to 6.6 are loaded to predicates directly so that
       it can be used across in-vector bitonic merge and in-vector bitonic sort steps w/o
       accessing VLM*/
    uchar256 tmp = v_u8_ld_tnsr_b(auxCoords, auxLutAndPredIdx);
    bool256 predTab6_1 = v_u8_cmp_eq_b(tmp, 1);
    auxCoords[0] += 256;
    tmp = v_u8_ld_tnsr_b(auxCoords, auxLutAndPredIdx);
    bool256 predTab6_2 = v_u8_cmp_eq_b(tmp, 1);
    auxCoords[0] += 256;
    tmp = v_u8_ld_tnsr_b(auxCoords, auxLutAndPredIdx);
    bool256 predTab6_3 = v_u8_cmp_eq_b(tmp, 1);
    auxCoords[0] += 256;
    tmp = v_u8_ld_tnsr_b(auxCoords, auxLutAndPredIdx);
    bool256 predTab6_4 = v_u8_cmp_eq_b(tmp, 1);
    auxCoords[0] += 256;
    tmp = v_u8_ld_tnsr_b(auxCoords, auxLutAndPredIdx);
    bool256 predTab6_5 = v_u8_cmp_eq_b(tmp, 1);
    auxCoords[0] += 256;
    tmp = v_u8_ld_tnsr_b(auxCoords, auxLutAndPredIdx);
    bool256 predTab6_6 = v_u8_cmp_eq_b(tmp, 1);

    #pragma loop_taken
    for (int f = fifthDimStart; f < fifthDimEnd; f += fifthDimStep)
    {
        fmCoords1[fifthDim] = f;
        fmCoords2[fifthDim] = f;
        fmCoords3[fifthDim] = f;
        fmCoords4[fifthDim] = f;

        #pragma loop_taken
        for (int b = batchStart; b < batchEnd; b += batchStep)
        {
            fmCoords1[batch] = b;
            fmCoords2[batch] = b;
            fmCoords3[batch] = b;
            fmCoords4[batch] = b;

            #pragma loop_taken
            for (int h = heightStart; h < heightEnd; h += heightStep)
            {
                fmCoords1[height] = h;
                fmCoords2[height] = h;
                fmCoords3[height] = h;
                fmCoords4[height] = h;


                #pragma loop_taken
                for (int w = widthStart; w < widthEnd; w += widthStep)
                {

                    fmCoords1[width] = w;
                    fmCoords2[width] = w;
                    fmCoords3[width] = w;
                    fmCoords4[width] = w;


                    /* This loop is activated to store the pad value,
                        when we have chunks fully out of bound*/
                    for (int d = sortDepthEnd; d < depthEndOrig; d += depthStep)
                    {
                        /*In squeezed case, sortDepthEnd is original validcount(nonPowerOf2)
                        aligned to chunksize, so that loop iterates in steps of chunksize upto
                        ofmSize which is a power of 2*/

                        fmCoords1[depth] = d;
                        fmCoords2[depth] = d + VECTOR_SIZE;
                        fmCoords3[depth] = d + 2 * VECTOR_SIZE;
                        fmCoords4[depth] = d + 3 * VECTOR_SIZE;
                        int64 padIdx = 0;

                        // Expect to get padValue from AGU
                        VECTOR padData = v_ld_tnsr_i(fmCoords1, ifmData);
                        for (int i = 0; i < nVecChunk; i+=4)
                        {
                            st_tnsr_i_v(fmCoords1, ofmDataIdx, padData);
                            st_tnsr_i_v_b(fmCoords2, ofmDataIdx, padData, predGt64, 0);
                            st_tnsr_i_v_b(fmCoords3, ofmDataIdx, padData, predGt128, 0);
                            st_tnsr_i_v_b(fmCoords4, ofmDataIdx, padData, predGt128, 0);

                            v_i32_st_tnsr(fmCoords1, ofmIndexIdx, padIdx);
                            v_i32_st_tnsr(fmCoords2, ofmIndexIdx, padIdx, 0, predGt64);
                            v_i32_st_tnsr(fmCoords3, ofmIndexIdx, padIdx, 0, predGt128);
                            v_i32_st_tnsr(fmCoords4, ofmIndexIdx, padIdx, 0, predGt128);

                            fmCoords1[depth] += (VECTOR_SIZE << 2);
                            fmCoords2[depth] += (VECTOR_SIZE << 2);
                            fmCoords3[depth] += (VECTOR_SIZE << 2);
                            fmCoords4[depth] += (VECTOR_SIZE << 2);
                        }
                    }

                    /*In squeezed case, sortDepthEnd is original validcount(nonPowerOf2)
                    aligned to chunksize, so that loop iterates in steps of chunksize upto
                    validcount aligned to chunksize instead of power of 2*/
                    for (int d = depthStart; d < sortDepthEnd; d += depthStep)
                    {
                        bool iDir = (d >> expChunkSize) & 1;
                        /* If chunk is a vector, use direction corresponding to index space element
                           Else, set direction as required direction*/
                        bool curDir = s_i1_mov(iDir, 0, sortDir, isChunkVector);

                        /* Set correct direction for predicate patterns used in BM64 phase
                           If curDir = 0, use as-is, else NOT.*/
                        bool256 predLastTab6_1 = v_i1_not_b(predTab6_1, 0, predTab6_1, curDir);
                        bool256 predLastTab6_2 = v_i1_not_b(predTab6_2, 0, predTab6_2, curDir);
                        bool256 predLastTab6_3 = v_i1_not_b(predTab6_3, 0, predTab6_3, curDir);
                        bool256 predLastTab6_4 = v_i1_not_b(predTab6_4, 0, predTab6_4, curDir);
                        bool256 predLastTab6_5 = v_i1_not_b(predTab6_5, 0, predTab6_5, curDir);
                        bool256 predLastTab6_6 = v_i1_not_b(predTab6_6, 0, predTab6_6, curDir);

    #ifdef BITONIC_BLOCKS_1_TO_6
                        /* Bitonic Sort stages 1.1 to 6.6
                           In-vector Sort (BS64) on all individual vectors
                           in the chunk and store into array */
                        fmCoords1[depth] = d;
                        fmCoords2[depth] = d + VECTOR_SIZE;
                        fmCoords3[depth] = d + 2 * VECTOR_SIZE;
                        fmCoords4[depth] = d + 3 * VECTOR_SIZE;

                        VECTOR curData2 = padVal;
                        VECTOR curData3 = padVal;
                        VECTOR curData4 = padVal;

                        for (int iVec = 0; iVec < nVecChunk; iVec += 4)
                        {
                            // Load data
                            VECTOR curData1 = v_ld_tnsr_i(fmCoords1, ifmData);
                            curData2 = v_ld_tnsr_i_b(fmCoords2, ifmData, curData2, predGt64, 0);
                            curData3 = v_ld_tnsr_i_b(fmCoords3, ifmData, curData3, predGt128, 0);
                            curData4 = v_ld_tnsr_i_b(fmCoords4, ifmData, curData4, predGt128, 0);

                            // Generate indices if index tensor is not provided as input
                            int startInd1 = fmCoords1[depth];
                            int startInd2 = fmCoords2[depth];
                            int startInd3 = fmCoords3[depth];
                            int startInd4 = fmCoords4[depth];

                            int64 curIdx1 =
                                    v_i32_add_b((int64)read_lane_id_4b_b(), startInd1, SW_SAT);
                            int64 curIdx2 =
                                    v_i32_add_b((int64)read_lane_id_4b_b(), startInd2, SW_SAT);
                            int64 curIdx3 =
                                    v_i32_add_b((int64)read_lane_id_4b_b(), startInd3, SW_SAT);
                            int64 curIdx4 =
                                    v_i32_add_b((int64)read_lane_id_4b_b(), startInd4, SW_SAT);

                            /* Perform Bitonic Sort of 64 on two index-data vectors
                               Here first vector is sorted in original direction and second vector
                              in opposite direction*/
                            /* In case of full bitonic (i.e., GBC node), use BM64 predicates
                               depending on curDir*/
                            double_index_data_pair_t dres1 = BitonicSort64(curData1, curIdx1,
                                curData2, curIdx2, lutTab0, lutTab1, lutTab2,
                                predLastTab6_1, predLastTab6_2, predLastTab6_3,
                                predLastTab6_4, predLastTab6_5, predLastTab6_6, iVec);
                            double_index_data_pair_t dres2 = BitonicSort64(curData3, curIdx3,
                                curData4, curIdx4, lutTab0, lutTab1, lutTab2,
                                predLastTab6_1, predLastTab6_2, predLastTab6_3,
                                predLastTab6_4, predLastTab6_5, predLastTab6_6, (iVec + 2));

                            vecListVlm[iVec]     = dres1.p1;
                            vecListVlm[iVec + 1] = dres1.p2;
                            vecListVlm[iVec + 2] = dres2.p1;
                            vecListVlm[iVec + 3] = dres2.p2;

                            fmCoords1[depth] += (VECTOR_SIZE << 2);
                            fmCoords2[depth] += (VECTOR_SIZE << 2);
                            fmCoords3[depth] += (VECTOR_SIZE << 2);
                            fmCoords4[depth] += (VECTOR_SIZE << 2);
                        }
    #endif //BITONIC_BLOCKS_1_TO_6

    /* NOTE: Bitonic blocks upto (N-2) need to be performed only when nVecChunk > 4. This is
       implicitly taken care in the first 'for' loop below. For nVecChunk <= 4,
       blocks N-1 and N are executed separately below.*/
    #ifdef BITONIC_BLOCKS_7_TO_N_MINUS_2
                        /* Bitonic Sort stages 7.1 to (N-2).(N-2)
                           Arrange vectors into group of nVecBGBox vectors and
                           do bitonic merge on each block in alternate directions.*/
                        int nVecChunkBy2 = (nVecChunk >> 1);
                        int totBGBoxCount = nVecChunkBy2; // initial value = (nVecChunk/2)
                        /* Iterate over all blue/green box sizes from 2 to nVecChunk/2. The size
                           doubles in each iteration.*/
                        for (int nVecBGBox = 2; nVecBGBox < nVecChunkBy2;
                                                                    nVecBGBox = (nVecBGBox << 1))
                        {
                            int redBoxCount = 1;

                            /* Vector-to-Vector comparison and swap
                               Iterate over all possible red box sizes within a blue/green box.
                               The size halves in each iteration */
                            for (int nVecRedBox = nVecBGBox; nVecRedBox > 1;
                                nVecRedBox = (nVecRedBox >> 1))
                            {
                                int v2offset = (nVecRedBox >> 1);

                                // Process each blue/green box
                                for (int indBGBox = 0; indBGBox < totBGBoxCount; indBGBox += 4)
                                {
                                    int indBGBoxByI = indBGBox * nVecBGBox;
                                    int indBGBoxByI1 = (indBGBox + 1) * nVecBGBox;
                                    int indBGBoxByI2 = (indBGBox + 2) * nVecBGBox;
                                    int indBGBoxByI3 = (indBGBox + 3) * nVecBGBox;
                                    // Process each red box
                                    for (int indRedBox = 0; indRedBox < redBoxCount; indRedBox++)
                                    {
                                        int indRedBoxByJ = indRedBox * nVecRedBox;

                                        int indBlueV1 = indBGBoxByI + indRedBoxByJ;
                                        int indBlueV2 = indBlueV1 + v2offset;
                                        int indBlueV3 = indBGBoxByI2 + indRedBoxByJ;
                                        int indBlueV4 = indBlueV3 + v2offset;

                                        int indGreenV1 = indBGBoxByI1 + indRedBoxByJ;
                                        int indGreenV2 = indGreenV1 + v2offset;
                                        int indGreenV3 = indBGBoxByI3 + indRedBoxByJ;
                                        int indGreenV4 = indGreenV3 + v2offset;
                                        // Perform all comparisons within the red box
                                        for (int indV2VComp = 0; indV2VComp < (nVecRedBox >> 1);
                                            indV2VComp++)
                                        {
                                            index_data_pair_t pairBlue1 = vecListVlm[indBlueV1];
                                            index_data_pair_t pairBlue2 = vecListVlm[indBlueV2];
                                            index_data_pair_t pairBlue3 = vecListVlm[indBlueV3];
                                            index_data_pair_t pairBlue4 = vecListVlm[indBlueV4];
                                            index_data_pair_t pairGreen1 = vecListVlm[indGreenV1];
                                            index_data_pair_t pairGreen2 = vecListVlm[indGreenV2];
                                            index_data_pair_t pairGreen3 = vecListVlm[indGreenV3];
                                            index_data_pair_t pairGreen4 = vecListVlm[indGreenV4];

                                            vecListVlm[indBlueV2] =  v_sel2_less_v_v_v_v(
                                                pairBlue1.v2, pairBlue2.v2,
                                                pairBlue1.v1, pairBlue2.v1);
                                            vecListVlm[indBlueV1] =  v_sel2_geq_v_v_v_v(
                                                pairBlue1.v2, pairBlue2.v2,
                                                pairBlue1.v1, pairBlue2.v1);
                                            vecListVlm[indBlueV4] =  v_sel2_less_v_v_v_v(
                                                pairBlue3.v2, pairBlue4.v2,
                                                pairBlue3.v1, pairBlue4.v1);
                                            vecListVlm[indBlueV3] =  v_sel2_geq_v_v_v_v(
                                                pairBlue3.v2, pairBlue4.v2,
                                                pairBlue3.v1, pairBlue4.v1);

                                            vecListVlm[indGreenV2] =  v_sel2_grt_v_v_v_v(
                                                pairGreen1.v2, pairGreen2.v2,
                                                pairGreen1.v1, pairGreen2.v1);
                                            vecListVlm[indGreenV1] =  v_sel2_leq_v_v_v_v(
                                                pairGreen1.v2, pairGreen2.v2,
                                                pairGreen1.v1, pairGreen2.v1);
                                            vecListVlm[indGreenV4] =  v_sel2_grt_v_v_v_v(
                                                pairGreen3.v2, pairGreen4.v2,
                                                pairGreen3.v1, pairGreen4.v1);
                                            vecListVlm[indGreenV3] =  v_sel2_leq_v_v_v_v(
                                                pairGreen3.v2, pairGreen4.v2,
                                                pairGreen3.v1, pairGreen4.v1);

                                            //advance offsets
                                            indBlueV1++;
                                            indBlueV2++;
                                            indBlueV3++;
                                            indBlueV4++;
                                            indGreenV1++;
                                            indGreenV2++;
                                            indGreenV3++;
                                            indGreenV4++;
                                        }
                                    }
                                }
                                // Red box count doubles in each iteration
                                redBoxCount = (redBoxCount << 1);
                            }

                            /* In-Vector comparison and swap on all vectors in the blue/green box
                               BM-64 on each vector (Steps 6.1 to 6.6) in alternate directions*/
                            for (int iVec = 0; iVec < nVecChunk; iVec += nVecBGBox)
                            {
                                for (int iMerge = 0; iMerge < nVecBGBox; iMerge += 2)
                                {
                                    double_index_data_pair_t res1 = BitonicMerge64DualDir(
                                        vecListVlm[iVec].v2, vecListVlm[iVec].v1,
                                        vecListVlm[iVec + nVecBGBox].v2,
                                        vecListVlm[iVec + nVecBGBox].v1,
                                        lutTab0, lutTab1, lutTab2,
                                        predTab6_1, predTab6_2, predTab6_3,
                                        predTab6_4, predTab6_5, predTab6_6);

                                    double_index_data_pair_t res2 = BitonicMerge64DualDir(
                                        vecListVlm[iVec + 1].v2, vecListVlm[iVec + 1].v1,
                                        vecListVlm[iVec + nVecBGBox + 1].v2,
                                        vecListVlm[iVec + nVecBGBox + 1].v1,
                                        lutTab0, lutTab1, lutTab2,
                                        predTab6_1, predTab6_2, predTab6_3,
                                        predTab6_4, predTab6_5, predTab6_6);

                                    vecListVlm[iVec]         = res1.p1;
                                    vecListVlm[iVec + nVecBGBox]     = res1.p2;
                                    vecListVlm[iVec + 1]     = res2.p1;
                                    vecListVlm[iVec + nVecBGBox + 1] = res2.p2;

                                    iVec += 2;
                                }
                            }

                            // Blue-Green box count decrements by half in each iteration
                            totBGBoxCount = (totBGBoxCount >> 1);
                        }
    #endif // BITONIC_BLOCKS_7_TO_N_MINUS_2

                        // Bitonic block (N-1) need to be performed only when nVecChunk > 2
                        if (nVecChunk > 2)
                        {
    #ifdef BITONIC_BLOCK_N_MINUS_1
                            /* Bitonic Sort stages (N-1).1 to (N-1).(N-1)
                              (N-1) has one blue and one green box.*/
                            int redBoxCount = 1;
                            int v2offset = (nVecChunkBy2 >> 1);
                            int indBlueV1 = 0;
                            int indBlueV2 = v2offset;
                            int indGreenV1 = nVecChunkBy2;
                            int indGreenV2 = nVecChunkBy2 + v2offset;
                            bool predE = s_i32_cmp_grt((nVecChunkBy2 >> 1), 1);
                            /* Perform all comparisons within the first red box of the
                            blue and green boxes */
                            for (int indV2VComp = 0; indV2VComp < (nVecChunkBy2 >> 1);
                                                                                    indV2VComp += 2)
                            {
                                index_data_pair_t pairBlue1 = vecListVlm[indBlueV1];
                                index_data_pair_t pairBlue2 = vecListVlm[indBlueV2];
                                index_data_pair_t pairGreen1 = vecListVlm[indGreenV1];
                                index_data_pair_t pairGreen2 = vecListVlm[indGreenV2];
                                index_data_pair_t pairBlue3 = vecListVlm[indBlueV1 + 1];
                                index_data_pair_t pairBlue4 = vecListVlm[indBlueV2 + 1];
                                index_data_pair_t pairGreen3 = vecListVlm[indGreenV1 + 1];
                                index_data_pair_t pairGreen4 = vecListVlm[indGreenV2 + 1];

                                index_data_pair_t pair1;
                                index_data_pair_t pair2;
                                index_data_pair_t pair3;
                                index_data_pair_t pair4;
                                index_data_pair_t pair5;
                                index_data_pair_t pair6;
                                index_data_pair_t pair7;
                                index_data_pair_t pair8;

                                pair1 =  v_sel2_less_v_v_v_v(
                                            pairBlue1.v2, pairBlue2.v2, pairBlue1.v1, pairBlue2.v1);
                                pair2 =  v_sel2_geq_v_v_v_v(
                                            pairBlue1.v2, pairBlue2.v2, pairBlue1.v1, pairBlue2.v1);
                                pair3 =  v_sel2_grt_v_v_v_v(
                                        pairGreen1.v2, pairGreen2.v2, pairGreen1.v1, pairGreen2.v1);
                                pair4 =  v_sel2_leq_v_v_v_v(
                                        pairGreen1.v2, pairGreen2.v2, pairGreen1.v1, pairGreen2.v1);
                                pair5 =  v_sel2_less_v_v_v_v(
                                            pairBlue3.v2, pairBlue4.v2, pairBlue3.v1, pairBlue4.v1);
                                pair6 =  v_sel2_geq_v_v_v_v(
                                            pairBlue3.v2, pairBlue4.v2, pairBlue3.v1, pairBlue4.v1);
                                pair7 =  v_sel2_grt_v_v_v_v(
                                        pairGreen3.v2, pairGreen4.v2, pairGreen3.v1, pairGreen4.v1);
                                pair8 =  v_sel2_leq_v_v_v_v(
                                        pairGreen3.v2, pairGreen4.v2, pairGreen3.v1, pairGreen4.v1);

                                vecListVlm[indBlueV1].v1  = pair2.v1;
                                vecListVlm[indBlueV1].v2  = pair2.v2;
                                vecListVlm[indBlueV2].v1  = pair1.v1;
                                vecListVlm[indBlueV2].v2  = pair1.v2;

                                vecListVlm[indGreenV2].v1 = pair3.v1;
                                vecListVlm[indGreenV2].v2 = pair3.v2;
                                vecListVlm[indGreenV1].v1 = pair4.v1;
                                vecListVlm[indGreenV1].v2 = pair4.v2;

                                vecListVlm[indBlueV1+1].v1 =
                                                        v_i32_mov_b(pair6.v1, 0, pair1.v1, predE);
                                vecListVlm[indBlueV1+1].v2 =
                                                        v_mov_v_b(pair6.v2, pair1.v2, predE, 0);
                                vecListVlm[indBlueV2+1].v1 =
                                                        v_i32_mov_b(pair5.v1, 0, pair4.v1, predE);
                                vecListVlm[indBlueV2+1].v2 =
                                                        v_mov_v_b(pair5.v2, pair4.v2, predE, 0);

                                vecListVlm[indGreenV2+1].v1 = pair7.v1;
                                vecListVlm[indGreenV2+1].v2 = pair7.v2;
                                vecListVlm[indGreenV1+1].v1 =
                                                        v_i32_mov_b(pair8.v1, 0, pair3.v1, predE);
                                vecListVlm[indGreenV1+1].v2 =
                                                        v_mov_v_b(pair8.v2, pair3.v2, predE, 0);

                                //advance offsets
                                indBlueV1 += 2;
                                indBlueV2 += 2;
                                indGreenV1 += 2;
                                indGreenV2 += 2;
                            }

                            // Red box count doubles in each iteration
                            redBoxCount = (redBoxCount << 1);

                            // Iterate over all the red box sizes within the blue and green boxes.
                            for (int nVecRedBox = (nVecChunkBy2 >> 1); nVecRedBox > 1;
                                nVecRedBox = (nVecRedBox >> 1))
                            {
                                v2offset = (nVecRedBox >> 1);

                                for (int indRedBox = 0; indRedBox < redBoxCount; indRedBox += 2)
                                {
                                    indBlueV1 = indRedBox * nVecRedBox;
                                    indBlueV2 = indBlueV1 + v2offset;
                                    indGreenV1 = nVecChunkBy2 + indRedBox * nVecRedBox;
                                    indGreenV2 = indGreenV1 + v2offset;
                                    int indBlueV3 = (indRedBox + 1) * nVecRedBox;
                                    int indBlueV4 = indBlueV3 + v2offset;
                                    int indGreenV3 = nVecChunkBy2 + (indRedBox + 1) * nVecRedBox;
                                    int indGreenV4 = indGreenV3 + v2offset;
                                    // Perform all comparisons within the red box
                                    for (int indV2VComp = 0; indV2VComp < (nVecRedBox >> 1);
                                        indV2VComp++)
                                    {
                                        index_data_pair_t pairBlue1 = vecListVlm[indBlueV1];
                                        index_data_pair_t pairBlue2 = vecListVlm[indBlueV2];
                                        index_data_pair_t pairBlue3 = vecListVlm[indBlueV3];
                                        index_data_pair_t pairBlue4 = vecListVlm[indBlueV4];
                                        index_data_pair_t pairGreen1 = vecListVlm[indGreenV1];
                                        index_data_pair_t pairGreen2 = vecListVlm[indGreenV2];
                                        index_data_pair_t pairGreen3 = vecListVlm[indGreenV3];
                                        index_data_pair_t pairGreen4 = vecListVlm[indGreenV4];

                                        vecListVlm[indBlueV2] =  v_sel2_less_v_v_v_v(pairBlue1.v2,
                                            pairBlue2.v2, pairBlue1.v1, pairBlue2.v1);
                                        vecListVlm[indBlueV1] =  v_sel2_geq_v_v_v_v(pairBlue1.v2,
                                            pairBlue2.v2, pairBlue1.v1, pairBlue2.v1);
                                        vecListVlm[indBlueV4] =  v_sel2_less_v_v_v_v(pairBlue3.v2,
                                            pairBlue4.v2, pairBlue3.v1, pairBlue4.v1);
                                        vecListVlm[indBlueV3] =  v_sel2_geq_v_v_v_v(pairBlue3.v2,
                                            pairBlue4.v2, pairBlue3.v1, pairBlue4.v1);

                                        vecListVlm[indGreenV2] =  v_sel2_grt_v_v_v_v(pairGreen1.v2,
                                            pairGreen2.v2, pairGreen1.v1, pairGreen2.v1);
                                        vecListVlm[indGreenV1] =  v_sel2_leq_v_v_v_v(pairGreen1.v2,
                                            pairGreen2.v2, pairGreen1.v1, pairGreen2.v1);
                                        vecListVlm[indGreenV4] =  v_sel2_grt_v_v_v_v(pairGreen3.v2,
                                            pairGreen4.v2, pairGreen3.v1, pairGreen4.v1);
                                        vecListVlm[indGreenV3] =  v_sel2_leq_v_v_v_v(pairGreen3.v2,
                                            pairGreen4.v2, pairGreen3.v1, pairGreen4.v1);

                                        //advance offsets
                                        indBlueV1++;
                                        indBlueV2++;
                                        indBlueV3++;
                                        indBlueV4++;
                                        indGreenV1++;
                                        indGreenV2++;
                                        indGreenV3++;
                                        indGreenV4++;
                                    }
                                }
                                // Red box count doubles in each iteration
                                redBoxCount = (redBoxCount << 1);
                            }

                            // BM-64 on each vector (Steps 6.1 to 6.6) in
                            // alternate directions
                            for (int iVec = 0; iVec < nVecChunkBy2; iVec += 2)
                            {
                                /* Second set of blocks in the opposite direction*/
                                double_index_data_pair_t res1 = BitonicMerge64DualDir(
                                    vecListVlm[iVec].v2, vecListVlm[iVec].v1,
                                    vecListVlm[iVec + nVecChunkBy2].v2,
                                    vecListVlm[iVec + nVecChunkBy2].v1,
                                    lutTab0, lutTab1, lutTab2,
                                    predTab6_1, predTab6_2, predTab6_3,
                                    predTab6_4, predTab6_5, predTab6_6);

                                double_index_data_pair_t res2 = BitonicMerge64DualDir(
                                    vecListVlm[iVec + 1].v2, vecListVlm[iVec + 1].v1,
                                    vecListVlm[iVec + nVecChunkBy2 + 1].v2,
                                    vecListVlm[iVec + nVecChunkBy2 + 1].v1,
                                    lutTab0, lutTab1, lutTab2,
                                    predTab6_1, predTab6_2, predTab6_3,
                                    predTab6_4, predTab6_5, predTab6_6);

                                vecListVlm[iVec + 1]           = res2.p1;
                                vecListVlm[iVec + nVecChunkBy2 + 1] = res2.p2;
                                vecListVlm[iVec]               = res1.p1;
                                vecListVlm[iVec + nVecChunkBy2]     = res1.p2;
                            }
    #endif //BITONIC_BLOCK_N_MINUS_1
                        } // if (nVecChunk > 2)

                        // Bitonic block N need to be performed only when nVecChunk > 1
                        if (nVecChunk > 1)
                        {
    #ifdef BITONIC_BLOCK_N_STAGE_1_TO_N_MINUS_6
                            /* Bitonic Sort stages N.1 to N.(N-6) */
                            // Block N has one blue OR green box.

                            // Set direction and predicate patterns
                            // Use sortDir if this is the last node, else use iDir
                            curDir = s_i1_mov(sortDir, 0, iDir, isLastNode);

                            // Set correct direction for predicate patterns used in BM64 phase
                            // If curDir = 0, use as-is, else NOT.
                            predLastTab6_1 = v_i1_not_b(predTab6_1, 0, predTab6_1, curDir);
                            predLastTab6_2 = v_i1_not_b(predTab6_2, 0, predTab6_2, curDir);
                            predLastTab6_3 = v_i1_not_b(predTab6_3, 0, predTab6_3, curDir);
                            predLastTab6_4 = v_i1_not_b(predTab6_4, 0, predTab6_4, curDir);
                            predLastTab6_5 = v_i1_not_b(predTab6_5, 0, predTab6_5, curDir);
                            predLastTab6_6 = v_i1_not_b(predTab6_6, 0, predTab6_6, curDir);

                            /* Vector-to-Vector comparison and swap */
                            int redBoxCount = 1;
                            int v2offset = (nVecChunk >> 1);
                            int indBlueV1 = 0;
                            int indBlueV2 = v2offset;

                            if (curDir)
                            {
                                /* Stage N.1 */
                                bool predE = s_i32_cmp_grt((nVecChunk >> 1), 1);
                                bool predE2 = s_i32_cmp_grt((nVecChunk >> 1), 2);
                                // Perform all comparisons within the first stage
                                // 4 vector-to-vector comparisons within the only red box
                                for (int indV2VComp = 0; indV2VComp < (nVecChunk >> 1);
                                    indV2VComp += 4)
                                {
                                    index_data_pair_t pairBlue1 = vecListVlm[indBlueV1];
                                    index_data_pair_t pairBlue2 = vecListVlm[indBlueV2];
                                    index_data_pair_t pairBlue3 = vecListVlm[indBlueV1 + 1];
                                    index_data_pair_t pairBlue4 = vecListVlm[indBlueV2 + 1];
                                    index_data_pair_t pairBlue5 = vecListVlm[indBlueV1 + 2];
                                    index_data_pair_t pairBlue6 = vecListVlm[indBlueV2 + 2];
                                    index_data_pair_t pairBlue7 = vecListVlm[indBlueV1 + 3];
                                    index_data_pair_t pairBlue8 = vecListVlm[indBlueV2 + 3];

                                    index_data_pair_t pair1;
                                    index_data_pair_t pair2;
                                    index_data_pair_t pair3;
                                    index_data_pair_t pair4;
                                    index_data_pair_t pair5;
                                    index_data_pair_t pair6;
                                    index_data_pair_t pair7;
                                    index_data_pair_t pair8;

                                    pair1 = v_sel2_less_v_v_v_v(pairBlue1.v2, pairBlue2.v2,
                                        pairBlue1.v1, pairBlue2.v1);
                                    pair2 =  v_sel2_geq_v_v_v_v(pairBlue1.v2, pairBlue2.v2,
                                        pairBlue1.v1, pairBlue2.v1);
                                    pair3 = v_sel2_less_v_v_v_v(pairBlue3.v2, pairBlue4.v2,
                                        pairBlue3.v1, pairBlue4.v1);
                                    pair4 =  v_sel2_geq_v_v_v_v(pairBlue3.v2, pairBlue4.v2,
                                        pairBlue3.v1, pairBlue4.v1);
                                    pair5 = v_sel2_less_v_v_v_v(pairBlue5.v2, pairBlue6.v2,
                                        pairBlue5.v1, pairBlue6.v1);
                                    pair6 =  v_sel2_geq_v_v_v_v(pairBlue5.v2, pairBlue6.v2,
                                        pairBlue5.v1, pairBlue6.v1);
                                    pair7 = v_sel2_less_v_v_v_v(pairBlue7.v2, pairBlue8.v2,
                                        pairBlue7.v1, pairBlue8.v1);
                                    pair8 =  v_sel2_geq_v_v_v_v(pairBlue7.v2, pairBlue8.v2,
                                        pairBlue7.v1, pairBlue8.v1);

                                    vecListVlm[indBlueV2].v1 = pair1.v1;
                                    vecListVlm[indBlueV2].v2 = pair1.v2;
                                    vecListVlm[indBlueV1].v1 = pair2.v1;
                                    vecListVlm[indBlueV1].v2 = pair2.v2;

                                    vecListVlm[indBlueV2 + 1].v1 = pair3.v1;
                                    vecListVlm[indBlueV2 + 1].v2 = pair3.v2;
                                    vecListVlm[indBlueV1 + 1].v1 =
                                                        v_i32_mov_b(pair4.v1, 0, pair1.v1, predE);
                                    vecListVlm[indBlueV1 + 1].v2 = v_mov_v_b(pair4.v2, pair1.v2,
                                        predE, 0);

                                    vecListVlm[indBlueV2 + 2].v1 = pair5.v1;
                                    vecListVlm[indBlueV2 + 2].v2 = pair5.v2;
                                    vecListVlm[indBlueV1 + 2].v1 =
                                                        v_i32_mov_b(pair6.v1, 0, pair1.v1, predE2);
                                    vecListVlm[indBlueV1 + 2].v2 = v_mov_v_b(pair6.v2, pair1.v2,
                                        predE2, 0);

                                    vecListVlm[indBlueV2 + 3].v1 = pair7.v1;
                                    vecListVlm[indBlueV2 + 3].v2 = pair7.v2;
                                    vecListVlm[indBlueV1 + 3].v1 =
                                                        v_i32_mov_b(pair8.v1, 0, pair3.v1, predE2);
                                    vecListVlm[indBlueV1 + 3].v2 = v_mov_v_b(pair8.v2, pair3.v2,
                                        predE2, 0);

                                    // advance offsets
                                    indBlueV1 += 4;
                                    indBlueV2 += 4;
                                }

                                /* Stage N.2 */
                                redBoxCount = (redBoxCount << 1);
                                nVecChunkBy2 = (nVecChunk >> 1);
                                int nVecChunkBy4 = (nVecChunk >> 2);
                                v2offset = (nVecChunkBy2 >> 1);
                                indBlueV1 = 0;
                                indBlueV2 = v2offset;
                                int indBlueV3 = nVecChunkBy2;
                                int indBlueV4 = nVecChunkBy2 + v2offset;

                                predE = s_i32_cmp_grt((nVecChunkBy2 >> 1), 1);
                                // Perform all comparisons within the second stage
                                // 2 vector-to-vector comparisons each within the two red boxes
                                for (int indV2VComp = 0; indV2VComp < (nVecChunkBy2 >> 1);
                                    indV2VComp += 2)
                                {
                                    index_data_pair_t pairBlue1 = vecListVlm[indBlueV1];
                                    index_data_pair_t pairBlue2 = vecListVlm[indBlueV2];

                                    index_data_pair_t pairBlue3 = vecListVlm[indBlueV3];
                                    index_data_pair_t pairBlue4 = vecListVlm[indBlueV4];

                                    index_data_pair_t pairBlue5 = vecListVlm[indBlueV1 + 1];
                                    index_data_pair_t pairBlue6 = vecListVlm[indBlueV2 + 1];

                                    index_data_pair_t pairBlue7 = vecListVlm[indBlueV3 + 1];
                                    index_data_pair_t pairBlue8 = vecListVlm[indBlueV4 + 1];

                                    index_data_pair_t pair1;
                                    index_data_pair_t pair2;
                                    index_data_pair_t pair3;
                                    index_data_pair_t pair4;
                                    index_data_pair_t pair5;
                                    index_data_pair_t pair6;
                                    index_data_pair_t pair7;
                                    index_data_pair_t pair8;

                                    pair1 = v_sel2_less_v_v_v_v(pairBlue1.v2, pairBlue2.v2,
                                        pairBlue1.v1, pairBlue2.v1);
                                    pair2 =  v_sel2_geq_v_v_v_v(pairBlue1.v2, pairBlue2.v2,
                                        pairBlue1.v1, pairBlue2.v1);
                                    pair3 = v_sel2_less_v_v_v_v(pairBlue3.v2, pairBlue4.v2,
                                        pairBlue3.v1, pairBlue4.v1);
                                    pair4 =  v_sel2_geq_v_v_v_v(pairBlue3.v2, pairBlue4.v2,
                                        pairBlue3.v1, pairBlue4.v1);
                                    pair5 = v_sel2_less_v_v_v_v(pairBlue5.v2, pairBlue6.v2,
                                        pairBlue5.v1, pairBlue6.v1);
                                    pair6 =  v_sel2_geq_v_v_v_v(pairBlue5.v2, pairBlue6.v2,
                                        pairBlue5.v1, pairBlue6.v1);
                                    pair7 = v_sel2_less_v_v_v_v(pairBlue7.v2, pairBlue8.v2,
                                        pairBlue7.v1, pairBlue8.v1);
                                    pair8 =  v_sel2_geq_v_v_v_v(pairBlue7.v2, pairBlue8.v2,
                                        pairBlue7.v1, pairBlue8.v1);

                                    vecListVlm[indBlueV2].v1 = pair1.v1;
                                    vecListVlm[indBlueV2].v2 = pair1.v2;
                                    vecListVlm[indBlueV1].v1 = pair2.v1;
                                    vecListVlm[indBlueV1].v2 = pair2.v2;

                                    vecListVlm[indBlueV4].v1 = pair3.v1;
                                    vecListVlm[indBlueV4].v2 = pair3.v2;
                                    vecListVlm[indBlueV3].v1 = pair4.v1;
                                    vecListVlm[indBlueV3].v2 = pair4.v2;

                                    vecListVlm[indBlueV2 + 1].v1 =
                                                        v_i32_mov_b(pair5.v1, 0, pair4.v1, predE);
                                    vecListVlm[indBlueV2 + 1].v2 = v_mov_v_b(pair5.v2, pair4.v2,
                                        predE, 0);
                                    vecListVlm[indBlueV1 + 1].v1 =
                                                        v_i32_mov_b(pair6.v1, 0, pair1.v1, predE);
                                    vecListVlm[indBlueV1 + 1].v2 = v_mov_v_b(pair6.v2, pair1.v2,
                                        predE, 0);

                                    vecListVlm[indBlueV4 + 1].v1 = pair7.v1;
                                    vecListVlm[indBlueV4 + 1].v2 = pair7.v2;
                                    vecListVlm[indBlueV3 + 1].v1 =
                                                        v_i32_mov_b(pair8.v1, 0, pair3.v1, predE);
                                    vecListVlm[indBlueV3 + 1].v2 = v_mov_v_b(pair8.v2, pair3.v2,
                                        predE, 0);

                                    // advance offsets
                                    indBlueV1 += 2;
                                    indBlueV2 += 2;
                                    indBlueV3 += 2;
                                    indBlueV4 += 2;
                                }

                                /* Stage N.3 to N.(N-6)*/
                                redBoxCount = (redBoxCount << 1);
                                // Iterate over all red box sizes
                                for (int nVecRedBox = nVecChunkBy4; nVecRedBox > 1;
                                    nVecRedBox = (nVecRedBox >> 1))
                                {
                                    v2offset = (nVecRedBox >> 1);
                                    // Iterate over all red boxes with current size of nVecRedBox
                                    for (int indRedBox = 0; indRedBox < redBoxCount; indRedBox += 4)
                                    {
                                        indBlueV1 = indRedBox * nVecRedBox;
                                        indBlueV2 = indBlueV1 + v2offset;
                                        indBlueV3 = (indRedBox + 1) * nVecRedBox;
                                        indBlueV4 = indBlueV3 + v2offset;
                                        int indBlueV5 = (indRedBox + 2) * nVecRedBox;
                                        int indBlueV6 = indBlueV5 + v2offset;
                                        int indBlueV7 = (indRedBox + 3) * nVecRedBox;
                                        int indBlueV8 = indBlueV7 + v2offset;
                                        /* Perform all comparisons within the current stage
                                           One vector-to-vector comparison each in the 4 red boxes*/
                                        for (int indV2VComp = 0; indV2VComp < (nVecRedBox >> 1);
                                            indV2VComp++)
                                        {
                                            index_data_pair_t pairBlue1 = vecListVlm[indBlueV1];
                                            index_data_pair_t pairBlue2 = vecListVlm[indBlueV2];

                                            index_data_pair_t pairBlue3 = vecListVlm[indBlueV3];
                                            index_data_pair_t pairBlue4 = vecListVlm[indBlueV4];

                                            index_data_pair_t pairBlue5 = vecListVlm[indBlueV5];
                                            index_data_pair_t pairBlue6 = vecListVlm[indBlueV6];

                                            index_data_pair_t pairBlue7 = vecListVlm[indBlueV7];
                                            index_data_pair_t pairBlue8 = vecListVlm[indBlueV8];

                                            vecListVlm[indBlueV2] =
                                                    v_sel2_less_v_v_v_v(pairBlue1.v2, pairBlue2.v2,
                                                                        pairBlue1.v1, pairBlue2.v1);
                                            vecListVlm[indBlueV1] =
                                                    v_sel2_geq_v_v_v_v(pairBlue1.v2, pairBlue2.v2,
                                                                        pairBlue1.v1, pairBlue2.v1);
                                            vecListVlm[indBlueV4] =
                                                    v_sel2_less_v_v_v_v(pairBlue3.v2, pairBlue4.v2,
                                                                        pairBlue3.v1, pairBlue4.v1);
                                            vecListVlm[indBlueV3] =
                                                    v_sel2_geq_v_v_v_v(pairBlue3.v2, pairBlue4.v2,
                                                                        pairBlue3.v1, pairBlue4.v1);

                                            vecListVlm[indBlueV6] =
                                                    v_sel2_less_v_v_v_v(pairBlue5.v2, pairBlue6.v2,
                                                                        pairBlue5.v1, pairBlue6.v1);
                                            vecListVlm[indBlueV5] =
                                                    v_sel2_geq_v_v_v_v(pairBlue5.v2, pairBlue6.v2,
                                                                        pairBlue5.v1, pairBlue6.v1);
                                            vecListVlm[indBlueV8] =
                                                    v_sel2_less_v_v_v_v(pairBlue7.v2, pairBlue8.v2,
                                                                        pairBlue7.v1, pairBlue8.v1);
                                            vecListVlm[indBlueV7] =
                                                    v_sel2_geq_v_v_v_v(pairBlue7.v2, pairBlue8.v2,
                                                                        pairBlue7.v1, pairBlue8.v1);

                                            // advance offsets
                                            indBlueV1 ++;
                                            indBlueV2 ++;
                                            indBlueV3 ++;
                                            indBlueV4 ++;
                                            indBlueV5 ++;
                                            indBlueV6 ++;
                                            indBlueV7 ++;
                                            indBlueV8 ++;
                                        }
                                    }

                                    // Red box count doubles in each iteration
                                    redBoxCount = (redBoxCount << 1);
                                }
                            }
                            else
                            {
                                /* Stage N.1 */
                                bool predE = s_i32_cmp_grt((nVecChunk >> 1), 1);
                                bool predE2 = s_i32_cmp_grt((nVecChunk >> 1), 2);
                                /* Perform all comparisons within the first stage
                                   4 vector-to-vector comparisons within the only red box*/
                                for (int indV2VComp = 0; indV2VComp < (nVecChunk >> 1);
                                    indV2VComp += 4)
                                {
                                    index_data_pair_t pairBlue1 = vecListVlm[indBlueV1];
                                    index_data_pair_t pairBlue2 = vecListVlm[indBlueV2];
                                    index_data_pair_t pairBlue3 = vecListVlm[indBlueV1 + 1];
                                    index_data_pair_t pairBlue4 = vecListVlm[indBlueV2 + 1];
                                    index_data_pair_t pairBlue5 = vecListVlm[indBlueV1 + 2];
                                    index_data_pair_t pairBlue6 = vecListVlm[indBlueV2 + 2];
                                    index_data_pair_t pairBlue7 = vecListVlm[indBlueV1 + 3];
                                    index_data_pair_t pairBlue8 = vecListVlm[indBlueV2 + 3];

                                    index_data_pair_t pair1;
                                    index_data_pair_t pair2;
                                    index_data_pair_t pair3;
                                    index_data_pair_t pair4;
                                    index_data_pair_t pair5;
                                    index_data_pair_t pair6;
                                    index_data_pair_t pair7;
                                    index_data_pair_t pair8;

                                    pair1 =  v_sel2_grt_v_v_v_v(pairBlue1.v2, pairBlue2.v2,
                                        pairBlue1.v1, pairBlue2.v1);
                                    pair2 =  v_sel2_leq_v_v_v_v(pairBlue1.v2, pairBlue2.v2,
                                        pairBlue1.v1, pairBlue2.v1);
                                    pair3 =  v_sel2_grt_v_v_v_v(pairBlue3.v2, pairBlue4.v2,
                                        pairBlue3.v1, pairBlue4.v1);
                                    pair4 =  v_sel2_leq_v_v_v_v(pairBlue3.v2, pairBlue4.v2,
                                        pairBlue3.v1, pairBlue4.v1);
                                    pair5 =  v_sel2_grt_v_v_v_v(pairBlue5.v2, pairBlue6.v2,
                                        pairBlue5.v1, pairBlue6.v1);
                                    pair6 =  v_sel2_leq_v_v_v_v(pairBlue5.v2, pairBlue6.v2,
                                        pairBlue5.v1, pairBlue6.v1);
                                    pair7 =  v_sel2_grt_v_v_v_v(pairBlue7.v2, pairBlue8.v2,
                                        pairBlue7.v1, pairBlue8.v1);
                                    pair8 =  v_sel2_leq_v_v_v_v(pairBlue7.v2, pairBlue8.v2,
                                        pairBlue7.v1, pairBlue8.v1);

                                    vecListVlm[indBlueV2].v1 = pair1.v1;
                                    vecListVlm[indBlueV2].v2 = pair1.v2;
                                    vecListVlm[indBlueV1].v1 = pair2.v1;
                                    vecListVlm[indBlueV1].v2 = pair2.v2;

                                    vecListVlm[indBlueV2 + 1].v1 = pair3.v1;
                                    vecListVlm[indBlueV2 + 1].v2 = pair3.v2;
                                    vecListVlm[indBlueV1 + 1].v1 =
                                                        v_i32_mov_b(pair4.v1, 0, pair1.v1, predE);
                                    vecListVlm[indBlueV1 + 1].v2 = v_mov_v_b(pair4.v2, pair1.v2,
                                        predE, 0);

                                    vecListVlm[indBlueV2 + 2].v1 = pair5.v1;
                                    vecListVlm[indBlueV2 + 2].v2 = pair5.v2;
                                    vecListVlm[indBlueV1 + 2].v1 =
                                                        v_i32_mov_b(pair6.v1, 0, pair1.v1, predE2);
                                    vecListVlm[indBlueV1 + 2].v2 = v_mov_v_b(pair6.v2, pair1.v2,
                                        predE2, 0);

                                    vecListVlm[indBlueV2 + 3].v1 = pair7.v1;
                                    vecListVlm[indBlueV2 + 3].v2 = pair7.v2;
                                    vecListVlm[indBlueV1 + 3].v1 =
                                                        v_i32_mov_b(pair8.v1, 0, pair3.v1, predE2);
                                    vecListVlm[indBlueV1 + 3].v2 = v_mov_v_b(pair8.v2, pair3.v2,
                                        predE2, 0);

                                    // advance offsets
                                    indBlueV1 += 4;
                                    indBlueV2 += 4;
                                }

                                /* Stage N.2 */
                                redBoxCount = (redBoxCount << 1);
                                nVecChunkBy2 = (nVecChunk >> 1);
                                int nVecChunkBy4 = (nVecChunk >> 2);
                                v2offset = (nVecChunkBy2 >> 1);
                                indBlueV1 = 0;
                                indBlueV2 = v2offset;
                                int indBlueV3 = nVecChunkBy2;
                                int indBlueV4 = nVecChunkBy2 + v2offset;

                                predE = s_i32_cmp_grt((nVecChunkBy2 >> 1), 1);
                                /* Perform all comparisons within the second stage
                                   2 vector-to-vector comparisons each within the two red boxes*/
                                for (int indV2VComp = 0; indV2VComp < (nVecChunkBy2 >> 1);
                                    indV2VComp += 2)
                                {
                                    index_data_pair_t pairBlue1 = vecListVlm[indBlueV1];
                                    index_data_pair_t pairBlue2 = vecListVlm[indBlueV2];

                                    index_data_pair_t pairBlue3 = vecListVlm[indBlueV3];
                                    index_data_pair_t pairBlue4 = vecListVlm[indBlueV4];

                                    index_data_pair_t pairBlue5 = vecListVlm[indBlueV1 + 1];
                                    index_data_pair_t pairBlue6 = vecListVlm[indBlueV2 + 1];

                                    index_data_pair_t pairBlue7 = vecListVlm[indBlueV3 + 1];
                                    index_data_pair_t pairBlue8 = vecListVlm[indBlueV4 + 1];

                                    index_data_pair_t pair1;
                                    index_data_pair_t pair2;
                                    index_data_pair_t pair3;
                                    index_data_pair_t pair4;
                                    index_data_pair_t pair5;
                                    index_data_pair_t pair6;
                                    index_data_pair_t pair7;
                                    index_data_pair_t pair8;

                                    pair1 =  v_sel2_grt_v_v_v_v(pairBlue1.v2, pairBlue2.v2,
                                        pairBlue1.v1, pairBlue2.v1);
                                    pair2 =  v_sel2_leq_v_v_v_v(pairBlue1.v2, pairBlue2.v2,
                                        pairBlue1.v1, pairBlue2.v1);
                                    pair3 =  v_sel2_grt_v_v_v_v(pairBlue3.v2, pairBlue4.v2,
                                        pairBlue3.v1, pairBlue4.v1);
                                    pair4 =  v_sel2_leq_v_v_v_v(pairBlue3.v2, pairBlue4.v2,
                                        pairBlue3.v1, pairBlue4.v1);
                                    pair5 =  v_sel2_grt_v_v_v_v(pairBlue5.v2, pairBlue6.v2,
                                        pairBlue5.v1, pairBlue6.v1);
                                    pair6 =  v_sel2_leq_v_v_v_v(pairBlue5.v2, pairBlue6.v2,
                                        pairBlue5.v1, pairBlue6.v1);
                                    pair7 =  v_sel2_grt_v_v_v_v(pairBlue7.v2, pairBlue8.v2,
                                        pairBlue7.v1, pairBlue8.v1);
                                    pair8 =  v_sel2_leq_v_v_v_v(pairBlue7.v2, pairBlue8.v2,
                                        pairBlue7.v1, pairBlue8.v1);

                                    vecListVlm[indBlueV2].v1 = pair1.v1;
                                    vecListVlm[indBlueV2].v2 = pair1.v2;
                                    vecListVlm[indBlueV1].v1 = pair2.v1;
                                    vecListVlm[indBlueV1].v2 = pair2.v2;

                                    vecListVlm[indBlueV4].v1 = pair3.v1;
                                    vecListVlm[indBlueV4].v2 = pair3.v2;
                                    vecListVlm[indBlueV3].v1 = pair4.v1;
                                    vecListVlm[indBlueV3].v2 = pair4.v2;

                                    vecListVlm[indBlueV2 + 1].v1 =
                                                        v_i32_mov_b(pair5.v1, 0, pair4.v1, predE);
                                    vecListVlm[indBlueV2 + 1].v2 =
                                                            v_mov_v_b(pair5.v2, pair4.v2, predE, 0);
                                    vecListVlm[indBlueV1 + 1].v1 =
                                                        v_i32_mov_b(pair6.v1, 0, pair1.v1, predE);
                                    vecListVlm[indBlueV1 + 1].v2 =
                                                            v_mov_v_b(pair6.v2, pair1.v2, predE, 0);

                                    vecListVlm[indBlueV4 + 1].v1 = pair7.v1;
                                    vecListVlm[indBlueV4 + 1].v2 = pair7.v2;
                                    vecListVlm[indBlueV3 + 1].v1 =
                                                        v_i32_mov_b(pair8.v1, 0, pair3.v1, predE);
                                    vecListVlm[indBlueV3 + 1].v2 =
                                                            v_mov_v_b(pair8.v2, pair3.v2, predE, 0);

                                    // advance offsets
                                    indBlueV1 += 2;
                                    indBlueV2 += 2;
                                    indBlueV3 += 2;
                                    indBlueV4 += 2;
                                }

                                /* Stage N.3*/
                                redBoxCount = (redBoxCount << 1);
                                // Iterate over all red box sizes
                                for (int nVecRedBox = nVecChunkBy4; nVecRedBox > 1;
                                    nVecRedBox = (nVecRedBox >> 1))
                                {
                                    v2offset = (nVecRedBox >> 1);
                                    // Iterate over all red boxes with current size of nVecRedBox
                                    for (int indRedBox = 0; indRedBox < redBoxCount; indRedBox += 4)
                                    {
                                        indBlueV1 = indRedBox * nVecRedBox;
                                        indBlueV2 = indBlueV1 + v2offset;
                                        indBlueV3 = (indRedBox + 1) * nVecRedBox;
                                        indBlueV4 = indBlueV3 + v2offset;
                                        int indBlueV5 = (indRedBox + 2) * nVecRedBox;
                                        int indBlueV6 = indBlueV5 + v2offset;
                                        int indBlueV7 = (indRedBox + 3) * nVecRedBox;
                                        int indBlueV8 = indBlueV7 + v2offset;
                                        /* Perform all comparisons within the current stage
                                           One vector-to-vector comparison each in the 4 red boxes*/
                                        for (int indV2VComp = 0; indV2VComp < (nVecRedBox >> 1);
                                            indV2VComp++)
                                        {
                                            index_data_pair_t pairBlue1 = vecListVlm[indBlueV1];
                                            index_data_pair_t pairBlue2 = vecListVlm[indBlueV2];

                                            index_data_pair_t pairBlue3 = vecListVlm[indBlueV3];
                                            index_data_pair_t pairBlue4 = vecListVlm[indBlueV4];

                                            index_data_pair_t pairBlue5 = vecListVlm[indBlueV5];
                                            index_data_pair_t pairBlue6 = vecListVlm[indBlueV6];

                                            index_data_pair_t pairBlue7 = vecListVlm[indBlueV7];
                                            index_data_pair_t pairBlue8 = vecListVlm[indBlueV8];

                                            vecListVlm[indBlueV2] = v_sel2_grt_v_v_v_v(pairBlue1.v2,
                                                pairBlue2.v2, pairBlue1.v1, pairBlue2.v1);
                                            vecListVlm[indBlueV1] = v_sel2_leq_v_v_v_v(pairBlue1.v2,
                                                pairBlue2.v2, pairBlue1.v1, pairBlue2.v1);
                                            vecListVlm[indBlueV4] = v_sel2_grt_v_v_v_v(pairBlue3.v2,
                                                pairBlue4.v2, pairBlue3.v1, pairBlue4.v1);
                                            vecListVlm[indBlueV3] = v_sel2_leq_v_v_v_v(pairBlue3.v2,
                                                pairBlue4.v2, pairBlue3.v1, pairBlue4.v1);

                                            vecListVlm[indBlueV6] = v_sel2_grt_v_v_v_v(pairBlue5.v2,
                                                pairBlue6.v2, pairBlue5.v1, pairBlue6.v1);
                                            vecListVlm[indBlueV5] = v_sel2_leq_v_v_v_v(pairBlue5.v2,
                                                pairBlue6.v2, pairBlue5.v1, pairBlue6.v1);
                                            vecListVlm[indBlueV8] = v_sel2_grt_v_v_v_v(pairBlue7.v2,
                                                pairBlue8.v2, pairBlue7.v1, pairBlue8.v1);
                                            vecListVlm[indBlueV7] = v_sel2_leq_v_v_v_v(pairBlue7.v2,
                                                pairBlue8.v2, pairBlue7.v1, pairBlue8.v1);

                                            // advance offsets
                                            indBlueV1 ++;
                                            indBlueV2 ++;
                                            indBlueV3 ++;
                                            indBlueV4 ++;
                                            indBlueV5 ++;
                                            indBlueV6 ++;
                                            indBlueV7 ++;
                                            indBlueV8 ++;
                                        }
                                    }

                                    // Red box count doubles in each iteration
                                    redBoxCount = (redBoxCount << 1);
                                }
                            }

    #endif // BITONIC_BLOCK_N_STAGE_1_TO_N_MINUS_6

    #ifdef BITONIC_BLOCK_N_LAST_6_STAGES
                            /* BM-64 on each vector (Steps 6.1 to 6.6) in
                               original direction and store result.*/
                            for (int iVec = 0; iVec < nVecChunk; iVec += 4)
                            {
                                index_data_pair_t res1, res2, res3, res4;
                                // BM64 in original direction only
                                res1 = BitonicMerge64OrigDir(vecListVlm[iVec].v2,
                                    vecListVlm[iVec].v1, lutTab0, lutTab1, lutTab2,
                                    predLastTab6_1, predLastTab6_2, predLastTab6_3,
                                    predLastTab6_4, predLastTab6_5, predLastTab6_6);

                                res2 = BitonicMerge64OrigDir(vecListVlm[iVec + 1].v2,
                                    vecListVlm[iVec + 1].v1, lutTab0, lutTab1, lutTab2,
                                    predLastTab6_1, predLastTab6_2, predLastTab6_3,
                                    predLastTab6_4, predLastTab6_5, predLastTab6_6);

                                res3 = BitonicMerge64OrigDir(vecListVlm[iVec + 2].v2,
                                    vecListVlm[iVec + 2].v1, lutTab0, lutTab1, lutTab2,
                                    predLastTab6_1, predLastTab6_2, predLastTab6_3,
                                    predLastTab6_4, predLastTab6_5, predLastTab6_6);

                                res4 = BitonicMerge64OrigDir(vecListVlm[iVec + 3].v2,
                                    vecListVlm[iVec + 3].v1, lutTab0, lutTab1, lutTab2,
                                    predLastTab6_1, predLastTab6_2, predLastTab6_3,
                                    predLastTab6_4, predLastTab6_5, predLastTab6_6);

                                fmCoords1[depth] = d + iVec*VECTOR_SIZE;
                                fmCoords2[depth] = d + (iVec + 1) * VECTOR_SIZE;
                                fmCoords3[depth] = d + (iVec + 2) * VECTOR_SIZE;
                                fmCoords4[depth] = d + (iVec + 3) * VECTOR_SIZE;

                                // Store result to output tensor
                                st_tnsr_i_v(fmCoords1, ofmDataIdx, res1.v2);
                                st_tnsr_i_v_b(fmCoords2, ofmDataIdx, res2.v2, predGt64, 0);
                                st_tnsr_i_v_b(fmCoords3, ofmDataIdx, res3.v2, predGt128, 0);
                                st_tnsr_i_v_b(fmCoords4, ofmDataIdx, res4.v2, predGt128, 0);

                                v_i32_st_tnsr(fmCoords1, ofmIndexIdx, res1.v1);
                                v_i32_st_tnsr(fmCoords2, ofmIndexIdx, res2.v1, 0, predGt64);
                                v_i32_st_tnsr(fmCoords3, ofmIndexIdx, res3.v1, 0, predGt128);
                                v_i32_st_tnsr(fmCoords4, ofmIndexIdx, res4.v1, 0, predGt128);

                            }

    #endif //BITONIC_BLOCK_N_LAST_6_STAGES
                        } // if (nVecChunk > 1)

                        /* If chunksize = VECTOR_SIZE, there is no BM64 phase. Hence load
                           BS64 output and write to tensor
                           We use depthStep to compute this predicate in order to handle static and
                           dynamic chunk size together*/
                        fmCoords1[depth] = d;
                        bool predChunkSize64 = s_i32_cmp_eq(depthStep, VECTOR_SIZE);
                        // Store result to output tensor
                        st_tnsr_i_v_b(fmCoords1, ofmDataIdx, vecListVlm[0].v2, predChunkSize64, 0);
                        v_i32_st_tnsr(fmCoords1, ofmIndexIdx, vecListVlm[0].v1, 0, predChunkSize64);
                    }
                }
            }
        }
    }
}


/*
* This function sorts a pair of index-data vectors in alternate directions and stores in VLM*
* Sort is done w.r.t. data and indices are moved along with data.
*/
double_index_data_pair_t BitonicSort64(VECTOR data1, int64 idx1, VECTOR data2, int64 idx2,
    uchar256 lutTab0, uchar256 lutTab1, uchar256 lutTab2,
    bool256 predTab6_1, bool256 predTab6_2, bool256 predTab6_3,
    bool256 predTab6_4, bool256 predTab6_5, bool256 predTab6_6,
    int iVec)
{
    VECTOR shuffData1;
    int64 shuffIdx1;
    index_data_pair_t res1;
    VECTOR shuffData2;
    int64 shuffIdx2;
    index_data_pair_t res2;
    // Step 1.1 (0 -> 1)
    shuffData1 = v_shuffle_v_v_b(data1, lutTab0, 0);
    shuffIdx1 = v_i32_shuffle_b(idx1, lutTab0, 0, 0);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab[0], 0);
    shuffData2 = v_shuffle_v_v_b(data2, lutTab0, 0);
    shuffIdx2 = v_i32_shuffle_b(idx2, lutTab0, 0, 0);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab[0], 1);

    // Step 2.1 (0:1 -> 2:3)
    data1 = res1.v2;
    idx1 = res1.v1;
    shuffData1 = v_shuffle_v_v_b(data1, lutTab1, shuffData1);
    shuffIdx1 = v_i32_shuffle_b(idx1, lutTab1, 0, shuffIdx1);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab[1], 0);
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData2 = v_shuffle_v_v_b(data2, lutTab1, shuffData2);
    shuffIdx2 = v_i32_shuffle_b(idx2, lutTab1, 0, shuffIdx2);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab[1], 1);

    // Step 2.2 (0 -> 1)
    data1 = res1.v2;
    idx1 = res1.v1;
    shuffData1 = v_shuffle_v_v_b(data1, lutTab0, shuffData1);
    shuffIdx1 = v_i32_shuffle_b(idx1, lutTab0, 0, shuffIdx1);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab[2], 0);
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData2 = v_shuffle_v_v_b(data2, lutTab0, shuffData2);
    shuffIdx2 = v_i32_shuffle_b(idx2, lutTab0, 0, shuffIdx2);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab[2], 1);

    // Step 3.1 (0:3 -> 4:7)
    data1 = res1.v2;
    idx1 = res1.v1;
    shuffData1 = v_shuffle_v_v_b(data1, lutTab2, shuffData1);
    shuffIdx1 = v_i32_shuffle_b(idx1, lutTab2, 0, shuffIdx1);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab[3], 0);
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData2 = v_shuffle_v_v_b(data2, lutTab2, shuffData2);
    shuffIdx2 = v_i32_shuffle_b(idx2, lutTab2, 0, shuffIdx2);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab[3], 1);

    // Step 3.2 (0:1 -> 2:3)
    data1 = res1.v2;
    idx1 = res1.v1;
    shuffData1 = v_shuffle_v_v_b(data1, lutTab1, shuffData1);
    shuffIdx1 = v_i32_shuffle_b(idx1, lutTab1, 0, shuffIdx1);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab[4], 0);
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData2 = v_shuffle_v_v_b(data2, lutTab1, shuffData2);
    shuffIdx2 = v_i32_shuffle_b(idx2, lutTab1, 0, shuffIdx2);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab[4], 1);

    // Step 3.3 (0 -> 1)
    data1 = res1.v2;
    idx1 = res1.v1;
    shuffData1 = v_shuffle_v_v_b(data1, lutTab0, shuffData1);
    shuffIdx1 = v_i32_shuffle_b(idx1, lutTab0, 0, shuffIdx1);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab[5], 0);
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData2 = v_shuffle_v_v_b(data2, lutTab0, shuffData2);
    shuffIdx2 = v_i32_shuffle_b(idx2, lutTab0, 0, shuffIdx2);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab[5], 1);

    // Step 4.1 (0:7 -> 8:15), i.e., group0 -> group1 in all dual groups
    data1 = res1.v2;
    idx1 = res1.v1;
    shuffData1 = v_mov_group_v(data1, 0xFFFFFFFF, shuffData1, 0b11, 0b1111);
    shuffIdx1 = v_i32_mov_group_b(idx1, 0xFFFFFFFF, 63, shuffIdx1);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab[6], 0);
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData2 = v_mov_group_v(data2, 0xFFFFFFFF, shuffData2, 0b11, 0b1111);
    shuffIdx2 = v_i32_mov_group_b(idx2, 0xFFFFFFFF, 63, shuffIdx2);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab[6], 1);


    // Step 4.2 (0:3 -> 4:7)
    data1 = res1.v2;
    idx1 = res1.v1;
    shuffData1 = v_shuffle_v_v_b(data1, lutTab2, shuffData1);
    shuffIdx1 = v_i32_shuffle_b(idx1, lutTab2, 0, shuffIdx1);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab[7], 0);
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData2 = v_shuffle_v_v_b(data2, lutTab2, shuffData2);
    shuffIdx2 = v_i32_shuffle_b(idx2, lutTab2, 0, shuffIdx2);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab[7], 1);

    // Step 4.3 (0:1 -> 2:3)
    data1 = res1.v2;
    idx1 = res1.v1;
    shuffData1 = v_shuffle_v_v_b(data1, lutTab1, shuffData1);
    shuffIdx1 = v_i32_shuffle_b(idx1, lutTab1, 0, shuffIdx1);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab[8], 0);
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData2 = v_shuffle_v_v_b(data2, lutTab1, shuffData2);
    shuffIdx2 = v_i32_shuffle_b(idx2, lutTab1, 0, shuffIdx2);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab[8], 1);

    // Step 4.4 (0 -> 1)
    data1 = res1.v2;
    idx1 = res1.v1;
    shuffData1 = v_shuffle_v_v_b(data1, lutTab0, shuffData1);
    shuffIdx1 = v_i32_shuffle_b(idx1, lutTab0, 0, shuffIdx1);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab[9], 0);
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData2 = v_shuffle_v_v_b(data2, lutTab0, shuffData2);
    shuffIdx2 = v_i32_shuffle_b(idx2, lutTab0, 0, shuffIdx2);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab[9], 1);

    // Step 5.1 (0:15 -> 16:31), i.e., dual group 0->1, 2->3
    data1 = res1.v2;
    idx1 = res1.v1;

    shuffData1 = v_mov_dual_group_v(data1, 0xFFFFFFFF, shuffData1, 0, 1, TRUE, TRUE);
    shuffIdx1 = v_i32_mov_dual_group_b(idx1, 0xFFFFFFFF, 0, 1, MkWr(TRUE, TRUE), shuffIdx1);
    shuffData1 = v_mov_dual_group_v(data1, 0xFFFFFFFF, shuffData1, 1, 0, TRUE, TRUE);
    shuffIdx1 = v_i32_mov_dual_group_b(idx1, 0xFFFFFFFF, 1, 0, MkWr(TRUE, TRUE), shuffIdx1);
    shuffData1 = v_mov_dual_group_v(data1, 0xFFFFFFFF, shuffData1, 2, 3, TRUE, TRUE);
    shuffIdx1 = v_i32_mov_dual_group_b(idx1, 0xFFFFFFFF, 2, 3, MkWr(TRUE, TRUE), shuffIdx1);
    shuffData1 = v_mov_dual_group_v(data1, 0xFFFFFFFF, shuffData1, 3, 2, TRUE, TRUE);
    shuffIdx1 = v_i32_mov_dual_group_b(idx1, 0xFFFFFFFF, 3, 2, MkWr(TRUE, TRUE), shuffIdx1);

    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab[10], 0);
    data2 = res2.v2;
    idx2 = res2.v1;

    shuffData2 = v_mov_dual_group_v(data2, 0xFFFFFFFF, shuffData2, 0, 1, TRUE, TRUE);
    shuffIdx2 = v_i32_mov_dual_group_b(idx2, 0xFFFFFFFF, 0, 1, MkWr(TRUE, TRUE), shuffIdx2);
    shuffData2 = v_mov_dual_group_v(data2, 0xFFFFFFFF, shuffData2, 1, 0, TRUE, TRUE);
    shuffIdx2 = v_i32_mov_dual_group_b(idx2, 0xFFFFFFFF, 1, 0, MkWr(TRUE, TRUE), shuffIdx2);
    shuffData2 = v_mov_dual_group_v(data2, 0xFFFFFFFF, shuffData2, 2, 3, TRUE, TRUE);
    shuffIdx2 = v_i32_mov_dual_group_b(idx2, 0xFFFFFFFF, 2, 3, MkWr(TRUE, TRUE), shuffIdx2);
    shuffData2 = v_mov_dual_group_v(data2, 0xFFFFFFFF, shuffData2, 3, 2, TRUE, TRUE);
    shuffIdx2 = v_i32_mov_dual_group_b(idx2, 0xFFFFFFFF, 3, 2, MkWr(TRUE, TRUE), shuffIdx2);

    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab[10], 1);

    // Step 5.2 (0:7 -> 8:15), i.e., group0 -> group1 in all dual groups
    data1 = res1.v2;
    idx1 = res1.v1;
    shuffData1 = v_mov_group_v(data1, 0xFFFFFFFF, shuffData1, 0b11, 0b1111);
    shuffIdx1 = v_i32_mov_group_b(idx1, 0xFFFFFFFF, 63, shuffIdx1);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab[11], 0);
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData2 = v_mov_group_v(data2, 0xFFFFFFFF, shuffData2, 0b11, 0b1111);
    shuffIdx2 = v_i32_mov_group_b(idx2, 0xFFFFFFFF, 63, shuffIdx2);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab[11], 1);

    // Step 5.3 (0:3 -> 4:7)
    data1 = res1.v2;
    idx1 = res1.v1;
    shuffData1 = v_shuffle_v_v_b(data1, lutTab2, shuffData1);
    shuffIdx1 = v_i32_shuffle_b(idx1, lutTab2, 0, shuffIdx1);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab[12], 0);
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData2 = v_shuffle_v_v_b(data2, lutTab2, shuffData2);
    shuffIdx2 = v_i32_shuffle_b(idx2, lutTab2, 0, shuffIdx2);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab[12], 1);

    // Step 5.4 (0:1 -> 2:3)
    data1 = res1.v2;
    idx1 = res1.v1;
    shuffData1 = v_shuffle_v_v_b(data1, lutTab1, shuffData1);
    shuffIdx1 = v_i32_shuffle_b(idx1, lutTab1, 0, shuffIdx1);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab[13], 0);
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData2 = v_shuffle_v_v_b(data2, lutTab1, shuffData2);
    shuffIdx2 = v_i32_shuffle_b(idx2, lutTab1, 0, shuffIdx2);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab[13], 1);

    // Step 5.5 (0 -> 1)
    data1 = res1.v2;
    idx1 = res1.v1;
    shuffData1 = v_shuffle_v_v_b(data1, lutTab0, shuffData1);
    shuffIdx1 = v_i32_shuffle_b(idx1, lutTab0, 0, shuffIdx1);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab[14], 0);
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData2 = v_shuffle_v_v_b(data2, lutTab0, shuffData2);
    shuffIdx2 = v_i32_shuffle_b(idx2, lutTab0, 0, shuffIdx2);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab[14], 1);


    // Step 6.1 (0:31 -> 32:63), i.e., dualgroup 0:1 -> 2:3
    data1 = res1.v2;
    idx1 = res1.v1;

    shuffData1 = v_mov_dual_group_v(data1, 0xFFFFFFFF, shuffData1, 0, 2, TRUE, TRUE);
    shuffIdx1 = v_i32_mov_dual_group_b(idx1, 0xFFFFFFFF, 0, 2, MkWr(TRUE, TRUE), shuffIdx1);
    shuffData1 = v_mov_dual_group_v(data1, 0xFFFFFFFF, shuffData1, 2, 0, TRUE, TRUE);
    shuffIdx1 = v_i32_mov_dual_group_b(idx1, 0xFFFFFFFF, 2, 0, MkWr(TRUE, TRUE), shuffIdx1);
    shuffData1 = v_mov_dual_group_v(data1, 0xFFFFFFFF, shuffData1, 1, 3, TRUE, TRUE);
    shuffIdx1 = v_i32_mov_dual_group_b(idx1, 0xFFFFFFFF, 1, 3, MkWr(TRUE, TRUE), shuffIdx1);
    shuffData1 = v_mov_dual_group_v(data1, 0xFFFFFFFF, shuffData1, 3, 1, TRUE, TRUE);
    shuffIdx1 = v_i32_mov_dual_group_b(idx1, 0xFFFFFFFF, 3, 1, MkWr(TRUE, TRUE), shuffIdx1);

    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab6_1, 0);
    data2 = res2.v2;
    idx2 = res2.v1;

    shuffData2 = v_mov_dual_group_v(data2, 0xFFFFFFFF, shuffData2, 0, 2, TRUE, TRUE);
    shuffIdx2 = v_i32_mov_dual_group_b(idx2, 0xFFFFFFFF, 0, 2, MkWr(TRUE, TRUE), shuffIdx2);
    shuffData2 = v_mov_dual_group_v(data2, 0xFFFFFFFF, shuffData2, 2, 0, TRUE, TRUE);
    shuffIdx2 = v_i32_mov_dual_group_b(idx2, 0xFFFFFFFF, 2, 0, MkWr(TRUE, TRUE), shuffIdx2);
    shuffData2 = v_mov_dual_group_v(data2, 0xFFFFFFFF, shuffData2, 1, 3, TRUE, TRUE);
    shuffIdx2 = v_i32_mov_dual_group_b(idx2, 0xFFFFFFFF, 1, 3, MkWr(TRUE, TRUE), shuffIdx2);
    shuffData2 = v_mov_dual_group_v(data2, 0xFFFFFFFF, shuffData2, 3, 1, TRUE, TRUE);
    shuffIdx2 = v_i32_mov_dual_group_b(idx2, 0xFFFFFFFF, 3, 1, MkWr(TRUE, TRUE), shuffIdx2);

    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab6_1, 1);

    // Step 6.2 (0:15 -> 16:31), i.e., dual group 0->1, 2->3
    data1 = res1.v2;
    idx1 = res1.v1;

    shuffData1 = v_mov_dual_group_v(data1, 0xFFFFFFFF, shuffData1, 0, 1, TRUE, TRUE);
    shuffIdx1 = v_i32_mov_dual_group_b(idx1, 0xFFFFFFFF, 0, 1, MkWr(TRUE, TRUE), shuffIdx1);
    shuffData1 = v_mov_dual_group_v(data1, 0xFFFFFFFF, shuffData1, 1, 0, TRUE, TRUE);
    shuffIdx1 = v_i32_mov_dual_group_b(idx1, 0xFFFFFFFF, 1, 0, MkWr(TRUE, TRUE), shuffIdx1);
    shuffData1 = v_mov_dual_group_v(data1, 0xFFFFFFFF, shuffData1, 2, 3, TRUE, TRUE);
    shuffIdx1 = v_i32_mov_dual_group_b(idx1, 0xFFFFFFFF, 2, 3, MkWr(TRUE, TRUE), shuffIdx1);
    shuffData1 = v_mov_dual_group_v(data1, 0xFFFFFFFF, shuffData1, 3, 2, TRUE, TRUE);
    shuffIdx1 = v_i32_mov_dual_group_b(idx1, 0xFFFFFFFF, 3, 2, MkWr(TRUE, TRUE), shuffIdx1);

    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab6_2, 0);
    data2 = res2.v2;
    idx2 = res2.v1;

    shuffData2 = v_mov_dual_group_v(data2, 0xFFFFFFFF, shuffData2, 0, 1, TRUE, TRUE);
    shuffIdx2 = v_i32_mov_dual_group_b(idx2, 0xFFFFFFFF, 0, 1, MkWr(TRUE, TRUE), shuffIdx2);
    shuffData2 = v_mov_dual_group_v(data2, 0xFFFFFFFF, shuffData2, 1, 0, TRUE, TRUE);
    shuffIdx2 = v_i32_mov_dual_group_b(idx2, 0xFFFFFFFF, 1, 0, MkWr(TRUE, TRUE), shuffIdx2);
    shuffData2 = v_mov_dual_group_v(data2, 0xFFFFFFFF, shuffData2, 2, 3, TRUE, TRUE);
    shuffIdx2 = v_i32_mov_dual_group_b(idx2, 0xFFFFFFFF, 2, 3, MkWr(TRUE, TRUE), shuffIdx2);
    shuffData2 = v_mov_dual_group_v(data2, 0xFFFFFFFF, shuffData2, 3, 2, TRUE, TRUE);
    shuffIdx2 = v_i32_mov_dual_group_b(idx2, 0xFFFFFFFF, 3, 2, MkWr(TRUE, TRUE), shuffIdx2);

    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab6_2, 1);

    // Step 6.3 (0:7 -> 8:15), i.e., group0 -> group1 in all dual groups
    data1 = res1.v2;
    idx1 = res1.v1;
    shuffData1 = v_mov_group_v(data1, 0xFFFFFFFF, shuffData1, 0b11, 0b1111);
    shuffIdx1 = v_i32_mov_group_b(idx1, 0xFFFFFFFF, 63, shuffIdx1);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab6_3, 0);
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData2 = v_mov_group_v(data2, 0xFFFFFFFF, shuffData2, 0b11, 0b1111);
    shuffIdx2 = v_i32_mov_group_b(idx2, 0xFFFFFFFF, 63, shuffIdx2);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab6_3, 1);

    // Step 6.4 (0:3 -> 4:7)
    data1 = res1.v2;
    idx1 = res1.v1;
    shuffData1 = v_shuffle_v_v_b(data1, lutTab2, shuffData1);
    shuffIdx1 = v_i32_shuffle_b(idx1, lutTab2, 0, shuffIdx1);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab6_4, 0);
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData2 = v_shuffle_v_v_b(data2, lutTab2, shuffData2);
    shuffIdx2 = v_i32_shuffle_b(idx2, lutTab2, 0, shuffIdx2);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab6_4, 1);

    // Step 6.5 (0:1 -> 2:3)
    data1 = res1.v2;
    idx1 = res1.v1;
    shuffData1 = v_shuffle_v_v_b(data1, lutTab1, shuffData1);
    shuffIdx1 = v_i32_shuffle_b(idx1, lutTab1, 0, shuffIdx1);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab6_5, 0);
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData2 = v_shuffle_v_v_b(data2, lutTab1, shuffData2);
    shuffIdx2 = v_i32_shuffle_b(idx2, lutTab1, 0, shuffIdx2);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab6_5, 1);

    // Step 6.6 (0 -> 1)
    data1 = res1.v2;
    idx1 = res1.v1;
    shuffData1 = v_shuffle_v_v_b(data1, lutTab0, shuffData1);
    shuffIdx1 = v_i32_shuffle_b(idx1, lutTab0, 0, shuffIdx1);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab6_6, 0);
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData2 = v_shuffle_v_v_b(data2, lutTab0, shuffData2);
    shuffIdx2 = v_i32_shuffle_b(idx2, lutTab0, 0, shuffIdx2);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab6_6, 1);

    double_index_data_pair_t p;
    p.p1 = res1;
    p.p2 = res2;

    return p;
}

double_index_data_pair_t BitonicMerge64DualDir(VECTOR data1, int64 idx1, VECTOR data2, int64 idx2,
    uchar256 lutTab0, uchar256 lutTab1, uchar256 lutTab2,
    bool256 predTab6_1, bool256 predTab6_2, bool256 predTab6_3,
    bool256 predTab6_4, bool256 predTab6_5, bool256 predTab6_6)
{
    VECTOR shuffData1;
    int64 shuffIdx1;
    VECTOR shuffData2;
    int64 shuffIdx2;
    index_data_pair_t res1, res2;
    // Step 6.1 (0:31 -> 32:63), i.e., dualgroup 0:1 -> 2:3
    shuffData1 = v_mov_dual_group_v(data1, 0xFFFFFFFF, 0, 0, 2, TRUE, TRUE);
    shuffIdx1 = v_i32_mov_dual_group_b(idx1, 0xFFFFFFFF, 0, 2, MkWr(TRUE, TRUE), 0);
    shuffData2 = v_mov_dual_group_v(data2, 0xFFFFFFFF, 0, 0, 2, TRUE, TRUE);
    shuffIdx2 = v_i32_mov_dual_group_b(idx2, 0xFFFFFFFF, 0, 2, MkWr(TRUE, TRUE), 0);
    shuffData1 = v_mov_dual_group_v(data1, 0xFFFFFFFF, shuffData1, 2, 0, TRUE, TRUE);
    shuffIdx1 = v_i32_mov_dual_group_b(idx1, 0xFFFFFFFF, 2, 0, MkWr(TRUE, TRUE), shuffIdx1);
    shuffData2 = v_mov_dual_group_v(data2, 0xFFFFFFFF, shuffData2, 2, 0, TRUE, TRUE);
    shuffIdx2 = v_i32_mov_dual_group_b(idx2, 0xFFFFFFFF, 2, 0, MkWr(TRUE, TRUE), shuffIdx2);
    shuffData1 = v_mov_dual_group_v(data1, 0xFFFFFFFF, shuffData1, 1, 3, TRUE, TRUE);
    shuffIdx1 = v_i32_mov_dual_group_b(idx1, 0xFFFFFFFF, 1, 3, MkWr(TRUE, TRUE), shuffIdx1);
    shuffData2 = v_mov_dual_group_v(data2, 0xFFFFFFFF, shuffData2, 1, 3, TRUE, TRUE);
    shuffIdx2 = v_i32_mov_dual_group_b(idx2, 0xFFFFFFFF, 1, 3, MkWr(TRUE, TRUE), shuffIdx2);
    shuffData1 = v_mov_dual_group_v(data1, 0xFFFFFFFF, shuffData1, 3, 1, TRUE, TRUE);
    shuffIdx1 = v_i32_mov_dual_group_b(idx1, 0xFFFFFFFF, 3, 1, MkWr(TRUE, TRUE), shuffIdx1);
    shuffData2 = v_mov_dual_group_v(data2, 0xFFFFFFFF, shuffData2, 3, 1, TRUE, TRUE);
    shuffIdx2 = v_i32_mov_dual_group_b(idx2, 0xFFFFFFFF, 3, 1, MkWr(TRUE, TRUE), shuffIdx2);

    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab6_1, 0);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab6_1, 1);

    // Step 6.2 (0:15 -> 16:31), i.e., dual group 0->1, 2->3
    data1 = res1.v2;
    idx1 = res1.v1;
    data2 = res2.v2;
    idx2 = res2.v1;

    shuffData1 = v_mov_dual_group_v(data1, 0xFFFFFFFF, shuffData1, 0, 1, TRUE, TRUE);
    shuffIdx1 = v_i32_mov_dual_group_b(idx1, 0xFFFFFFFF, 0, 1, MkWr(TRUE, TRUE), shuffIdx1);
    shuffData2 = v_mov_dual_group_v(data2, 0xFFFFFFFF, shuffData2, 0, 1, TRUE, TRUE);
    shuffIdx2 = v_i32_mov_dual_group_b(idx2, 0xFFFFFFFF, 0, 1, MkWr(TRUE, TRUE), shuffIdx2);
    shuffData1 = v_mov_dual_group_v(data1, 0xFFFFFFFF, shuffData1, 1, 0, TRUE, TRUE);
    shuffIdx1 = v_i32_mov_dual_group_b(idx1, 0xFFFFFFFF, 1, 0, MkWr(TRUE, TRUE), shuffIdx1);
    shuffData2 = v_mov_dual_group_v(data2, 0xFFFFFFFF, shuffData2, 1, 0, TRUE, TRUE);
    shuffIdx2 = v_i32_mov_dual_group_b(idx2, 0xFFFFFFFF, 1, 0, MkWr(TRUE, TRUE), shuffIdx2);
    shuffData1 = v_mov_dual_group_v(data1, 0xFFFFFFFF, shuffData1, 2, 3, TRUE, TRUE);
    shuffIdx1 = v_i32_mov_dual_group_b(idx1, 0xFFFFFFFF, 2, 3, MkWr(TRUE, TRUE), shuffIdx1);
    shuffData2 = v_mov_dual_group_v(data2, 0xFFFFFFFF, shuffData2, 2, 3, TRUE, TRUE);
    shuffIdx2 = v_i32_mov_dual_group_b(idx2, 0xFFFFFFFF, 2, 3, MkWr(TRUE, TRUE), shuffIdx2);
    shuffData1 = v_mov_dual_group_v(data1, 0xFFFFFFFF, shuffData1, 3, 2, TRUE, TRUE);
    shuffIdx1 = v_i32_mov_dual_group_b(idx1, 0xFFFFFFFF, 3, 2, MkWr(TRUE, TRUE), shuffIdx1);
    shuffData2 = v_mov_dual_group_v(data2, 0xFFFFFFFF, shuffData2, 3, 2, TRUE, TRUE);
    shuffIdx2 = v_i32_mov_dual_group_b(idx2, 0xFFFFFFFF, 3, 2, MkWr(TRUE, TRUE), shuffIdx2);

    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab6_2, 0);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab6_2, 1);

    // Step 6.3 (0:7 -> 8:15), i.e., group0 -> group1 in all dual groups
    data1 = res1.v2;
    idx1 = res1.v1;
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData1 = v_mov_group_v(data1, 0xFFFFFFFF, shuffData1, 0b11, 0b1111);
    shuffIdx1 = v_i32_mov_group_b(idx1, 0xFFFFFFFF, 63, shuffIdx1);
    shuffData2 = v_mov_group_v(data2, 0xFFFFFFFF, shuffData2, 0b11, 0b1111);
    shuffIdx2 = v_i32_mov_group_b(idx2, 0xFFFFFFFF, 63, shuffIdx2);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab6_3, 0);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab6_3, 1);

    // Step 6.4 (0:3 -> 4:7)
    data1 = res1.v2;
    idx1 = res1.v1;
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData1 = v_shuffle_v_v_b(data1, lutTab2, shuffData1);
    shuffIdx1 = v_i32_shuffle_b(idx1, lutTab2, 0, shuffIdx1);
    shuffData2 = v_shuffle_v_v_b(data2, lutTab2, shuffData2);
    shuffIdx2 = v_i32_shuffle_b(idx2, lutTab2, 0, shuffIdx2);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab6_4, 0);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab6_4, 1);

    // Step 6.5 (0:1 -> 2:3)
    data1 = res1.v2;
    idx1 = res1.v1;
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData1 = v_shuffle_v_v_b(data1, lutTab1, shuffData1);
    shuffIdx1 = v_i32_shuffle_b(idx1, lutTab1, 0, shuffIdx1);
    shuffData2 = v_shuffle_v_v_b(data2, lutTab1, shuffData2);
    shuffIdx2 = v_i32_shuffle_b(idx2, lutTab1, 0, shuffIdx2);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab6_5, 0);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab6_5, 1);

    // Step 6.6 (0 -> 1)
    data1 = res1.v2;
    idx1 = res1.v1;
    data2 = res2.v2;
    idx2 = res2.v1;
    shuffData1 = v_shuffle_v_v_b(data1, lutTab0, shuffData1);
    shuffIdx1 = v_i32_shuffle_b(idx1, lutTab0, 0, shuffIdx1);
    shuffData2 = v_shuffle_v_v_b(data2, lutTab0, shuffData2);
    shuffIdx2 = v_i32_shuffle_b(idx2, lutTab0, 0, shuffIdx2);
    res1 =  v_sel2_leq_v_v_v_v(data1, shuffData1, idx1, shuffIdx1);
    res1 = v_sel2_geq_v_v_v_v_vb(data1, shuffData1, idx1, shuffIdx1, res1, predTab6_6, 0);
    res2 =  v_sel2_leq_v_v_v_v(data2, shuffData2, idx2, shuffIdx2);
    res2 = v_sel2_geq_v_v_v_v_vb(data2, shuffData2, idx2, shuffIdx2, res2, predTab6_6, 1);

    double_index_data_pair_t res;
    res.p1 = res1;
    res.p2 = res2;

    return res;
}

index_data_pair_t BitonicMerge64OrigDir(VECTOR data, int64 idx,
    uchar256 lutTab0, uchar256 lutTab1, uchar256 lutTab2,
    bool256 predTab6_1, bool256 predTab6_2, bool256 predTab6_3,
    bool256 predTab6_4, bool256 predTab6_5, bool256 predTab6_6)
{
    VECTOR shuffData;
    int64 shuffIdx;
    index_data_pair_t res;
    // Step 6.1 (0:31 -> 32:63), i.e., dualgroup 0:1 -> 2:3

    shuffData = v_mov_dual_group_v(data, 0xFFFFFFFF, 0, 0, 2, TRUE, TRUE);
    shuffIdx = v_i32_mov_dual_group_b(idx, 0xFFFFFFFF, 0, 2, MkWr(TRUE, TRUE), 0);
    shuffData = v_mov_dual_group_v(data, 0xFFFFFFFF, shuffData, 2, 0, TRUE, TRUE);
    shuffIdx = v_i32_mov_dual_group_b(idx, 0xFFFFFFFF, 2, 0, MkWr(TRUE, TRUE), shuffIdx);
    shuffData = v_mov_dual_group_v(data, 0xFFFFFFFF, shuffData, 1, 3, TRUE, TRUE);
    shuffIdx = v_i32_mov_dual_group_b(idx, 0xFFFFFFFF, 1, 3, MkWr(TRUE, TRUE), shuffIdx);
    shuffData = v_mov_dual_group_v(data, 0xFFFFFFFF, shuffData, 3, 1, TRUE, TRUE);
    shuffIdx = v_i32_mov_dual_group_b(idx, 0xFFFFFFFF, 3, 1, MkWr(TRUE, TRUE), shuffIdx);

    res =  v_sel2_leq_v_v_v_v(data, shuffData, idx, shuffIdx);
    res = v_sel2_geq_v_v_v_v_vb(data, shuffData, idx, shuffIdx, res, predTab6_1, ORIG_DIR);

    // Step 6.2 (0:15 -> 16:31), i.e., dual group 0->1, 2->3
    data = res.v2;
    idx = res.v1;

    shuffData = v_mov_dual_group_v(data, 0xFFFFFFFF, shuffData, 0, 1, TRUE, TRUE);
    shuffIdx = v_i32_mov_dual_group_b(idx, 0xFFFFFFFF, 0, 1, MkWr(TRUE, TRUE), shuffIdx);
    shuffData = v_mov_dual_group_v(data, 0xFFFFFFFF, shuffData, 1, 0, TRUE, TRUE);
    shuffIdx = v_i32_mov_dual_group_b(idx, 0xFFFFFFFF, 1, 0, MkWr(TRUE, TRUE), shuffIdx);
    shuffData = v_mov_dual_group_v(data, 0xFFFFFFFF, shuffData, 2, 3, TRUE, TRUE);
    shuffIdx = v_i32_mov_dual_group_b(idx, 0xFFFFFFFF, 2, 3, MkWr(TRUE, TRUE), shuffIdx);
    shuffData = v_mov_dual_group_v(data, 0xFFFFFFFF, shuffData, 3, 2, TRUE, TRUE);
    shuffIdx = v_i32_mov_dual_group_b(idx, 0xFFFFFFFF, 3, 2, MkWr(TRUE, TRUE), shuffIdx);


    res =  v_sel2_leq_v_v_v_v(data, shuffData, idx, shuffIdx);
    res = v_sel2_geq_v_v_v_v_vb(data, shuffData, idx, shuffIdx, res, predTab6_2, ORIG_DIR);

    // Step 6.3 (0:7 -> 8:15), i.e., group0 -> group1 in all dual groups
    data = res.v2;
    idx = res.v1;
    shuffData = v_mov_group_v(data, 0xFFFFFFFF, shuffData, 0b11, 0b1111);
    shuffIdx = v_i32_mov_group_b(idx, 0xFFFFFFFF, 63, shuffIdx);
    res =  v_sel2_leq_v_v_v_v(data, shuffData, idx, shuffIdx);
    res = v_sel2_geq_v_v_v_v_vb(data, shuffData, idx, shuffIdx, res, predTab6_3, ORIG_DIR);

    // Step 6.4 (0:3 -> 4:7)
    data = res.v2;
    idx = res.v1;
    shuffData = v_shuffle_v_v_b(data, lutTab2, shuffData);
    shuffIdx = v_i32_shuffle_b(idx, lutTab2, 0, shuffIdx);
    res =  v_sel2_leq_v_v_v_v(data, shuffData, idx, shuffIdx);
    res = v_sel2_geq_v_v_v_v_vb(data, shuffData, idx, shuffIdx, res, predTab6_4, ORIG_DIR);

    // Step 6.5 (0:1 -> 2:3)
    data = res.v2;
    idx = res.v1;
    shuffData = v_shuffle_v_v_b(data, lutTab1, shuffData);
    shuffIdx = v_i32_shuffle_b(idx, lutTab1, 0, shuffIdx);
    res =  v_sel2_leq_v_v_v_v(data, shuffData, idx, shuffIdx);
    res = v_sel2_geq_v_v_v_v_vb(data, shuffData, idx, shuffIdx, res, predTab6_5, ORIG_DIR);

    // Step 6.6 (0 -> 1)
    data = res.v2;
    idx = res.v1;
    shuffData = v_shuffle_v_v_b(data, lutTab0, shuffData);
    shuffIdx = v_i32_shuffle_b(idx, lutTab0, 0, shuffIdx);
    res =  v_sel2_leq_v_v_v_v(data, shuffData, idx, shuffIdx);
    res = v_sel2_geq_v_v_v_v_vb(data, shuffData, idx, shuffIdx, res, predTab6_6, ORIG_DIR);

    return res;
}