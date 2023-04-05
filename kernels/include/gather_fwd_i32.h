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


/* This kernel supports output rank upto 5D and has st_partial usage instead of config_add_one_dim
 * and st_low usage */
//#pragma tpc_printf (enable)
void main(tensor input, tensor indices, tensor output, int axis)
{
    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd   = get_index_space_size() + indexSpaceStart;

    int5 dataCoords0 = { 0 };
    int5 dataCoords1 = { 0 };
    int5 dataCoords2 = { 0 };
    int5 dataCoords3 = { 0 };

    const int dim0 = 0;
    const int dim1 = 1;
    const int dim2 = 2;
    const int dim3 = 3;
    const int dim4 = 4;

    const int depthStep   = 64;
    const int depthStart  = indexSpaceStart[dim0] * depthStep;
    const int depthEnd    = indexSpaceEnd[dim0] * depthStep;

    const int widthStep   = 4;
    const int widthStart  = indexSpaceStart[dim1] * widthStep;
    const int widthEnd    = indexSpaceEnd[dim1] * widthStep;

    const int heightStep  = 1;
    const int heightStart = indexSpaceStart[dim2];
    const int heightEnd   = indexSpaceEnd[dim2];

    const int batchStep   = 1;
    const int batchStart  = indexSpaceStart[dim3];
    const int batchEnd    = indexSpaceEnd[dim3];

    const int fifthDimStep   = 1;
    const int fifthDimStart  = indexSpaceStart[dim4];
    const int fifthDimEnd    = indexSpaceEnd[dim4];

    int5 outCoords0 = {0};
    
    int5 idxCoords0 = {0};
    int5 idxCoords1 = {0};
    int5 idxCoords2 = {0};
    int5 idxCoords3 = {0};


    const int indxSize = get_dim_size(indices, dim0);

    uint64 lane0 = read_lane_id_4b_b() - 0;
    uint64 lane1 = read_lane_id_4b_b() - 1;
    uint64 lane2 = read_lane_id_4b_b() - 2;
    uint64 lane3 = read_lane_id_4b_b() - 3;

    #pragma loop_taken
    for (int f = fifthDimStart; f < fifthDimEnd; f += fifthDimStep)
    {
        outCoords0[dim4] = f; 
        dataCoords0[dim4] = f; idxCoords0[dim4] = f;
        dataCoords1[dim4] = f; idxCoords1[dim4] = f;
        dataCoords2[dim4] = f; idxCoords2[dim4] = f;
        dataCoords3[dim4] = f; idxCoords3[dim4] = f;

        #pragma loop_taken
        for (int b = batchStart; b < batchEnd; b += batchStep)
        {
            outCoords0[dim3] = b; 
            dataCoords0[dim3] = b; idxCoords0[dim3] = b;
            dataCoords1[dim3] = b; idxCoords1[dim3] = b;
            dataCoords2[dim3] = b; idxCoords2[dim3] = b;
            dataCoords3[dim3] = b; idxCoords3[dim3] = b;

            #pragma loop_taken
            for (int h = heightStart; h < heightEnd; h += heightStep)
            {
                outCoords0[dim2] = h; 
                dataCoords0[dim2] = h; idxCoords0[dim2] = h;
                dataCoords1[dim2] = h; idxCoords1[dim2] = h;
                dataCoords2[dim2] = h; idxCoords2[dim2] = h;
                dataCoords3[dim2] = h; idxCoords3[dim2] = h;

                #pragma loop_taken
                for (int w = widthStart; w < widthEnd; w += 1)
                {
                    outCoords0[dim1] = w; 
                    dataCoords0[dim1] = w; idxCoords0[dim1] = w;
                    dataCoords1[dim1] = w; idxCoords1[dim1] = w;
                    dataCoords2[dim1] = w; idxCoords2[dim1] = w;
                    dataCoords3[dim1] = w; idxCoords3[dim1] = w;

                    #pragma loop_taken
                    for (int d = depthStart; d < depthEnd; d += depthStep)
                    {
                        int vecEnd = s_i32_min(depthStep, indxSize-d);
                        int vecTail = vecEnd & 3;
                        int vecEndUnrolled = vecEnd - vecTail;

                        outCoords0[dim0] = d;

                        dataCoords0[dim0] = d;
                        dataCoords1[dim0] = d + 1;
                        dataCoords2[dim0] = d + 2;
                        dataCoords3[dim0] = d + 3;

                        idxCoords0[dim0] = d;
                        idxCoords1[dim0] = d + 1;
                        idxCoords2[dim0] = d + 2;
                        idxCoords3[dim0] = d + 3;

                         __global__ int* idxAddr0 = ( __global__ int*)gen_addr(idxCoords0, indices); idxCoords0[dim0] += 4;
                         __global__ int* idxAddr1 = ( __global__ int*)gen_addr(idxCoords1, indices); idxCoords1[dim0] += 4;
                         __global__ int* idxAddr2 = ( __global__ int*)gen_addr(idxCoords2, indices); idxCoords2[dim0] += 4;
                         __global__ int* idxAddr3 = ( __global__ int*)gen_addr(idxCoords3, indices); idxCoords3[dim0] += 4;

                         __global__ int* outAddr0;
                         __global__ int* outAddr1;
                         __global__ int* outAddr2;
                         __global__ int* outAddr3;

                        bool64 pred0;
                        bool64 pred1;
                        bool64 pred2;
                        bool64 pred3;

                        int64 temp0;
                        for (int vec_d = 0; vec_d < vecEndUnrolled; vec_d += 4)
                        {
                            // load index value from indices tensor
                            int index0 = s_i32_ld_g((__global__ int*)idxAddr0);
                            int index1 = s_i32_ld_g((__global__ int*)idxAddr1);
                            int index2 = s_i32_ld_g((__global__ int*)idxAddr2);
                            int index3 = s_i32_ld_g((__global__ int*)idxAddr3);

                            idxAddr0 = ( __global__ int*)gen_addr(idxCoords0, indices); idxCoords0[dim0] += 4;
                            idxAddr1 = ( __global__ int*)gen_addr(idxCoords1, indices); idxCoords1[dim0] += 4;
                            idxAddr2 = ( __global__ int*)gen_addr(idxCoords2, indices); idxCoords2[dim0] += 4;
                            idxAddr3 = ( __global__ int*)gen_addr(idxCoords3, indices); idxCoords3[dim0] += 4;


                            // overwrite the coords value along axis
                            dataCoords0[AXIS] = index0;
                            dataCoords1[AXIS] = index1;
                            dataCoords2[AXIS] = index2;
                            dataCoords3[AXIS] = index3;


                            // load corresponding value from the input tensor
                            outAddr0 = ( __global__ int*)gen_addr(dataCoords0, input);
                            outAddr1 = ( __global__ int*)gen_addr(dataCoords1, input);
                            outAddr2 = ( __global__ int*)gen_addr(dataCoords2, input);
                            outAddr3 = ( __global__ int*)gen_addr(dataCoords3, input);

                            pred0 = v_u32_cmp_eq_b(lane0, vec_d);
                            pred1 = v_u32_cmp_eq_b(lane1, vec_d);
                            pred2 = v_u32_cmp_eq_b(lane2, vec_d);
                            pred3 = v_u32_cmp_eq_b(lane3, vec_d);

                            int64 value0 = v_i32_ld_g((__global__ int*)outAddr0);
                            int64 value1 = v_i32_ld_g((__global__ int*)outAddr1);
                            int64 value2 = v_i32_ld_g((__global__ int*)outAddr2);
                            int64 value3 = v_i32_ld_g((__global__ int*)outAddr3);

                            temp0 = v_i32_mov_vb(value0, 0, temp0, pred0, 0);
                            temp0 = v_i32_mov_vb(value1, 0, temp0, pred1, 0);
                            temp0 = v_i32_mov_vb(value2, 0, temp0, pred2, 0);
                            temp0 = v_i32_mov_vb(value3, 0, temp0, pred3, 0);

// no need to increment if we override it with index value
#if AXIS != 0
                            dataCoords0[dim0] += 4;
                            dataCoords1[dim0] += 4;
                            dataCoords2[dim0] += 4;
                            dataCoords3[dim0] += 4;
#endif
                        }

                        // process tail cases
                        int ld0en = vecTail > 0;
                        int ld1en = vecTail > 1;
                        int ld2en = vecTail > 2;

                        // load index value from indices tensor
                        int index0 = s_i32_ld_g((__global__ int*)idxAddr0, 0, 0, ld0en);
                        int index1 = s_i32_ld_g((__global__ int*)idxAddr1, 0, 0, ld1en);
                        int index2 = s_i32_ld_g((__global__ int*)idxAddr2, 0, 0, ld2en);

                        dataCoords0[AXIS] = index0;
                        dataCoords1[AXIS] = index1;
                        dataCoords2[AXIS] = index2;

                        outAddr0 = ( __global__ int*)gen_addr(dataCoords0, input);
                        outAddr1 = ( __global__ int*)gen_addr(dataCoords1, input);
                        outAddr2 = ( __global__ int*)gen_addr(dataCoords2, input);

                        pred0 = v_u32_cmp_eq_b(lane0, vecEndUnrolled);
                        pred1 = v_u32_cmp_eq_b(lane1, vecEndUnrolled);
                        pred2 = v_u32_cmp_eq_b(lane2, vecEndUnrolled);

                        int64 value0 = v_i32_ld_g((__global__ int*)outAddr0, 0, (int64)0, ld0en, 0);
                        int64 value1 = v_i32_ld_g((__global__ int*)outAddr1, 0, (int64)0, ld1en, 0);
                        int64 value2 = v_i32_ld_g((__global__ int*)outAddr2, 0, (int64)0, ld2en, 0);

                        temp0 = v_i32_mov_vb(value0, 0, temp0, pred0, 0);
                        temp0 = v_i32_mov_vb(value1, 0, temp0, pred1, 0);
                        temp0 = v_i32_mov_vb(value2, 0, temp0, pred2, 0);

// no need to increment if we override it with index value
#if AXIS != 0
                        dataCoords0[dim0] += 4;
                        dataCoords1[dim0] += 4;
                        dataCoords2[dim0] += 4;
#endif
                        v_i32_st_tnsr_partial(outCoords0, output, temp0, vecEnd-1, 0);
                    }
                }
            }
        }
    }
}

