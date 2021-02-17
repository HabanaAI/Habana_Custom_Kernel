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
void main(tensor ifm,
          tensor ofm,
          float scaleToI16)
{
    const int depth = 0;
    const int width = 1;
    const int height = 2;
    const int batch = 3;

    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end = get_index_space_size() + index_space_start;

    // DEPTH
    const int depthStep = 128;
    const int depthStart = index_space_start[depth] * depthStep;
    const int depthEnd = index_space_end[depth] * depthStep;

    // WIDTH
    const int widthStep = 1;
    const int widthStart = index_space_start[width] * widthStep;
    const int widthEnd = index_space_end[width] * widthStep;

    // HEIGHT
    const int heightStep = 1;
    const int heightStart = index_space_start[height] * heightStep;
    const int heightEnd = index_space_end[height] * heightStep;

    // BATCH
    const int batchStep = 1;
    const int batchStart = index_space_start[batch] * batchStep;
    const int batchtEnd = index_space_end[batch] * batchStep;

    int5 ifmCoords0, ifmCoords1, ofmCoords;

    float64_pair_t x;
    int64 y_i32;

    for (int d = depthStart; d < depthEnd; d += depthStep)
    {
        ifmCoords0[0] = ifmCoords1[0] = d;
        ofmCoords[0] = d;

        for (int b = batchStart; b < batchtEnd; b += batchStep)
        {
            ifmCoords0[3] =  ifmCoords1[3] = b;
            ofmCoords[3] = b;

            for (int h = heightStart; h < heightEnd; h += heightStep)
            {
                ifmCoords0[2] = ifmCoords1[2] = h;
                ofmCoords[2] = h;

                ifmCoords1[0] = ifmCoords0[0] + 64;

                for (int w = widthStart; w < widthEnd; w += widthStep)
                {

                    ifmCoords0[1] = ifmCoords1[1] = w;
                    ofmCoords[1] = w;

                    // Load input elements
                    x.v1 = v_f32_ld_tnsr_b(ifmCoords0, ifm, 0, 0, 1, 0);
                    x.v2 = v_f32_ld_tnsr_b(ifmCoords1, ifm, 0, 0, 1, 0);

                    // Multiply by scale
                    x.v1 = x.v1 * scaleToI16;
                    x.v2 = x.v2 * scaleToI16;

                    // Convert f32 to i16 in two non-contiguous vectors
                    short128 t0 = 0, t1 = 0;
                    y_i32 = v_convert_f32_to_i32_b(x.v1, (e_round_half_ne <<16), 0, 1, 0);
                    // Store converted elements in lane 0 of t0
                    t0 = v_convert_int32_to_i16_v_v(y_i32, 0, t0, e_round_half_ne, 0);
                    // Store converted elements in lane 0 of t1
                    y_i32 = v_convert_f32_to_i32_b(x.v2, (e_round_half_ne <<16), 0, 1, 0);
                    t1 = v_convert_int32_to_i16_v_v(y_i32, 0, t1, e_round_half_ne, 0);

                    short128 t = 0, y = 0;

                    // Pack the two non-contiguous vectors into a sigle contiguous vector

                    /* Packs group0 in lower half of groups and group1 in upper half of groups
                    for all 4 dual groups */
                    y = v_i16_pack_b(t0, ((e_group_0) << 8) | ((e_every_second_element) << 9), y, 1, 0);
                    y = v_i16_pack_b(t0, ((e_group_1) << 8) | ((e_every_second_element) << 9), y, 1, 0);

                    // Note: int128 vector contains 4 dual groups and each group has 16 elements
                    // Move elements from dualgroup1 of y to upper half of dualgroup0 of y
                    // 0..15, 16..31
                    y = v_i16_mov_dual_group_b(y, 0xFFFFFFFF, 1, 0, MkWr(0, 1), y, 1, 0);
                    // Move elements from dualgroup2 of y to lower half of dualgroup1 of y
                    // 0..15, 16..31, 32..47
                    y = v_i16_mov_dual_group_b(y, 0xFFFFFFFF, 2, 1, MkWr(1, 0), y, 1, 0);
                    // Move elements from dualgroup3 of y to upper half of dualgroup1 of y
                    // 0..15, 16..31, 32..47, 48..63
                    y = v_i16_mov_dual_group_b(y, 0xFFFFFFFF, 3, 1, MkWr(0, 1), y, 1, 0);

                    t = v_i16_pack_b(t1, ((e_group_0) << 8) | ((e_every_second_element) << 9), t, 1, 0);
                    t = v_i16_pack_b(t1, ((e_group_1) << 8) | ((e_every_second_element) << 9), t, 1, 0);
                    // Move elements from dualgroup0 of t to lower half of dualgroup2 of y
                    // 0..15, 16..31, 32..47, 48..63, 64..79
                    y = v_i16_mov_dual_group_b(t, 0xFFFFFFFF, 0, 2, MkWr(1, 0), y, 1, 0);
                    // Move elements from dualgroup1 of t to upper half of dualgroup2 of y
                    // 0..15, 16..31, 32..47, 48..63, 64..79, 80..95
                    y = v_i16_mov_dual_group_b(t, 0xFFFFFFFF, 1, 2, MkWr(0, 1), y, 1, 0);
                    // Move elements from dualgroup2 of t to lower half of dualgroup3 of y
                    // 0..15, 16..31, 32..47, 48..63, 64..79, 80..95, 96..111
                    y = v_i16_mov_dual_group_b(t, 0xFFFFFFFF, 2, 3, MkWr(1, 0), y, 1, 0);
                    // Move elements from dualgroup3 of t to upper half of dualgroup3 of y
                    // 0..15, 16..31, 32..47, 48..63, 64..79, 80..95, 96..111, 112..127
                    y = v_i16_mov_dual_group_b(t, 0xFFFFFFFF, 3, 3, MkWr(0, 1), y, 1, 0); 

                    // Store output data
                    v_i16_st_tnsr(ofmCoords, ofm, y, 0, 1, 0);
                }
            }
        }
    }
}

