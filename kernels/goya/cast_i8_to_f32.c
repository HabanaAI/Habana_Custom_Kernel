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
void main(tensor ifm,
          tensor ofm,
          float scaleToF32)
{

    const int depth = 0;
    const int width = 1;
    const int height = 2;
    const int batch = 3;

    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end = get_index_space_size() + index_space_start;

    // DEPTH
    const int depthStep = 256;
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

    int5 ifmCoords, ofmCoords0, ofmCoords1, ofmCoords2, ofmCoords3;

    char256 x;
    int256 y;
    int64 t0, t1, t2;
    float64 y0, y1, y2, y3;

    for (int d = depthStart; d < depthEnd; d += depthStep)
    {
        ifmCoords[0] = d;
        ofmCoords0[0] = d;

        for (int b = batchStart; b < batchtEnd; b += batchStep)
        {
            ifmCoords[3] = b;
            ofmCoords0[3] =ofmCoords1[3] = ofmCoords2[3] = ofmCoords3[3] = b;

            for (int h = heightStart; h < heightEnd; h += heightStep)
            {
                ifmCoords[2] = h;
                ofmCoords0[2] = ofmCoords1[2] = ofmCoords2[2] = ofmCoords3[2] = h;

                ofmCoords1[0] = ofmCoords0[0] + 64;
                ofmCoords2[0] = ofmCoords0[0] + 128;
                ofmCoords3[0] = ofmCoords0[0] + 192;

                for (int w = widthStart; w < widthEnd; w += widthStep)
                {

                    ifmCoords[1] = w;
                    ofmCoords0[1] =ofmCoords1[1] = ofmCoords2[1] = ofmCoords3[1] = w;
                    x = v_i8_ld_tnsr_b(ifmCoords, ifm);

                    // Cast char256 into int256
                    // Note: char256 vector contains 4 dual groups and each group has 32 elements
                    // Unpacks lower half of first group of all 4 dual gorups
                    // 0..15, 64..79, 128..143, 192..207
                    y.v1 = (int64) v_i8_unpack_b(x, ((e_group_0) << 8) | ((e_every_forth_element) << 9) | ((e_lower_half_group) << 10), 0);
                    // Unpacks upper half of first group of all 4 dual gorups
                    // 16..31, 80..95, 144..159, 208..223
                    t1 = (int64) v_i8_unpack_b(x, ((e_group_0) << 8) | ((e_every_forth_element) << 9) | ((e_higher_half_group) << 10), 0);
                    // Unpacks lower half of second group of all 4 dual gorups
                    // 32..47, 96..111, 160..175, 224..239
                    t2 = (int64) v_i8_unpack_b(x, ((e_group_1) << 8) | ((e_every_forth_element) << 9) | ((e_lower_half_group) << 10), 0);
                    // Unpacks upper half of second group of all 4 dual gorups
                    // 48..63, 112..127, 176..191, 240..255
                    y.v4 = (int64) v_i8_unpack_b(x, ((e_group_1) << 8) | ((e_every_forth_element) << 9) | ((e_higher_half_group) << 10), 0);

                    // Rearranges the vector in correct order
                    // 0..15
                    t0 = y.v1;
                    // Move dualgroup0 of t1 to dualgroup1 of y.v1
                    // 0..15, 16..31
                    y.v1 = v_i32_mov_dual_group_b(t1, 0xFFFFFFFF, 0, 1, MkWr(1, 1), y.v1);
                    // Move dualgroup0 of t2 to dualgroup2 of y.v1
                    // 0..15, 16..31, 32..47
                    y.v1 = v_i32_mov_dual_group_b(t2, 0xFFFFFFFF, 0, 2, MkWr(1, 1), y.v1);
                    // Move dualgroup0 of y.v4 to dualgroup3 of y.v1
                    // 0..15, 16..31, 32..47, 48..63
                    y.v1 = v_i32_mov_dual_group_b(y.v4, 0xFFFFFFFF, 0, 3, MkWr(1, 1), y.v1);

                    // Move dualgroup1 of t0 to dualgroup0 of y.v2
                    // 64..79
                    y.v2 = v_i32_mov_dual_group_b(t0, 0xFFFFFFFF, 1, 0, MkWr(1, 1), y.v2);
                    // Move dualgroup1 of t1 to dualgroup1 of y.v2
                    // 64..79, 80..95
                    y.v2 = v_i32_mov_dual_group_b(t1, 0xFFFFFFFF, 1, 1, MkWr(1, 1), y.v2);
                    // Move dualgroup1 of t2 to dualgroup2 of y.v2
                    // 64..79, 80..95, 96..111
                    y.v2 = v_i32_mov_dual_group_b(t2, 0xFFFFFFFF, 1, 2, MkWr(1, 1), y.v2);
                    // Move dualgroup1 of y.v4 to dualgroup3 of y.v2
                    // 64..79, 80..95, 96..111, 112..127
                    y.v2 = v_i32_mov_dual_group_b(y.v4, 0xFFFFFFFF, 1, 3, MkWr(1, 1), y.v2);

                    // Move dualgroup2 of t0 to dualgroup0 of y.v3
                    // 128..143
                    y.v3 = v_i32_mov_dual_group_b(t0, 0xFFFFFFFF, 2, 0, MkWr(1, 1), y.v3);
                    // Move dualgroup2 of t1 to dualgroup1 of y.v3
                    // 128..143, 144..159
                    y.v3 = v_i32_mov_dual_group_b(t1, 0xFFFFFFFF, 2, 1, MkWr(1, 1), y.v3);
                    // Move dualgroup2 of t2 to dualgroup2 of y.v3
                    // 128..143, 144..159, 160..175
                    y.v3 = v_i32_mov_dual_group_b(t2, 0xFFFFFFFF, 2, 2, MkWr(1, 1), y.v3);
                    // Move dualgroup2 of y.v4 to dualgroup3 of y.v3
                    // 128..143, 144..159, 160..175, 176..191
                    y.v3 = v_i32_mov_dual_group_b(y.v4, 0xFFFFFFFF, 2, 3, MkWr(1, 1), y.v3);

                    // Move dualgroup3 of t0 to dualgroup0 of y.v4
                    // 192..207
                    y.v4 = v_i32_mov_dual_group_b(t0, 0xFFFFFFFF, 3, 0, MkWr(1, 1), y.v4);
                    // Move dualgroup3 of t1 to dualgroup1 of y.v4
                    // 192..207, 208..223
                    y.v4 = v_i32_mov_dual_group_b(t1, 0xFFFFFFFF, 3, 1, MkWr(1, 1), y.v4);
                    // Move dualgroup3 of t2 to dualgroup2 of y.v4
                    // 192..207, 208..223, 224..239
                    y.v4 = v_i32_mov_dual_group_b(t2, 0xFFFFFFFF, 3, 2, MkWr(1, 1), y.v4);

                    // 0..15, 16..31, 32..47, 48..63
                    y0 = v_convert_i8_to_f32_b((char256)y.v1);
                    y0 = y0 * scaleToF32;
                    v_f32_st_tnsr(ofmCoords0, ofm, y0);

                    // 64..79, 80..95, 96..111, 112..127
                    y1 = v_convert_i8_to_f32_b((char256)y.v2);
                    y1 = y1 * scaleToF32;
                    v_f32_st_tnsr(ofmCoords1, ofm, y1);

                    // 128..143, 144..159, 160..175, 176..191
                    y2 = v_convert_i8_to_f32_b((char256)y.v3);
                    y2 = y2 * scaleToF32;
                    v_f32_st_tnsr(ofmCoords2, ofm, y2);

                    // 192..207, 208..223, 224..239, 240..255
                    y3 = v_convert_i8_to_f32_b((char256)y.v4);
                    y3 = y3 * scaleToF32;
                    v_f32_st_tnsr(ofmCoords3, ofm, y3);
                }
            }
        }
    }
}

