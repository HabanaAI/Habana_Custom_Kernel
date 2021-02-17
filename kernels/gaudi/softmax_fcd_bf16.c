/**********************************************************************
Copyright (c) 2020 Habana Labs.

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

void main(
    tensor ifm,
    tensor ofm,
    // LUT is an aux tensor, it should be the last argument
    tensor lut_tab
)
{
    const int depth  = 0;
    const int width  = 1;
    const int height = 2;
    const int batch  = 3;

    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end = get_index_space_size() + index_space_start;

    // depth
    const int depthStep  = 128;
    const int depthStart = 0;
    // Returns the dim0 size of ifm
    const int depthEnd   = get_dim_size(ifm, 0);

    // width
    const int widthStep  = 1;
    const int widthStart = index_space_start[width] * widthStep;
    const int widthEnd   = index_space_end[width]   * widthStep;

    // height
    const int heightStep  = 1;
    const int heightStart = index_space_start[height] * heightStep;
    const int heightEnd   = index_space_end[height]   * heightStep;

    // batch
    const int batchStep  = 1;
    const int batchStart = index_space_start[batch] * batchStep;
    const int batchEnd   = index_space_end[batch]   * batchStep;

    int5 ifmCoords = { depthStart, widthStart, heightStart, batchStart, 0 };

    char256 lut0, lut1, lut2, lut3;

    bfloat128 zero_bf16 = v_bf16_mov_s(0.f);

    bfloat128 x;
    bfloat128 y;
    bfloat128 sum;

    int5 lutCoords = { 0 };

    // Load lut values for reduction operation
    lut0 = v_i8_ld_tnsr_b(lutCoords, lut_tab, 0, 0, 1, 0);    lutCoords[depth] += 256;
    lut1 = v_i8_ld_tnsr_b(lutCoords, lut_tab, 0, 0, 1, 0);    lutCoords[depth] += 256;
    lut2 = v_i8_ld_tnsr_b(lutCoords, lut_tab, 0, 0, 1, 0);    lutCoords[depth] += 256;
    lut3 = v_i8_ld_tnsr_b(lutCoords, lut_tab, 0, 0, 1, 0);    lutCoords[depth] += 256;

    for (int b = batchStart; b < batchEnd; b += batchStep)
    {
        ifmCoords[batch] = b;

        for (int h = heightStart; h < heightEnd; h += heightStep)
        {
            ifmCoords[height] = h;

            for (int w = widthStart; w < widthEnd; w += widthStep)
            {
                ifmCoords[width] = w;

                sum = zero_bf16;

                for (int d = depthStart; d < depthEnd; d += depthStep)
                {
                    ifmCoords[depth] = d;

                    // load input tensors
                    x = v_bf16_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);
                    float64_pair_t  xf32, yf32;
                    xf32 = v_convert_bf16_to_f32_all_b(x, 0, (float128){0}, 1, 0);
                    // exp_bf16(bfloat128 input)
                    yf32.v1 = v_exp_cephes_f32(xf32.v1);
                    yf32.v2 = v_exp_cephes_f32(xf32.v2);
                    y = v_convert_f32_to_bf16_all_b(yf32, (0 <<16), 0, 1, 0);

                    // Move zero for out of bound co-ordinates
                    bool256 pred = from_bool128(v_u16_cmp_geq_b(d + V_LANE_ID_16, (unsigned)depthEnd, 0, to_bool128((bool256){0}), 1, 0));
                    y = v_bf16_mov_vb(zero_bf16, 0, y, to_bool128(pred), 0);

                    // Sum up the values in a vector
                    sum = sum + y;
                }


               // Sum across the vector
               // Final sum will be stored in first element/first group which will be broadcasted later
               // Note: Each vector contains 4 dual groups and each group has 16 elements
               bfloat128 dgroup, group, temp;
               // Get data from 2nd dual group
               dgroup = v_bf16_mov_dual_group_b(sum, 0xFFFFFFFF, 1, 0, MkWr(1, 1), 0, 1, 0);
               // Dual group (0 + 1)
               sum = sum + dgroup;
               // Get data from 3rd dual group
               dgroup = v_bf16_mov_dual_group_b(sum, 0xFFFFFFFF, 2, 0, MkWr(1, 1), 0, 1, 0);
               // Dual group (0 + 1 + 2)
               sum = sum + dgroup;
               // Get data from 4th dual group
               dgroup = v_bf16_mov_dual_group_b(sum, 0xFFFFFFFF, 3, 0, MkWr(1, 1), 0, 1, 0);
               // Dual group (0 + 1 + 2 + 3)
               sum = sum + dgroup;

               // Exchange hi and lo groups
               group = v_bf16_mov_group_b(sum, 0xFFFFFFFF, (0b1111 << 2) | 0b11, 0, 1, 0);
               // Dual group (0 + 1 + 2 + 3) + lo_hi groups
               sum = sum + group;

               // lut values are used to shuffle within a group
               // 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14
               temp = v_bf16_shuffle_b(sum, (uchar256)lut0, 0, sum, 1, 0);
               // Dual group (0 + 1 + 2 + 3) + lo_hi groups + element (0 + 1)
               sum = sum + temp;
               // 0+1, .., 2+3, .., 4+5, .., 6+7, .., 8+9, .., 10+11, .., 12+13, .., 14+15, ..
               temp = v_bf16_shuffle_b(sum, (uchar256)lut1, 0, sum, 1, 0);
               // Dual group (0 + 1 + 2 + 3) + lo_hi groups + element (0 + 1 + 2)
               sum = sum + temp;
               // 0+1+2+3,...4+5+6+7,... 8+9+10+11, ..., 12+13+14+15
               temp = v_bf16_shuffle_b(sum, (uchar256)lut2, 0, sum, 1, 0);
               // Dual group (0 + 1 + 2 + 3) + lo_hi groups + element (0 + 1 + 2 + 3)
               sum = sum + temp;
               // 0+1+2+3+4+5+6+7,.......... 8+9+10+11+12+13+14+15,....
               temp = v_bf16_shuffle_b(sum, (uchar256)lut3, 0, sum, 1, 0);
               // Dual group (0 + 1 + 2 + 3) + lo_hi groups + element (0 + 1 + 2 + 3)
               sum = sum + temp;
               // 0+1+2+3+4+5+6+7+8+9+10+11+12+13+14+15,....

               // Broadcast sum to 2nd dual_group
               sum = v_bf16_mov_dual_group_b(sum, 0xFFFFFFFF, 0, 1, MkWr(1, 1), sum, 1, 0);
               // Broadcast sum to 3rd dual_group
               sum = v_bf16_mov_dual_group_b(sum, 0xFFFFFFFF, 0, 2, MkWr(1, 1), sum, 1, 0);
               // Broadcast sum to 4th dual_group
               sum = v_bf16_mov_dual_group_b(sum, 0xFFFFFFFF, 0, 3, MkWr(1, 1), sum, 1, 0);


                ifmCoords[width] = w;
                float64_pair_t sumf32;
                // calculate 1/sum by using float
                sumf32 = v_convert_bf16_to_f32_all_b(sum, 0, (float128){0}, 1, 0);
                sumf32.v1 = v_reciprocal_f32(sumf32.v1);
                sumf32.v2 = v_reciprocal_f32(sumf32.v2);
                sum = v_convert_f32_to_bf16_all_b(sumf32, (0 <<16), 0, 1, 0);


                for (int d = depthStart; d < depthEnd; d += depthStep)
                {
                    ifmCoords[depth] = d;

                    x = v_bf16_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);
                    float64_pair_t xf32, yf32;
                    xf32 = v_convert_bf16_to_f32_all_b(x, 0, (float128){0}, 1, 0);
                    // exp_bf16(bfloat128 input)
                    yf32.v1 = v_exp_cephes_f32(xf32.v1);
                    yf32.v2 = v_exp_cephes_f32(xf32.v2);
                    y = v_convert_f32_to_bf16_all_b(yf32, (0 <<16), 0, 1, 0);

                    // Multiply exp(x) * 1/(sum_of_exponents)
                    x = y * sum;

                    v_bf16_st_tnsr(ifmCoords, ofm, x, 0, 1, 0);
                }
            }
        }
    }
}
