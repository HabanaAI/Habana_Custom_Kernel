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

void main(
    tensor ifm,
    tensor ofm
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
    const int depthStart = index_space_start[depth] * depthStep;
    const int depthEnd   = index_space_end[depth]   * depthStep;

    // width
    const int widthStep  = 1;
    const int widthStart = 0;
    const int widthEnd   = get_dim_size(ifm, 1);

    // height
    const int heightStep  = 1;
    const int heightStart = index_space_start[height] * heightStep;
    const int heightEnd   = index_space_end[height]   * heightStep;

    // batch
    const int batchStep  = 1;
    const int batchStart = index_space_start[batch] * batchStep;
    const int batchEnd   = index_space_end[batch]   * batchStep;

    int5 ifmCoords = { depthStart, widthStart, heightStart, batchStart, 0 };

    bfloat128 zero_bf16 = 0.f;

    // definition of -inf in bf16
    static const unsigned short minusInfBf16 = 0xff80;
    const short128 minusInfShort = minusInfBf16;
    const bfloat128   neg_inf_bf16 = *((bfloat128*)&minusInfShort);

    bfloat128 x;
    bfloat128 y;
    bfloat128 sum;
    bfloat128 max;
    // for numerical sbaility reasons we will implement the following softmax calculation EXP(X-Xmax)/ SUM(X-Xmax)
    for (int d = depthStart; d < depthEnd; d += depthStep)
    {
        ifmCoords[depth] = d;
        for (int b = batchStart; b < batchEnd; b += batchStep)
        {
            ifmCoords[batch] = b;
            for (int h = heightStart; h < heightEnd; h += heightStep)
            {
                ifmCoords[height] = h;
                max = neg_inf_bf16;
                for (int w = widthStart; w < widthEnd; w += widthStep)
                {
                    ifmCoords[width] = w;
                    // load input pixel
                    x = v_bf16_ld_tnsr_b(ifmCoords, ifm);
                    // Move -inf for out of bound co-ordinates
                    bool256 pred = from_bool128(v_u16_cmp_geq_b(d + V_LANE_ID_16, (unsigned)depthEnd, 0, to_bool128((bool256){0})));
                    y = v_bf16_mov_vb(neg_inf_bf16, 0, x, to_bool128(pred), 0);
                    // Get max values
                    max = v_bf16_max_b(max,y);
                }

                sum = zero_bf16;
                for (int w = widthStart; w < widthEnd; w += widthStep)
                {
                    ifmCoords[width] = w;
                    // load input pixel
                    x = v_bf16_ld_tnsr_b(ifmCoords, ifm);
                    x = x - max;
                    float64_pair_t  xf32, yf32;
                    xf32 = v_convert_bf16_to_f32_all_b(x);
                    // exp_f32(float64 input)
                    yf32.v1 = v_exp_cephes_f32(xf32.v1);
                    yf32.v2 = v_exp_cephes_f32(xf32.v2);
                    y = v_convert_f32_to_bf16_all_b(yf32);
                    // Move zero for out of bound co-ordinates
                    bool256 pred = from_bool128(v_u16_cmp_geq_b(d + V_LANE_ID_16, (unsigned)depthEnd, 0, to_bool128((bool256){0})));
                    y = v_bf16_mov_vb(zero_bf16, 0, y, to_bool128(pred), 0);
                    // Sum up the values in a vector
                    sum = sum + y;

                }


                 float64_pair_t sumf32;
                // calculate 1/sum by using float
                sumf32 = v_convert_bf16_to_f32_all_b(sum);
                sumf32.v1 = v_reciprocal_f32(sumf32.v1);
                sumf32.v2 = v_reciprocal_f32(sumf32.v2);
                sum = v_convert_f32_to_bf16_all_b(sumf32);

                for (int w = widthStart; w < widthEnd; w += widthStep)
                {
                    ifmCoords[width] = w;

                    x = v_bf16_ld_tnsr_b(ifmCoords, ifm);
                    x = x - max;
                    float64_pair_t xf32, yf32;
                    xf32 = v_convert_bf16_to_f32_all_b(x);
                    // exp_bf16(bfloat128 input)
                    yf32.v1 = v_exp_cephes_f32(xf32.v1);
                    yf32.v2 = v_exp_cephes_f32(xf32.v2);
                    y = v_convert_f32_to_bf16_all_b(yf32);
                    // Multiply exp(x) * 1/(sum_of_exponents)
                    x = y * sum;
                    v_bf16_st_tnsr(ifmCoords, ofm, x);
               }
            }
        }
    }

}
