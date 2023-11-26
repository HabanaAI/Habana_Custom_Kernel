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
#define MINUS_INF (0xFF80)
//unroll factor
#define UNROLL_FACTOR 4

// VLM  CONFIGURATION

// space left for register spill
#define REG_SPILL_REDUCTION 20
// space assigned for data. as we do not have LUT, out total space is 320 2Byte vectors
#define VLM_MAX_VECTOR (320 - REG_SPILL_REDUCTION)
// number of data compartments we will use
#define VLM_INPUTS      1
// calculation of the number of vectors we can place along the depth
#define VLM_VECTORS_IN_DEPTH (VLM_MAX_VECTOR/(VLM_INPUTS*UNROLL_FACTOR))
#define INPUT_TENSOR    0
#define WIDTH_0         0
#define WIDTH_1         1
#define WIDTH_2         2
#define WIDTH_3         3


//local memory definition
__local__ bfloat128 vlm[VLM_INPUTS][UNROLL_FACTOR][VLM_VECTORS_IN_DEPTH];

//reciprocal function (without LUT)
float64 reciprocal_cephes_fast_f32(float64 input)
{
    float64 result, temp0, temp1, temp2;
    const float a = 2.58586f;
    const float b = -5.81818f;
    const float c = 4.24242f;

    int64 significand = 0;
    significand = v_i32_and_b(*((int64*)&input), 0x007fffff);
    significand = v_i32_or_b(significand, 0x3f000000);
    result = *((float64*)&significand);

    int64 exponent = 0;
    exponent =  v_i32_shr_b(*((int64*)&input), 23);
    exponent = v_i32_and_b(exponent, 0x000000ff);
    exponent -= 0x7e;

    temp0 = v_f32_mac_b(result, a, b);
    temp1 = v_f32_mac_b(result, temp0, c);
    temp2 = v_f32_mac_b(-result, temp1, 2);
    temp2 *= temp1;
    temp0 = v_f32_mac_b(-result, temp2, 2);
    temp0 *= temp2;

    int64 exp =  v_i32_shr_b(*((int64*)&temp0), 23);
    exp = v_i32_and_b(exp, 0x000000ff);
    exp = v_i32_add_b(exp, -exponent);
    exp = v_i32_and_b(exp, 0xff);
    result = v_f32_form_fp_num_ie_b((char256)exp, input, temp0, SW_EXP_IS_NUM);

    return result;
}


float64 reciprocal_cephes_f32(float64 input)
{
    float64 result = reciprocal_cephes_fast_f32(input);


    // ====================================
    //  Processing special values: denorm, +-0. +-inf, nan
    float64 abs_x = v_f32_abs_b(input);

    const uint64 flt_max = 0x7f7fffff;
    const float64 flt_max_fp32 = *((float64*)&flt_max);

    float64 fclass = v_f32_fclass_b(input);
    result = v_f32_calc_fp_special_b(fclass, fclass, e_fp_recip, result);
    result = v_f32_sel_geq_f32_b(abs_x, flt_max_fp32, 0.0f, result);
    // ====================================

    return result;
}

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
    const int depthStart = 0;
    // Returns the dim0 size of ifm
    const int depthEnd   = get_dim_size(ifm, 0);

    //unroll factor
    const int unroll_factor = UNROLL_FACTOR;

    // width
    const int widthStep  = unroll_factor;
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
    int5 ifmCoords1 = { depthStart, widthStart, heightStart, batchStart, 0 };
    int5 ifmCoords2 = { depthStart, widthStart, heightStart, batchStart, 0 };
    int5 ifmCoords3 = { depthStart, widthStart, heightStart, batchStart, 0 };
    int5 ifmCoords4 = { depthStart, widthStart, heightStart, batchStart, 0 };

    const bfloat128 zero_bf16 = 0.f;
    // definition of -inf in bf16
    static const unsigned short minusInfBf16 = 0xff80;
    const short128 minusInfShort = minusInfBf16;
    const bfloat128   neg_inf_bf16 = *((bfloat128*)&minusInfShort);
    // calculation of the vlm_data_end in the depth dimension
    const int vlm_data_end =  (VLM_VECTORS_IN_DEPTH*depthStep) > depthEnd ? depthEnd : (VLM_VECTORS_IN_DEPTH*depthStep)  ;

    // as our code in manually unrolled 4 time, all intermidiate variables are duplicated 4 times as well
    bfloat128 x1;
    bfloat128 x2;
    bfloat128 x3;
    bfloat128 x4;

    bfloat128 y1;
    bfloat128 y2;
    bfloat128 y3;
    bfloat128 y4;

    bfloat128 sum1;
    bfloat128 sum2;
    bfloat128 sum3;
    bfloat128 sum4;

    bfloat128 max1;
    bfloat128 max2;
    bfloat128 max3;
    bfloat128 max4;

    // we will use the following formula for numerical stability: EXP(X - Xmax) / SUM(EXP(X-Xmax))
#pragma loop_taken
    for (int b = batchStart; b < batchEnd; b += batchStep)
    {
        ifmCoords1[batch] = b;
        ifmCoords2[batch] = b;
        ifmCoords3[batch] = b;
        ifmCoords4[batch] = b;

#pragma loop_taken
        for (int h = heightStart; h < heightEnd; h += heightStep)
        {
            ifmCoords1[height] = h;
            ifmCoords2[height] = h;
            ifmCoords3[height] = h;
            ifmCoords4[height] = h;

#pragma loop_taken
            for (int w = widthStart; w < widthEnd; w += widthStep)
            {

                ifmCoords1[width] = w;
                ifmCoords2[width] = w+1;
                ifmCoords3[width] = w+2;
                ifmCoords4[width] = w+3;

                sum1 = zero_bf16;
                sum2 = zero_bf16;
                sum3 = zero_bf16;
                sum4 = zero_bf16;

                max1 = neg_inf_bf16;
                max2 = neg_inf_bf16;
                max3 = neg_inf_bf16;
                max4 = neg_inf_bf16;

#pragma loop_taken
                // in this loop we iterate over vectors that fit inside the vlm
                for (int d = depthStart, i = 0 ; d < vlm_data_end ; d += depthStep , i++)
                {
                    ifmCoords1[depth] = d;
                    ifmCoords2[depth] = d;
                    ifmCoords3[depth] = d;
                    ifmCoords4[depth] = d;


                    // load input tensors
                    x1 = v_bf16_ld_tnsr_b(ifmCoords1, ifm);
                    x2 = v_bf16_ld_tnsr_b(ifmCoords2, ifm);
                    x3 = v_bf16_ld_tnsr_b(ifmCoords3, ifm);
                    x4 = v_bf16_ld_tnsr_b(ifmCoords4, ifm);
                    // save to vlm for later use
                    vlm[INPUT_TENSOR][WIDTH_0][i]=x1;
                    vlm[INPUT_TENSOR][WIDTH_1][i]=x2;
                    vlm[INPUT_TENSOR][WIDTH_2][i]=x3;
                    vlm[INPUT_TENSOR][WIDTH_3][i]=x4;
                    // Move zero for out of bound co-ordinates
                    bool256 pred = from_bool128(v_u16_cmp_geq_b(d + V_LANE_ID_16, (unsigned)depthEnd, 0, to_bool128((bool256){0})));
                    x1 = v_bf16_mov_vb(neg_inf_bf16, 0, x1, to_bool128(pred), 0);
                    x2 = v_bf16_mov_vb(neg_inf_bf16, 0, x2, to_bool128(pred), 0);
                    x3 = v_bf16_mov_vb(neg_inf_bf16, 0, x3, to_bool128(pred), 0);
                    x4 = v_bf16_mov_vb(neg_inf_bf16, 0, x4, to_bool128(pred), 0);
                    // calculate max values vector
                    max1 = v_bf16_max_b(max1,x1);
                    max2 = v_bf16_max_b(max2,x2);
                    max3 = v_bf16_max_b(max3,x3);
                    max4 = v_bf16_max_b(max4,x4);
                }
                for (int d = vlm_data_end  ; d < depthEnd; d += depthStep )
                {
                    // this loop is outside the scope of the vlm
                    ifmCoords1[depth] = d;
                    ifmCoords2[depth] = d;
                    ifmCoords3[depth] = d;
                    ifmCoords4[depth] = d;


                    // load input tensors
                    x1 = v_bf16_ld_tnsr_b(ifmCoords1, ifm);
                    x2 = v_bf16_ld_tnsr_b(ifmCoords2, ifm);
                    x3 = v_bf16_ld_tnsr_b(ifmCoords3, ifm);
                    x4 = v_bf16_ld_tnsr_b(ifmCoords4, ifm);

                    // Move zero for out of bound co-ordinates
                    bool256 pred = from_bool128(v_u16_cmp_geq_b(d + V_LANE_ID_16, (unsigned)depthEnd, 0, to_bool128((bool256){0})));

                    x1 = v_bf16_mov_vb(neg_inf_bf16, 0, x1, to_bool128(pred), 0);
                    x2 = v_bf16_mov_vb(neg_inf_bf16, 0, x2, to_bool128(pred), 0);
                    x3 = v_bf16_mov_vb(neg_inf_bf16, 0, x3, to_bool128(pred), 0);
                    x4 = v_bf16_mov_vb(neg_inf_bf16, 0, x4, to_bool128(pred), 0);

                    // calculate max values vector
                    max1 = v_bf16_max_b(max1,x1);
                    max2 = v_bf16_max_b(max2,x2);
                    max3 = v_bf16_max_b(max3,x3);
                    max4 = v_bf16_max_b(max4,x4);
                }
                // reduce max values vectors
                max1=v_bf16_reduce_nolut_max(max1);
                max2=v_bf16_reduce_nolut_max(max2);
                max3=v_bf16_reduce_nolut_max(max3);
                max4=v_bf16_reduce_nolut_max(max4);
                //calculate the  SUM(EXP(X-Xmax))
#pragma loop_taken
                for (int d = depthStart, i = 0 ; d < vlm_data_end ; d += depthStep , i++)
                {
                    // this loop iterate over vectors in the vlm
                    ifmCoords1[depth] = d;
                    ifmCoords2[depth] = d;
                    ifmCoords3[depth] = d;
                    ifmCoords4[depth] = d;

                    // load input tensors flom vlm
                    x1 = vlm[INPUT_TENSOR][WIDTH_0][i];
                    x2 = vlm[INPUT_TENSOR][WIDTH_1][i];
                    x3 = vlm[INPUT_TENSOR][WIDTH_2][i];
                    x4 = vlm[INPUT_TENSOR][WIDTH_3][i];


                    // substract maximum values that were calculated earlier
                    x1=x1-max1;
                    x2=x2-max2;
                    x3=x3-max3;
                    x4=x4-max4;

                    // in order to calc exponent, we will cast to  f32
                    float64_pair_t  xf32_1, yf32_1;
                    float64_pair_t  xf32_2, yf32_2;
                    float64_pair_t  xf32_3, yf32_3;
                    float64_pair_t  xf32_4, yf32_4;

                    xf32_1 = v_convert_bf16_to_f32_all_b(x1);
                    xf32_2 = v_convert_bf16_to_f32_all_b(x2);
                    xf32_3 = v_convert_bf16_to_f32_all_b(x3);
                    xf32_4 = v_convert_bf16_to_f32_all_b(x4);

                    yf32_1.v1 = v_exp_cephes_f32(xf32_1.v1);
                    yf32_1.v2 = v_exp_cephes_f32(xf32_1.v2);

                    yf32_2.v1 = v_exp_cephes_f32(xf32_2.v1);
                    yf32_2.v2 = v_exp_cephes_f32(xf32_2.v2);

                    yf32_3.v1 = v_exp_cephes_f32(xf32_3.v1);
                    yf32_3.v2 = v_exp_cephes_f32(xf32_3.v2);

                    yf32_4.v1 = v_exp_cephes_f32(xf32_4.v1);
                    yf32_4.v2 = v_exp_cephes_f32(xf32_4.v2);
                    // cast back to bf16
                    y1 = v_convert_f32_to_bf16_all_b(yf32_1);
                    y2 = v_convert_f32_to_bf16_all_b(yf32_2);

                    y3 = v_convert_f32_to_bf16_all_b(yf32_3);
                    y4 = v_convert_f32_to_bf16_all_b(yf32_4);

                    // Move zero for out of bound co-ordinates
                    bool256 pred = from_bool128(v_u16_cmp_geq_b(d + V_LANE_ID_16, (unsigned)depthEnd, 0, to_bool128((bool256){0})));

                    y1 = v_bf16_mov_vb(zero_bf16, 0, y1, to_bool128(pred), 0);
                    y2 = v_bf16_mov_vb(zero_bf16, 0, y2, to_bool128(pred), 0);
                    y3 = v_bf16_mov_vb(zero_bf16, 0, y3, to_bool128(pred), 0);
                    y4 = v_bf16_mov_vb(zero_bf16, 0, y4, to_bool128(pred), 0);
                    // save values to vlm for later use
                    vlm[INPUT_TENSOR][WIDTH_0][i]=y1;
                    vlm[INPUT_TENSOR][WIDTH_1][i]=y2;
                    vlm[INPUT_TENSOR][WIDTH_2][i]=y3;
                    vlm[INPUT_TENSOR][WIDTH_3][i]=y4;

                    // Sum up the values in a vector
                    sum1 = sum1 + y1;
                    sum2 = sum2 + y2;
                    sum3 = sum3 + y3;
                    sum4 = sum4 + y4;
                }
                for (int d = vlm_data_end ; d < depthEnd; d += depthStep )
                {
                    // this loop iterate over vectors that are not in the vlm
                    ifmCoords1[depth] = d;
                    ifmCoords2[depth] = d;
                    ifmCoords3[depth] = d;
                    ifmCoords4[depth] = d;


                    // load input tensors
                    x1 = v_bf16_ld_tnsr_b(ifmCoords1, ifm);
                    x2 = v_bf16_ld_tnsr_b(ifmCoords2, ifm);
                    x3 = v_bf16_ld_tnsr_b(ifmCoords3, ifm);
                    x4 = v_bf16_ld_tnsr_b(ifmCoords4, ifm);


                    // substract maximum values
                    x1=x1-max1;
                    x2=x2-max2;
                    x3=x3-max3;
                    x4=x4-max4;


                    float64_pair_t  xf32_1, yf32_1;
                    float64_pair_t  xf32_2, yf32_2;
                    float64_pair_t  xf32_3, yf32_3;
                    float64_pair_t  xf32_4, yf32_4;

                    xf32_1 = v_convert_bf16_to_f32_all_b(x1);
                    xf32_2 = v_convert_bf16_to_f32_all_b(x2);
                    xf32_3 = v_convert_bf16_to_f32_all_b(x3);
                    xf32_4 = v_convert_bf16_to_f32_all_b(x4);

                    yf32_1.v1 = v_exp_cephes_f32(xf32_1.v1);
                    yf32_1.v2 = v_exp_cephes_f32(xf32_1.v2);

                    yf32_2.v1 = v_exp_cephes_f32(xf32_2.v1);
                    yf32_2.v2 = v_exp_cephes_f32(xf32_2.v2);

                    yf32_3.v1 = v_exp_cephes_f32(xf32_3.v1);
                    yf32_3.v2 = v_exp_cephes_f32(xf32_3.v2);

                    yf32_4.v1 = v_exp_cephes_f32(xf32_4.v1);
                    yf32_4.v2 = v_exp_cephes_f32(xf32_4.v2);

                    y1 = v_convert_f32_to_bf16_all_b(yf32_1);
                    y2 = v_convert_f32_to_bf16_all_b(yf32_2);

                    y3 = v_convert_f32_to_bf16_all_b(yf32_3);
                    y4 = v_convert_f32_to_bf16_all_b(yf32_4);

                    // Move zero for out of bound co-ordinates
                    bool256 pred = from_bool128(v_u16_cmp_geq_b(d + V_LANE_ID_16, (unsigned)depthEnd, 0, to_bool128((bool256){0})));

                    y1 = v_bf16_mov_vb(zero_bf16, 0, y1, to_bool128(pred), 0);
                    y2 = v_bf16_mov_vb(zero_bf16, 0, y2, to_bool128(pred), 0);
                    y3 = v_bf16_mov_vb(zero_bf16, 0, y3, to_bool128(pred), 0);
                    y4 = v_bf16_mov_vb(zero_bf16, 0, y4, to_bool128(pred), 0);


                    // Sum up the values in a vector
                    sum1 = sum1 + y1;
                    sum2 = sum2 + y2;
                    sum3 = sum3 + y3;
                    sum4 = sum4 + y4;
                }

                // Sum across  the vector (reduce) 
                sum1 = v_bf16_reduce_nolut_add(sum1);
                sum2 = v_bf16_reduce_nolut_add(sum2);
                sum3 = v_bf16_reduce_nolut_add(sum3);
                sum4 = v_bf16_reduce_nolut_add(sum4);

                float64_pair_t sumf32_1;
                float64_pair_t sumf32_2;
                float64_pair_t sumf32_3;
                float64_pair_t sumf32_4;

                // calculate 1/sum by using float
                sumf32_1 = v_convert_bf16_to_f32_all_b(sum1);
                sumf32_2 = v_convert_bf16_to_f32_all_b(sum2);
                sumf32_3 = v_convert_bf16_to_f32_all_b(sum3);
                sumf32_4 = v_convert_bf16_to_f32_all_b(sum4);

                sumf32_1.v1 = reciprocal_cephes_f32(sumf32_1.v1);
                sumf32_1.v2 = sumf32_1.v1;//reciprocal_cephes_f32(sumf32_1.v2);

                sumf32_2.v1 = reciprocal_cephes_f32(sumf32_2.v1);
                sumf32_2.v2 = sumf32_2.v1;//reciprocal_cephes_f32(sumf32_2.v2);

                sumf32_3.v1 = reciprocal_cephes_f32(sumf32_3.v1);
                sumf32_3.v2 = sumf32_3.v1;//reciprocal_cephes_f32(sumf32_3.v2);

                sumf32_4.v1 = reciprocal_cephes_f32(sumf32_4.v1);
                sumf32_4.v2 = sumf32_4.v1;//reciprocal_cephes_f32(sumf32_4.v2);

                sum1 = v_convert_f32_to_bf16_all_b(sumf32_1);
                sum2 = v_convert_f32_to_bf16_all_b(sumf32_2);
                sum3 = v_convert_f32_to_bf16_all_b(sumf32_3);
                sum4 = v_convert_f32_to_bf16_all_b(sumf32_4);


#pragma loop_taken
                for (int d = depthStart, i = 0 ; d < vlm_data_end  ; d += depthStep , i++)
                {
                    // go over vlm fetch data and calculate 
                    ifmCoords1[depth] = d;
                    ifmCoords2[depth] = d;
                    ifmCoords3[depth] = d;
                    ifmCoords4[depth] = d;
                    // load input tensors

                    y1 = vlm[INPUT_TENSOR][WIDTH_0][i];
                    y2 = vlm[INPUT_TENSOR][WIDTH_1][i];
                    y3 = vlm[INPUT_TENSOR][WIDTH_2][i];
                    y4 = vlm[INPUT_TENSOR][WIDTH_3][i];


                    // Multiply exp(X-Xmax) * 1/(sum_of_exponents)
                    x1 = y1 * sum1;
                    x2 = y2 * sum2;
                    x3 = y3 * sum3;
                    x4 = y4 * sum4;

                    v_bf16_st_tnsr(ifmCoords1, ofm, x1,0,ifmCoords1[width] < widthEnd);
                    v_bf16_st_tnsr(ifmCoords2, ofm, x2,0,ifmCoords2[width] < widthEnd);
                    v_bf16_st_tnsr(ifmCoords3, ofm, x3,0,ifmCoords3[width] < widthEnd);
                    v_bf16_st_tnsr(ifmCoords4, ofm, x4,0,ifmCoords4[width] < widthEnd);
                }
                for (int d = vlm_data_end  ; d < depthEnd; d += depthStep )
                {
                    // go over vectors outside the vlm
                    ifmCoords1[depth] = d;
                    ifmCoords2[depth] = d;
                    ifmCoords3[depth] = d;
                    ifmCoords4[depth] = d;

                    // load input tensors from memory
                    x1 = v_bf16_ld_tnsr_b(ifmCoords1, ifm);
                    x2 = v_bf16_ld_tnsr_b(ifmCoords2, ifm);
                    x3 = v_bf16_ld_tnsr_b(ifmCoords3, ifm);
                    x4 = v_bf16_ld_tnsr_b(ifmCoords4, ifm);
                    // here intermidiate values were not cached in the vlm, we will need to calculate again
                    x1=x1-max1;
                    x2=x2-max2;
                    x3=x3-max3;
                    x4=x4-max4;

                    float64_pair_t xf32_1, yf32_1;
                    float64_pair_t xf32_2, yf32_2;
                    float64_pair_t xf32_3, yf32_3;
                    float64_pair_t xf32_4, yf32_4;

                    xf32_1 = v_convert_bf16_to_f32_all_b(x1);
                    xf32_2 = v_convert_bf16_to_f32_all_b(x2);
                    xf32_3 = v_convert_bf16_to_f32_all_b(x3);
                    xf32_4 = v_convert_bf16_to_f32_all_b(x4);

                    yf32_1.v1 = v_exp_cephes_f32(xf32_1.v1);
                    yf32_1.v2 = v_exp_cephes_f32(xf32_1.v2);
                    yf32_2.v1 = v_exp_cephes_f32(xf32_2.v1);
                    yf32_2.v2 = v_exp_cephes_f32(xf32_2.v2);
                    yf32_3.v1 = v_exp_cephes_f32(xf32_3.v1);
                    yf32_3.v2 = v_exp_cephes_f32(xf32_3.v2);
                    yf32_4.v1 = v_exp_cephes_f32(xf32_4.v1);
                    yf32_4.v2 = v_exp_cephes_f32(xf32_4.v2);

                    y1 = v_convert_f32_to_bf16_all_b(yf32_1);
                    y2 = v_convert_f32_to_bf16_all_b(yf32_2);
                    y3 = v_convert_f32_to_bf16_all_b(yf32_3);
                    y4 = v_convert_f32_to_bf16_all_b(yf32_4);

                    // Multiply exp(X-Xmax) * 1/(sum_of_exponents)
                    x1 = y1 * sum1;
                    x2 = y2 * sum2;
                    x3 = y3 * sum3;
                    x4 = y4 * sum4;

                    v_bf16_st_tnsr(ifmCoords1, ofm, x1,0,ifmCoords1[width] < widthEnd);
                    v_bf16_st_tnsr(ifmCoords2, ofm, x2,0,ifmCoords2[width] < widthEnd);
                    v_bf16_st_tnsr(ifmCoords3, ofm, x3,0,ifmCoords3[width] < widthEnd);
                    v_bf16_st_tnsr(ifmCoords4, ofm, x4,0,ifmCoords4[width] < widthEnd);
                }
            }
        }
    }
}
