/**********************************************************************
Copyright (c) 2021 Habana Labs. All rights reserved.

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

#if defined(FLOAT32)
#define VECTOR                      float64
#define VECTOR_SIZE                 64
#define V_LANE_ID                   V_LANE_ID_32
#define VECTOR_SUM                  float64
#define v_ld_tnsr_i(a,b)            v_f32_ld_tnsr_i(a,b)
#define v_sel_less_v_s_v_v(a,b,c,d) v_f32_sel_less_f32_b(a,b,c,d)
#define v_sel_geq_v_s_v_v(a,b,c,d)  v_f32_sel_geq_f32_b(a,b,c,d)
#define v_sel_grt_v_s_v_v(a,b,c,d)  v_f32_sel_grt_f32_b(a,b,c,d)
#define st_tnsr_i_v(a,b,c)          f32_st_tnsr_i_v(a,b,c)
#define bv_cmp_eq_v_v(a, b)         from_bool64(v_f32_cmp_eq_b(a, b))
#define bv_u_cmp_geq_v_s(a, b)      from_bool64(v_u32_cmp_geq_b(a, b))
#define v_mov_v_vb(a, b, c, d)      v_f32_mov_vb(a, 0, b, to_bool64(c), d)
#define v_mov_s_vb(a, b, c, d)      v_f32_mov_vb(a, 0, b, to_bool64(c), d)
#define v_add_v_v_b(a, b, c, d, e)  v_f32_add_b(a, b, 0, c, d, e)
#define v_mul_v_s(a, b)             v_f32_mul_b(a, b)
#define v_mul_v_v(a,b)              v_f32_mul_b(a, b)
#define s_ld_g_a(a)         s_f32_ld_g(a)
#define v_mov_s(a)                  v_f32_mov_b(a)
#define log(a)                      log_f32(a)

float64 log_f32(float64 input)
{
    //  log32_cephes: log(1+x) = x - 0.5 x**2 + x**3 P(x)
    const float log2_e_m_1            = 0.44269504088896340735992; // log2(e) - 1.0
    const float ln_2                  = 0.69314718056;             // ln(2)        (0x3f317218)
    const float one_sqrt_2            = 0.70710677;                // 1/sqrt(2)    (0x3f3504f3)
    const int   poly_tab_size         = 9;
    const float coeffs[poly_tab_size] = {7.0376836292E-2,
                                         -1.1514610310E-1,
                                         1.1676998740E-1,
                                         -1.2420140846E-1,
                                         1.4249322787E-1,
                                         -1.6668057665E-1,
                                         2.0000714765E-1,
                                         -2.4999993993E-1,
                                         3.3333331174E-1};

    int64      exponent    = v_f32_extract_exp_b(input, 0) + 1; // 1, 2
    const char exp_126     = 126;
    float64    fraction    = v_f32_form_fp_num_ie_b(exp_126, input, input, SW_EXP_IS_NUM); // 3
    float64    fl_exponent = v_convert_i32_to_f32_b(exponent, e_round_half_ne << 16);      // 4

    float64 diff      = fraction - one_sqrt_2;                                             // 5
    int64   diff_sign = v_i32_shr_b((*(int64*)&diff), 31);                                 // 6
    exponent -= diff_sign;                                                                 // 7
    fl_exponent          = v_convert_i32_to_f32_b(exponent, e_round_half_ne << 16);        // 8
    float64 fl_diff_sign = v_convert_i32_to_f32_b(diff_sign, e_round_half_ne << 16);       // 9
    fraction += fl_diff_sign * fraction - 1.0f;                                            // 10, 11

    float64 x_sqr = fraction * fraction; // 12
    float64 poly  = coeffs[0];
    for (int i = 1; i < poly_tab_size; i++)                          // 13, 14, 15, 16
        poly = v_f32_mac_b(poly, fraction, coeffs[i], (0) << 1); // 17, 18, 19, 20

    float64 tailor = fraction * (x_sqr * poly); // 21, 22
    tailor -= 0.5 * x_sqr;                      // 23

    float64 result = tailor * log2_e_m_1; // 24
    result += fraction * log2_e_m_1;      // 25
    result += tailor;                     // 26
    result += fraction;                   // 27

    result += fl_exponent; // 28
    result *= ln_2;        // 29

    float64 fclass = v_f32_fclass_b(input);
    result         = v_f32_calc_fp_special_b(fclass, fclass, e_fp_log, result);

    return result;
} // log_f32_cephes
#endif

#if defined(BFLOAT16)
#define VECTOR                      bfloat128
#define VECTOR_SIZE                 128
#define V_LANE_ID                   V_LANE_ID_16
#define LU_DT16                     SW_BV16
#define VECTOR_SUM                  float128    // for special use double bfloat size
#define v_ld_tnsr_i(a,b)            v_bf16_ld_tnsr_i(a,b)
#define v_sel_less_v_s_v_v(a,b,c,d) v_bf16_sel_less_bf16_b(a,b,c,d)
#define v_sel_geq_v_s_v_v(a,b,c,d)  v_bf16_sel_geq_bf16_b(a,b,c,d)
#define v_sel_grt_v_s_v_v(a,b,c,d)  v_bf16_sel_grt_bf16_b(a,b,c,d)
#define st_tnsr_i_v(a,b,c)          bf16_st_tnsr_i_v(a,b,c)
#define bv_cmp_eq_v_v(a, b)         from_bool128(v_bf16_cmp_eq_b(a, b))
#define bv_u_cmp_geq_v_s(a, b)      from_bool128(v_u16_cmp_geq_b(a, b))
#define v_mov_v_vb(a, b, c, d)      v_bf16_mov_vb(a, 0, b, to_bool128(c), d)
#define v_mov_s_vb(a, b, c, d)      v_bf16_mov_vb(a, 0, b, to_bool128(c), d)
#define v_add_v_v_b(a, b, c, d, e)  v_bf16_add_b(a, b, 0, c, d, e)
#define v_mul_v_s(a, b)             v_bf16_mul_b(a, b)
#define v_mul_v_v(a,b)              v_bf16_mul_b(a, b)
#define s_ld_g_a(a)         s_bf16_ld_g(a)
#define v_mov_s(a)                  v_bf16_mov_b(a)
#define log(a)                      log_bf16(a)
// the following only for BF16
#define av_mac_acc32_b(a, b, acc, neg, c, d) v_bf16_mac_acc32_b(a, b, acc, (neg) << 1, c, d)
#define v_convert_f32_to_vec(a, rnd)    v_convert_f32_to_bf16_all_b(a, (rnd <<16))
#define av_mac_acc32(a, b, acc, neg)    v_bf16_mac_acc32_b(a, b, acc, neg)


bfloat128 log_bf16(bfloat128 input)
{
    const int C0C1_func_id = e_bf16_log2_linear_interleaved_m5;

    bfloat128 result;

    // FIXME: logic of second input parameter in v_bf16_extract_exp_) is inverted.
    short128 exp = v_bf16_extract_exp_b(input, /*true*/ 0);                                 // 1
    bfloat128 masked_inp = (bfloat128) v_u16_and_b((ushort128)input, 0xFFFCu);              // 2

    ushort128_bfloat128_pair_t all_coeffs_tab;
    all_coeffs_tab = v_bf16_get_lut_entry_and_interval_start_b(masked_inp,
                                                               0,
                                                            // TODO: NEED modif e_func_variant enum
                                                               4 << 13);

    ushort128 masked_exp = v_u16_and_b((ushort128)all_coeffs_tab.v1, 0x0003u);              // 4

    bfloat128 interval;

    // V_Dest = SIGN(Src2) | EXPONENT(Src1) | MANTISSA(Src3)
    masked_inp = v_bf16_form_fp_num_b(1., all_coeffs_tab.v2, all_coeffs_tab.v2);                // 5
    interval = (bfloat128) v_bf16_sel_eq_u16_b(masked_exp, 0x2u, all_coeffs_tab.v2, masked_inp); //6
    // V_Dest = SIGN(Src2) | EXPONENT(Src1) | MANTISSA(Src3)
    masked_inp = v_bf16_form_fp_num_b(interval, input, input);                                  // 7
    interval = v_bf16_sub_b(masked_inp, interval, e_no_negation << 1);                          // 8

    bfloat128_pair_t C0C1 = {0};
    C0C1 = v_bf16_lookup_2c((ushort128) all_coeffs_tab.v1, C0C1_func_id, LU_DT16, C0C1);  // 9
    result = v_bf16_mac_b(C0C1.v2, interval, C0C1.v1);                                          //10

    bfloat128 floorExp = v_convert_i16_to_bf16_b((short128)exp, e_round_half_ne << 16);         //11
    bool256 IsZero = from_bool128(v_u16_cmp_eq_b(masked_exp, 0.));                              //12

    C0C1.v1 = v_bf16_sub_b(masked_inp, 1., e_no_negation << 1);                             //13
    result = v_bf16_mul_vb(result, C0C1.v1, 0, result, to_bool128(IsZero), 1);              //14

    result = v_bf16_add_vb(result, floorExp, 0, result, to_bool128(IsZero));                //15

    result = v_bf16_calc_fp_special_b(v_bf16_fclass_b(input),
                                    v_bf16_fclass_b(input), e_fp_log, result);              // 16,17

    return v_bf16_mul_b(result, 0.69314718); 
}

#endif
