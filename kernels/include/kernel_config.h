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
typedef float                       SCALAR;
#define v_ld_tnsr_i(a,b)            v_f32_ld_tnsr_b(a,b)
#define v_sel_less_v_s_v_v(a,b,c,d) v_f32_sel_less_f32_b(a,b,c,d)
#define v_sel_geq_v_s_v_v(a,b,c,d)    v_f32_sel_geq_f32_b(a,b,c,d)
#define v_sel_grt_v_s_v_v(a,b,c,d)    v_f32_sel_grt_f32_b(a,b,c,d)
#define v_sel_leq_v_s_v_v(a, b, c, d) v_f32_sel_leq_f32_b(a, b, c, d)
#define v_sel_geq_v_s_v_v_b(a, b, c, d, i, p, o)  v_f32_sel_geq_f32_b(a, b, c, d, 0, i, p, o)
#define v_sel_less_v_s_v_v_b(a, b, c, d, i, p, o) v_f32_sel_less_f32_b(a, b, c, d, 0, i, p, o)
#define st_tnsr_i_v(a,b,c)          v_f32_st_tnsr(a,b,c)
#define bv_cmp_eq_v_v(a, b)         from_bool64(v_f32_cmp_eq_b(a, b))
#define bv_u_cmp_geq_v_s(a, b)      from_bool64(v_u32_cmp_geq_b(a, b))
#define v_mov_v_vb(a, b, c, d)      v_f32_mov_vb(a, 0, b, to_bool64(c), d)
#define v_mov_s_vb(a, b, c, d)      v_f32_mov_vb(a, 0, b, to_bool64(c), d)
#define v_add_v_v_b(a, b, c, d, e)  v_f32_add_b(a, b, 0, c, d, e)
#define v_mul_v_s(a, b)             v_f32_mul_b(a, b)
#define v_mul_v_v(a,b)              v_f32_mul_b(a, b)
#define s_ld_g_a(a)                 s_f32_ld_g(a)
#define v_ld_g_a(a)                 v_f32_ld_g(a)
#define v_mov_s(a)                  v_f32_mov_b(a)
#define log(a)                      log_f32(a)
#define exp(a)                      exp_f32(a)
#define V_LANE_ID                   read_lane_id_4b_b()
#define v_sel_leq_v_v_v_v(a, b, c, d) v_f32_sel_leq_f32_b(a, b, c, d)

#endif

#if defined(BFLOAT16)
#define VECTOR                      bfloat128
#define VECTOR_SIZE                 128
typedef bf16                        SCALAR;
#define v_ld_tnsr_i(a,b)            v_bf16_ld_tnsr_b(a,b)
#define v_sel_less_v_s_v_v(a,b,c,d) v_bf16_sel_less_bf16_b(a,b,c,d)
#define v_sel_geq_v_s_v_v(a,b,c,d)    v_bf16_sel_geq_bf16_b(a,b,c,d)
#define v_sel_grt_v_s_v_v(a,b,c,d)    v_bf16_sel_grt_bf16_b(a,b,c,d)
#define v_sel_leq_v_s_v_v(a, b, c, d) v_bf16_sel_leq_bf16_b(a, b, c, d)
#define v_sel_geq_v_s_v_v_b(a, b, c, d, i, p, o)  v_bf16_sel_geq_bf16_b(a, b, c, d, 0, i, p, o)
#define v_sel_less_v_s_v_v_b(a, b, c, d, i, p, o) v_bf16_sel_less_bf16_b(a, b, c, d, 0, i, p, o)
#define st_tnsr_i_v(a,b,c)          v_bf16_st_tnsr(a,b,c)

#endif
