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

//////////////////////////////////////// SIN_F32 /////////////////////////////////////////////////
// sin_fast_f32 
float64 sin_fast_f32(float64 input)
{
    float64 sign_res = v_f32_sel_grt_f32_b(input, 0.0f, 1.0f, -1.0f); // 1
    const float four_by_pi = 1.27323949;       /* 4/pi = 0x3fa2f983 */                        
    const float pi4_1      = 7.85156250e-01;   /* pi/4 = 0x3f490000 */                        
    const float pi4_2      = 2.41875648498e-4; /* 0x397da000 */                               
    const float pi4_3      = 3.7748949774e-8;  /* 0x3fa2f983*/                                
                                                                                              
    float64 abs_x        = v_f32_abs_b(input);                                                
    float64 fl_pi4_shift = abs_x * four_by_pi;                                                
    int64   pi4_shift    = v_convert_f32_to_i32_b(fl_pi4_shift, e_round_down << 16);          
    pi4_shift += pi4_shift & 1; /* Shift x in [-pi/4, +pi/4] */                               
    fl_pi4_shift          = v_convert_i32_to_f32_b(pi4_shift, e_round_half_ne << 16);         
    float64 reduced_x     = abs_x;                                                            
    reduced_x             = v_f32_mac_b(fl_pi4_shift, pi4_1, reduced_x, (1) << 1);         
    reduced_x             = v_f32_mac_b(fl_pi4_shift, pi4_2, reduced_x, (1) << 1);         
    reduced_x             = v_f32_mac_b(fl_pi4_shift, pi4_3, reduced_x, (1) << 1);         
    float64 abs_reduced_x = v_f32_abs_b(reduced_x);                                           
                                                                                              
    int64   pi2_shift     = (pi4_shift >> 1) & 3; /* remove shift by 2*pi: x in [0, 2*pi) */  
    float64 fl_sign_shift = v_convert_i32_to_f32_b(pi2_shift & 2, e_round_half_ne << 16);     
    sign_res -= fl_sign_shift * sign_res; /* x>pi? -> shift by pi: cos(pi-x) = -cos(x) */     
    pi2_shift -= pi2_shift & 2;           /* remove shift by pi -> pi2_shift in [0, 1] */     
                                                                                              
    const int COEFF_TAB_SHIFT = 17; /* 23 - (m = 6) */                                        
    const int FUNC_ID         = e_fp32_sin_cos;                                               
                                                                                              
    bool256               sin_x = from_bool64(v_i32_cmp_eq_b(pi2_shift, 0));         
    uint64_float64_pair_t all_coeffs_tab;                                                     
    all_coeffs_tab = v_f32_get_lut_entry_and_interval_start_b(                                
        abs_reduced_x, COEFF_TAB_SHIFT, (e_func_variant_sin_cos) << 13,                       
                                     (uint64_float64_pair_t){0}, 1, 0);                       
    uint64 intervals = all_coeffs_tab.v1;                                                     
    intervals        = v_u32_add_vb(intervals, 64, e_no_saturation, intervals,                
                                    to_bool64(sin_x), 1);                                     
    float64 value    = abs_reduced_x - all_coeffs_tab.v2;                                     
    float64 result;                                                                           
    
    //lookup and mac
    float64        C0 = v_f32_lookup_1c(intervals, FUNC_ID, SW_BV32, (float64){0} );    
    float64_pair_t C1C2;                                                                
    C1C2 = v_f32_lookup_2c(intervals, FUNC_ID, SW_BV32, (float64_pair_t){0} );          

    result = C1C2.v1;                                                                   
    result = v_f32_mac_b(C1C2.v2, value, result, (0) << 1);                         
    C0     = v_f32_mac_b(result, value, C0, (0) << 1);                              
    result = C0;  
                                                                                              
    result = v_f32_mul_vb(abs_reduced_x, result, 0, result, to_bool64(sin_x));

    sign_res =
        v_f32_sel_less_f32_vb(reduced_x, 0.0f, -sign_res, sign_res, 0, sign_res, to_bool64(sin_x));
    result = v_f32_sel_less_f32_b(sign_res, 0.0f, -result, result); // 30
    return result;
}

// sin_f32 VPU ops = sin_fast_f32 VPU ops + ABS + SIN_SPECIAL_VALUES = 30+1+5=36
float64 sin_f32(float64 input)
{
    float64 abs_x  = v_f32_abs_b(input);
    float64 result = sin_fast_f32(input);
    // ====================================
    //  Processing special values: +-inf, nan, sin/cos limits

    const uint64  nan_int          = 0x7fffffff;                                              
    const float64 nan_fp32         = *((float64*)&nan_int);                                 
    const float   sin_max_arg      = s_convert_i32_to_f32(0xffffff, e_round_half_ne << 16);
    const float   sin_accuracy_limit = s_convert_i32_to_f32(0x2000, e_round_half_ne << 16);   
                                                                                            
    result = v_f32_sel_grt_f32_b(abs_x, sin_accuracy_limit, 0.0f, result);                  
    result = v_f32_sel_grt_f32_b(abs_x, sin_max_arg, nan_fp32, result);                     
    result = v_f32_sel_geq_u32_b(*((uint64*)&abs_x), 0x7f800000, nan_fp32, result);
    // ====================================

    return result;
}

void main(tensor input, tensor output)
{
    const int depth     = 0;
    const int width     = 1;
    const int height    = 2;
    const int batch     = 3;
    const int rank4     = 4;

    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end = get_index_space_size() + index_space_start;

    int5 coords = { 0, 0, 0, 0, 0 };

    // DEPTH
    const int depthStep     = 64;
    const int depthStart    = index_space_start[depth] * depthStep;
    const int depthEnd      = index_space_end[depth] * depthStep;

    // WIDTH
    const int widthStep     = 4;
    const int widthStart    = index_space_start[width] * widthStep;
    const int widthEnd      = index_space_end[width] * widthStep;

    // HEIGHT
    const int heightStep    = 1;
    const int heightStart   = index_space_start[height];
    const int heightEnd     = index_space_end[height];

    // BATCH
    const int batchStep     = 1;
    const int batchStart    = index_space_start[batch];
    const int batchtEnd     = index_space_end[batch];

    // 5th dimension
    const int rank4Step     = 1;
    const int rank4Start    = index_space_start[rank4];
    const int rank4End     = index_space_end[rank4];

    for (int r = rank4Start; r < rank4End; r += rank4Step)
    {
        coords[rank4] = r;
        for (int b = batchStart; b < batchtEnd; b += batchStep)
        {
            coords[batch] = b;

            for (int h = heightStart; h < heightEnd; h += heightStep)
            {
                coords[height] = h;
                for (int d = depthStart; d < depthEnd; d += depthStep)
                {
                    coords[depth] = d;
                    #pragma loop_unroll(4)
                    for (int w = widthStart; w < widthEnd; w += 1)
                    {
                        coords[width] = w;

                        float64 x = v_f32_ld_tnsr_b(coords, input);

                        float64 y = sin_f32(x);

                        v_f32_st_tnsr(coords, output, y);
                    }
                }
            }
        }
    }
}
