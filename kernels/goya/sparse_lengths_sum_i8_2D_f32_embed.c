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
// Cast char256 into int256
int256 cast_i8_to_32bits_lin_order(char256 x)
{
    int256 y;
    int64 t0, t1, t2;

    // 0..15, 64..79, 128..143, 192..207
    y.v1 = (int64) v_i8_unpack_b(x, ((e_group_0) << 8) | ((e_every_forth_element) << 9) | ((e_lower_half_group) << 10), 0);
    // 16..31, 80..95, 144..159, 208..223
    t1 = (int64) v_i8_unpack_b(x, ((e_group_0) << 8) | ((e_every_forth_element) << 9) | ((e_higher_half_group) << 10), 0);
    // 32..47, 96..111, 160..175, 224..239
    t2 = (int64) v_i8_unpack_b(x, ((e_group_1) << 8) | ((e_every_forth_element) << 9) | ((e_lower_half_group) << 10), 0);
    // 48..63, 112..127, 176..191, 240..255
    y.v4 = (int64) v_i8_unpack_b(x, ((e_group_1) << 8) | ((e_every_forth_element) << 9) | ((e_higher_half_group) << 10), 0);

    // 0..15
    t0 = y.v1;
    // 0..15, 16..31
    y.v1 = v_i32_mov_dual_group_b(t1, 0xFFFFFFFF, 0, 1, MkWr(1, 1), y.v1);
    // 0..15, 16..31, 32..47, 48..63
    y.v1 = v_i32_mov_dual_group_b(t2, 0xFFFFFFFF, 0, 2, MkWr(1, 1), y.v1);
    // 0..15, 64..79, 128..143, 192..207
    y.v1 = v_i32_mov_dual_group_b(y.v4, 0xFFFFFFFF, 0, 3, MkWr(1, 1), y.v1);

    // 64..79
    y.v2 = v_i32_mov_dual_group_b(t0, 0xFFFFFFFF, 1, 0, MkWr(1, 1), y.v2);
    // 64..79, 80..95
    y.v2 = v_i32_mov_dual_group_b(t1, 0xFFFFFFFF, 1, 1, MkWr(1, 1), y.v2);
    // 64..79, 80..95, 96..111
    y.v2 = v_i32_mov_dual_group_b(t2, 0xFFFFFFFF, 1, 2, MkWr(1, 1), y.v2);
    // 64..79, 80..95, 96..111, 112..127
    y.v2 = v_i32_mov_dual_group_b(y.v4, 0xFFFFFFFF, 1, 3, MkWr(1, 1), y.v2);

    // 128..143
    y.v3 = v_i32_mov_dual_group_b(t0, 0xFFFFFFFF, 2, 0, MkWr(1, 1), y.v3);
    // 128..143, 144..159
    y.v3 = v_i32_mov_dual_group_b(t1, 0xFFFFFFFF, 2, 1, MkWr(1, 1), y.v3);
    // 128..143, 144..159, 160..175
    y.v3 = v_i32_mov_dual_group_b(t2, 0xFFFFFFFF, 2, 2, MkWr(1, 1), y.v3);
    // 128..143, 144..159, 160..175, 176..191
    y.v3 = v_i32_mov_dual_group_b(y.v4, 0xFFFFFFFF, 2, 3, MkWr(1, 1), y.v3);

    // 192..207
    y.v4 = v_i32_mov_dual_group_b(t0, 0xFFFFFFFF, 3, 0, MkWr(1, 1), y.v4);
    // 192..207, 208..223
    y.v4 = v_i32_mov_dual_group_b(t1, 0xFFFFFFFF, 3, 1, MkWr(1, 1), y.v4);
    // 192..207, 208..223, 224..239
    y.v4 = v_i32_mov_dual_group_b(t2, 0xFFFFFFFF, 3, 2, MkWr(1, 1), y.v4);

    return y;
}

#ifndef __HABANA_TOOL_VERSION
    #define __HABANA_TOOL_VERSION 0
    #define VERSION2DEC(a,b,c) 1
#endif

#if (__HABANA_TOOL_VERSION < VERSION2DEC(0,2,0))
typedef struct _float256
{
    float64 v1;
    float64 v2;
    float64 v3;
    float64 v4;
} float256;
#endif

//macro to add two float256 vector with a scalar predicate
#define av_f32_add_v_v_b(acc_out, acc_in1, acc_in2, predicate) \
(acc_out).v1 = v_f32_add_b((acc_in1).v1, (acc_in2).v1, 0, (acc_out).v1, predicate); \
(acc_out).v2 = v_f32_add_b((acc_in1).v2, (acc_in2).v2, 0, (acc_out).v2, predicate); \
(acc_out).v3 = v_f32_add_b((acc_in1).v3, (acc_in2).v3, 0, (acc_out).v3, predicate); \
(acc_out).v4 = v_f32_add_b((acc_in1).v4, (acc_in2).v4, 0, (acc_out).v4, predicate);

void main(tensor input_tensor,
          tensor indices_tensor,
          tensor lengths_tensor,
          tensor output_tensor)
{
    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end = get_index_space_size() + index_space_start;

    // DEPTH
    const int depth_step  = 256;
    const int depth_start = index_space_start[0] * depth_step;
    const int depth_end   = index_space_end[0] * depth_step;

    // WIDTH
    const int width_step  = 2;
    const int width_start = index_space_start[1] * width_step;
    const int width_end   = index_space_end[1] * width_step;

    int5 in_coord_1 = {0};
    int5 in_coord_2 = {0};
    int5 idx_coord_1 = {0};
    int5 idx_coord_2 = {0};
    int5 lengths_coord_1 = {0};
    int5 lengths_coord_2 = {0};
    int5 out_coord_1 = {0};
    int5 out_coord_2 = {0};
    int5 scale_bias_coord_1 = {0};
    int5 scale_bias_coord_2 = {0};

    const int input_dim0_len = get_dim_size(input_tensor, 0);
    //the column number for scale and bias (the two float values take up 4 int8 pockets each)
    const int scale_column = input_dim0_len - 8;

    //assigning scale column
    scale_bias_coord_1[0] = scale_bias_coord_2[0] = scale_column;

    // LUT1 for shuffling first element to all 64 elements of the first dual group
    const uchar256 lut1 = 0x80;
    // LUT2 for shuffling second element to all 64 elements of the first dual group
    const uchar256 lut2 = 0x81;

    int256  v_zero;
    v_zero.v1 = v_zero.v2 = v_zero.v3 = v_zero.v4 = v_i32_mov_b(0);

    int index_offset = 0;

    //finding the sum of length tensor upto the current element
    for(int segment_no = 0; segment_no < width_start; segment_no++)
    {
        lengths_coord_1[0] = segment_no;
        __global__ int* len_coord_ptr = gen_addr(lengths_coord_1, lengths_tensor);
        index_offset += s_i32_ld_g(len_coord_ptr);
    }
    //this is the index tensor offset
    const int index_offset_orig = index_offset;

    for (int depth = depth_start; depth < depth_end; depth += depth_step)
    {
        in_coord_1[0] = out_coord_1[0] = depth;
        in_coord_2[0] = out_coord_2[0] = depth;
        //after each iteration in depth, offset_1 is reset to the original length tensor offset
        int index_offset_1 = index_offset_orig;
        int index_offset_2;

        //iterating along the length tensor (i.e. the width of the output)
        for (int segment_no = width_start; segment_no < width_end; segment_no += width_step)
        {
            //processing two length elements at a time
            out_coord_1[1] = lengths_coord_1[0] = segment_no + 0;
            out_coord_2[1] = lengths_coord_2[0] = segment_no + 1;

            //address generation
            __global__ int* len_coord_ptr_1 = gen_addr(lengths_coord_1, lengths_tensor);
            __global__ int* len_coord_ptr_2 = gen_addr(lengths_coord_2, lengths_tensor);
            //obtaining the two segment lengths
            const int segment_length_1 = s_i32_ld_g(len_coord_ptr_1);
            const int segment_length_2 = s_i32_ld_g(len_coord_ptr_2);
            //offset_2 = offset_1 + segment_length_1
            index_offset_2 = index_offset_1 + segment_length_1;
            //finding the larger of the two lengths to iterate until
            int max_length = s_i32_max(segment_length_2, segment_length_1);
            //augmented float vector to hold the result
            float256 out_value_1 = {0};
            float256 out_value_2 = {0};
            //intermediate int augmented vectors for the conversion from char256 to float256
            int256 in_value_1_int_av;
            int256 in_value_2_int_av;

            idx_coord_1[0] = index_offset_1;
            idx_coord_2[0] = index_offset_2;
            //prologue
            __global__ int* idx_coord_ptr_1 = gen_addr(idx_coord_1, indices_tensor);
            __global__ int* idx_coord_ptr_2 = gen_addr(idx_coord_2, indices_tensor);

            scale_bias_coord_1[1] = in_coord_1[1] = s_i32_ld_g(idx_coord_ptr_1);
            scale_bias_coord_2[1] = in_coord_2[1] = s_i32_ld_g(idx_coord_ptr_2);
            //loading value from input
            char256 in_value_1 = v_i8_ld_tnsr_b(in_coord_1, input_tensor);
            char256 in_value_2 = v_i8_ld_tnsr_b(in_coord_2, input_tensor);

            //loading the vector containing the bias and the zp
            float64 scale_zp_1 = v_f32_ld_tnsr_low_b(scale_bias_coord_1, input_tensor);
            float64 scale_zp_2 = v_f32_ld_tnsr_low_b(scale_bias_coord_2, input_tensor);

            //extracting scale and zp into separate vectors
            /* Extract scale and broadcast to the whole vector */
            // Shuffle first element of vector to dual group 0
            float64 scale_1_v = v_f32_shuffle_b(scale_zp_1, lut1, 0, scale_zp_1);
            float64 scale_2_v = v_f32_shuffle_b(scale_zp_2, lut1, 0, scale_zp_2);
            // Move dual group 0 to dual group 1
            scale_1_v = v_f32_mov_dual_group_b(scale_1_v, 0xFFFFFFFF, 0, 1, MkWr(1, 1), scale_1_v);
            scale_2_v = v_f32_mov_dual_group_b(scale_2_v, 0xFFFFFFFF, 0, 1, MkWr(1, 1), scale_2_v);
            // Move dual group 0 to dual group 2
            scale_1_v = v_f32_mov_dual_group_b(scale_1_v, 0xFFFFFFFF, 0, 2, MkWr(1, 1), scale_1_v);
            scale_2_v = v_f32_mov_dual_group_b(scale_2_v, 0xFFFFFFFF, 0, 2, MkWr(1, 1), scale_2_v);
            // Move dual group 0 to dual group 3
            scale_1_v = v_f32_mov_dual_group_b(scale_1_v, 0xFFFFFFFF, 0, 3, MkWr(1, 1), scale_1_v);
            scale_2_v = v_f32_mov_dual_group_b(scale_2_v, 0xFFFFFFFF, 0, 3, MkWr(1, 1), scale_2_v);

            //assuming that the second column of the scale-bias tensor is filled with -sc*zp values
            float64 neg_scale_x_bias_1_v = v_f32_shuffle_b(scale_zp_1, lut2, 0, scale_zp_1);
            float64 neg_scale_x_bias_2_v = v_f32_shuffle_b(scale_zp_2, lut2, 0, scale_zp_2);

            // Move dual group 0 to dual group 1
            neg_scale_x_bias_1_v = v_f32_mov_dual_group_b(neg_scale_x_bias_1_v, 0xFFFFFFFF, 0, 1, MkWr(1, 1), \
                 neg_scale_x_bias_1_v);
            neg_scale_x_bias_2_v = v_f32_mov_dual_group_b(neg_scale_x_bias_2_v, 0xFFFFFFFF, 0, 1, MkWr(1, 1), \
                 neg_scale_x_bias_2_v);
            // Move dual group 0 to dual group 2
            neg_scale_x_bias_1_v = v_f32_mov_dual_group_b(neg_scale_x_bias_1_v, 0xFFFFFFFF, 0, 2, MkWr(1, 1), \
                 neg_scale_x_bias_1_v);
            neg_scale_x_bias_2_v = v_f32_mov_dual_group_b(neg_scale_x_bias_2_v, 0xFFFFFFFF, 0, 2, MkWr(1, 1), \
                 neg_scale_x_bias_2_v);
            // Move dual group 0 to dual group 3
            neg_scale_x_bias_1_v = v_f32_mov_dual_group_b(neg_scale_x_bias_1_v, 0xFFFFFFFF, 0, 3, MkWr(1, 1), \
                 neg_scale_x_bias_1_v);
            neg_scale_x_bias_2_v = v_f32_mov_dual_group_b(neg_scale_x_bias_2_v, 0xFFFFFFFF, 0, 3, MkWr(1, 1), \
                 neg_scale_x_bias_2_v);

            //char256 to int256
            in_value_1_int_av = cast_i8_to_32bits_lin_order(in_value_1);
            in_value_2_int_av = cast_i8_to_32bits_lin_order(in_value_2);

            float256 in_value_1_float;
            float256 in_value_2_float;

            //iterating through the elements to be accumulated
            for (int element_no = 1; element_no < max_length; element_no++)
            {
                //this predicate lets us stop accumulating past the the segment length selectively
                char pred_1 = s_i32_cmp_leq(element_no, segment_length_1);
                char pred_2 = s_i32_cmp_leq(element_no, segment_length_2);

                //conversion to f32
                in_value_1_float.v1 = v_convert_i8_to_f32_b((char256)in_value_1_int_av.v1);
                in_value_1_float.v2 = v_convert_i8_to_f32_b((char256)in_value_1_int_av.v2);
                in_value_1_float.v3 = v_convert_i8_to_f32_b((char256)in_value_1_int_av.v3);
                in_value_1_float.v4 = v_convert_i8_to_f32_b((char256)in_value_1_int_av.v4);
                //application of scale and bias
                in_value_1_float.v1 = v_f32_mac_b(in_value_1_float.v1, scale_1_v, neg_scale_x_bias_1_v, (e_no_negation) << 1);
                in_value_1_float.v2 = v_f32_mac_b(in_value_1_float.v2, scale_1_v, neg_scale_x_bias_1_v, (e_no_negation) << 1);
                in_value_1_float.v3 = v_f32_mac_b(in_value_1_float.v3, scale_1_v, neg_scale_x_bias_1_v, (e_no_negation) << 1);
                in_value_1_float.v4 = v_f32_mac_b(in_value_1_float.v4, scale_1_v, neg_scale_x_bias_1_v, (e_no_negation) << 1);
                //conversion to f32
                in_value_2_float.v1 = v_convert_i8_to_f32_b((char256)in_value_2_int_av.v1);
                in_value_2_float.v2 = v_convert_i8_to_f32_b((char256)in_value_2_int_av.v2);
                in_value_2_float.v3 = v_convert_i8_to_f32_b((char256)in_value_2_int_av.v3);
                in_value_2_float.v4 = v_convert_i8_to_f32_b((char256)in_value_2_int_av.v4);
                //application of scale and bias
                in_value_2_float.v1 = v_f32_mac_b(in_value_2_float.v1, scale_2_v, neg_scale_x_bias_2_v, (e_no_negation) << 1);
                in_value_2_float.v2 = v_f32_mac_b(in_value_2_float.v2, scale_2_v, neg_scale_x_bias_2_v, (e_no_negation) << 1);
                in_value_2_float.v3 = v_f32_mac_b(in_value_2_float.v3, scale_2_v, neg_scale_x_bias_2_v, (e_no_negation) << 1);
                in_value_2_float.v4 = v_f32_mac_b(in_value_2_float.v4, scale_2_v, neg_scale_x_bias_2_v, (e_no_negation) << 1);

                //next index coordinate
                idx_coord_1[0]++;
                idx_coord_2[0]++;

                idx_coord_ptr_1 = gen_addr(idx_coord_1, indices_tensor);
                idx_coord_ptr_2 = gen_addr(idx_coord_2, indices_tensor);

                scale_bias_coord_1[1] = in_coord_1[1] = s_i32_ld_g(idx_coord_ptr_1);
                scale_bias_coord_2[1] = in_coord_2[1] = s_i32_ld_g(idx_coord_ptr_2);
                //loading value from input
                in_value_1 = v_i8_ld_tnsr_b(in_coord_1, input_tensor);
                in_value_2 = v_i8_ld_tnsr_b(in_coord_2, input_tensor);

                //char256 to float256
                in_value_1_int_av = cast_i8_to_32bits_lin_order(in_value_1);
                in_value_2_int_av = cast_i8_to_32bits_lin_order(in_value_2);
                //accumulating
                av_f32_add_v_v_b(out_value_1, in_value_1_float, out_value_1, pred_1);
                av_f32_add_v_v_b(out_value_2, in_value_2_float, out_value_2, pred_2);

                //scale is loaded from input tensor in the embedded version
                //loading the vector containing the scale and the zp
                float64 scale_zp_1 = v_f32_ld_tnsr_b(scale_bias_coord_1, input_tensor);
                float64 scale_zp_2 = v_f32_ld_tnsr_b(scale_bias_coord_2, input_tensor);

                //extracting scale and zp into separate vectors
                /* Extract scale and broadcast to the whole vector */
                // Shuffle first element of vector to dual group 0
                scale_1_v = v_f32_shuffle_b(scale_zp_1, lut1, 0, scale_zp_1);
                scale_2_v = v_f32_shuffle_b(scale_zp_2, lut1, 0, scale_zp_2);
                // Move dual group 0 to dual group 1
                scale_1_v = v_f32_mov_dual_group_b(scale_1_v, 0xFFFFFFFF, 0, 1, MkWr(1, 1), scale_1_v);
                scale_2_v = v_f32_mov_dual_group_b(scale_2_v, 0xFFFFFFFF, 0, 1, MkWr(1, 1), scale_2_v);
                // Move dual group 0 to dual group 2
                scale_1_v = v_f32_mov_dual_group_b(scale_1_v, 0xFFFFFFFF, 0, 2, MkWr(1, 1), scale_1_v);
                scale_2_v = v_f32_mov_dual_group_b(scale_2_v, 0xFFFFFFFF, 0, 2, MkWr(1, 1), scale_2_v);
                // Move dual group 0 to dual group 3
                scale_1_v = v_f32_mov_dual_group_b(scale_1_v, 0xFFFFFFFF, 0, 3, MkWr(1, 1), scale_1_v);
                scale_2_v = v_f32_mov_dual_group_b(scale_2_v, 0xFFFFFFFF, 0, 3, MkWr(1, 1), scale_2_v);

                //assuming that the second column of the scale-bias tensor is filled with -sc*zp values
                neg_scale_x_bias_1_v = v_f32_shuffle_b(scale_zp_1, lut2, 0, scale_zp_1);
                neg_scale_x_bias_2_v = v_f32_shuffle_b(scale_zp_2, lut2, 0, scale_zp_2);

                // Move dual group 0 to dual group 1
                neg_scale_x_bias_1_v = v_f32_mov_dual_group_b(neg_scale_x_bias_1_v, 0xFFFFFFFF, 0, 1, MkWr(1, 1), \
                     neg_scale_x_bias_1_v);
                neg_scale_x_bias_2_v = v_f32_mov_dual_group_b(neg_scale_x_bias_2_v, 0xFFFFFFFF, 0, 1, MkWr(1, 1), \
                     neg_scale_x_bias_2_v);
                // Move dual group 0 to dual group 2
                neg_scale_x_bias_1_v = v_f32_mov_dual_group_b(neg_scale_x_bias_1_v, 0xFFFFFFFF, 0, 2, MkWr(1, 1), \
                     neg_scale_x_bias_1_v);
                neg_scale_x_bias_2_v = v_f32_mov_dual_group_b(neg_scale_x_bias_2_v, 0xFFFFFFFF, 0, 2, MkWr(1, 1), \
                     neg_scale_x_bias_2_v);
                // Move dual group 0 to dual group 3
                neg_scale_x_bias_1_v = v_f32_mov_dual_group_b(neg_scale_x_bias_1_v, 0xFFFFFFFF, 0, 3, MkWr(1, 1), \
                     neg_scale_x_bias_1_v);
                neg_scale_x_bias_2_v = v_f32_mov_dual_group_b(neg_scale_x_bias_2_v, 0xFFFFFFFF, 0, 3, MkWr(1, 1), \
                     neg_scale_x_bias_2_v);

            }
            //epilogue

            char pred_1 = s_i32_cmp_leq(max_length, segment_length_1);
            char pred_2 = s_i32_cmp_leq(max_length, segment_length_2);

            in_value_1_float.v1 = v_convert_i8_to_f32_b((char256)in_value_1_int_av.v1);
            in_value_1_float.v2 = v_convert_i8_to_f32_b((char256)in_value_1_int_av.v2);
            in_value_1_float.v3 = v_convert_i8_to_f32_b((char256)in_value_1_int_av.v3);
            in_value_1_float.v4 = v_convert_i8_to_f32_b((char256)in_value_1_int_av.v4);
            in_value_1_float.v1 = v_f32_mac_b(in_value_1_float.v1, scale_1_v, neg_scale_x_bias_1_v, (e_no_negation) << 1);
            in_value_1_float.v2 = v_f32_mac_b(in_value_1_float.v2, scale_1_v, neg_scale_x_bias_1_v, (e_no_negation) << 1);
            in_value_1_float.v3 = v_f32_mac_b(in_value_1_float.v3, scale_1_v, neg_scale_x_bias_1_v, (e_no_negation) << 1);
            in_value_1_float.v4 = v_f32_mac_b(in_value_1_float.v4, scale_1_v, neg_scale_x_bias_1_v, (e_no_negation) << 1);

            in_value_2_float.v1 = v_convert_i8_to_f32_b((char256)in_value_2_int_av.v1);
            in_value_2_float.v2 = v_convert_i8_to_f32_b((char256)in_value_2_int_av.v2);
            in_value_2_float.v3 = v_convert_i8_to_f32_b((char256)in_value_2_int_av.v3);
            in_value_2_float.v4 = v_convert_i8_to_f32_b((char256)in_value_2_int_av.v4);
            in_value_2_float.v1 = v_f32_mac_b(in_value_2_float.v1, scale_2_v, neg_scale_x_bias_2_v, (e_no_negation) << 1);
            in_value_2_float.v2 = v_f32_mac_b(in_value_2_float.v2, scale_2_v, neg_scale_x_bias_2_v, (e_no_negation) << 1);
            in_value_2_float.v3 = v_f32_mac_b(in_value_2_float.v3, scale_2_v, neg_scale_x_bias_2_v, (e_no_negation) << 1);
            in_value_2_float.v4 = v_f32_mac_b(in_value_2_float.v4, scale_2_v, neg_scale_x_bias_2_v, (e_no_negation) << 1);

            av_f32_add_v_v_b(out_value_1, in_value_1_float, out_value_1, pred_1);
            av_f32_add_v_v_b(out_value_2, in_value_2_float, out_value_2, pred_2);
            //epilogue ends here

            //for next iteration, offset is calculated from the last segment of the current iteration
            index_offset_1 = index_offset_2 + segment_length_2;

            //store the output vectors
            v_f32_st_tnsr(out_coord_1, output_tensor, out_value_1.v1); out_coord_1[0] += 64;
            v_f32_st_tnsr(out_coord_1, output_tensor, out_value_1.v2); out_coord_1[0] += 64;
            v_f32_st_tnsr(out_coord_1, output_tensor, out_value_1.v3); out_coord_1[0] += 64;
            v_f32_st_tnsr(out_coord_1, output_tensor, out_value_1.v4); out_coord_1[0] -= 192;
            v_f32_st_tnsr(out_coord_2, output_tensor, out_value_2.v1); out_coord_2[0] += 64;
            v_f32_st_tnsr(out_coord_2, output_tensor, out_value_2.v2); out_coord_2[0] += 64;
            v_f32_st_tnsr(out_coord_2, output_tensor, out_value_2.v3); out_coord_2[0] += 64;
            v_f32_st_tnsr(out_coord_2, output_tensor, out_value_2.v4); out_coord_2[0] -= 192;
        }
    }
}
