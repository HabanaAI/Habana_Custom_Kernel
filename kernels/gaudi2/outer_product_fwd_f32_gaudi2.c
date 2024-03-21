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
#pragma tpc_printf (enable)
// #define PRINTF_ENABLE 1

#define VECTORLENGTH 64
#define A_LEN 64
#define PART_WIDTH 2
#define PART_HEIGHT 1

// __local__ float a_vals[A_LEN];
__local__ float64 b_v[1];

int64 v_broadcast_element_32bits(int64 x, int N)
{
    /////////////////////////////////////////////////////////////////////////////////////
    // extract bits 0-2 of N to determine the intra-group of shuffle (0-7)
    /////////////////////////////////////////////////////////////////////////////////////
    char offset = N & 0x7; //bits [2:0]

    // add enable bit to shuffle ptrn;
    uchar256 shflPtrn = 0x80 + offset;

    /////////////////////////////////////////////////////////////////////////////////////
    // extract bit 3 of N to determine the inter group of shuffle (group 0/1)
    /////////////////////////////////////////////////////////////////////////////////////
    int gSel = N & 0x8; //bit [3]

    // if group == 1: enable bit for group1
    shflPtrn = v_u8_add_b(shflPtrn, 0x20, 0, shflPtrn, (gSel == 0x8));

    // shuffle the data in all dual groups
    int64 shflData = v_i32_shuffle_b(x, shflPtrn, 0, x);

    /////////////////////////////////////////////////////////////////////////////////////
    // extract bits 4-5 of N to determine the dual group for _mov_dual_group (dg 0/1/2/3)
    /////////////////////////////////////////////////////////////////////////////////////
    int dgSel = N & 0x30; //bits [5:4]

    // move the chosen shflData to all the dual groups
    shflData = v_i32_mov_dual_group_all_b(shflData, 0xFFFFFFFF, 0, 0, 0, 0, MkWrA(0b11, 0b11, 0b11, 0b11), shflData, (dgSel == 0x00));
    shflData = v_i32_mov_dual_group_all_b(shflData, 0xFFFFFFFF, 1, 1, 1, 1, MkWrA(0b11, 0b11, 0b11, 0b11), shflData, (dgSel == 0x10));
    shflData = v_i32_mov_dual_group_all_b(shflData, 0xFFFFFFFF, 2, 2, 2, 2, MkWrA(0b11, 0b11, 0b11, 0b11), shflData, (dgSel == 0x20));
    shflData = v_i32_mov_dual_group_all_b(shflData, 0xFFFFFFFF, 3, 3, 3, 3, MkWrA(0b11, 0b11, 0b11, 0b11), shflData, (dgSel == 0x30));

    return shflData;
}

void main(tensor a_mat, tensor b_mat, tensor o_mat)
{
    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end = get_index_space_size() + index_space_start;
    int5 a_coords = {0};
    int5 b_coords = {0};
    int5 o_coords = {0};

    float64 a_f32 = 0;
    int64 a_int32 = 0;

    for(int batch = index_space_start[3];
            batch < index_space_end[3];
            batch++)
    {
        a_coords[3] = batch;
        b_coords[3] = batch;
        o_coords[3] = batch;

    for(int head = index_space_start[2];
            head < index_space_end[2];
            head++)
    {
        a_coords[2] = head;
        b_coords[2] = head;
        o_coords[2] = head;

        for(int block_y = index_space_start[1] * PART_HEIGHT;
                block_y < index_space_end[1] * PART_HEIGHT;
                block_y += PART_HEIGHT)
        {
            a_coords[1] = block_y;
            a_coords[0] = 0;
            b_coords[1] = block_y;
            o_coords[1] = block_y;

            // for (int c = 0; c < A_LEN; c++)
            // {
            //     __global__ char * a_addr = gen_addr(a_coords, a_mat);
            //     a_coords[0] += 1;
            //     a_vals[c] = s_f32_ld_g(a_addr, a_mat);
            // }

            a_f32 = v_f32_ld_tnsr_b(a_coords, a_mat);
            a_int32 = (int64)a_f32; 
            b_v[0] = v_f32_ld_tnsr_b(b_coords, b_mat);

// #ifdef PRINTF_ENABLE
//             printf("a_vals: \n");
//             for (int j=0;j<16;j++)
//             {
//                 printf("%f ", a_vals[j]);
//             }
//             printf("\n");
//             printf("b_v[0]: \n");
//             for (int j=0;j<16;j++)
//             {
//                 printf("%f ", b_v[0][j]);
//             }
// #endif   

            float64 out_vecs[PART_WIDTH];
            float64 a_vecs[PART_WIDTH];

            for(int block_x = 0; block_x < A_LEN; block_x += PART_WIDTH)
            {
                // a_vecs[0] = a_vals[block_x];
                int64 x0 = v_broadcast_element_32bits(a_int32, block_x);
                a_vecs[0] = (float64)x0;
                out_vecs[0] = v_f32_mul_b(a_vecs[0], b_v[0]);

                // a_vecs[1] = a_vals[block_x+1];
                int64 x1 = v_broadcast_element_32bits(a_int32, block_x+1);
                a_vecs[1] = (float64)x1;
                out_vecs[1] = v_f32_mul_b(a_vecs[1], b_v[0]);

                #pragma unroll (PART_WIDTH)
                for(int i = 0; i < PART_WIDTH; i++)
                {
                    o_coords[0] = (block_x + i) * A_LEN;
                    v_f32_st_tnsr(o_coords, o_mat, out_vecs[i]);
                }
            }
        }
    }
    }
}

