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

#define NUM_UNROLL 4
#include "kernel_config.h"

// invLen is 1, reduction is 'sum'.
// invLen is actual 1/len, reduction is 'mean'
void main(tensor input, tensor target, tensor divTensor,
          float invLen)
{
    const int depth   = 0;
    const int width   = 1;
    const int height  = 2;
    const int batch   = 3;

    const int depthEnd  = get_dim_size(input, 0);
    const int widthEnd  = get_dim_size(input, 1);
    const int heightEnd = get_dim_size(input, 2);
    const int batchEnd  = get_dim_size(input, 3);

    int5 ifmCoords = {0};
    int5 ofmCoords = {0,0,0,0,0};

    VECTOR_SUM sum[NUM_UNROLL];

#pragma unroll(NUM_UNROLL)
    for (int k = 0; k < NUM_UNROLL; k++)
    {
#ifdef FLOAT32
        sum[k] = 0;
#else
        sum[k].v1 = 0;
        sum[k].v2 = 0;
#endif
    }

    for (int d = 0; d < depthEnd; d += VECTOR_SIZE)
    {
        ifmCoords[depth] = d;

        for (int b = 0; b < batchEnd; b += 1)
        {
            ifmCoords[batch] = b;

            for (int h = 0; h < heightEnd; h += 1)
            {
                ifmCoords[height] = h;

                for (int w = 0; w < widthEnd; w += NUM_UNROLL)
                {
                    ifmCoords[width] = w;

                    #pragma unroll(NUM_UNROLL)
                    for (int k = 0; k < NUM_UNROLL; k++)
                    {
                        VECTOR x = v_ld_tnsr_i(ifmCoords, input);
                        VECTOR y = v_ld_tnsr_i(ifmCoords, target);

                        // t = y *(log(y) - x)
                        VECTOR input_y =  log(y); 
                        VECTOR logYminusX = input_y - x;
                        VECTOR t = y * logYminusX;

                        bool256 pred = bv_u_cmp_geq_v_s(d + V_LANE_ID, (unsigned)depthEnd);
                        t = v_mov_s_vb(0, t, pred, 0);
#ifdef FLOAT32
                        sum[k] = v_add_v_v_b(sum[k], t, sum[k], (w+k) < widthEnd, 0);
#else
                        sum[k] = av_mac_acc32_b(t, 1.f, sum[k], e_no_negation, (w+k) < widthEnd, 0);
#endif
                        ifmCoords[width] += 1;
                    }
                }
            }
        }
    }

#ifdef FLOAT32

    #pragma unroll(NUM_UNROLL)
    for (int k = 1; k < NUM_UNROLL; k++)
    {
        sum[0] += sum[k];
    }

    sum[0] = v_f32_reduce_add(sum[0]);
    sum[0] = v_mul_v_s(sum[0], invLen);
    v_f32_st_tnsr(ofmCoords, divTensor, sum[0]);

#else

    #pragma unroll(NUM_UNROLL)
    for (int k = 1; k < NUM_UNROLL; k++)
    {
        sum[0].v1 += sum[k].v1;
        sum[0].v2 += sum[k].v2;
    }
    sum[0].v1 += sum[0].v2;

    sum[0].v1 = v_f32_reduce_add(sum[0].v1);
    sum[0].v1 = v_f32_mul_b(sum[0].v1, invLen);
    v_bf16_st_tnsr(ofmCoords, divTensor, v_convert_f32_to_vec(sum[0], (e_round_half_ne<<16)));

#endif
}
