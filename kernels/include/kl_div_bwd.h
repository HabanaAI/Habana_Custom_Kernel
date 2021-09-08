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
void main(tensor gradin, tensor input,
          tensor target, tensor outTensor,
          float invLen)
{
    const int depth     = 0;
    const int width     = 1;
    const int height    = 2;
    const int batch     = 3;

    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end = get_index_space_size() + index_space_start;

    int5 coords = { 0, 0, 0, 0, 0 };

    // DEPTH
    const int depthStep     = VECTOR_SIZE;
    const int depthStart    = index_space_start[depth] * depthStep;
    const int depthEnd      = index_space_end[depth] * depthStep;

    // WIDTH
    const int widthStep     = 1;
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

    for (int b = batchStart; b < batchtEnd; b += batchStep)
    {
        coords[batch] = b;

        for (int h = heightStart; h < heightEnd; h += heightStep)
        {
            coords[height] = h;
            for (int d = depthStart; d < depthEnd; d += depthStep)
            {
                coords[depth] = d;
                //#pragma loop_unroll(4)
                for (int w = widthStart; w < widthEnd; w += widthStep)
                {
                    coords[width] = w;

                    VECTOR t = v_ld_tnsr_i(coords, target);
                    VECTOR g = v_ld_tnsr_i(coords, gradin);

#ifdef FLOAT32
                    VECTOR out = (-t);
                    out = v_mul_v_v(out, invLen);
                    out = v_mul_v_v(out, g);
                    st_tnsr_i_v(coords, outTensor, out);
#else
                    float128 out;
                    out.v1 = 0.0f; out.v2=0.0f;
                    t = v_mul_v_v(t, g);
                    out = av_mac_acc32(t, 1.f, out, SW_NEG);
                    out.v1 = v_f32_mul_b(out.v1, invLen);
                    out.v2 = v_f32_mul_b(out.v2, invLen);

                    st_tnsr_i_v(coords, outTensor, v_convert_f32_to_vec(out, e_round_half_ne));
#endif
                    
                }
            }
        }
    }
}
