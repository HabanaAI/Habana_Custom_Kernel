/*****************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 * Authors:
 * Zhongkai Zhang <zhzhang@habana.ai>
 ******************************************************************************
*/

#define NUM_UNROLL 4
#include "kernel_config.h"

// invLen is 1, reduction is 'sum'.
// invLen is actual 1/len, reduction is 'mean'
void main(tensor input, tensor target, tensor gradin,
          tensor outTensor, float invLen, bool log_target)
{
    const int depth     = 0;
    const int width     = 1;
    const int height    = 2;
    const int batch     = 3;
    const int fifdim    = 4;

    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end = get_index_space_size() + index_space_start;

    int5 coords = { 0, 0, 0, 0, 0 };

    // DEPTH
    const int depthStep     = VECTOR_SIZE;
    const int depthStart    = index_space_start[depth] * depthStep;
    const int depthEnd      = index_space_end[depth] * depthStep;

    // WIDTH
    const int widthStep     = NUM_UNROLL;
    const int widthStart    = index_space_start[width] * widthStep;
    const int widthEnd      = index_space_end[width] * widthStep;

    // HEIGHT
    const int heightStep    = 1;
    const int heightStart   = index_space_start[height];
    const int heightEnd     = index_space_end[height];

    // BATCH
    const int batchStep     = 1;
    const int batchStart    = index_space_start[batch];
    const int batchEnd      = index_space_end[batch];

    const int fifdimStep    = 1;
    const int fifdimStart   = index_space_start[fifdim];
    const int fifdimEnd     = index_space_end[fifdim];

#ifndef Kl_DIV_NON_REDUCTION
    __global__ float* addr = (__global__ float*)gen_addr(coords, gradin);
    VECTOR g = v_ld_g_a(addr);
#endif

    if(log_target)
    {
        for (int f = fifdimStart; f < fifdimEnd; f += fifdimStep)
        {
            coords[fifdim] = f;

            for (int b = batchStart; b < batchEnd; b += batchStep)
            {
                coords[batch] = b;

                for (int h = heightStart; h < heightEnd; h += heightStep)
                {
                    coords[height] = h;
                    for (int d = depthStart; d < depthEnd; d += depthStep)
                    {
                        coords[depth] = d;
                        #pragma unroll(NUM_UNROLL)
                        for (int w = widthStart; w < widthEnd; w += 1)
                        {
                            coords[width] = w;
                            // x is dummy read, we don't need x tensor, just for simulation check
                            VECTOR x = v_ld_tnsr_i(coords, input);
                            VECTOR t = v_ld_tnsr_i(coords, target);
#ifdef Kl_DIV_NON_REDUCTION
                            VECTOR g = v_ld_tnsr_i(coords, gradin);
#endif

#ifdef FLOAT32
                            VECTOR out = exp(t);
                            // x is dummy read, we don't need x tensor, just for simulation check
                            out = (-out) + x * (0.0);
                            out = out * invLen;
                            out = out * g;
                            st_tnsr_i_v(coords, outTensor, out);
#else
                            float128 out;
                            out.v1 = 0.0f; out.v2=0.0f;
                            VECTOR input_t = exp(t);
                            t = input_t * g;
                            // x is dummy read, we don't need x tensor, just for simulation check
                            t = t + x * 0;
                            out = av_mac_acc32(t, 1.f, out, SW_NEG);
                            out.v1 = v_f32_mul_b(out.v1, invLen);
                            out.v2 = v_f32_mul_b(out.v2, invLen);

                            st_tnsr_i_v(coords, outTensor, v_convert_f32_to_vec(out, SW_RHNE));
#endif
                        }
                    }
                }
            }
        }
    }
    else
    {
        for (int f = fifdimStart; f < fifdimEnd; f += fifdimStep)
        {
            coords[fifdim] = f;

            for (int b = batchStart; b < batchEnd; b += batchStep)
            {
                coords[batch] = b;

                for (int h = heightStart; h < heightEnd; h += heightStep)
                {
                    coords[height] = h;
                    for (int d = depthStart; d < depthEnd; d += depthStep)
                    {
                        coords[depth] = d;
                        #pragma unroll(NUM_UNROLL)
                        for (int w = widthStart; w < widthEnd; w += 1)
                        {
                            coords[width] = w;
                            // x is dummy read, we don't need x tensor, just for simulation check
                            VECTOR x = v_ld_tnsr_i(coords, input);
                            VECTOR t = v_ld_tnsr_i(coords, target);
#ifdef Kl_DIV_NON_REDUCTION
                            VECTOR g = v_ld_tnsr_i(coords, gradin);
#endif
#ifdef FLOAT32
                            VECTOR out = (-t);
                            out = out * invLen;
                            out = out * g;
                            out = out + x * (0.0);
                            st_tnsr_i_v(coords, outTensor, out);
#else
                            float128 out;
                            out.v1 = 0.0f; out.v2 = 0.0f;
                            t= t * g;
                            t = t + x*0;
                            out = av_mac_acc32(t, 1.f, out, SW_NEG);
                            out.v1 = v_f32_mul_b(out.v1, invLen);
                            out.v2 = v_f32_mul_b(out.v2, invLen);

                            st_tnsr_i_v(coords, outTensor, v_convert_f32_to_vec(out, SW_RHNE));
#endif
                        }
                    }
                }
            }
        }
    }
}
