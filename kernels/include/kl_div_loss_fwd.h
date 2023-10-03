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
void main(tensor input, tensor target, tensor divTensor,
          float invLen, bool log_target)
{
    const int depth   = 0;
    const int width   = 1;
    const int height  = 2;
    const int batch   = 3;
    const int fifdim  = 4;

    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end = get_index_space_size() + index_space_start;

    // DEPTH
    const int depthStep     = VECTOR_SIZE;
    int depthStart    = index_space_start[depth] * depthStep;
    int depthEnd      = index_space_end[depth] * depthStep;

    // WIDTH
    const int widthStep     = NUM_UNROLL;
    int widthStart    = index_space_start[width] * widthStep;
    int widthEnd      = index_space_end[width] * widthStep;

    // HEIGHT
    const int heightStep    = 1;
    int heightStart   = index_space_start[height];
    int heightEnd     = index_space_end[height];

    // BATCH
    const int batchStep     = 1;
    int batchStart    = index_space_start[batch];
    int batchEnd     = index_space_end[batch];

    // fifdim
    const int fifdimStep     = 1;
    int fifdimStart    = index_space_start[fifdim];
    int fifdimEnd     = index_space_end[fifdim];

#ifndef Kl_DIV_NON_REDUCTION
    depthStart = 0;
    depthEnd = get_dim_size(input, 0);
    widthStart = 0;
    widthEnd = get_dim_size(input, 1);
    heightStart = 0;
    heightEnd = get_dim_size(input, 2);
    batchStart = 0;
    batchEnd = get_dim_size(input, 3);
    fifdimStart = 0;
    fifdimEnd = get_dim_size(input, 4);
#endif

    int5 ifmCoords = {0};

#ifndef Kl_DIV_NON_REDUCTION
    int5 ofmCoords = {0,0,0,0,0};
  #ifdef FLOAT32
    float64 sum[NUM_UNROLL];
  #else
    float128 sum[NUM_UNROLL];
  #endif

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
#endif

    if(log_target)
    {
        for (int f = fifdimStart; f < fifdimEnd; f += fifdimStep)
        {
            ifmCoords[fifdim] = f;

            for (int b = batchStart; b < batchEnd; b += batchStep)
            {
                ifmCoords[batch] = b;

                for (int h = heightStart; h < heightEnd; h += heightStep)
                {
                    ifmCoords[height] = h;

                    for (int d = depthStart; d < depthEnd; d += depthStep)
                    {
                        ifmCoords[depth] = d;
#ifdef Kl_DIV_NON_REDUCTION
                        #pragma unroll(NUM_UNROLL)
                        for (int w = widthStart; w < widthEnd; w += 1)
                        {
                            ifmCoords[width] = w;
                            VECTOR x = v_ld_tnsr_i(ifmCoords, input);
                            VECTOR y = v_ld_tnsr_i(ifmCoords, target);

                            // t = exp(y) *(y - x)
                            VECTOR input_y =  exp(y);
                            VECTOR YminusX = y - x;
                            VECTOR t = input_y * YminusX;

                            st_tnsr_i_v(ifmCoords, divTensor, t);
                        }
#else
                        for (int w = widthStart; w < widthEnd; w += widthStep)
                        {
                            ifmCoords[width] = w;
                            #pragma unroll(NUM_UNROLL)
                            for (int k = 0; k < NUM_UNROLL; k++)
                            {
                                VECTOR x = v_ld_tnsr_i(ifmCoords, input);
                                VECTOR y = v_ld_tnsr_i(ifmCoords, target);

                                // t = exp(y) *(y - x)
                                VECTOR input_y =  exp(y);
                                VECTOR YminusX = y - x;
                                VECTOR t = input_y * YminusX;
                                bool256 pred = bv_u_cmp_geq_v_s(d + V_LANE_ID, (unsigned)depthEnd);
                                t = v_mov_s_vb(0, t, pred, 0);
  #ifdef FLOAT32
                                sum[k] = v_add_v_v_b(sum[k], t, sum[k], (w+k) < widthEnd, 0);
  #else
                                sum[k] = av_mac_acc32_b(t, 1.f, sum[k], SW_NO_NEG, (w+k) < widthEnd, 0);
  #endif
                                ifmCoords[width] += 1;
                            }
                        }
#endif
                    }
                }
            }
        }
    }
    else
    {
        for (int f = fifdimStart; f < fifdimEnd; f += fifdimStep)
        {
            ifmCoords[fifdim] = f;

            for (int b = batchStart; b < batchEnd; b += batchStep)
            {
                ifmCoords[batch] = b;

                for (int h = heightStart; h < heightEnd; h += heightStep)
                {
                    ifmCoords[height] = h;

                    for (int d = depthStart; d < depthEnd; d += depthStep)
                    {
                        ifmCoords[depth] = d;
#ifdef Kl_DIV_NON_REDUCTION
                        #pragma unroll(NUM_UNROLL)
                        for (int w = widthStart; w < widthEnd; w += 1)
                        {
                            ifmCoords[width] = w;

                            VECTOR x = v_ld_tnsr_i(ifmCoords, input);
                            VECTOR y = v_ld_tnsr_i(ifmCoords, target);

                            // t = y *(log(y) - x)
                            y = v_sel_leq_v_v_v_v(y, 0, 1, y);
                            VECTOR input_y =  log(y);
                            VECTOR logYminusX = input_y - x;
                            VECTOR t = y * logYminusX;

                            st_tnsr_i_v(ifmCoords, divTensor, t);
                        }
#else
                        for (int w = widthStart; w < widthEnd; w += widthStep)
                        {
                            ifmCoords[width] = w;
                            #pragma unroll(NUM_UNROLL)
                            for (int k = 0; k < NUM_UNROLL; k++)
                            {
                                VECTOR x = v_ld_tnsr_i(ifmCoords, input);
                                VECTOR y = v_ld_tnsr_i(ifmCoords, target);

                                // t = y *(log(y) - x)
                                y = v_sel_leq_v_v_v_v(y, 0, 1, y);
                                VECTOR input_y =  log(y);
                                VECTOR logYminusX = input_y - x;
                                VECTOR t = y * logYminusX;
                                bool256 pred = bv_u_cmp_geq_v_s(d + V_LANE_ID, (unsigned)depthEnd);
                                t = v_mov_s_vb(0, t, pred, 0);
  #ifdef FLOAT32
                                sum[k] = v_add_v_v_b(sum[k], t, sum[k], (w+k) < widthEnd, 0);
  #else
                                sum[k] = av_mac_acc32_b(t, 1.f, sum[k], SW_NO_NEG, (w+k) < widthEnd, 0);
  #endif
                                ifmCoords[width] += 1;

                            }
                        }
#endif
                    }
                }
            }
        }
    }

#ifndef Kl_DIV_NON_REDUCTION
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
    st_tnsr_reg_i_s_v(ofmCoords, divTensor, v_convert_f32_to_vec(sum[0], SW_RHNE));
  #endif
#endif
}
