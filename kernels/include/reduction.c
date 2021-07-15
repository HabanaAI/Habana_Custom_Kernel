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

#include "kernel_config.h"
#include "special_all.h"

#ifdef BFLOAT16
    #include "config_reduction_type_bf16.h"
    #include "special_all_bf16.h"
#else
    #include "config_reduction_type.h"
#endif

#include "reduction_functions.h"

void main(tensor ifm,
          tensor ofm,
          int vecsPerLoop)
{
    const int depth  = 0;
    const int width  = 1;
    const int height = 2;
    const int batch  = 3;

    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end   = get_index_space_size() + index_space_start;

    // DEPTH
    const int depthStep  = VECTOR_SIZE * vecsPerLoop;
    const int depthStart = index_space_start[depth] * depthStep;
    const int depthEnd   = index_space_end[depth] * depthStep;

    // WIDTH
    const int widthStep  = 1;
    const int widthStart = index_space_start[width] * widthStep;
    const int widthEnd   = index_space_end[width] * widthStep;

    // HEIGHT
    const int heightStep  = 1;
    const int heightStart = index_space_start[height] * heightStep;
    const int heightEnd   = index_space_end[height] * heightStep;

    // BATCH
    const int batchStep = 1;
    const int batchStart = index_space_start[batch] * batchStep;
    const int batchtEnd = index_space_end[batch] * batchStep;

    VECTOR acc;
    VECTOR ifmVal;
    int5 ifmCoords = {0}, ofmCoords = {0};

    acc = v_init_v();

#if defined(USING_RMW)
    for (int b = batchStart; b < batchtEnd; b += batchStep)
    {
        ifmCoords[batch] = b;
        ofmCoords[batch] = b;

        for (int h = heightStart; h < heightEnd; h += heightStep)
        {
            ifmCoords[height] = h;
            ofmCoords[height] = h;

            for (int w = widthStart; w < widthEnd; w += widthStep)
            {
                ifmCoords[width]  = w;
                ofmCoords[width]  = w;
                ofmCoords[depth]  = 0;

                for (int d = depthStart; d < depthEnd; d += VECTOR_SIZE)
                {
                    ifmCoords[depth]  = d;

                    ifmVal = v_ld_tnsr_i(ifmCoords, ifm);
                    acc = v_acc_v_v(acc, ifmVal);
                }

                acc = v_reduce_op(acc);
                f32_st_tnsr_rmw_i_v(ofmCoords, ofm, acc, e_rmw_fp32, e_rmw_add, e_rmw_atomic, e_tnsr_dt_srf);
            }
        }
    }

#else  
   for (int b = batchStart; b < batchtEnd; b += batchStep)
    {
        ifmCoords[batch] = b;
        ofmCoords[batch] = b;

        for (int h = heightStart; h < heightEnd; h += heightStep)
        {
            ifmCoords[height] = h;
            ofmCoords[height] = h;

            for (int w = widthStart; w < widthEnd; w += widthStep)
            {
                ifmCoords[width]  = w;
                ofmCoords[width]  = w;
                output_depth_idx  = index_space_start[depth];

                for (int d = depthStart; d < depthEnd; d += depthStep, output_depth_idx++)
                {
                    acc = v_init_v();
                    for(int d0 = d; d0 < d + depthStep ; d0 += VECTOR_SIZE)
                    {
                        ifmCoords[depth]  = d0;
                        ifmVal = v_ld_tnsr_i(ifmCoords, ifm);
                        acc = v_acc_v_v(acc, ifmVal);
                    }
                    acc = v_reduce_op(acc);
                    ofmCoords[depth]  = output_depth_idx;
                    f32_st_tnsr_partial_i_v(ofmCoords, ofm, acc, 0, 1);
                }
            }
        }
    }
#endif
}