/**********************************************************************
Copyright (c) 2022 Habana Labs. All rights reserved.

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


#if defined(FLOAT32)
    #define PLUS_INF        0x7f800000
    #define  v_sel_grt_b(grad, output) \
                                v_i32_sel_grt_i32_b((int64)grad, PLUS_INF, (int64)grad, (int64)output);
#elif defined(BFLOAT16)
    #define PLUS_INF        0x7F80
    #define  v_sel_grt_b(grad, output) \
                        v_i16_sel_grt_i16_b((short128)grad, PLUS_INF, (short128)grad, (short128)output);
#endif

void main(tensor grad, tensor input, tensor output)
{
    const int depth = 0;
    const int width = 1;
    const int height = 2;
    const int batch = 3;
    const int fifthDim = 4;    
    /*Special cases handled:
      input = nan,        grad = validValue, TF output = 0
      input = validValue, grad = nan,        TF output = nan
      input = nan,        grad = nan,        TF output = nan*/
    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

    // DEPTH
    const int depthStep = VECTOR_SIZE;
    const int depthStart = indexSpaceStart[depth] * depthStep;
    const int depthEnd = indexSpaceEnd[depth] * depthStep;

    // WIDTH
    const int widthStep = 4;
    const int widthStart = indexSpaceStart[width] * widthStep;
    const int widthEnd = indexSpaceEnd[width] * widthStep;

    // HEIGHT
    const int heightStep = 1;
    const int heightStart = indexSpaceStart[height];
    const int heightEnd = indexSpaceEnd[height];

    // BATCH
    const int batchStep = 1;
    const int batchStart = indexSpaceStart[batch];
    const int batchEnd = indexSpaceEnd[batch];

    // fifthDim
    const int fifthDimStep = 1;
    const int fifthDimStart = indexSpaceStart[fifthDim];
    const int fifthDimEnd = indexSpaceEnd[fifthDim];

    // bool256 pred0, pred1, pred2, pred3;
    VECTOR threshold_v = 0.f;
    VECTOR x0, x1, x2, x3;
    VECTOR outp0, outp1, outp2, outp3;
    VECTOR grad0, grad1, grad2, grad3;

    int5 ifmCoords = { 0, widthStart, 0, 0, 0 };
    int5 ifmCoords1 = { 0, widthStart, 0, 0, 0 };
    int5 ofmCoords = { 0, widthStart, 0, 0, 0 };

    #pragma loop_taken
    for (int d = depthStart; d < depthEnd; d += depthStep)
    {
        ifmCoords[depth] = d;   ofmCoords[depth] = d;   ifmCoords1[depth] = d;
        #pragma loop_taken
        for (int f = fifthDimStart; f < fifthDimEnd; f += fifthDimStep)
        {
            ifmCoords[fifthDim] = f;   ofmCoords[fifthDim] = f;   ifmCoords1[fifthDim] = f;
            #pragma loop_taken
            for (int b = batchStart; b < batchEnd; b += batchStep)
            {
                ifmCoords[batch] = b;   ofmCoords[batch] = b;   ifmCoords1[batch] = b;
                #pragma loop_taken
                for (int h = heightStart; h < heightEnd; h += heightStep)
                {
                    ifmCoords[height] = h;   ofmCoords[height] = h; ifmCoords1[height] = h;

                    #pragma loop_taken
                    for (int w = widthStart; w < widthEnd; w += widthStep)
                    {
                        x0 = v_ld_tnsr_i(ifmCoords1, input);    ifmCoords1[width] += 1;
                        x1 = v_ld_tnsr_i(ifmCoords1, input);    ifmCoords1[width] += 1;
                        x2 = v_ld_tnsr_i(ifmCoords1, input);    ifmCoords1[width] += 1;
                        x3 = v_ld_tnsr_i(ifmCoords1, input);    ifmCoords1[width] += 1;

                        grad0 = v_ld_tnsr_i(ifmCoords, grad);  ifmCoords[width] += 1;
                        grad1 = v_ld_tnsr_i(ifmCoords, grad);  ifmCoords[width] += 1;
                        grad2 = v_ld_tnsr_i(ifmCoords, grad);  ifmCoords[width] += 1;
                        grad3 = v_ld_tnsr_i(ifmCoords, grad);  ifmCoords[width] += 1;

                        outp0 = v_sel_grt_v_s_v_v(x0, threshold_v, grad0, 0);
                        outp1 = v_sel_grt_v_s_v_v(x1, threshold_v, grad1, 0);
                        outp2 = v_sel_grt_v_s_v_v(x2, threshold_v, grad2, 0);
                        outp3 = v_sel_grt_v_s_v_v(x3, threshold_v, grad3, 0);
#if defined(USE_RELU6)
                        outp0 = v_sel_less_v_s_v_v_b(x0, (SCALAR)6.0, outp0, 0, outp0, 1, 0);
                        outp1 = v_sel_less_v_s_v_v_b(x1, (SCALAR)6.0, outp1, 0, outp1, 1, 0);
                        outp2 = v_sel_less_v_s_v_v_b(x2, (SCALAR)6.0, outp2, 0, outp2, 1, 0);
                        outp3 = v_sel_less_v_s_v_v_b(x3, (SCALAR)6.0, outp3, 0, outp3, 1, 0);
#else
                        outp0 = v_sel_less_v_s_v_v_b(x0, (SCALAR)6.0, outp0, 0, outp0, 0, 0);
                        outp1 = v_sel_less_v_s_v_v_b(x1, (SCALAR)6.0, outp1, 0, outp1, 0, 0);
                        outp2 = v_sel_less_v_s_v_v_b(x2, (SCALAR)6.0, outp2, 0, outp2, 0, 0);
                        outp3 = v_sel_less_v_s_v_v_b(x3, (SCALAR)6.0, outp3, 0, outp3, 0, 0);
#endif

                        //return NAN if grad == NAN, checking it by comparing grad greater than INF
                        outp0 = (VECTOR)v_sel_grt_b(grad0, outp0);
                        outp1 = (VECTOR)v_sel_grt_b(grad1, outp1);
                        outp2 = (VECTOR)v_sel_grt_b(grad2, outp2);
                        outp3 = (VECTOR)v_sel_grt_b(grad3, outp3);

                        st_tnsr_i_v(ofmCoords, output, outp0);    ofmCoords[width] += 1;
                        st_tnsr_i_v(ofmCoords, output, outp1);    ofmCoords[width] += 1;
                        st_tnsr_i_v(ofmCoords, output, outp2);    ofmCoords[width] += 1;
                        st_tnsr_i_v(ofmCoords, output, outp3);    ofmCoords[width] += 1;

                    }
                    ifmCoords[width] = widthStart;  ofmCoords[width] = widthStart;
                    ifmCoords1[width] = widthStart;
                }
            }
        }
    }
}
