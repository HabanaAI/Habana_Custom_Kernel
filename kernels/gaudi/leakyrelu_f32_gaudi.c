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


void main(tensor input, tensor output, float alpha)
{
    const int depth     = 0;
    const int width     = 1;
    const int height    = 2;
    const int batch     = 3;

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

                    float64 x = v_f32_ld_tnsr_b(coords, input, 0, 0, 1, 0);

                    bool256 cond = from_bool64(v_f32_cmp_grt_b(x, 0.0, 0, to_bool64((bool256){0}), 1, 0));

                    float64 y = v_f32_mul_vb(x, alpha, 0, x, to_bool64(cond), 1);

                    v_f32_st_tnsr(coords, output, y, 0, 1, 0);
                }
            }
        }
    }
}
