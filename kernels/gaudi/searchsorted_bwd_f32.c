/**********************************************************************
Copyright (c) 2023 Habana Labs.

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
void main(
    tensor ifm,
    tensor ofm
)
{
    const int depth  = 0;
    const int width  = 1;
    const int height = 2;
    const int batch  = 3;
    const int fifdim  = 4;

    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end = get_index_space_size() + index_space_start;

    // depth
    const int depthStep  = 64;
    const int depthStart = index_space_start[depth] * depthStep;
    const int depthEnd   = index_space_end[depth] * depthStep;

    // width
    const int widthStep  = 1;
    const int widthStart = index_space_start[width] * widthStep;
    const int widthEnd   = index_space_end[width]   * widthStep;

    // height
    const int heightStep  = 1;
    const int heightStart = index_space_start[height] * heightStep;
    const int heightEnd   = index_space_end[height]   * heightStep;

    // batch
    const int batchStep  = 1;
    const int batchStart = index_space_start[batch] * batchStep;
    const int batchEnd   = index_space_end[batch]   * batchStep;

    // fifdim
    const int fifdimStep  = 1;
    const int fifdimStart = index_space_start[fifdim] * fifdimStep;
    const int fifdimEnd   = index_space_end[fifdim]   * fifdimStep;

    int5 ifmCoords = { depthStart, widthStart, heightStart, batchStart, fifdimStart };

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

                    for (int w = widthStart; w < widthEnd; w += widthStep)
                    {
                        ifmCoords[width] = w;
                        float64 value = v_f32_ld_tnsr_b(ifmCoords, ifm);

                        v_f32_st_tnsr(ifmCoords, ofm, value);
                    }
                }
            }
        }
    }
}
