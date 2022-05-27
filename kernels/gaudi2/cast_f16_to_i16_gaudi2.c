/**********************************************************************
Copyright (c) 2022 Habana Labs.

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

#define DEPTH_STEP                        (128)
#define NUM_UNROLL                        (4)

#define ROUND_MODE             5
#define CONV_ROUND_CSR_OFFSET  0xCA8

void main(tensor ifm,
          tensor ofm,
          unsigned int roundingMode
          )
{
    // save backup and update rounding mode to CONV_ROUND_CSR
    unsigned int  convCsrBackup = s_u32_ld_l(CONV_ROUND_CSR_OFFSET, SW_MMIO);
    s_u32_st_l(CONV_ROUND_CSR_OFFSET, roundingMode, SW_MMIO);

    const int depth = 0;
    const int width = 1;
    const int height = 2;
    const int batch = 3;
    const int fifthDim = 4;

    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

    int5 ifmCoords = { 0, 0, 0, 0, 0 };

    // DEPTH
    const int depthStep = DEPTH_STEP;
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
    const int batchtEnd = indexSpaceEnd[batch];

    // fifthDim
    const int fifthDimStep  = 1;
    const int fifthDimStart  = indexSpaceStart[fifthDim];
    const int fifthDimEnd    = indexSpaceEnd[fifthDim];

    half128 vec_in;
    short128 vec_out;

    #pragma loop_taken
    for (int d = depthStart; d < depthEnd; d += depthStep)
    {
        ifmCoords[depth] = d;

        #pragma loop_taken
        for (int f = fifthDimStart; f < fifthDimEnd; f += fifthDimStep)
        {
            ifmCoords[fifthDim] = f;

            #pragma loop_taken
            for (int b = batchStart; b < batchtEnd; b += batchStep)
            {
                ifmCoords[batch] = b;
                #pragma loop_taken
                for (int h = heightStart; h < heightEnd; h += heightStep)
                {
                    ifmCoords[height] = h;
                    //#pragma loop_taken
                    //#pragma unroll(NUM_UNROLL)
                    for (int w = widthStart ; w < widthEnd; w += 1)
                    {
                        ifmCoords[width] = w;
                        vec_in = v_f16_ld_tnsr_b(ifmCoords, ifm);
                        vec_out = v_convert_f16_to_i16_b(vec_in, (ROUND_MODE << 16));
                        v_i16_st_tnsr(ifmCoords, ofm, vec_out);
                    }
                }
            }
        }
    }
    // Restore rounding mode to CONV_ROUND_CSR
    s_u32_st_l(CONV_ROUND_CSR_OFFSET, convCsrBackup, SW_MMIO);

}

