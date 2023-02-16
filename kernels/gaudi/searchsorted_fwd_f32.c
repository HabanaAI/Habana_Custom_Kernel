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
#define bv_cmp_eq_v_v(a, b) from_bool64(v_f32_cmp_eq_b(a, b))

void main(
    tensor ifm_seq,
    tensor ifm_val,
    tensor ofm_idx,
    bool   side
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
    const int widthStart = 0;
    const int widthEnd   = get_dim_size(ifm_seq, 1);

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

    // value width
    const int valueWidthStep  = 1;
    const int valueWidthStart  = 0;
    // Returns the dim0 size of ifm
    const int valueWidthEnd   = get_dim_size(ifm_val, 1);

    int64 one = 0;

    int5 ifmCoords = { depthStart, widthStart, heightStart, batchStart, fifdimStart };
    int5 ofmCoords = { depthStart, valueWidthStart, heightStart, batchStart, fifdimStart };

    // side is right
    if(side == 1)
    {
        for (int f = fifdimStart; f < fifdimEnd; f += fifdimStep)
        {
            ifmCoords[fifdim] = ofmCoords[fifdim] = f;

            for (int b = batchStart; b < batchEnd; b += batchStep)
            {
                ifmCoords[batch] = ofmCoords[batch] = b;

                for (int h = heightStart; h < heightEnd; h += heightStep)
                {
                    ifmCoords[height] = ofmCoords[height] = h;

                    for (int d = depthStart; d < depthEnd; d += depthStep)
                    {
                        ifmCoords[depth] = ofmCoords[depth] = d;

                        for (int vw = valueWidthStart; vw < valueWidthEnd; vw += valueWidthStep)
                        {
                            ofmCoords[width] = vw;
                            float64 value = v_f32_ld_tnsr_b(ofmCoords, ifm_val);
                            int64 index = 0;

                            for (int w = widthStart; w < widthEnd; w += widthStep)    
                            {
                                ifmCoords[width] = w;
                                float64 sequence = v_f32_ld_tnsr_b(ifmCoords, ifm_seq);

                                float64 cmps = v_f32_sel_leq_f32_b(sequence, value, 0, 1);
                                bool256 pred = bv_cmp_eq_v_v(cmps, (float64) one);
                                index = v_i32_mov_vb(w+1, 0, index, to_bool64(pred),0);
                            }
                            v_i32_st_tnsr(ofmCoords, ofm_idx, index);
                        }
                    }
                }
            }
        }
    }
    // side is left
    else
    {
        for (int f = fifdimStart; f < fifdimEnd; f += fifdimStep)
        {
            ifmCoords[fifdim] = ofmCoords[fifdim] = f;

            for (int b = batchStart; b < batchEnd; b += batchStep)
            {
                ifmCoords[batch] = ofmCoords[batch] = b;

                for (int h = heightStart; h < heightEnd; h += heightStep)
                {
                    ifmCoords[height] = ofmCoords[height] = h;

                    for (int d = depthStart; d < depthEnd; d += depthStep)
                    {
                        ifmCoords[depth] = ofmCoords[depth] = d;

                        for (int vw = valueWidthStart; vw < valueWidthEnd; vw += valueWidthStep)
                        {
                            ofmCoords[width] = vw;
                            float64 value = v_f32_ld_tnsr_b(ofmCoords, ifm_val);
                            int64 index = 0;

                            for (int w = widthStart; w < widthEnd; w += widthStep)    
                            {
                                ifmCoords[width] = w;
                                float64 sequence = v_f32_ld_tnsr_b(ifmCoords, ifm_seq);

                                float64 cmps = v_f32_sel_less_f32_b(sequence, value, 0, 1);
                                bool256 pred = bv_cmp_eq_v_v(cmps, (float64) one);
                                index = v_i32_mov_vb(w+1, 0, index, to_bool64(pred),0);
                            }
                            v_i32_st_tnsr(ofmCoords, ofm_idx, index);
                        }
                    }
                }
            }
        }
    }
}
