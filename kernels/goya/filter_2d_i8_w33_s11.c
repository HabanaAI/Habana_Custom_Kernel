/**********************************************************************
Copyright (c) 2018 Habana Labs.

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

// vector allocation in vlm

void main(tensor ifm,
          tensor filter,
          tensor bias,
          tensor ofm,
          char padw,
          char padh,
          char scale_factor
)
{
    const int depth = 0;
    const int width = 1;
    const int height = 2;
    const int batch = 3;

    const int width_mask = 1 << width;
    const int height_mask = 1 << height;

    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end = get_index_space_size() + index_space_start;

    // DEPTH
    const int depthStep = 256;
    const int depthStart = index_space_start[depth] * depthStep;
    const int depthEnd = index_space_end[depth] * depthStep;

    // WIDTH
    const int widthStep = 4;
    const int widthStart = index_space_start[width] * widthStep;
    const int widthEnd = index_space_end[width] * widthStep;

    // HEIGHT
    const int heightStep = 1;
    const int heightStart = index_space_start[height];
    const int heightEnd = index_space_end[height];

    // BATCH
    const int batchStep = 1;
    const int batchStart = index_space_start[batch];
    const int batchtEnd = index_space_end[batch];

    int256 acc0; // V0, V1, V2, V3
    int256 acc1; // V4, V5, V6, V7
    int256 acc2; // V8, V9, V10, V11
    int256 acc3; // V12, V13, V14, V15

    char256 h00, h01, h02; // V16, V17, V18
    char256 h10, h11, h12; // V19, V20, V21,
    char256 h20, h21; // V22, V23

    char256 x00, x01, x02, x03, x04, x05; // V24, V25, V26, V27, V28, V29
    char256 x10, x11, x12, x13, x14, x15; // V30, V31, V32, V33, V34, V35
    char256 x20, x21, x22, x23; // V36, V37, V38, V39

    int64 biasVector[4];
    char256 h22;

    int5 ifmCoords    = { depthStart, 0, 0, 0, 0 };
    int5 filterCoords = { depthStart, 0, 0, 0, 0 };
    int5 biasCoords   = { depthStart, 0, 0, 0, 0 };
    int5 ofmCoords    = { depthStart, 0, 0, 0, 0 };

    int widthMinusPadw = widthStart - padw;
    int MulStrideMinusPadH = heightStart - padh;

    // Set input pointer to the start of the input as per the padding values
    ifmCoords[width] = widthMinusPadw;
    ifmCoords[height] = MulStrideMinusPadH;

    for (int d = depthStart; d <  depthEnd; d += depthStep)
    {
        ifmCoords[depth] = d; ofmCoords[depth] = d;

        // Load bias values into accumalator
        acc0.v1 = v_i32_ld_tnsr_b(biasCoords, bias, 0, 0, 1, 0); biasCoords[depth] += 64;
        acc0.v2 = v_i32_ld_tnsr_b(biasCoords, bias, 0, 0, 1, 0); biasCoords[depth] += 64;
        acc0.v3 = v_i32_ld_tnsr_b(biasCoords, bias, 0, 0, 1, 0); biasCoords[depth] += 64;
        acc0.v4 = v_i32_ld_tnsr_b(biasCoords, bias, 0, 0, 1, 0); biasCoords[depth] += 64;

        // Replicate the same bias values in all the accumalator
        acc1.v1 = acc0.v1;
        acc1.v2 = acc0.v2;
        acc1.v3 = acc0.v3;
        acc1.v4 = acc0.v4;

        acc2.v1 = acc0.v1;
        acc2.v2 = acc0.v2;
        acc2.v3 = acc0.v3;
        acc2.v4 = acc0.v4;

        acc3.v1 = acc0.v1;
        acc3.v2 = acc0.v2;
        acc3.v3 = acc0.v3;
        acc3.v4 = acc0.v4;

        // store in vlm for fast access
        biasVector[0] = acc0.v1;
        biasVector[1] = acc0.v2;
        biasVector[2] = acc0.v3;
        biasVector[3] = acc0.v4;

        // Load 3X3 filter which has to be multiplied across the input
        h00 = v_i8_ld_tnsr_b(filterCoords, filter, 0, 0, 1, 0);   filterCoords[width] += 1;
        h01 = v_i8_ld_tnsr_b(filterCoords, filter, 0, 0, 1, 0);   filterCoords[width] += 1;
        h02 = v_i8_ld_tnsr_b(filterCoords, filter, 0, 0, 1, 0);   filterCoords[height] += 1;
        h12 = v_i8_ld_tnsr_b(filterCoords, filter, 0, 0, 1, 0);   filterCoords[width] -= 1;
        h11 = v_i8_ld_tnsr_b(filterCoords, filter, 0, 0, 1, 0);   filterCoords[width] -= 1;
        h10 = v_i8_ld_tnsr_b(filterCoords, filter, 0, 0, 1, 0);   filterCoords[height] += 1;
        h20 = v_i8_ld_tnsr_b(filterCoords, filter, 0, 0, 1, 0);   filterCoords[width] += 1;
        h21 = v_i8_ld_tnsr_b(filterCoords, filter, 0, 0, 1, 0);   filterCoords[width] += 1;
        h22 = v_i8_ld_tnsr_b(filterCoords, filter, 0, 0, 1, 0);

        // Set the filter pointer to the start for next channel
        filterCoords = i_i32_add(-2, filterCoords, width_mask | height_mask, 0, filterCoords, 1, 0);

        for (int b = batchStart ; b < batchtEnd; b += batchStep)
        {
            ifmCoords[batch] = b; ofmCoords[batch] = b;

            for (int w = widthStart; w < widthEnd; w += widthStep)
            {

                /* Input [00 01 02 03 04 05
                          10 11 12 13 14 15
                          20 21 22 23 24 25]
                   Load first two rows of input elements x00-x05 x10-15*/
                // Loop unrolled for 4 elements
                x00 = v_i8_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);   ifmCoords[width] += 1;
                x01 = v_i8_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);   ifmCoords[width] += 1;
                x02 = v_i8_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);   ifmCoords[width] += 1;
                x03 = v_i8_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);   ifmCoords[width] += 1;
                x04 = v_i8_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);   ifmCoords[width] += 1;
                x05 = v_i8_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);   ifmCoords[height] += 1;
                x15 = v_i8_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);   ifmCoords[width] -= 1;
                x14 = v_i8_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);   ifmCoords[width] -= 1;
                x13 = v_i8_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);   ifmCoords[width] -= 1;
                x12 = v_i8_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);   ifmCoords[width] -= 1;
                x11 = v_i8_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);   ifmCoords[width] -= 1;
                x10 = v_i8_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);   ifmCoords[height] += 1;

                for (int h = heightStart; h < heightEnd; h += heightStep)
                {

                    ofmCoords[height] = h;
                    ofmCoords[width] = w;

                    // Load the third row of input elements x20-x23
                    x20 = v_i8_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);   ifmCoords[width] += 1;
                    x21 = v_i8_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);   ifmCoords[width] += 1;
                    x22 = v_i8_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);   ifmCoords[width] += 1;
                    x23 = v_i8_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);   ifmCoords[width] += 1;

                    // Multiply inputs with first column of filter
                    // proces hw = 00
                    acc0 = v_i8_mac_b(x00, h00, acc0, 1, 1, 0);
                    acc1 = v_i8_mac_b(x01, h00, acc1, 1, 1, 0);
                    acc2 = v_i8_mac_b(x02, h00, acc2, 1, 1, 0);
                    acc3 = v_i8_mac_b(x03, h00, acc3, 1, 1, 0);

                    // proces hw = 10
                    acc0 = v_i8_mac_b(x10, h10, acc0, 1, 1, 0);
                    acc1 = v_i8_mac_b(x11, h10, acc1, 1, 1, 0);
                    acc2 = v_i8_mac_b(x12, h10, acc2, 1, 1, 0);
                    acc3 = v_i8_mac_b(x13, h10, acc3, 1, 1, 0);

                    // proces hw = 20
                    acc0 = v_i8_mac_b(x20, h20, acc0, 1, 1, 0);
                    acc1 = v_i8_mac_b(x21, h20, acc1, 1, 1, 0);
                    acc2 = v_i8_mac_b(x22, h20, acc2, 1, 1, 0);
                    acc3 = v_i8_mac_b(x23, h20, acc3, 1, 1, 0);

                    x00 = x10; x10 = x20;
                    // Load x24
                    x20 = v_i8_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);   ifmCoords[width] += 1;

                    // Multiply inputs with second column of filter
                    // proces hw = 01
                    acc0 = v_i8_mac_b(x01, h01, acc0, 1, 1, 0);
                    acc1 = v_i8_mac_b(x02, h01, acc1, 1, 1, 0);
                    acc2 = v_i8_mac_b(x03, h01, acc2, 1, 1, 0);
                    acc3 = v_i8_mac_b(x04, h01, acc3, 1, 1, 0);

                    // proces hw = 11
                    acc0 = v_i8_mac_b(x11, h11, acc0, 1, 1, 0);
                    acc1 = v_i8_mac_b(x12, h11, acc1, 1, 1, 0);
                    acc2 = v_i8_mac_b(x13, h11, acc2, 1, 1, 0);
                    acc3 = v_i8_mac_b(x14, h11, acc3, 1, 1, 0);

                    // proces hw = 21
                    acc0 = v_i8_mac_b(x21, h21, acc0, 1, 1, 0);
                    acc1 = v_i8_mac_b(x22, h21, acc1, 1, 1, 0);
                    acc2 = v_i8_mac_b(x23, h21, acc2, 1, 1, 0);
                    acc3 = v_i8_mac_b(x20, h21, acc3, 1, 1, 0);

                    x01 = x11; x11 = x21;
                    // Load x25
                    x21 = v_i8_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);   ifmCoords[height] += 1;
                    ifmCoords[width] -= 5;

                    // Multiply inputs with third column of filter
                    // proces hw = 02
                    acc0 = v_i8_mac_b(x02, h02, acc0, 1, 1, 0);
                    acc1 = v_i8_mac_b(x03, h02, acc1, 1, 1, 0);
                    acc2 = v_i8_mac_b(x04, h02, acc2, 1, 1, 0);
                    acc3 = v_i8_mac_b(x05, h02, acc3, 1, 1, 0);

                    // proces hw = 12
                    acc0 = v_i8_mac_b(x12, h12, acc0, 1, 1, 0);
                    acc1 = v_i8_mac_b(x13, h12, acc1, 1, 1, 0);
                    acc2 = v_i8_mac_b(x14, h12, acc2, 1, 1, 0);
                    acc3 = v_i8_mac_b(x15, h12, acc3, 1, 1, 0);

                    // proces hw = 22
                    acc0 = v_i8_mac_b(x22, h22, acc0, 1, 1, 0);
                    acc1 = v_i8_mac_b(x23, h22, acc1, 1, 1, 0);
                    acc2 = v_i8_mac_b(x20, h22, acc2, 1, 1, 0);
                    acc3 = v_i8_mac_b(x21, h22, acc3, 1, 1, 0);


                    /* Copy last two rows of input values to first two rows
                       for next iteration along height */
                    x02 = x12; x03 = x13; x04 = x14; x05 = x15;
                    x12 = x22; x13 = x23; x14 = x20; x15 = x21;

                    // normalize the result
                    // convert accumalator int32 to int8  and compress them in one vector
                    // v_convert_int32_to_i8_b(acc, scale_factor, lane_id, ((rounding mode(rne)) << 16), out_source, 1, 0);
                    x22 = v_convert_int32_to_i8_v_s(acc0.v1, scale_factor, 0, 0, 0);
                    x22 = v_convert_int32_to_i8_v_s(acc0.v2, scale_factor, x22, 0, 1);
                    x22 = v_convert_int32_to_i8_v_s(acc0.v3, scale_factor, x22, 0, 2);
                    x22 = v_convert_int32_to_i8_v_s(acc0.v4, scale_factor, x22, 0, 3);                    
                    //Store the result
                    v_i8_st_tnsr(ofmCoords, ofm, x22, 0, 1, 0);          ofmCoords[width] += 1;

                    x22 = v_convert_int32_to_i8_v_s(acc1.v1, scale_factor, x22, 0, 0);
                    x22 = v_convert_int32_to_i8_v_s(acc1.v2, scale_factor, x22, 0, 1);
                    x22 = v_convert_int32_to_i8_v_s(acc1.v3, scale_factor, x22, 0, 2);
                    x22 = v_convert_int32_to_i8_v_s(acc1.v4, scale_factor, x22, 0, 3);
                    v_i8_st_tnsr(ofmCoords, ofm, x22, 0, 1, 0);          ofmCoords[width] += 1;

                    x22 = v_convert_int32_to_i8_v_s(acc2.v1, scale_factor, x22, 0, 0);
                    x22 = v_convert_int32_to_i8_v_s(acc2.v2, scale_factor, x22, 0, 1);
                    x22 = v_convert_int32_to_i8_v_s(acc2.v3, scale_factor, x22, 0, 2);
                    x22 = v_convert_int32_to_i8_v_s(acc2.v4, scale_factor, x22, 0, 3);
                    v_i8_st_tnsr(ofmCoords, ofm, x22, 0, 1, 0);          ofmCoords[width] += 1;

                    x22 = v_convert_int32_to_i8_v_s(acc3.v1, scale_factor, x22, 0, 0);
                    x22 = v_convert_int32_to_i8_v_s(acc3.v2, scale_factor, x22, 0, 1);
                    x22 = v_convert_int32_to_i8_v_s(acc3.v3, scale_factor, x22, 0, 2);
                    x22 = v_convert_int32_to_i8_v_s(acc3.v4, scale_factor, x22, 0, 3);
                    v_i8_st_tnsr(ofmCoords, ofm, x22, 0, 1, 0);          ofmCoords[width] += 1;

                    // Initialize all the accumalators with bias values from vlm
                    acc0.v1 = biasVector[0];
                    acc0.v2 = biasVector[1];
                    acc0.v3 = biasVector[2];
                    acc0.v4 = biasVector[3];

                    acc1.v1 = biasVector[0];
                    acc1.v2 = biasVector[1];
                    acc1.v3 = biasVector[2];
                    acc1.v4 = biasVector[3];

                    acc2.v1 = biasVector[0];
                    acc2.v2 = biasVector[1];
                    acc2.v3 = biasVector[2];
                    acc2.v4 = biasVector[3];

                    acc3.v1 = biasVector[0];
                    acc3.v2 = biasVector[1];
                    acc3.v3 = biasVector[2];
                    acc3.v4 = biasVector[3];

                }
                ifmCoords[width] += widthStep;
                // set input poiner to height start
                ifmCoords[height] = MulStrideMinusPadH;
            }
            // set input poiner to width start
            ifmCoords[width] = widthMinusPadw;
        }
        // Set filter pointer for next set of channels
        filterCoords[depth] += depthStep;
    }
}
