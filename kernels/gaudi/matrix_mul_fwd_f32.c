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
#pragma tpc_printf (enable)
//#define PRINTF_ENABLE 1

#define VECTORLENGTH 64
#define ACCUMWIDTH  1
#define PARTWIDTH  (ACCUMWIDTH * VECTORLENGTH)
#define PARTHEIGHT  6


void main(tensor aMatrix,
          tensor bMatrix,
          tensor cMatrix)
{
    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

    const int commonSize = get_dim_size(aMatrix, 0);
    const int commonSizeMinusOne = commonSize - 1;

    const int dim1Mask = 1 << 1;

    int5 aCoords[PARTHEIGHT] = {0};
    int5 bCoords = {0};
    int5 cCoords = {0};

    for(int batch = indexSpaceStart[2];
            batch < indexSpaceEnd[2];
            batch++)
    {
        aCoords[0][2] = batch;
        aCoords[1][2] = batch;
        aCoords[2][2] = batch;
        aCoords[3][2] = batch;
        aCoords[4][2] = batch;
        aCoords[5][2] = batch;
        bCoords[2] = batch;
        cCoords[2] = batch;

        for(int blockStartY = indexSpaceStart[1] * PARTHEIGHT;
                blockStartY < indexSpaceEnd[1] * PARTHEIGHT;
                blockStartY += PARTHEIGHT)
        {
            for(int blockStartX = indexSpaceStart[0] * PARTWIDTH;
                    blockStartX < indexSpaceEnd[0] * PARTWIDTH;
                    blockStartX += PARTWIDTH)
            {
                float64 accums[PARTHEIGHT * ACCUMWIDTH];
                float64 bValue[ACCUMWIDTH];
                float64 aValue[PARTHEIGHT];

                float64 bias = {0};
                accums[0] = bias;
                accums[1] = bias;
                accums[2] = bias;
                accums[3] = bias;
                accums[4] = bias;
                accums[5] = bias;

                aCoords[0][1] = blockStartY + 0;
                aCoords[1][1] = blockStartY + 1;
                aCoords[2][1] = blockStartY + 2;
                aCoords[3][1] = blockStartY + 3;
                aCoords[4][1] = blockStartY + 4;
                aCoords[5][1] = blockStartY + 5;

                aCoords[0][0] = 0;
                aCoords[1][0] = 0;
                aCoords[2][0] = 0;
                aCoords[3][0] = 0;
                aCoords[4][0] = 0;
                aCoords[5][0] = 0;

                __global__ char * p_aValue0 = gen_addr(aCoords[0], aMatrix);
                __global__ char * p_aValue1 = gen_addr(aCoords[1], aMatrix);
                __global__ char * p_aValue2 = gen_addr(aCoords[2], aMatrix);
                __global__ char * p_aValue3 = gen_addr(aCoords[3], aMatrix);
                __global__ char * p_aValue4 = gen_addr(aCoords[4], aMatrix);
                __global__ char * p_aValue5 = gen_addr(aCoords[5], aMatrix);

                aValue[0] = v_f32_ld_g(p_aValue0);
                aValue[1] = v_f32_ld_g(p_aValue1);
                aValue[2] = v_f32_ld_g(p_aValue2);
                aValue[3] = v_f32_ld_g(p_aValue3);
                aValue[4] = v_f32_ld_g(p_aValue4);
                aValue[5] = v_f32_ld_g(p_aValue5);

#ifdef PRINTF_ENABLE
                for(int ii=0;ii<6;ii++)
                {
                    for (int j=0;j<2;j++)
                    {
                        printf("aValue[%d]", ii);
                        printf("[%d] ", j);
                        printf("is %f \n", aValue[ii][j]);
                    }
                }
#endif

                bCoords[1] = 0;
                bCoords[0] = blockStartX;

                bValue[0] = v_f32_ld_tnsr_b(bCoords, bMatrix);
                bCoords[0] += VECTORLENGTH;

#ifdef PRINTF_ENABLE
                for (int j=0;j<4;j++)
                {
                    printf("bValue[0][%d]", j);
                    printf("is %f \n", bValue[0][j]);
                }

                printf("Start new loop \n");
#endif                
                for(int c = 0; c < commonSize; c++)
                {
                    accums[0] = v_f32_mac_b(bValue[0], aValue[0], accums[0]);
                    accums[1] = v_f32_mac_b(bValue[0], aValue[1], accums[1]);
                    accums[2] = v_f32_mac_b(bValue[0], aValue[2], accums[2]);
                    accums[3] = v_f32_mac_b(bValue[0], aValue[3], accums[3]);
                    accums[4] = v_f32_mac_b(bValue[0], aValue[4], accums[4]);
                    accums[5] = v_f32_mac_b(bValue[0], aValue[5], accums[5]);

#ifdef PRINTF_ENABLE
                    for(int ii=0;ii<6;ii++)
                    {
                        for (int j=0;j<4;j++)
                        {
                            printf("accums[%d]", ii);
                            printf("[%d] ", j);
                            printf("is %f \n", accums[ii][j]);
                        }
                    }
#endif
                    char pred = s_i32_cmp_less(c, commonSizeMinusOne);
                    bCoords = i_i32_add(1, bCoords, dim1Mask, 0, bCoords, pred);

                    bCoords[0] = blockStartX;
                    bValue[0] = v_f32_ld_tnsr_b(bCoords, bMatrix);
                    bCoords[0] += VECTORLENGTH;

#ifdef PRINTF_ENABLE
                    for (int j=0;j<4;j++)
                    {
                        printf("next bValue[0][%d]", j);
                        printf("is %f \n", bValue[0][j]);
                    }
#endif
                    aCoords[0][0] += 1;
                    aCoords[1][0] += 1;
                    aCoords[2][0] += 1;
                    aCoords[3][0] += 1;
                    aCoords[4][0] += 1;
                    aCoords[5][0] += 1;

                    p_aValue0 = gen_addr(aCoords[0], aMatrix);
                    p_aValue1 = gen_addr(aCoords[1], aMatrix);
                    p_aValue2 = gen_addr(aCoords[2], aMatrix);
                    p_aValue3 = gen_addr(aCoords[3], aMatrix);
                    p_aValue4 = gen_addr(aCoords[4], aMatrix);
                    p_aValue5 = gen_addr(aCoords[5], aMatrix);

                    aValue[0] = v_f32_ld_g(p_aValue0, 0, aValue[0], pred);
                    aValue[1] = v_f32_ld_g(p_aValue1, 0, aValue[1], pred);
                    aValue[2] = v_f32_ld_g(p_aValue2, 0, aValue[2], pred);
                    aValue[3] = v_f32_ld_g(p_aValue3, 0, aValue[3], pred);
                    aValue[4] = v_f32_ld_g(p_aValue4, 0, aValue[4], pred);
                    aValue[5] = v_f32_ld_g(p_aValue5, 0, aValue[5], pred);

#ifdef PRINTF_ENABLE
                    for(int ii=0;ii<6;ii++)
                    {
                        for (int j=0;j<2;j++)
                        {
                            printf("[%d] ", j);
                            printf("is %f \n", aValue[ii][j]);
                        }
                    }                    

                    printf("Next common loop \n");
#endif                    
                }

                cCoords[0] = blockStartX;
                #pragma unroll (PARTHEIGHT)
                for(int i = 0; i < PARTHEIGHT; i++)
                {
                    cCoords[1] = blockStartY + i;
                    v_f32_st_tnsr(cCoords, cMatrix, accums[i]);
                }
            }
        }
    }
}
