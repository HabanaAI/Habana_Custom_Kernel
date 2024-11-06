/**********************************************************************
Copyright (c) 2024 Habana Labs.

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
//#pragma tpc_printf (enable)
#define print_vec_float(vec, str)  { printf(str); for (int ii=0; ii<5; ii++) {  printf("%f, ", vec[ii]);  }; printf("\n"); }

//A [batch, seq_len, dstate, dim]
void main(tensor ifm_in, tensor ifm_start, tensor ofm_out, unsigned int rng_size)
{
    const int mem = 0;
    const int dim = 1;
    const int head = 2;
    const int batch = 3;

    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;
    const int memlen_size = get_dim_size(ifm_dt, 0); 

    int5 ifm_in_Coords      = { 0, 0, 0, 0, 0 };
    int5 ifm_start_Coords   = { 0, 0, 0, 0, 0 };
    int5 ofm_out_Coords     = { 0, 0, 0, 0, 0 };

    // memlen (FCD)
    const int memStep = VECTOR_SIZE;
    const int memStart = indexSpaceStart[mem] * memStep;
    const int memEnd = indexSpaceEnd[mem] * memStep;
    // dim
    const int dimStep = 1;
    const int dimStart = indexSpaceStart[dim] * dimStep;
    const int dimEnd = indexSpaceEnd[dim] * dimStep;
    // head
    const int headStep = 1;
    const int headStart = indexSpaceStart[head];
    const int headEnd = indexSpaceEnd[head];
    // BATCH
    const int batchStep = 1;
    const int batchStart = indexSpaceStart[batch];
    const int batchtEnd = indexSpaceEnd[batch];

    VECTOR vec_in;
    int idx_start;

    #pragma loop_taken
    for (int b = batchStart; b < batchtEnd; b += batchStep)
    {
        ifm_in_Coords[batch] = 0;
        ifm_start_Coords[batch] = b;
        ofm_out_Coords[batch] = b;

        for (int h = headStart ; n < headEnd; n += headStep)
        {
            ifm_in_Coords[head] = h;
            ifm_start_Coords[head] = 0; 
            ofm_out_Coords[dstate] = h; 

            for (int d = dimStart; d < dimEnd; d += dimStep)
            {
                ifm_in_Coords[dim] = d;
                ifm_start_Coords[dim] = 0;
                ofm_out_Coords[dim] = d;

                for (int m = 0; m < rng_size; m += memStep)
                {
                    //ifm_state_Coords[seq] = 0;  // input state doesn't hace seq
                    ifm_in_Coords[mem] = m;
                    ifm_start_Coords[mem] = 0;
                    ofm_out_Coords[mem] = m;

                    idx_start = s_i32_ld_g(gen_addr(ifm_start_Coords, ifm_start));
                    ifm_in_Coords[mem] = m + idx_start;
                    vec_in = v_ld_tnsr_i(ifm_in_Coords, ifm_in);

                    st_tnsr_i_v(ofm_out_Coords, ofm_out, vec_in);

                } // end of mem_len loop

            } // end of dim loop
        }
    }
}