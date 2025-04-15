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

void main(tensor ifm_state, tensor ifm_x, tensor ifm_C, tensor ifm_D, tensor ifm_z, 
        tensor ofm_out)
{
    const int dim = 0;
    const int dstate = 1;
    const int seq = 2;
    const int batch = 3;

    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;
    const int dstate_size = get_dim_size(ifm_state, 1);

    int5 ifm_state_Coords   = { 0, 0, 0, 0, 0 };
    int5 ifm_x_Coords       = { 0, 0, 0, 0, 0 };
    int5 ifm_C_Coords       = { 0, 0, 0, 0, 0 };
    int5 ifm_D_Coords       = { 0, 0, 0, 0, 0 };
    int5 ifm_z_Coords       = { 0, 0, 0, 0, 0 };
    int5 ofm_out_Coords     = { 0, 0, 0, 0, 0 };

    // dim (FCD)
    const int dimStep = VECTOR_SIZE;
    const int dimStart = indexSpaceStart[dim] * dimStep;
    const int dimEnd = indexSpaceEnd[dim] * dimStep;
    // dstate
    //const int dstateStep = 1;
    //const int dstateStart = indexSpaceStart[dstate] * dstateStep;
    //const int dstateEnd = indexSpaceEnd[dstate] * dstateStep;
    // nhead
    const int seqStep = 1;
    const int seqStart = indexSpaceStart[seq];
    const int seqEnd = indexSpaceEnd[seq];
    // BATCH
    const int batchStep = 1;
    const int batchStart = indexSpaceStart[batch];
    const int batchtEnd = indexSpaceEnd[batch];

    VECTOR vec_state, vec_x, scl_C, vec_D, vec_z;
    VECTOR vec_out;

    #pragma loop_taken
    for (int b = batchStart; b < batchtEnd; b += batchStep)
    {
        ifm_state_Coords[batch] = b;
        ifm_x_Coords[batch] = b;
        ifm_C_Coords[batch] = b;
        ifm_D_Coords[batch] = 0; // D doesn't have batch
        ifm_z_Coords[batch] = b;
        ofm_out_Coords[batch] = b;
        #pragma loop_taken
        for (int h = seqStart; h < seqEnd; h += seqStep)
        {
            ifm_state_Coords[seq] = h;
            ifm_x_Coords[seq] = h;
            ifm_C_Coords[seq] = h;
            ifm_D_Coords[seq] = 0; // D doens't have seq len
            ifm_z_Coords[seq] = h;
            ofm_out_Coords[seq] = h;
            //#pragma loop_taken
            //#pragma unroll(NUM_UNROLL)
            for (int d = dimStart; d < dimEnd; d += dimStep)
            {
                ifm_state_Coords[dim] = d;
                ifm_x_Coords[dim] = d;
                ifm_C_Coords[dim] = 0; // C doesn't have dim
                ifm_D_Coords[dim] = d; 
                ifm_z_Coords[dim] = d;
                ofm_out_Coords[dim] = d;

                vec_out = 0; // reset to 0 before dstate loop
                VECTOR temp;
                //print_vec_float(vec_x, "vec_x HPU: ")
                //for (int n = dstateStart ; n < dstateEnd; n += 1)
                for (int n = 0 ; n < dstate_size; n += 1)
                {
                    ifm_state_Coords[dstate] = n;
                    ifm_C_Coords[dstate] = n;
                    ifm_D_Coords[dstate] = 0; // D doesn't have dstate
                    ifm_z_Coords[dstate] = 0; // z doesn't have dstate
                    ofm_out_Coords[dstate] = 0; // out doesn't have dstate

                    vec_state = v_ld_tnsr_i(ifm_state_Coords, ifm_state);
                    scl_C = v_ld_g_a(gen_addr(ifm_C_Coords, ifm_C));

                    VECTOR vec_out_partial;
                    vec_out_partial = v_mul_v_v(vec_state, scl_C);

                    vec_out = v_add_v_v(vec_out_partial, vec_out, 0);

                } // end of dstate loop

                vec_x = v_ld_tnsr_i(ifm_x_Coords, ifm_x);
                vec_D = v_ld_tnsr_i(ifm_D_Coords, ifm_D);
                vec_out = v_mac_v_v(vec_D, vec_x, vec_out);

                VECTOR tp_sig;
                temp = v_ld_tnsr_i(ifm_z_Coords, ifm_z);
                tp_sig = sigmoid(temp);
                vec_z = v_mul_v_v(tp_sig, temp);
                vec_out = v_mul_v_v(vec_out, vec_z);

                st_tnsr_i_v(ofm_out_Coords, ofm_out, vec_out);
            } // end of dim loop
        }
    }
}