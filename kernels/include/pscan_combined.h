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
#pragma tpc_printf (enable)
#define print_vec_float(vec, str)  { printf(str); for (int ii=0; ii<5; ii++) {  printf("%f, ", vec[ii]);  }; printf("\n"); }

#define aso_init()                                                                  \
    int count = 0;                                                                  \
    set_semaphore_value(0);

#define aso_wait()                                                                  \
    aso(SW_INC | SW_VPU);                                     \
    volatile int a = get_semaphore_value();                                         \
    while (a == count)                                                              \
    {                                                                               \
        a = get_semaphore_value();                                                  \
    }                                                                               \
    count++;

//A [batch, seq_len, dstate, dim]
void main(tensor ifm_state, tensor ifm_x, tensor ifm_dt, tensor ifm_A, tensor ifm_B, tensor ifm_C, tensor ifm_D, tensor ifm_z,
        tensor ofm_state, tensor ofm_out)
{
    const int dim = 0;
    const int dstate = 1;
    const int seq = 2;
    const int batch = 3;

    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;
    const int seq_size = get_dim_size(ifm_dt, 2); 
    const int dstate_size = get_dim_size(ifm_A, 1); 

    int5 ifm_state_Coords   = { 0, 0, 0, 0, 0 };
    int5 ifm_x_Coords       = { 0, 0, 0, 0, 0 };
    int5 ifm_dt_Coords      = { 0, 0, 0, 0, 0 };
    int5 ifm_A_Coords       = { 0, 0, 0, 0, 0 };
    int5 ifm_B_Coords       = { 0, 0, 0, 0, 0 };
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
    // sequence len
    //const int seqStep = 1;
    //const int seqStart = indexSpaceStart[seq];
    //const int seqEnd = indexSpaceEnd[seq];
    // BATCH
    const int batchStep = 1;
    const int batchStart = indexSpaceStart[batch];
    const int batchtEnd = indexSpaceEnd[batch];

    VECTOR vec_state, vec_x, vec_dt, vec_A, scl_B, scl_C, vec_D, vec_z;
    VECTOR vec_out;
    aso_init();

    #pragma loop_taken
    for (int b = batchStart; b < batchtEnd; b += batchStep)
    {
        ifm_state_Coords[batch] = b;
        ifm_x_Coords[batch] = b;
        ifm_dt_Coords[batch] = b;
        ifm_A_Coords[batch] = 0; // A doesn't have batch
        ifm_B_Coords[batch] = b;
        ifm_C_Coords[batch] = b;
        ifm_D_Coords[batch] = 0; // D doesn't have batch
        ifm_z_Coords[batch] = b;
        ofm_out_Coords[batch] = b;


        for (int d = dimStart; d < dimEnd; d += dimStep)
        {
            ifm_state_Coords[dim] = d;
            ifm_x_Coords[dim] = d;
            ifm_dt_Coords[dim] = d;
            ifm_A_Coords[dim] = d; 
            ifm_B_Coords[dim] = 0; // B doesn't have dim
            ifm_C_Coords[dim] = 0; // C doesn't have dim
            ifm_D_Coords[dim] = d; 
            ifm_z_Coords[dim] = d;
            ofm_out_Coords[dim] = d;

            // first loop for seq_len loop
            ifm_state_Coords[seq] = 0;  // input state doesn't hace seq
            ifm_x_Coords[seq] = 0;//h0
            ifm_dt_Coords[seq] = 0;//h0
            ifm_A_Coords[seq] = 0;      // A doesn't have seq
            ifm_B_Coords[seq] = 0;//h0
            ifm_C_Coords[seq] = 0;//h0
            ifm_D_Coords[seq] = 0; // D doens't have seq len
            ifm_z_Coords[seq] = 0;//h0
            ofm_out_Coords[seq] = 0;//h0

            VECTOR temp, dA, dB, vec_out_partial;
            vec_out = 0; // reset to 0 before dstate loop
            vec_dt = v_ld_tnsr_i(ifm_dt_Coords, ifm_dt);
            VECTOR vec_one = 1;
            temp = exp(vec_dt);
            temp = vec_one + temp;
            vec_dt = log(temp);

            for (int n = 0 ; n < dstate_size; n += 1)
            {
                ifm_state_Coords[dstate] = n;
                ifm_x_Coords[dstate] = 0;  // x doesn't have dstate
                ifm_dt_Coords[dstate] = 0; // dt doesn't have dstate
                ifm_A_Coords[dstate] = n; 
                ifm_B_Coords[dstate] = n;
                ifm_C_Coords[dstate] = n;
                ifm_D_Coords[dstate] = 0; // D doesn't have dstate
                ifm_z_Coords[dstate] = 0; // z doesn't have dstate
                ofm_out_Coords[dstate] = 0; 

                vec_A = v_ld_tnsr_i(ifm_A_Coords, ifm_A);
                vec_x = v_ld_tnsr_i(ifm_x_Coords, ifm_x);
                temp = v_mul_v_v(vec_dt, vec_A);
                dA = exp(temp);                    

                scl_B = v_ld_g_a(gen_addr(ifm_B_Coords, ifm_B));
                dB = v_mul_v_v(vec_dt, scl_B);

                vec_state = v_ld_tnsr_i(ifm_state_Coords, ifm_state);

                temp = v_mul_v_v(vec_state, dA);
                vec_state = v_mac_v_v(dB, vec_x, temp);
                st_tnsr_i_v(ifm_state_Coords, ofm_state, vec_state);
                aso_wait();

                scl_C = v_ld_g_a(gen_addr(ifm_C_Coords, ifm_C));
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

            // start from loop 1, we get state from ofm_state instead on ifm_state
            for (int h = 1; h < seq_size; h += 1)
            {
                ifm_state_Coords[seq] = 0;  // input state doesn't hace seq
                ifm_x_Coords[seq] = h;
                ifm_dt_Coords[seq] = h;
                ifm_A_Coords[seq] = 0;      // A doesn't have seq
                ifm_B_Coords[seq] = h;
                ifm_C_Coords[seq] = h;
                ifm_D_Coords[seq] = 0; // D doens't have seq len
                ifm_z_Coords[seq] = h;
                ofm_out_Coords[seq] = h;

                vec_out = 0; // reset to 0 before dstate loop
                vec_dt = v_ld_tnsr_i(ifm_dt_Coords, ifm_dt);
                VECTOR vec_one = 1;
                temp = exp(vec_dt);
                temp = vec_one + temp;
                vec_dt = log(temp);
                
                for (int n = 0 ; n < dstate_size; n += 1)
                {
                    ifm_state_Coords[dstate] = n;
                    ifm_x_Coords[dstate] = 0;  // x doesn't have dstate
                    ifm_dt_Coords[dstate] = 0; // dt doesn't have dstate
                    ifm_A_Coords[dstate] = n; 
                    ifm_B_Coords[dstate] = n;
                    ifm_C_Coords[dstate] = n;
                    ifm_D_Coords[dstate] = 0; // D doesn't have dstate
                    ifm_z_Coords[dstate] = 0; // z doesn't have dstate
                    ofm_out_Coords[dstate] = 0; 

                    
                    vec_A = v_ld_tnsr_i(ifm_A_Coords, ifm_A);
                    vec_x = v_ld_tnsr_i(ifm_x_Coords, ifm_x);
                    temp = v_mul_v_v(vec_dt, vec_A);
                    dA = exp(temp);                    

                    scl_B = v_ld_g_a(gen_addr(ifm_B_Coords, ifm_B));
                    dB = v_mul_v_v(vec_dt, scl_B);

                    vec_state = v_ld_tnsr_i(ifm_state_Coords, ofm_state);

                    temp = v_mul_v_v(vec_state, dA);
                    vec_state = v_mac_v_v(dB, vec_x, temp);
                    st_tnsr_i_v(ifm_state_Coords, ofm_state, vec_state);
                    aso_wait();

                    scl_C = v_ld_g_a(gen_addr(ifm_C_Coords, ifm_C));
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

            } // end of seq_len loop
        }
    }
}

