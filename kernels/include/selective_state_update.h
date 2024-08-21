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

void main(tensor ifm_state, tensor ifm_x, tensor ifm_dt, tensor ifm_A, tensor ifm_B, tensor ifm_C, tensor ifm_D, tensor ifm_dt_bias,
#if defined(USE_Z_TENSOR)
        tensor ifm_z, 
#endif        
        tensor ofm_out,
        unsigned int with_D, 
        unsigned int with_dt_bias
        )
{
    const int dim = 0;
    const int dstate = 1;
    const int nhead = 2;
    const int batch = 3;

    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

    int5 ifm_state_Coords   = { 0, 0, 0, 0, 0 };
    int5 ifm_x_Coords       = { 0, 0, 0, 0, 0 };
    int5 ifm_dt_Coords      = { 0, 0, 0, 0, 0 };
    int5 ifm_A_Coords       = { 0, 0, 0, 0, 0 };
    int5 ifm_B_Coords       = { 0, 0, 0, 0, 0 };
    int5 ifm_C_Coords       = { 0, 0, 0, 0, 0 };
    int5 ifm_D_Coords       = { 0, 0, 0, 0, 0 };
    int5 ifm_z_Coords       = { 0, 0, 0, 0, 0 };
    int5 ifm_dt_bias_Coords = { 0, 0, 0, 0, 0 };
    int5 ofm_out_Coords     = { 0, 0, 0, 0, 0 };

    // dim (FCD)
    const int dimStep = VECTOR_SIZE;
    const int dimStart = indexSpaceStart[dim] * dimStep;
    const int dimEnd = indexSpaceEnd[dim] * dimStep;
    // dstate
    const int dstateStep = 1;
    const int dstateStart = indexSpaceStart[dstate] * dstateStep;
    const int dstateEnd = indexSpaceEnd[dstate] * dstateStep;
    // nhead
    const int nheadStep = 1;
    const int nheadStart = indexSpaceStart[nhead];
    const int nheadEnd = indexSpaceEnd[nhead];
    // BATCH
    const int batchStep = 1;
    const int batchStart = indexSpaceStart[batch];
    const int batchtEnd = indexSpaceEnd[batch];

    VECTOR vec_state, vec_x, vec_dt, vec_A, scl_B, scl_C, vec_D, vec_dt_bias;
    VECTOR vec_out;

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
        ifm_dt_bias_Coords[batch] = 0; // dt_bias doesn't have batch
        ofm_out_Coords[batch] = b;
        #pragma loop_taken
        for (int h = nheadStart; h < nheadEnd; h += nheadStep)
        {
            ifm_state_Coords[nhead] = h;
            ifm_x_Coords[nhead] = h;
            ifm_dt_Coords[nhead] = h;
            ifm_A_Coords[nhead] = h; 
            ifm_B_Coords[nhead] = h;
            ifm_C_Coords[nhead] = h;
            ifm_D_Coords[nhead] = h; 
            ifm_z_Coords[nhead] = h;
            ifm_dt_bias_Coords[nhead] = h; 
            ofm_out_Coords[nhead] = h;
            //#pragma loop_taken
            //#pragma unroll(NUM_UNROLL)
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
                ifm_dt_bias_Coords[dim] = d; 
                ofm_out_Coords[dim] = d;

                vec_x = v_ld_tnsr_i(ifm_x_Coords, ifm_x);
                vec_out = 0; // reset to 0 before dstate loop
                VECTOR temp;
                for (int n = dstateStart ; n < dstateEnd; n += 1)
                {
                    ifm_state_Coords[dstate] = n;
                    ifm_x_Coords[dstate] = 0;  // x doesn't have dstate
                    ifm_dt_Coords[dstate] = 0; // dt doesn't have dstate
                    ifm_A_Coords[dstate] = n; 
                    ifm_B_Coords[dstate] = n;
                    ifm_C_Coords[dstate] = n;
                    ifm_D_Coords[dstate] = 0; // D doesn't have dstate
                    ifm_z_Coords[dstate] = 0; // z doesn't have dstate
                    ifm_dt_bias_Coords[dstate] = 0; // dt_bias doesn't have dstate
                    ofm_out_Coords[dstate] = 0; // out doesn't have dstate

                    vec_dt = v_ld_tnsr_i(ifm_dt_Coords, ifm_dt);
                    vec_dt_bias = v_ld_tnsr_i(ifm_dt_bias_Coords, ifm_dt_bias);
                    int8_t Pred_dt = s_i32_cmp_neq(with_dt_bias, 0);
                    vec_dt = v_add_v_v_b(vec_dt, vec_dt_bias, vec_dt, Pred_dt, 0);
                    
#if defined(USE_SOFTPLUS_DT) 
                    VECTOR vec_one = 1;
                    temp = exp(vec_dt);
                    temp = vec_one + temp;
                    vec_dt = log(temp);
#endif
                    VECTOR dA;
                    
                    vec_A = v_ld_tnsr_i(ifm_A_Coords, ifm_A);
                    temp = v_mul_v_v(vec_dt, vec_A);
                    dA = exp(temp);                    

                    VECTOR dB;
                    scl_B = v_ld_g_a(gen_addr(ifm_B_Coords, ifm_B));
                    scl_C = v_ld_g_a(gen_addr(ifm_C_Coords, ifm_C));
                    dB = v_mul_v_v(vec_dt, scl_B);

                    vec_state = v_ld_tnsr_i(ifm_state_Coords, ifm_state);
                    
                    temp = v_mul_v_v(vec_state, dA);
                    vec_state = v_mac_v_v(dB, vec_x, temp);

                    VECTOR vec_out_partial;
                    vec_out_partial = v_mul_v_v(vec_state, scl_C);
                    vec_out = v_add_v_v(vec_out_partial, vec_out, 0);

                } // end of dstate loop

                int8_t Pred_D = s_i32_cmp_neq(with_D, 0);
                vec_D = v_ld_tnsr_i(ifm_D_Coords, ifm_D);
                vec_out = v_mac_v_v_b(vec_D, vec_x, vec_out, 0, Pred_D, 0);
#if defined(USE_Z_TENSOR)
                VECTOR tp_sig, vec_z;
                temp = v_ld_tnsr_i(ifm_z_Coords, ifm_z);
                tp_sig = sigmoid(temp);
                vec_z = v_mul_v_v(tp_sig, temp);
                vec_out = v_mul_v_v(vec_out, vec_z);
#endif
                st_tnsr_rmw_i_v(ofm_out_Coords, ofm_out, vec_out, e_rmw_add, e_rmw_atomic, e_tnsr_dt_srf);          
            } // end of dim loop
        }
    }
}

