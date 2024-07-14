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

#pragma tpc_printf (enable)

#define print_vec_int(vec, str)  { printf(str); for (int ii=0; ii<64; ii++) {  printf("%d, ", vec[ii]);  }; printf("\n"); }
#define print_vec_float(vec, str)  { printf(str); for (int ii=0; ii<64; ii++) {  printf("%f, ", vec[ii]);  }; printf("\n"); }

void main(tensor input0, tensor input1, tensor output, tensor aux)
{
    set_lut_32(aux);

    // form indices vector in range [0,31]
    uint64 indx = read_lane_id_4b_b(); 
    indx = v_u32_and_b(indx, 31);
    print_vec_int(indx, "indx: ");

    // get table 0
    int funcId0 = (0x0 << 2) | 0x3; 

    float64 c = v_f32_lookup(indx, funcId0, SW_LUT_PTR, 0);

    printf("\ntable0: \n");
    print_vec_float(c, "c:");

    // get table 1
    int funcId1 = (0x1 << 2) | 0x3; 

    float128 c1c2 = {0};
    float64 c0 = 0;
    c1c2 = v_f32_lookup_2c(indx, funcId1, SW_LUT_PTR, c1c2);
    c0 = v_f32_lookup_1c(indx, funcId1, SW_LUT_PTR, c0);

    printf("\ntable1: \n");
    print_vec_float(c0, "c0: ");
    print_vec_float(c1c2.v1, "c1: ");
    print_vec_float(c1c2.v2, "c2: ");

    // perform computations using lut coefficients
    int5 ifmCoords = {0};
    float64 x00 = v_f32_ld_tnsr_b(ifmCoords, input0);
    float64 x01 = v_f32_ld_tnsr_b(ifmCoords, input1);

    float64 res = c0;
    res += x00 * c1c2.v1;
    res += x01 * c1c2.v2;
    res *= c;

    int5 ofmCoords = {0};
    v_f32_st_tnsr(ofmCoords, output, res);
}
