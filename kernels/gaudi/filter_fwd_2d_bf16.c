/**********************************************************************
Copyright (c) 2020 Habana Labs.

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
void main(tensor ifm,
          tensor filter,
          tensor ofm,
            int padw,
            int padh,
            int kernel_w,
            int kernel_h,
            int stride_w,
            int stride_h,
            int dilation_w,
            int dilation_h)
{
    int5 index_space_start = get_index_space_offset();
    int5 index_space_end = get_index_space_size() + index_space_start;
    int5 output_coords = {0};
    for (int b = index_space_start[3] ; b <  index_space_end[3]; b += 1)
    {
        output_coords[3] = b;
        for (int h = index_space_start[2] ; h <  index_space_end[2]; h += 1)
        {
            output_coords[2] = h;
            for (int w = index_space_start[1] ; w <  index_space_end[1]; w += 1)
            {
                output_coords[1] = w;
                for (int d = index_space_start[0]*128 ; d <  index_space_end[0]*128; d += 128)
                {
                    int5 filterCoords = {0};
                    output_coords[0] = d;
                    filterCoords[0] = d;
                    float128 accum = {0};
                    for (int kh = 0 ; kh <  kernel_h; kh++)
                    {
                        filterCoords[2] = kh;
                        //int kw = 0;
                        for (int kw = 0 ; kw <  kernel_w; kw++)
                        {
                            filterCoords[1] = kw;
                            int5 ifmCoords = {d,
                                              (stride_w*w) - padw + (kw*dilation_w),
                                              (stride_h*h) - padh + (kh*dilation_h) ,
                                               b,
                                               0};
                            bfloat128 filterVector = v_bf16_ld_tnsr_b(filterCoords, filter, 0, 0, 1, 0);
                            bfloat128 ifmVector = v_bf16_ld_tnsr_b(ifmCoords, ifm, 0, 0, 1, 0);
                            accum = v_bf16_mac_acc32_b(filterVector, ifmVector, accum, (e_no_negation) << 1, 1, 0);
                        }
                    }
                    bfloat128 out = v_convert_f32_to_bf16_all_b (accum,0,0,1,0);
                    v_bf16_st_tnsr(output_coords, ofm, out, 0, 1, 0);
                }
            }
        }
    }
}
