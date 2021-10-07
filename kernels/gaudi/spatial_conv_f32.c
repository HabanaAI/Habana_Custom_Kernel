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
  int channelSize = get_dim_size(ifm, 0);
  int5 output_coords = {0};
    for (int b = index_space_start[4] ; b <  index_space_end[4] ; b += 1)
    {
        output_coords[3] = b;
        for (int h = index_space_start[3] ; h <  index_space_end[3] ; h += 1)
        {
            output_coords[2] = h;
            for (int w = index_space_start[2] ; w <  index_space_end[2] ; w += 1)
            {
                output_coords[1] = w;
                for(int k = index_space_start[1] ; k < index_space_end[1] ; k += 1)
                {
                    output_coords[0] = k;
                    float64 accum_all = {0};
                    for (int d = 0 ; d < channelSize; d += 64)
                    {
                         int5 filterCoords = {0};
                         filterCoords[0] = d;
                         filterCoords[1] = k;
                         float64 accum = {0};
                         for (int kh = 0 ; kh <  kernel_h; kh++)
                        {
                             filterCoords[3] = kh;
                             for (int kw = 0 ; kw <  kernel_w; kw++)
                             {
                                 filterCoords[2] = kw;
                                 int5 ifmCoords = {d,
                                             (stride_w*w) - padw + (kw*dilation_w),
                                             (stride_h*h) - padh + (kh*dilation_h) ,
                                             b,
                                             0};
                                 float64 filterVector = v_f32_ld_tnsr_b(filterCoords, filter);
                                 float64 ifmVector = v_f32_ld_tnsr_b(ifmCoords, ifm);
                                 accum = v_f32_mac_b(filterVector, ifmVector, accum, (e_no_negation) << 1);
                             }
                         }
                         accum_all += accum;
                     }
                     accum_all = v_f32_reduce_add(accum_all);
                     v_f32_st_tnsr_partial(output_coords, ofm, accum_all, 0, 0);
                 }
            }
        }
    }
    
}