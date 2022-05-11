/**********************************************************************
Copyright (c) 2022 Habana Labs.

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

#define max(a, b) ((a > b) ? a : b)
#define min(a, b) ((a < b) ? a : b)

#define CACHE_LINE_VECTOR_SIZE  (32)

#define st_tnsr_rmw_i_v(a, b, c, rmw_op, rmw, tnsr_dt_location)\
                v_f32_st_tnsr_rmw(a, b, c, MkRMW(\
                    e_rmw_fp32, \
                    rmw_op, \
                    rmw, \
                    tnsr_dt_location\
                ), 0, 1, 0);
void main(const tensor ifm,
          const tensor validCount,        // valid count
          tensor ofm,
          const tensor reciprocal_tab,  // reciprocal_tab is an aux tensor
          const int pad_w,
          const int pad_h,
          const int kernel_w,
          const int kernel_h,
          const int stride_w,
          const int stride_h,
          const int dilation_w,
          const int dilation_h,
          const int include_pads, // not_used
          const int    numTpc,
          const float  invNumTpc)
{

    const int channels = 0;
    const int width    = 1;
    const int height   = 2;
    const int batch    = 3;

    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end   = get_index_space_size() + index_space_start;

    /* CWHB */
    const int channels_step  = 64;
    const int channels_start = index_space_start[channels] * channels_step;
    const int channels_end   = index_space_end[channels] * channels_step;

    const int width_step  = 1;
    const int width_start = index_space_start[width] * width_step;
    const int width_end   = index_space_end[width] * width_step;

    const int height_step  = 1;
    const int height_start = index_space_start[height] * height_step;
    const int height_end   = index_space_end[height] * height_step;

    int5 validCountCoord = {0,0,0,0,0};
    __global__ int* validCountAddr = (__global__ int*)gen_addr(validCountCoord, validCount);
    const int validCounter = s_i32_ld_g(validCountAddr);
    const int tpcWidth = (int)(validCounter * invNumTpc);

    const int idxStart = index_space_start[batch] * tpcWidth;
    int idxEnd = s_i32_min(index_space_end[batch] * tpcWidth, get_dim_size(ifm, 3));
    // last TPC should take care of not multiple of 8 tensor length
    if (index_space_end[batch] == numTpc)
    {
        idxEnd = validCounter;
    }
    // Get prefetch stride value for tensor 0
    int t0Stride = get_tensor_hwpref_stride(0);

    /* Prefetch indices to Scalar cache lines only once,
       H/W Prefetch logic will tick in later */
    int5 prefetchCoords = {0,0,0,idxStart,0};
    __global__ int *indexPtr = (__global__ int*)gen_addr(prefetchCoords, ifm);
    for (int i = 0; i < t0Stride; i++)
    {
        prefetch(indexPtr);

        prefetchCoords[3] += CACHE_LINE_VECTOR_SIZE;
        indexPtr = (__global__ int*)gen_addr(prefetchCoords, ifm);
    }


    const int ofm_w = get_dim_size(ofm, width);
    const int ofm_h = get_dim_size(ofm, height);

    float64 reciprocal = v_f32_mov_b(0); // filled by reciprocal aux tensor

    float64 fInput;

    // Iterate over IFM channels/width/height/batch
    int numOfIndices  = idxEnd - idxStart;
    int idx = idxStart;

    #pragma loop_taken
    for (int b = idx; b < idx + numOfIndices; b++)
    {
        #pragma loop_taken
        for (int c = channels_start; c < channels_end; c += channels_step)
        {
            #pragma loop_taken
            for (int h = height_start; h < height_end; h += height_step)
            {
                int start_h = (h * stride_h) - pad_h;

                #pragma loop_taken
                for (int w = width_start; w < width_end; w += width_step)
                {
                    int start_w = (w * stride_w) - pad_w;
                    int5 input_coord = {c, w, h, b};

                    // calculate, the value for back propagation, which depends on the
                    // number of pixels that participated in averaging in the forward
                    float64 prop_vec = v_f32_ld_tnsr_b(input_coord, ifm);
                    // count number of pixels which are into ifm and into kernel at the same time
                    long int  hend = min(start_h + kernel_h, ofm_h);
                    long int  wend = min(start_w + kernel_w, ofm_w);

                    long int  hstart = max(start_h, (long int ) 0);
                    long int  wstart = max(start_w, (long int ) 0);

                    int pixels_in_area = (hend - hstart) * (wend - wstart);

                    //VECTOR recip = reciprocal(pixels_in_area);
                    int5 reciprocal_coord = {pixels_in_area, 0, 0, 0, 0};
                    __global__ void* reciprocal_addr= gen_addr(reciprocal_coord, reciprocal_tab);
                    reciprocal = v_f32_ld_g(reciprocal_addr);

                    fInput = v_f32_mul_b(prop_vec, reciprocal);

                    #pragma loop_taken
                    for (int kh = 0; kh < kernel_h; kh++)
                    {
                        int ifm_h_index = start_h + (kh * dilation_h);

                        #pragma loop_taken
                        for(int kw = 0; kw < kernel_w; kw++)
                        {
                            int ifm_w_index = start_w + (kw * dilation_w);

                            int5 ifm_coords = {c, ifm_w_index, ifm_h_index, b, 0};

                            st_tnsr_rmw_i_v(ifm_coords, ofm, fInput,
                                    e_rmw_add,
                                    e_rmw_atomic,
                                    e_tnsr_dt_srf
                            );
                        }
                    }
                }
            }
        }
    }
}
