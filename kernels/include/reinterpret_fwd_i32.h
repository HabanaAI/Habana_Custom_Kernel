#include "kernel_config.h"

void main(const tensor input, tensor output) {
    printf("Hello from Reinterpret kernel\n");
    const int dim0 = 0;
    const int dim1 = 1;
    const int dim2 = 2;
    const int dim3 = 3;
    const int dim4 = 4;

    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end   = get_index_space_size() + index_space_start;

    const int dim0Step     = 64; // we consume 64 float elements at a time (float64)
    const int dim0Start    = index_space_start[dim0] * dim0Step;
    const int dim0End      = index_space_end[dim0] * dim0Step;

    const int dim1Step     = 1;
    const int dim1Start    = index_space_start[dim1] * dim1Step;
    const int dim1End      = index_space_end[dim1] * dim1Step;

    const int dim2Step     = 1;
    const int dim2Start    = index_space_start[dim2] * dim2Step;
    const int dim2End      = index_space_end[dim2] * dim2Step;

    const int dim3Step     = 1;
    const int dim3Start    = index_space_start[dim3] * dim3Step;
    const int dim3End      = index_space_end[dim3] * dim3Step;

    const int dim4Step     = 1;
    const int dim4Start    = index_space_start[dim4] * dim4Step;
    const int dim4End      = index_space_end[dim4] * dim4Step;

    int5 coords = {0, 0, 0, 0, 0};

    for (int d0 = dim0Start; d0 < dim0End; d0 += dim0Step) {
        coords[dim0] = d0;
        for (int d1 = dim1Start; d1 < dim1End; d1 += dim1Step) {
            coords[dim1] = d1;
            for (int d2 = dim2Start; d2 < dim2End; d2 += dim2Step) {
                coords[dim2] = d2;
                for (int d3 = dim3Start; d3 < dim3End; d3 += dim3Step) {
                    coords[dim3] = d3;
                    for (int d4 = dim4Start; d4 < dim4End; d4 += dim4Step) {
                        coords[dim4] = d4;
                        // read the bytes as i32 and store in the output tensor. It's as easy as that
                        v_i32_st_tnsr(coords, output, v_i32_ld_tnsr_b(coords, input));
                    }
                }
            }
        }
    }
}
