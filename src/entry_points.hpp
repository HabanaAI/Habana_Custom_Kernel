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

#ifndef _ENTRY_POINTS_HPP_
#define _ENTRY_POINTS_HPP_

extern "C"
{

typedef enum 
{
    GOYA_KERNEL_FILTER_2D_F32 = 0,
    GOYA_KERNEL_PRINTF_TEST,
    GOYA_KERNEL_SPARSE_LEN_SUM_F32,
    GOYA_KERNEL_FILTER_2D_I8_W33,
    GOYA_KERNEL_SOFTMAX_FCD_F32,
    GOYA_KERNEL_SOFTMAX_NONFCD_F32,
    GOYA_KERNEL_CAST_I8_F32,
    GOYA_KERNEL_CAST_F32_I16,
    GOYA_KERNEL_LEAKY_RELU_F32,

    GOYA_KERNEL_MAX_EXAMPLE_KERNEL

} Goya_Kernel_Name_e;

typedef enum 
{
    GAUDI_KERNEL_BATCH_NORM_F32 = 0,
    GAUDI_KERNEL_CAST_BF16_F32,
    GAUDI_KERNEL_CAST_F32_BF16,
    GAUDI_KERNEL_FILTER_FWD_2D_BF16,
    GAUDI_KERNEL_LEAKU_RELU_F32,
    GAUDI_KERNEL_SOFTMAX_FCD_BF16,
    GAUDI_KERNEL_SOFTMAX_NONFCD_BF16,
    GAUDI_KERNEL_SPARSE_LEN_SUM_BF16,
    GAUDI_KERNEL_CUSTOMDIV_FWD_F32,
    GAUDI_KERNEL_RELU6_FWD_F32,
    GAUDI_KERNEL_RELU6_BWD_F32,
    GAUDI_KERNEL_RELU6_FWD_BF16,
    GAUDI_KERNEL_RELU6_BWD_BF16,
    GAUDI_KERNEL_MATRIXMUL_FWD_F32,

    GAUDI_KERNEL_MAX_EXAMPLE_KERNEL

} Gaudi_Kernel_Name_e;
/*
 ***************************************************************************************************
 *   @brief This function returns exported kernel names
 *
 *   @param names       [out]  List of strings to be filled with kernel names.
 *   @param kernelCount [in/out] The number of strings in 'names' argument.
 *                      If the list is too short, the library will return the
 *                      required list length.
 *   @param deviceId    [in] The type of device E.g. dali/gaudi etc.
 *
 *   @return                  The status of the operation.
 ***************************************************************************************************
 */
gcapi::GlueCodeReturn_t GetKernelNames(_OUT_ char**         names,
                                       unsigned*            kernelCount,
                                       gcapi::DeviceId_t    deviceId);

/*
 ***************************************************************************************************
 *   @brief This kernel library main entry point, it returns all necessary
 *          information about a kernel to execute on device.
 *
 *
 *   @return                  The status of the operation.
 ***************************************************************************************************
 */
gcapi::GlueCodeReturn_t
HabanaKernel(_IN_  gcapi::HabanaKernelParams_t* params,
             _OUT_ gcapi::HabanaKernelInstantiation_t*instance);

} // extern "C"
#endif
