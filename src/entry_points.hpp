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

#ifndef _ENTRY_POINTS_HPP_
#define _ENTRY_POINTS_HPP_

extern "C"
{

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
    GAUDI_KERNEL_RELU_FWD_F32,
    GAUDI_KERNEL_RELU_BWD_F32,
    GAUDI_KERNEL_RELU_FWD_BF16,
    GAUDI_KERNEL_RELU_BWD_BF16,
    GAUDI_KERNEL_MATRIXMUL_FWD_F32,
    GAUDI_KERNEL_SPATIAL_CONV_F32,
    GAUDI_KERNEL_SIN_F32,
    GAUDI_KERNEL_ADD_F32,
    GAUDI_KERNEL_AVG_POOL_2D_FWD_F32,
    GAUDI_KERNEL_AVG_POOL_2D_BWD_F32,
    GAUDI_KERNEL_SEARCH_SORTED_FWD_F32,
    GAUDI_KERNEL_GATHER_FWD_DIM0_I32,
    GAUDI_KERNEL_GATHER_FWD_DIM1_I32,
    GAUDI_KERNEL_KL_DIV_FWD_F32,
    GAUDI_KERNEL_KL_DIV_BWD_F32,

    GAUDI_KERNEL_MAX_EXAMPLE_KERNEL

} Gaudi_Kernel_Name_e;

typedef enum
{
    GAUDI2_KERNEL_AVG_POOL_2D_FWD_F32 = 0,
    GAUDI2_KERNEL_AVG_POOL_2D_BWD_F32,
    GAUDI2_KERNEL_CAST_F16_TO_I16,
    GAUDI2_KERNEL_KL_DIV_FWD_F32,
    GAUDI2_KERNEL_SOFTMAX_FCD_BF16,
    GAUDI2_KERNEL_SOFTMAX_NONFCD_BF16,
    GAUDI2_KERNEL_ADD_F32,
    GAUDI2_KERNEL_RELU_FWD_F32,
    GAUDI2_KERNEL_RELU_BWD_F32,
    GAUDI2_KERNEL_RELU_FWD_BF16,
    GAUDI2_KERNEL_RELU_BWD_BF16,    
    GAUDI2_KERNEL_USER_LUT,

    GAUDI2_KERNEL_MAX_EXAMPLE_KERNEL

} Gaudi2_Kernel_Name_e;

/*
 ***************************************************************************************************
 *   @brief This function returns exported kernel names
 *
 *   @param deviceId    [in] The type of device E.g. dali/gaudi etc.*
 *   @param kernelCount [in/out] The number of strings in 'names' argument.
 *                      If the list is too short, the library will return the
 *                      required list length.
 *   @param guids       [out]  List of structure to be filled with kernel guids.
 *
 *   @return                  The status of the operation.
 ***************************************************************************************************
 */
tpc_lib_api::GlueCodeReturn GetKernelGuids( _IN_    tpc_lib_api::DeviceId        deviceId,
                                            _INOUT_ uint32_t*                    kernelCount,
                                            _OUT_   tpc_lib_api::GuidInfo*       guids);

/*
 ***************************************************************************************************
 *   @brief This kernel library main entry point, it returns all necessary
 *          information about a kernel to execute on device.
 *
 *
 *   @return                  The status of the operation.
 ***************************************************************************************************
 */
tpc_lib_api::GlueCodeReturn
InstantiateTpcKernel(_IN_  tpc_lib_api::HabanaKernelParams* params,
             _OUT_ tpc_lib_api::HabanaKernelInstantiation*instance);

tpc_lib_api::GlueCodeReturn
GetShapeInference(_IN_ tpc_lib_api::DeviceId deviceId,  _IN_ tpc_lib_api::ShapeInferenceParams* inputParams,  _OUT_ tpc_lib_api::ShapeInferenceOutput* outputData);

} // extern "C"
#endif
