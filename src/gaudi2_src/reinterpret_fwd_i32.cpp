#include <iostream>
#include <cstring>
#include "reinterpret_fwd_i32.hpp" // Include the header file for your kernel

extern unsigned char _binary___reinterpret_fwd_i32_o_start;
extern unsigned char _binary___reinterpret_fwd_i32_o_end;

tpc_lib_api::GlueCodeReturn ReinterpretFwdI32::GetKernelName(
             char kernelName [tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName, "reinterpret_fwd_i32");
    return tpc_lib_api::GLUE_SUCCESS;
}


tpc_lib_api::GlueCodeReturn ReinterpretFwdI32::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* in_defs,
            tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    // Validate correct amount of input tensors
    if (in_defs->inputTensorNr != 1)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    // Validate correct amount of output tensors
    if (in_defs->outputTensorNr != 1)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    // Validate input data type is float and output data type is int
    if (in_defs->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32 ||
        in_defs->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_I32)
    {
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    // Define index space geometry based on the output tensor dimensions
    // Assuming the kernel processes the tensor in 64-element chunks
    int elementsInVec = 64;
    uint64_t outputSizes[gcapi::MAX_TENSOR_DIM] = {0};
    memcpy(outputSizes, in_defs->inputTensors[0].geometry.maxSizes, sizeof(outputSizes));

    // Round up to elementsInVec and divide by elementsInVec
    unsigned depthIndex = (outputSizes[0]) / elementsInVec;
    out_defs->indexSpaceRank = 5;
    out_defs->indexSpaceGeometry[0] = depthIndex;
    out_defs->indexSpaceGeometry[1] = outputSizes[1];
    out_defs->indexSpaceGeometry[2] = outputSizes[2];
    out_defs->indexSpaceGeometry[3] = outputSizes[3];
    out_defs->indexSpaceGeometry[4] = outputSizes[4];

    // Define index space mapping for input and output tensors
    // The mapping is direct since this kernel does not change the data layout
    for (uint32_t i = 0; i < out_defs->indexSpaceRank; ++i)
    {
        out_defs->inputTensorAccessPattern[0].mapping[i].indexSpaceDim = i;
        out_defs->inputTensorAccessPattern[0].mapping[i].a = 1;
        out_defs->inputTensorAccessPattern[0].mapping[i].start_b = 0;
        out_defs->inputTensorAccessPattern[0].mapping[i].end_b = 0;

        out_defs->outputTensorAccessPattern[0].mapping[i].indexSpaceDim = i;
        out_defs->outputTensorAccessPattern[0].mapping[i].a = 1;
        out_defs->outputTensorAccessPattern[0].mapping[i].start_b = 0;
        out_defs->outputTensorAccessPattern[0].mapping[i].end_b = 0;
    }
    // Load the ISA binary into the descriptor
    unsigned IsaSize = (&_binary___reinterpret_fwd_i32_o_end - &_binary___reinterpret_fwd_i32_o_start);
    unsigned givenBinarySize = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        memcpy(out_defs->kernel.kernelElf,
               &_binary___reinterpret_fwd_i32_o_start,
               IsaSize);
    }
    else
    {
        return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}
