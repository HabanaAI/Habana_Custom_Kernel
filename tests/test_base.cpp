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

#include <algorithm>
#include <sstream>

#include "TPC.h" //TPC simulator header
#include "test_base.hpp"
#include "tpc_elf_api.hpp"
#include "tpc_test_core_api.h"

int c_divide_by_all_indices        = 5;
bool TestBase::s_printfIsUsed = false;

void TestBase::SetUp()
{
    // clear in/out structures.
    memset(&m_in_defs,0,sizeof(m_in_defs));
    memset(&m_out_defs,0,sizeof(m_out_defs));

    // allocate memory for ISA
    m_out_defs.elfSize = c_default_isa_buffer_size;
    m_out_defs.kernelElf = malloc(c_default_isa_buffer_size);
}

void TestBase::TearDown()
{
    free (m_out_defs.kernelElf);
    m_out_defs.kernelElf = NULL;
    m_out_defs.elfSize = 0;
}

TensorDescriptorGaudi TestBase::DaliTensorDescToGaudiDesc(const TensorDescriptor * desc)
{
    TensorDescriptorGaudi gaudiDesc = {};
    gaudiDesc.paddingValue = desc->paddingValue;
    gaudiDesc.configuration = desc->configuration;
    gaudiDesc.baseAddrUnion.baseAddr = desc->baseAddrUnion.baseAddr;
    for (int i =0 ; i < TestBase::num_dims_in_irf; i++)
    {
        gaudiDesc.dimDescriptors[i].size = desc->dimDescriptors[i].size;
        gaudiDesc.dimDescriptors[i].stride = desc->dimDescriptors[i].stride;
    }
    return gaudiDesc;
}


unsigned int TestBase::RunSimulation(   std::vector<TensorDescriptor>& descriptors,
                                        const gcapi::HabanaKernelParams_t& gc_input,
                                        const gcapi::HabanaKernelInstantiation_t& gc_output,
                                        IndexSpaceMappingTest_t testMode)
{
    
   return tpc_tests::RunSimulation(gc_input, gc_output, descriptors);

}

static const std::string dataType[] = {"float32", "float16", "int32", "int16", "int8", "uint8", "bfloat16"};

void TestBase::PrintKernelInputParams(const gcapi::HabanaKernelParams_t* gc_input)
{
    std::stringstream ss;
    ss << "Kernel Input Params:" << std::endl;
    ss << "\tinputTensorNr = "  << gc_input->inputTensorNr          << std::endl;
    for (unsigned i = 0; i < gc_input->inputTensorNr; i++)
    {
        ss << "\tinputTensors[" << i << "]."
           << dataType[gc_input->inputTensors[i].dataType] << "_"
           << gc_input->inputTensors[i].geometry.dims     << "DTensor[] = {"
           << gc_input->inputTensors[i].geometry.sizes[0] << ", "
           << gc_input->inputTensors[i].geometry.sizes[1] << ", "
           << gc_input->inputTensors[i].geometry.sizes[2] << ", "
           << gc_input->inputTensors[i].geometry.sizes[3] << ", "
           << gc_input->inputTensors[i].geometry.sizes[4] << "}"
           << std::endl;
        if (gc_input->inputTensors[i].dataType != gcapi::DATA_F32)
        {
            ss << "\tinputTensors[" << i << "].scale = "
               << gc_input->inputTensors[i].quantizationParam.scale          << std::endl;
            ss << "\tinputTensors[" << i << "].zeroPoint = "
               << (int)gc_input->inputTensors[i].quantizationParam.zeroPoint << std::endl;
        }
    }
    ss << "\toutputTensorNr = " << gc_input->outputTensorNr << std::endl;
    for (unsigned i = 0; i < gc_input->outputTensorNr; i++)
    {
        ss << "\toutputTensors[" << i << "]."
           << dataType[gc_input->outputTensors[i].dataType] << "_"
           << gc_input->outputTensors[i].geometry.dims     << "DTensor[] = {"
           << gc_input->outputTensors[i].geometry.sizes[0] << ", "
           << gc_input->outputTensors[i].geometry.sizes[1] << ", "
           << gc_input->outputTensors[i].geometry.sizes[2] << ", "
           << gc_input->outputTensors[i].geometry.sizes[3] << ", "
           << gc_input->outputTensors[i].geometry.sizes[4] << "}"
           << std::endl;
        if (gc_input->inputTensors[i].dataType != gcapi::DATA_F32)
        {
            ss << "\toutputTensors[" << i << "].scale = "
               << gc_input->outputTensors[i].quantizationParam.scale          << std::endl;
            ss << "\toutputTensors[" << i << "].zeroPoint = "
               << (int)gc_input->outputTensors[i].quantizationParam.zeroPoint << std::endl;
        }
    }
    ss << "\tdebugFlags = " << gc_input->debugFlags << std::endl << std::endl;


    std::cout << ss.str();
}

void TestBase::PrintKernelOutputParams(const gcapi::HabanaKernelParams_t* gc_input,
                             const gcapi::HabanaKernelInstantiation_t*gc_output)
{
    std::stringstream ss;
    ss << "Glue code outputs:"  << std::endl;
    ss << "\tindexSpaceGeometry.dims  = " << gc_output->indexSpaceGeometry.dims << std::endl;
    ss << "\tindexSpaceGeometry.sizes = "
       << gc_output->indexSpaceGeometry.sizes[0] << ", "
       << gc_output->indexSpaceGeometry.sizes[1] << ", "
       << gc_output->indexSpaceGeometry.sizes[2] << ", "
       << gc_output->indexSpaceGeometry.sizes[3] << ", "
       << gc_output->indexSpaceGeometry.sizes[4]
       << std::endl;
    for (unsigned i = 0; i < gc_input->inputTensorNr; i++)
    {
        ss << "\tinputTensorAccessPattern[" << i << "].allRequired = "
           << gc_output->inputTensorAccessPattern[i].allRequired
           << std::endl;

        for (unsigned j = 0; j < gc_output->indexSpaceGeometry.dims; j++)
        {
            ss << "\tinputTensorAccessPattern[" << i << "].tensorDim[" << j << "].indexSpaceDim = "
               << gc_output->inputTensorAccessPattern[i].dim[j].dim     << std::endl;
            ss << "\tinputTensorAccessPattern[" << i << "].tensorDim[" << j << "].start_a = "
               << gc_output->inputTensorAccessPattern[i].dim[j].start_a << std::endl;
            ss << "\tinputTensorAccessPattern[" << i << "].tensorDim[" << j << "].start_b = "
               << gc_output->inputTensorAccessPattern[i].dim[j].start_b << std::endl;
            ss << "\tinputTensorAccessPattern[" << i << "].tensorDim[" << j << "].end_a = "
               << gc_output->inputTensorAccessPattern[i].dim[j].end_a   << std::endl;
            ss << "\tinputTensorAccessPattern[" << i << "].tensorDim[" << j << "].end_b = "
               << gc_output->inputTensorAccessPattern[i].dim[j].end_b   << std::endl;
        }
        ss << "\tinputPadValues[" << i << "].i32Value = "
           << gc_output->inputPadValues[i].i32Value
           << std::endl;
    }
    for (unsigned i = 0; i < gc_input->outputTensorNr; i++)
    {
        ss << "\toutputTensorAccessPattern[" << i << "].allRequired = "
           << gc_output->outputTensorAccessPattern[i].allRequired
           << std::endl;

        for (unsigned j = 0; j < gc_output->indexSpaceGeometry.dims; j++)
        {
            ss << "\toutputTensorAccessPattern[" << i << "].tensorDim[" << j << "].indexSpaceDim = "
               << gc_output->outputTensorAccessPattern[i].dim[j].dim     << std::endl;
            ss << "\toutputTensorAccessPattern[" << i << "].tensorDim[" << j << "].start_a = "
               << gc_output->outputTensorAccessPattern[i].dim[j].start_a << std::endl;
            ss << "\toutputTensorAccessPattern[" << i << "].tensorDim[" << j << "].start_b = "
               << gc_output->outputTensorAccessPattern[i].dim[j].start_b << std::endl;
            ss << "\toutputTensorAccessPattern[" << i << "].tensorDim[" << j << "].end_a = "
               << gc_output->outputTensorAccessPattern[i].dim[j].end_a   << std::endl;
            ss << "\toutputTensorAccessPattern[" << i << "].tensorDim[" << j << "].end_b = "
               << gc_output->outputTensorAccessPattern[i].dim[j].end_b   << std::endl;
        }
    }
    ss << "\tauxiliaryTensorCount = " << gc_output->auxiliaryTensorCount << std::endl;
    ss << "\tkernel.kernelBinary = " << gc_output->kernel.kernelBinary << std::endl;
    ss << "\tkernel.binarySize = "   << gc_output->kernel.binarySize   << std::endl;
    ss << "\tkernel.paramsNr = "     << gc_output->kernel.paramsNr     << std::endl;
    for (unsigned i = 0; i < gc_output->kernel.paramsNr; i++)
    {
        ss << "\tkernel.scalarParams[" << i << "] = " << gc_output->kernel.scalarParams[i] << std::endl;
    }
    ss << "\tflags.Value = " << gc_output->flags.Value << std::endl;

     std::cout << ss.str();
}
