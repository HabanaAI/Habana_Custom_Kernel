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

#include <algorithm>
#include <sstream>


#include "test_base.hpp"
#include "tpc_test_core_api.h"

#define MAX_ALLOCATED_TENSOR 16

void TestBase::SetUp()
{

    // clear in/out structures.
   memset(&m_in_defs,0,sizeof(m_in_defs));
   memset(&m_out_defs,0,sizeof(m_out_defs));

   // allocate the tensor space
   m_in_defs.inputTensors = (tpc_lib_api::Tensor*)calloc(MAX_ALLOCATED_TENSOR,  sizeof(tpc_lib_api::Tensor));
   m_in_defs.outputTensors = (tpc_lib_api::Tensor*)calloc(MAX_ALLOCATED_TENSOR,  sizeof(tpc_lib_api::Tensor));

   // allocate the accesspatten
   m_out_defs.inputTensorAccessPattern = (tpc_lib_api::TensorAccessPattern*)calloc(MAX_ALLOCATED_TENSOR, sizeof(tpc_lib_api::TensorAccessPattern));
   m_out_defs.outputTensorAccessPattern = (tpc_lib_api::TensorAccessPattern*)calloc(MAX_ALLOCATED_TENSOR, sizeof(tpc_lib_api::TensorAccessPattern));

   m_out_defs.auxiliaryTensors = (tpc_lib_api::AuxTensor*)calloc(MAX_ALLOCATED_TENSOR, sizeof(tpc_lib_api::AuxTensor));
   for(int i=0;i< MAX_ALLOCATED_TENSOR;i++)
      m_out_defs.auxiliaryTensors[i].pData =NULL;
   // allocate memory for ISA
   m_out_defs.kernel.elfSize = c_default_isa_buffer_size;
   m_out_defs.kernel.kernelElf = malloc(c_default_isa_buffer_size);
}

void TestBase::TearDown()
{
   for(int i=0;i< MAX_ALLOCATED_TENSOR;i++)
   {
      if(m_out_defs.auxiliaryTensors[i].pData)
         free(m_out_defs.auxiliaryTensors[i].pData);
   }
   free(m_in_defs.inputTensors);
   free(m_in_defs.outputTensors);

   free(m_out_defs.inputTensorAccessPattern);
   free(m_out_defs.outputTensorAccessPattern);

   free(m_out_defs.auxiliaryTensors);

   free (m_out_defs.kernel.kernelElf);
   m_out_defs.kernel.kernelElf = NULL;
   m_out_defs.kernel.elfSize = 0;
}

unsigned int TestBase::RunSimulation(   std::vector<TensorDesc2>& descriptors,
                                        const tpc_lib_api::HabanaKernelParams& gc_input,
                                        const tpc_lib_api::HabanaKernelInstantiation& gc_output,
                                        IndexSpaceMappingTest_t testMode)
{
   unsigned int  retVal= 0;
   //debug prints of glue code input and output.
 
   PrintKernelInputParams(&gc_input);
   PrintKernelOutputParams(&gc_input,&gc_output);    

   VPEStats stat;

   retVal = tpc_tests::RunSimulation(gc_input, gc_output, descriptors, stat);
   const char* env = getenv("TPC_RUNNER");
   if(env != nullptr && strcmp(env, "1") == 0)
      printf("Program executed using Habana device\n");
   else
      printf("Program executed in %u cycles using simulation\n", retVal);
   return retVal;

}

static const std::string dataType[] = {"float32", "float16", "int32", "int16", "int8", "uint8", "bfloat16"};


void TestBase::PrintKernelInputParams(const tpc_lib_api::HabanaKernelParams* gc_input)
{
    std::stringstream ss;
    ss << "Kernel Input Params:" << std::endl;
    ss << "\tinputTensorNr = "  << gc_input->inputTensorNr          << std::endl;
    for (unsigned i = 0; i < gc_input->inputTensorNr; i++)
    {
        ss << "\tinputTensors[" << i << "]."
           //<< dataType[gc_input->inputTensors[i].geometry.dataType] << "_"
           << gc_input->inputTensors[i].geometry.dims     << "DTensor[] = {"
           << gc_input->inputTensors[i].geometry.maxSizes[0] << ", "
           << gc_input->inputTensors[i].geometry.maxSizes[1] << ", "
           << gc_input->inputTensors[i].geometry.maxSizes[2] << ", "
           << gc_input->inputTensors[i].geometry.maxSizes[3] << ", "
           << gc_input->inputTensors[i].geometry.maxSizes[4] << "}"
           << std::endl;
        if (gc_input->inputTensors[i].geometry.dataType != tpc_lib_api::DATA_F32)
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
           //<< dataType[gc_input->outputTensors[i].geometry.dataType] << "_"
           << gc_input->outputTensors[i].geometry.dims     << "DTensor[] = {"
           << gc_input->outputTensors[i].geometry.maxSizes[0] << ", "
           << gc_input->outputTensors[i].geometry.maxSizes[1] << ", "
           << gc_input->outputTensors[i].geometry.maxSizes[2] << ", "
           << gc_input->outputTensors[i].geometry.maxSizes[3] << ", "
           << gc_input->outputTensors[i].geometry.maxSizes[4] << "}"
           << std::endl;
        if (gc_input->inputTensors[i].geometry.dataType != tpc_lib_api::DATA_F32)
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

void TestBase::PrintKernelOutputParams(const tpc_lib_api::HabanaKernelParams* gc_input,
                             const tpc_lib_api::HabanaKernelInstantiation*gc_output)
{
    std::stringstream ss;
    ss << "Glue code outputs:"  << std::endl;
    ss << "\tindexSpaceGeometry.dims  = " << gc_output->indexSpaceRank << std::endl;
    ss << "\tindexSpaceGeometry.maxSizes = "
       << gc_output->indexSpaceGeometry[0] << ", "
       << gc_output->indexSpaceGeometry[1] << ", "
       << gc_output->indexSpaceGeometry[2] << ", "
       << gc_output->indexSpaceGeometry[3] << ", "
       << gc_output->indexSpaceGeometry[4]
       << std::endl;
    for (unsigned i = 0; i < gc_input->inputTensorNr; i++)
    {
        ss << "\tinputTensorAccessPattern[" << i << "].allRequired = "
           << gc_output->inputTensorAccessPattern[i].allRequired
           << std::endl;

        for (unsigned j = 0; j < gc_output->indexSpaceRank; j++)
        {
            ss << "\tinputTensorAccessPattern[" << i << "].tensorDim[" << j << "].indexSpaceDim = "
               << gc_output->inputTensorAccessPattern[i].mapping[j].indexSpaceDim << std::endl;
            ss << "\tinputTensorAccessPattern[" << i << "].tensorDim[" << j << "].a = "
               << gc_output->inputTensorAccessPattern[i].mapping[j].a << std::endl;
            ss << "\tinputTensorAccessPattern[" << i << "].tensorDim[" << j << "].start_b = "
               << gc_output->inputTensorAccessPattern[i].mapping[j].start_b << std::endl;
            ss << "\tinputTensorAccessPattern[" << i << "].tensorDim[" << j << "].end_b = "
               << gc_output->inputTensorAccessPattern[i].mapping[j].end_b   << std::endl;
        }
    }
    for (unsigned i = 0; i < gc_input->outputTensorNr; i++)
    {
        ss << "\toutputTensorAccessPattern[" << i << "].allRequired = "
           << gc_output->outputTensorAccessPattern[i].allRequired
           << std::endl;

        for (unsigned j = 0; j < gc_output->indexSpaceRank; j++)
        {
            ss << "\toutputTensorAccessPattern[" << i << "].tensorDim[" << j << "].indexSpaceDim = "
               << gc_output->outputTensorAccessPattern[i].mapping[j].indexSpaceDim << std::endl;
            ss << "\toutputTensorAccessPattern[" << i << "].tensorDim[" << j << "].a = "
               << gc_output->outputTensorAccessPattern[i].mapping[j].a << std::endl;
            ss << "\toutputTensorAccessPattern[" << i << "].tensorDim[" << j << "].start_b = "
               << gc_output->outputTensorAccessPattern[i].mapping[j].start_b << std::endl;
            ss << "\toutputTensorAccessPattern[" << i << "].tensorDim[" << j << "].end_b = "
               << gc_output->outputTensorAccessPattern[i].mapping[j].end_b   << std::endl;
        }
    }
    ss << "\tauxiliaryTensorNr = " << gc_output->auxiliaryTensorNr << std::endl;
    ss << "\tkernel.kernelElf = " << gc_output->kernel.kernelElf << std::endl;
    ss << "\tkernel.elfSize = "   << gc_output->kernel.elfSize   << std::endl;
    ss << "\tkernel.paramsNr = "     << gc_output->kernel.paramsNr     << std::endl;
    for (unsigned i = 0; i < gc_output->kernel.paramsNr; i++)
    {
        ss << "\tkernel.scalarParams[" << i << "] = " << gc_output->kernel.scalarParams[i] << std::endl;
    }
    
     std::cout << ss.str();
}
