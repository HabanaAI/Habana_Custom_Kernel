/**********************************************************************
Copyright (c) 2018 Habana Labs.

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
#include "test_base_tpc_callback.hpp"
#include "tpc_elf_api.hpp"

int c_divide_by_all_indices        = 5;
bool TestBase::s_printfIsUsed = false;
std::shared_ptr<test::PrintfTensor> TestBase::s_ptr_printfTensor;

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


#define PRINTF_SIZE_BYTES (32*1024) /* 32kB */

void TestBase::AllocatePrintfTensor( std::vector<TensorDescriptor>& descriptors,
                                     const gcapi::HabanaKernelParams_t& in_defs,
                                     const gcapi::HabanaKernelInstantiation_t& out_defs,
                                     const TpcElfTools::TPCProgramHeader & programHeader)
{
    /* first, turn off s_printfIsUsed flag for cases of consecutive tests execution. */
    TestBase::s_printfIsUsed = false;

    if (programHeader.printfUsed)
    {
        /* in case of repeated call of AllocatePrintfTensor function and reinitialization of
           s_ptr_printfTensor, the previous PrintfTensor will be automatically correctly destroyed
           due to functionality of smart pointers.*/
        TestBase::s_ptr_printfTensor = std::make_shared<test::PrintfTensor>(PRINTF_SIZE_BYTES);

        descriptors.push_back(TestBase::s_ptr_printfTensor->GetTensorDescriptor());
        TestBase::s_printfIsUsed = true;
    }
}

void TestBase::ShowMessageFromPrintfTensor(std::string title)
{
    if (TestBase::s_printfIsUsed)
    {
        if(title != "")
        {
            std::cout << "\n" << title << ":\n";
        }

        std::cout << TestBase::s_ptr_printfTensor->GetAllMessages();
    }
}

unsigned int TestBase::RunSimulation(   std::vector<TensorDescriptor>& descriptors,
                                        const gcapi::HabanaKernelParams_t& gc_input,
                                        const gcapi::HabanaKernelInstantiation_t& gc_output,
                                        IndexSpaceMappingTest_t testMode)
{

    TpcElfTools::TPCProgramHeader programHeader = {};

    TpcElfTools::ExtractTpcProgramHeaderFromElf(gc_output.kernelElf,
                                                    gc_output.elfSize,
                                                    programHeader);
    TestBase::AllocatePrintfTensor(descriptors, gc_input, gc_output, programHeader);

    //debug prints of glue code input and output.
    PrintKernelInputParams(&gc_input);
    PrintKernelOutputParams(&gc_input,&gc_output);

    unsigned retVal = 0;
    gcapi::HabanaKernelInstantiation_t gcOutputInternal;
    memcpy(&gcOutputInternal, &gc_output, sizeof(gc_output));
    memcpy(&gcOutputInternal, &gc_output, sizeof(gc_output));

    const int c_maxPartition = 32;
    const int c_partitionFactor = 2;

    IndexSpace indexSpace = {{0}};
    memcpy(&(indexSpace.size[0]),
            &(gcOutputInternal.indexSpaceGeometry.sizes[0]),
            gcOutputInternal.indexSpaceGeometry.dims * sizeof(uint32_t));

    // here we partition the index space to simulate un-predictable execution
    // order of the index space members.
    std::vector<IndexSpace> partition;
    DivideIndexSpace(c_maxPartition,
                     5,
                     c_partitionFactor,
                     gcOutputInternal.indexSpaceGeometry.dims,
                     indexSpace,
                     partition);

    printf("Program will be executed in %lu chunks\n",partition.size());
    for (unsigned i = 0; i < partition.size(); i++)
    {
        memcpy(&(gcOutputInternal.indexSpaceGeometry.sizes[0]),
                &(partition[i].size[0]),
                gcOutputInternal.indexSpaceGeometry.dims * sizeof(uint32_t));

        retVal += RunSimulationInternal(descriptors, gc_input, gcOutputInternal, partition[i].offset);

        ShowMessageFromPrintfTensor("instance "+ std::to_string(i));
    }

    printf("Program executed in %u cycles\n", retVal);

    return retVal;
}

unsigned int TestBase::RunSimulationInternal(const std::vector<TensorDescriptor>& descriptors,
                                    const gcapi::HabanaKernelParams_t& gc_input,
                                    const gcapi::HabanaKernelInstantiation_t& gc_output,
                                    int offsets [5],
                                    IndexSpaceMappingTest_t testMode)
{
    std::shared_ptr<TPCCallbackInterface> pCallbackInterface
                    = std::make_shared<TestBaseTPCCallback>(&gc_input, &gc_output, offsets);

    TPCGenerations generation = TPCGenerations::DALI;
    if (gc_input.deviceId == gcapi::DEVICE_ID_GAUDI)
    {
        generation = TPCGenerations::GAUDI;
    }

    TPC tpcSim(0,0,
               pCallbackInterface,
               gc_output.flags.specialFunctionsUsed, //false == large VLM
               generation);

    // generate and load tensor descriptors
    if (gc_input.deviceId == gcapi::DEVICE_ID_GOYA)
    {
        for (unsigned i =0  ; i < descriptors.size(); i++)
        {
            tpcSim.loadTensorDescriptor(i,(TensorDescriptor*) &(descriptors[i]));
        }
    }
    else if (gc_input.deviceId == gcapi::DEVICE_ID_GAUDI)
    {
        // generate and load tensor descriptors
        for (unsigned i = 0 ; i < descriptors.size(); i++)
        {
            TensorDescriptorGaudi gaudi = DaliTensorDescToGaudiDesc(&(descriptors[i]));
            tpcSim.loadTensorDescriptor(i, &gaudi);
        }
    }


    memcpy (tpcSim.getSRF().data(),
            gc_output.kernel.scalarParams,
            sizeof(uint32_t)* gc_output.kernel.paramsNr);
    // load IRF offsets
    memcpy (&(tpcSim.getIRF()[0][0]),
            offsets,
            gc_output.indexSpaceGeometry.dims * sizeof(uint32_t));

   // load IRF sizes
    memcpy (&(tpcSim.getIRF()[1][0]),
            gc_output.indexSpaceGeometry.sizes,
            gc_output.indexSpaceGeometry.dims * sizeof(uint32_t));
    // set program binary.
    tpcSim.loadVpeProgramElf( gc_output.kernelElf, gc_output.elfSize);

    // execute the kernel
    tpcSim.start();

    // collect statistics.
    VPEStats stat = tpcSim.getVpeStatistics();

    std::shared_ptr<TestBaseTPCCallback> pTestBaseCallback =
                std::dynamic_pointer_cast<TestBaseTPCCallback>(pCallbackInterface);
    // validate the index space mapping is correct and optimized.
    pTestBaseCallback->ValidateAccessPattern(testMode);

    return stat.instructionsExecuted;
}

// Divide all dimensions of the index-space by 2.
void TestBase::DivideIndexSpaceRecursive(const int maxPartition,
                                         const int currentDim,
                                         int& partitionCount,
                                         IndexSpace input,
                                         std::vector<IndexSpace>& output)
{
    if ((partitionCount == maxPartition) || (currentDim == -1))
    {
        output.push_back(input);
        return;
    }
    // If size == 1, can't split on this dimension.
    if (input.size[currentDim] <= 1)
    {
        DivideIndexSpaceRecursive(maxPartition, currentDim - 1, partitionCount, input, output);
        return;
    }

    IndexSpace item = input;
    item.size[currentDim] = input.size[currentDim] / 2;
    input.size[currentDim] -=  item.size[currentDim];
    input.offset[currentDim] = item.size[currentDim];
    partitionCount++;
    DivideIndexSpaceRecursive(maxPartition, currentDim - 1, partitionCount, item, output);
    DivideIndexSpaceRecursive(maxPartition, currentDim - 1, partitionCount, input, output);

}

// Divide single dimension of the index-space by a given factor.
void TestBase::DivideIndexSpaceByDim(const int maxPartition,
                                     const int partitionDim,
                                     const int partitionFactor,
                                     IndexSpace input,
                                     std::vector<IndexSpace>& output)
{
    int partitionCount = std::min(std::min(partitionFactor, maxPartition),
                                    input.size[partitionDim]);

    int size = input.size[partitionDim] / partitionCount;
    int rem = input.size[partitionDim] % partitionCount;

    for (int i = 0; i < partitionCount; i++)
    {
        IndexSpace item = input;
        item.size[partitionDim] = size;
        if (i == partitionCount - 1)
        {
            item.size[partitionDim] += rem;
        }
        item.offset[partitionDim] = i * size;
        output.push_back(item);
    }
}

void TestBase::DivideIndexSpace(const int maxPartition,
                                const int partitionDim,
                                const int partitionFactor,
                                const int indexSpaceDim,
                                IndexSpace input,
                                std::vector<IndexSpace>& output)
{
    output.clear();
    int partitionCount = 1;
    if (partitionDim == c_divide_by_all_indices)
    {
        DivideIndexSpaceRecursive(maxPartition, indexSpaceDim - 1, partitionCount, input, output);
    }
    else if (partitionDim >= 0 && partitionDim < indexSpaceDim)
    {
        DivideIndexSpaceByDim(maxPartition, partitionDim, partitionFactor, input, output);
    }
    // For -1 or other invalid parameters, no partition
    else
    {
        output.push_back(input);
    }

    std::random_shuffle(std::begin(output), std::end(output));
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
