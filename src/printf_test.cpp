/*****************************************++*****************************
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

#include <cstring>
#include "printf_test.hpp"

extern unsigned char _binary___printf_test_o_start;
extern unsigned char _binary___printf_test_o_end;


gcapi::GlueCodeReturn_t PrintfTestKernel::GetGcDefinitions(
                        gcapi::HabanaKernelParams_t* in_defs,
                        gcapi::HabanaKernelInstantiation_t* out_defs)
{
    gcapi::GlueCodeReturn_t retVal = gcapi::GLUE_SUCCESS;

    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (in_defs->inputTensorNr != 1)
    {
        in_defs->inputTensorNr  = 1;
        return gcapi::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }

    //validate correct amount of output tensors
    if (in_defs->outputTensorNr != 0)
    {
        in_defs->outputTensorNr = 0;
        return gcapi::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }


    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example only one index space is
    *    used for printing
    **************************************************************************************/
    out_defs->indexSpaceGeometry.dims = 1;
    out_defs->indexSpaceGeometry.sizes[0] = 1;


    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    // To avoid partition by depth, set size of depth equal 1.
    // TPC can determine actual depth dimension by call get_dim_size(ifm, 0).
    // ElementsInVector for float datatype is 64
    int min_used_index = 0;
    int max_used_index = 64;

    out_defs->inputTensorAccessPattern[0].dim[0].dim        = 0;
    out_defs->inputTensorAccessPattern[0].dim[0].start_a    = 0;
    out_defs->inputTensorAccessPattern[0].dim[0].end_a      = 0;
    out_defs->inputTensorAccessPattern[0].dim[0].start_b    = min_used_index;
    out_defs->inputTensorAccessPattern[0].dim[0].end_b      = max_used_index - 1;


    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
    // special scalar structure for loadSrf function
    PrintfTestParams* params = static_cast<PrintfTestParams*>(in_defs->NodeParams);
    // copy out SRF content
    out_defs->kernel.paramsNr = sizeof(*params)/ sizeof(int);
    memcpy(&( out_defs->kernel.scalarParams[0]),params, sizeof(*params));

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    // Load ISA into the descriptor.
    unsigned IsaSize = (&_binary___printf_test_o_end - &_binary___printf_test_o_start);
    unsigned givenBinarySize = out_defs->elfSize;
    out_defs->elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (out_defs->kernelElf ,
               &_binary___printf_test_o_start,
               IsaSize);
    }
    else
    {
        retVal = gcapi::GLUE_INSUFICIENT_ELF_BUFFER;
        return retVal;
    }
    return retVal;
}

gcapi::GlueCodeReturn_t PrintfTestKernel::GetKernelName(char kernelName [gcapi::MAX_NODE_NAME])
{
    strcpy(kernelName,"printf_test");
    return gcapi::GLUE_SUCCESS;
}


