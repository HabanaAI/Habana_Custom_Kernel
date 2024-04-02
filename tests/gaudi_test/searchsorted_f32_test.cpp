/**********************************************************************
Copyright (c) 2023 Habana Labs.

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

#include "searchsorted_f32_test.hpp"
#include "entry_points.hpp"

void SearchSortedF32Test::searchsorted_fwd_f32_reference_implementation(
        const float_5DTensor& input0,
        const float_5DTensor& input1,
        int32_5DTensor& output,
        const SearchSortedF32::SearchSortedParam& def)
{
    int coords[5] = {0};
    int coords_out[5] = {0};
    for (unsigned r4 = 0; r4 < input0.Size(4); r4 += 1)
    {
        coords[4] = r4;
        coords_out[4] = r4;
        for (unsigned b = 0; b < input0.Size(3); b += 1)
        {
            coords[3] = b;
            coords_out[3] = b;
            for (unsigned h = 0; h < input0.Size(2); h += 1)
            {
                coords[2] = h;
                coords_out[2] = h;
                for (unsigned d = 0; d < input0.Size(0); d += 1)
                {
                    coords[0] = d;
                    coords_out[0] = d;
                    int32_t index = 0;
                    for (unsigned vw = 0; vw < input1.Size(1); vw += 1)
                    {
                        coords_out[1] = vw;
                        for (unsigned w = 0; w < input0.Size(1); w += 1)
                        {
                            coords[1] = w;
                            float sequence = input0.ElementAt(coords);
                            float value = input1.ElementAt(coords_out);
                            if(def.side)
                            {
                                if(sequence <= value)
                                    index = w+1;
                            }
                            else{
                                if(sequence < value)
                                    index = w+1;                                
                            }
                            
                        }
                        output.SetElement(coords_out, index);
                    }
                }
            }
        }
    }
}

int SearchSortedF32Test::runTest()
{
    const int height = 1;
    const int width  = 5;
    const int width_val  = 3;
    const int depth  = 3;
    const int batch  = 1;
    const int rank4  = 1;

    uint64_t ifmInitializer[] = {depth, width, height, batch, rank4};
    uint64_t ofmInitializer[] = {depth, width_val, height, batch, rank4};

    float_5DTensor input0(ifmInitializer);
    input0.FillWithSortedValue(1);

    float_5DTensor input1(ofmInitializer);
    input1.FillWithSortedValue(0);

    int32_5DTensor output(ofmInitializer);
    int32_5DTensor output_ref(ofmInitializer);
    SearchSortedF32::SearchSortedParam def;
    def.side = 1; //1:right, 0:left

    // execute reference implementation of the kernel.
    searchsorted_fwd_f32_reference_implementation(input0, input1, output_ref, def);

    // generate input for query call
    m_in_defs.deviceId = tpc_lib_api::DEVICE_ID_GAUDI;
    m_in_defs.outputTensorNr = 1;
    m_in_defs.nodeParams.nodeParams = &def;
    m_in_defs.inputTensorNr = 2;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), input0);
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), input1);
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), output);

    tpc_lib_api::GuidInfo *guids = nullptr;
    unsigned kernelCount = 0;
    tpc_lib_api::GlueCodeReturn result = GetKernelGuids(tpc_lib_api::DEVICE_ID_GAUDI, &kernelCount, guids);
    guids = new tpc_lib_api::GuidInfo[kernelCount];
    result = GetKernelGuids(tpc_lib_api::DEVICE_ID_GAUDI, &kernelCount, guids);
    if (result != tpc_lib_api::GLUE_SUCCESS)
    {
        std::cout << "Can't get kernel name!! " << result << std::endl;
        ReleaseKernelNames(guids, kernelCount);
        return -1;
    }

    strcpy(m_in_defs.guid.name, guids[GAUDI_KERNEL_SEARCH_SORTED_FWD_F32].name);
    result  = InstantiateTpcKernel(&m_in_defs,&m_out_defs);
    if (result != tpc_lib_api::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(guids, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc2> vec;
    vec.push_back(input0.GetTensorDescriptor());
    vec.push_back(input1.GetTensorDescriptor());
    vec.push_back(output.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(guids, kernelCount);
    output.Print(0);
    output_ref.Print(0);
    for (int element = 0 ; element <  output_ref.ElementCount() ; element++)
    {
        if (abs(output.Data()[element] - output_ref.Data()[element]) > 1e-6)
        {
            std::cout << "Search Sorted F32 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "Search Sorted F32 test pass!!" << std::endl;
    return 0;
}

