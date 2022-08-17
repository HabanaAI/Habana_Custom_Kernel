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

#include "sparse_lengths_sum_bf16_test.hpp"
#include "entry_points.hpp"

void SparseLengthsSumBF16Test::SparseLengthsSumRefImplementation(
        bfloat16_2DTensor &input_tensor,
        int32_1DTensor &indices_tensor,
        int32_1DTensor &lengths_tensor,
        float_2DTensor &output_tensor)
{
    const int32_t lengths_size = lengths_tensor.Size(0);
    const int32_t in_dim0_size = input_tensor.Size(0);     //The Input Tensor axis lengths

    int32_t in_coord[5]         = {0};
    int32_t idx_coord[1]        = {0};
    int32_t lengths_coord[1]    = {0};
    int32_t out_coord[5]        = {0};
    int32_t idx = 0;

    for (int32_t in_dim0 = 0; in_dim0 < in_dim0_size; in_dim0++)
    {
        idx = 0;
        in_coord[0] = out_coord[0] = in_dim0;
        for (int32_t segment_no = 0; segment_no < lengths_size; segment_no++)
        {
            out_coord[1] = lengths_coord[0] = segment_no;
            int32_t segment_length = lengths_tensor.ElementAt(lengths_coord);
            float out_value_float = 0;

            for (int32_t element_no = 0; element_no < segment_length; element_no++)
            {
                idx_coord[0] = idx;
                idx++;
                in_coord[1] = indices_tensor.ElementAt(idx_coord);

                float* scale_ptr = (float*)((bfloat16*)input_tensor.Data()
                                    + ((in_coord[1] + 1) * in_dim0_size - 8));
                float* bias_ptr  = (float*)((bfloat16*)input_tensor.Data()
                                    + ((in_coord[1] + 1) * in_dim0_size - 4));

                float scale = *scale_ptr;
                float neg_scale_x_bias = *bias_ptr;

                float input_val = (float)input_tensor.ElementAt(in_coord);
                input_val = std::fmaf(input_val, scale, neg_scale_x_bias);
                out_value_float += input_val;
            }
            output_tensor.SetElement(out_coord, out_value_float);
        }
    }
}


int SparseLengthsSumBF16Test::runTest()
{

    uint32_t input_size[2]      = { 23, 15 };
    uint32_t indices_size[1]    = {19};
    uint32_t lengths_size[1]    = { 5 };
    uint32_t output_size[4]     = { 15, 5};

    bfloat16_2DTensor input_tensor;
    int32_1DTensor indices_tensor;
    int32_1DTensor lengths_tensor;
    float_2DTensor out_tensor;
    float_2DTensor out_tensor_ref;

    // initializing tensors with the sizes
    input_tensor.Init(input_size);
    indices_tensor.Init(indices_size);
    lengths_tensor.Init(lengths_size);
    out_tensor.Init(output_size);
    out_tensor_ref.Init(output_size);

    // generate input for query call
    m_in_defs.inputTensorNr = 3;
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), input_tensor);
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), indices_tensor);
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[2]), lengths_tensor);

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), out_tensor);

    input_tensor.InitRand(
            std::numeric_limits<bfloat16 >::min(),
            std::numeric_limits<bfloat16>::max(), 0);

    indices_tensor.InitRand(0, input_size[1] - 1, 3);

    // filling scale-bias input tensor
    for(int32_t i = 0; i < (int32_t)input_size[1]; i++)
    {
        float scale = ((rand() % 100) + 1) / (float)50;
        float bias = (rand() % 100) - 50;

        float* scale_ptr = (float*)((bfloat16*)input_tensor.Data()
                                + ((i + 1) * input_size[0] - 8));
        float* bias_ptr  = (float*)((bfloat16*)input_tensor.Data()
                                + ((i + 1) * input_size[0] - 4));

        *scale_ptr = scale;
        *bias_ptr = bias;
    }

    // filling the length tensor (all segments are of equal length)
    int32_t no_of_segments = lengths_size[0];
    int32_t no_of_indices_left = indices_size[0];
    int32_t segment_length = no_of_indices_left / no_of_segments;
    int32_t lengths_coord[1];
    // initializing length tensor with all zeroes
    for(int32_t i = 0; i < no_of_segments; i++)
    {
        lengths_coord[0] = i;
        lengths_tensor.SetElement(lengths_coord, 0);
    }
    // setting the segment lengths in the length tensor
    lengths_coord[0] = 0;
    // filling till the last but one element
    for(int32_t i = 0; i < no_of_segments - 1 ; i++)
    {
        lengths_coord[0] = i;
        no_of_indices_left -= (segment_length);
        lengths_tensor.SetElement(lengths_coord, segment_length);
        if(no_of_indices_left <= 0)
        {
            break;
        }
    }
    lengths_coord[0] = no_of_segments -1;
    lengths_tensor.SetElement(lengths_coord, no_of_indices_left);

    SparseLengthsSumRefImplementation(
            input_tensor, indices_tensor, lengths_tensor, out_tensor_ref);

    char**   kernelNames = nullptr;
    unsigned kernelCount = 0;
    gcapi::GlueCodeReturn_t result = GetKernelNames(kernelNames, &kernelCount, gcapi::DEVICE_ID_GAUDI);
    kernelNames = new char*[kernelCount];
    for (unsigned i = 0; i < kernelCount; i++)
    {
        kernelNames[i] = new char[gcapi::MAX_NODE_NAME];
    }    
    result = GetKernelNames(kernelNames, &kernelCount, gcapi::DEVICE_ID_GAUDI);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Can't get kernel name!! " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI_KERNEL_SPARSE_LEN_SUM_BF16]);
    result  = HabanaKernel(&m_in_defs,&m_out_defs);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel!! " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc> vec;
    vec.push_back(input_tensor.GetTensorDescriptor());
    vec.push_back(indices_tensor.GetTensorDescriptor());
    vec.push_back(lengths_tensor.GetTensorDescriptor());
    vec.push_back(out_tensor.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount);    
    out_tensor.Print(0);
    out_tensor_ref.Print(0);
    for (int element = 0 ; element <  out_tensor_ref.ElementCount() ; element++)
    {
        if (out_tensor.Data()[element] != out_tensor_ref.Data()[element])
        {
            std::cout << "Sparse length Sum BF16 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "Sparse length Sum BF16 test pass!!" << std::endl;
    return 0;
}
