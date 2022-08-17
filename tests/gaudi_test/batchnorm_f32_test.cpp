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

#include "batchnorm_f32_test.hpp"
#include "entry_points.hpp"

template<typename T, typename T_res>
void Mac(T* op1, T* op2, T_res* op3, T_res* res1, T_res* res2)
{
    *res1 = (T_res)(*op3) + (T_res)(*op1) * (T_res)(*op2);
}

template<typename T, typename T_res>
void MacNeg(T* op1, T* op2, T_res* op3, T_res* res1, T_res* res2)
{
    *res1 = (T_res)(*op3) - (T_res)(*op1) * (T_res)(*op2);
}

void BatchNormF32Test::batchnorm_fwd_reference_implementation(
                test::Tensor<float, 4> &ofm,
                test::Tensor<float, 1> &mean,
                test::Tensor<float, 1> &istd,
                const test::Tensor<float, 4> &ifm,
                const test::Tensor<float, 1> &beta,
                const test::Tensor<float, 1> &gamma,
                const float momentum)
{

        // DEPTH
        const int depthStart = 0;
        const int depthEnd = ifm.Size(0);

        // WIDTH
        const int widthStart = 0;
        const int widthEnd = ifm.Size(1);

        // HEIGHT
        const int heightStart = 0;
        const int heightEnd = ifm.Size(2);

        // BATCH
        const int batchStart = 0;
        const int batchEnd = ifm.Size(3);

        int N = (batchEnd - batchStart) *
                (widthEnd - widthStart) *
                (heightEnd - heightStart);

        int coords[4] = {0};
        for (int d = depthStart; d < depthEnd; d += 1)
        {
            coords[0] = d;

            // calculate mean and variance
            float mean_v = 0, var_v = 0;
            for (int b = batchStart; b < batchEnd; b += 1)
            {
                coords[3] = b;
                for (int h = heightStart; h < heightEnd; h += 1)
                {
                    coords[2] = h;
                    for (int w = widthStart; w < widthEnd; w += 1)
                    {
                        coords[1] = w;

                        float ifmVal = ifm.ElementAt(coords);
                        mean_v += ifmVal;
                        //var_v += ifmVal * ifmVal;
                        Mac<float, float>(&ifmVal, &ifmVal, &var_v, &var_v, NULL);

                    }
                }
            }

            int depth_coord[1] = {d};

            // Calculate mean and variance
            mean_v /= (float)N;
            float tmp = var_v / (float)N;
            MacNeg<float, float>(&mean_v, &mean_v, &tmp, &var_v, NULL);

            // Norm Val = ((x-mean) * gamma / sqrt(variance + epsilon)) + beta
            // Norm Val = x * (gamma / sqrt(variance + eps))
            //                    - mean * (gamma / sqrt(variance + eps)) + beta
            // Norm Val = x * scale  + bias
            // where scale = (gamma / sqrt(variance + eps)) and bias = (-mean * scale + bias)

            float beta_v = beta.ElementAt(depth_coord);
            float inv_std = 1.0 / sqrt(var_v + 1e-5);
            float scale = gamma.ElementAt(depth_coord) * inv_std;
            float bias;

            MacNeg<float, float>(&scale, &mean_v, &beta_v, &bias, NULL);

            istd.SetElement(depth_coord, inv_std);
            mean.SetElement(depth_coord, mean_v);

            for (int b = batchStart; b < batchEnd; b += 1)
            {
                coords[3] = b;
                for (int h = heightStart; h < heightEnd; h += 1)
                {
                    coords[2] = h;
                    for (int w = widthStart; w < widthEnd; w += 1)
                    {
                        coords[1] = w;

                        float ifmVal = ifm.ElementAt(coords);
                        float outputValue;

                        // Norm Value = x *scale + bias
                        Mac<float, float>(&ifmVal, &scale, &bias, &outputValue, NULL);

                        ofm.SetElement(coords, outputValue);
                    }
                }
            }
        }
}

int BatchNormF32Test::runTest()
{
    // Initialize input data
    const int fm_height = 8;
    const int fm_width  = 3;
    const int fm_depth = 100;
    const int fm_batch = 1;

    unsigned int fmInitializer[] = {fm_depth, fm_width, fm_height, fm_batch};
    unsigned int fmInitializer_dep[] = {fm_depth};
    float_4DTensor input(fmInitializer);
    float_1DTensor beta(fmInitializer_dep);
    float_1DTensor gamma(fmInitializer_dep);
    input.FillWithData(0, 10);
    gamma.FillWithData(-5.0, 5.0);
    beta.FillWithData(50.0, 100.0);

    float_4DTensor output(fmInitializer);
    float_1DTensor mean(fmInitializer_dep);
    float_1DTensor istd(fmInitializer_dep);
    float_4DTensor output_ref(fmInitializer);
    float_1DTensor mean_ref(fmInitializer_dep);
    float_1DTensor istd_ref(fmInitializer_dep);

    BatchNormF32::BatchNormParams def;

    // Set momentum = 0.9 to calculate running mean and variance
    def.momentum = 0.9;

    // execute reference implementation of the kernel.
    batchnorm_fwd_reference_implementation(output_ref,
                                     mean_ref,
                                     istd_ref,
                                     input,
                                     beta,
                                     gamma,
                                     def.momentum);
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI;

    // generate input for query call
    m_in_defs.inputTensorNr = 3;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]),input );
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]),beta );
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[2]),gamma );

    m_in_defs.outputTensorNr = 3;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]),output );
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[1]),mean );
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[2]),istd );

    m_in_defs.NodeParams = &def;

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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI_KERNEL_BATCH_NORM_F32]);
    result  = HabanaKernel(&m_in_defs,&m_out_defs);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }


    // generate and load tensor descriptors
    std::vector<TensorDesc> vec;
    vec.push_back(input.GetTensorDescriptor());
    vec.push_back(beta.GetTensorDescriptor());
    vec.push_back(gamma.GetTensorDescriptor());;
    vec.push_back(output.GetTensorDescriptor());
    vec.push_back(mean.GetTensorDescriptor());
    vec.push_back(istd.GetTensorDescriptor());

    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount);
    output.Print(0);
    output_ref.Print(0);
    for (int element = 0 ; element <  output_ref.ElementCount() ; element++)
    {
        if (std::abs(output.Data()[element] - output_ref.Data()[element])  > 1e-5)
        {
            std::cout << "test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "test pass!!" << std::endl;
    return 0;
}


