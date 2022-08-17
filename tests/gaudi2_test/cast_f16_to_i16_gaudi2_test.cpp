/**********************************************************************
Copyright (c) 2022 Habana Labs.

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

#include "cast_f16_to_i16_gaudi2_test.hpp"
#include "entry_points.hpp"

void CastF16toI16Gaudi2Test::cast_f16_to_i16_ref(
         const test::Tensor<float16,5>& input,
         test::Tensor<int16_t,5>& output,
         const IndexSpace& indexSpace, unsigned int rounding)
{
    int eig =128;

    int coords[5] = { 0 };
    for (int d = indexSpace.offset[0]; d < (indexSpace.offset[0] + indexSpace.size[0]) * eig; d += 1)
    {
        coords[0] = d;
        for (int fif = indexSpace.offset[4]; fif < (indexSpace.offset[4] + indexSpace.size[4]); fif += 1)        
        {
            coords[4] = fif;
            for (int b = indexSpace.offset[3]; b < indexSpace.offset[3] + indexSpace.size[3]; b += 1)
            {
                coords[3] = b;
                for (int h = indexSpace.offset[2]; h < indexSpace.offset[2] + indexSpace.size[2]; h += 1)
                {
                    coords[2] = h;
                    for (int w = indexSpace.offset[1]; w < indexSpace.offset[1] + indexSpace.size[1]; w += 1)
                    {
                        coords[1] = w;
                        float ofmVal_f;
                        int16_t ofmVal;
                        float16 ifmVal = input.ElementAt(coords);
                        fp16_to_fp32(ifmVal.get_val(), &ofmVal_f);
                        if(rounding == RND_TO_NE)
                            ofmVal = (int16_t)(ofmVal_f+0.5);
                        else if(rounding == RND_TO_0)
                            ofmVal = (int16_t)(ofmVal_f);
                        else if(rounding == RND_TO_PINF)
                            ofmVal = (int16_t)ceil(ofmVal_f);
                        else if(rounding == RND_TO_NINF)
                            ofmVal = (int16_t)floor(ofmVal_f);
                        else
                            ofmVal = (int16_t)(ofmVal_f+0.5);
                        output.SetElement(coords, ofmVal);
                    }
                }
            }
        }
    }
}

int CastF16toI16Gaudi2Test::runTest()
 {

    /**********************Test for cast bf16 to f32************************/
    // Initalize input size
    const int ifm_height = 5;
    const int ifm_width  = 8;
    const int ofmifm_depth = 128;
    const int batch = 2;
    const int fifdim = 1;


    // Initalize inputs
    unsigned int ifmofmInitializer[] = {ofmifm_depth,ifm_width,ifm_height,batch,fifdim};
    float16_5DTensor ifm(ifmofmInitializer);
    ifm.FillWithData_f16();
    int16_5DTensor ofm(ifmofmInitializer);
    int16_5DTensor ofm_ref(ifmofmInitializer);

    IndexSpace indexSpace = {{0}};
    int depthIS = (ofmifm_depth + 127) / 128 ;
    indexSpace.size[0] = depthIS;
    indexSpace.size[1] = ifm_width;
    indexSpace.size[2] = ifm_height;
    indexSpace.size[3] = batch;
    indexSpace.size[4] = fifdim;

    // Define input and output scale for quantization
    Castf16toi16Gaudi2::Castf16toi16Param def;
    def.roundingMode = RND_TO_NINF;
    //def.roundingMode = RND_TO_NE;

    // execute reference implementation of the kernel.
    this->cast_f16_to_i16_ref(ifm, ofm_ref, indexSpace, def.roundingMode);

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI2;
    m_in_defs.NodeParams = &def;
    m_in_defs.inputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]),ifm );

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]),ofm );

    char**   kernelNames = nullptr;
    unsigned kernelCount = 0;
    gcapi::GlueCodeReturn_t result = GetKernelNames(kernelNames, &kernelCount, gcapi::DEVICE_ID_GAUDI2);
    kernelNames = new char*[kernelCount];
    for (unsigned i = 0; i < kernelCount; i++)
    {
        kernelNames[i] = new char[gcapi::MAX_NODE_NAME];
    }    
    result = GetKernelNames(kernelNames, &kernelCount, gcapi::DEVICE_ID_GAUDI2);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Can't get kernel name!! " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI2_KERNEL_CAST_F16_TO_I16]);
    result  = HabanaKernel(&m_in_defs,&m_out_defs);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc> vec;
    vec.push_back(ifm.GetTensorDescriptor());
    vec.push_back(ofm.GetTensorDescriptor());

    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ofm.Print(0);
    ofm_ref.Print(0);
    for (int element = 0 ; element <  ofm_ref.ElementCount() ; element++)
    {
        if (ofm.Data()[element] != ofm_ref.Data()[element])
        {
            std::cout << "Cast F16_to_I16 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "Cast F16_to_I16 test pass!!" << std::endl;


    return 0;
 }

