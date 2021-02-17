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

#ifndef CAST_TEST_HPP
#define CAST_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "cast.hpp"
#include "ConvInterface.h"

class CastTest : public TestBase
{
public:
    CastTest() {}
    ~CastTest() {}
    int runTest();

    inline static void cast_int8_to_f32_ref(
         const test::Tensor<int8_t,4>& input,
         test::Tensor<float,4>& output,
         const IndexSpace& indexSpace,
         float scaleToFloat);

    inline static void cast_f32_to_int16_ref(
        const test::Tensor<float,4>& ifm,
        test::Tensor<int16_t,4>& ofm,
        const IndexSpace& indexSpace,
        float scaleToInt);
private:
    CastTest(const CastTest& other) = delete;
    CastTest& operator=(const CastTest& other) = delete;

};

inline void CastTest::cast_int8_to_f32_ref(
         const test::Tensor<int8_t,4>& input,
         test::Tensor<float,4>& output,
         const IndexSpace& indexSpace,
         float scaleToFloat)
{
    int eig =256;

    int coords[4] = { 0 };
    for (int d = indexSpace.offset[0]; d < (indexSpace.offset[0] + indexSpace.size[0]) * eig; d += 1)
    {
        coords[0] = d;
        for (int b = indexSpace.offset[3]; b < indexSpace.offset[3] + indexSpace.size[3]; b += 1)
        {
            coords[3] = b;
            for (int h = indexSpace.offset[2]; h < indexSpace.offset[2] + indexSpace.size[2]; h += 1)
            {
                coords[2] = h;
                for (int w = indexSpace.offset[1]; w < indexSpace.offset[1] + indexSpace.size[1]; w += 1)
                {
                    coords[1] = w;
                    float ofmVal;
                    int8_t ifmVal = input.ElementAt(coords);
                    // Convert from int8 to f32 datatype
                    tpc_dali::tpcsim_int8_to_fp32(&ifmVal, &ofmVal, 0);
                    ofmVal *= scaleToFloat;
                    output.SetElement(coords, ofmVal);
                }
            }
        }
    }
}

inline void CastTest::cast_f32_to_int16_ref(
    const test::Tensor<float,4>& ifm,
    test::Tensor<int16_t,4>& ofm,
    const IndexSpace& indexSpace,
    float scaleToInt)
{
    int eig = 128;

    int coords[4] = { 0 };
    for (int d = indexSpace.offset[0]; d < (indexSpace.offset[0] + indexSpace.size[0]) * eig; d += 1)
    {
        coords[0] = d;
        for (int b = indexSpace.offset[3]; b < indexSpace.offset[3] + indexSpace.size[3]; b += 1)
        {
            coords[3] = b;
            for (int h = indexSpace.offset[2]; h < indexSpace.offset[2] + indexSpace.size[2]; h += 1)
            {
                coords[2] = h;
                for (int w = indexSpace.offset[1]; w < indexSpace.offset[1] + indexSpace.size[1]; w += 1)
                {
                    coords[1] = w;
                    float ifmVal = ifm.ElementAt(coords);
                    ifmVal *= scaleToInt;
                    int16_t ofmVal = 0;
                    // Convert f32 -> i32 -> i16
                    int32_t result_int = 0;
                    tpc_dali::tpcsim_fp32_to_int32(&ifmVal, &result_int, 0);
                    tpc_dali::tpcsim_int32_to_int16(&result_int, &ofmVal, 0, 0, 0);
                    ofm.SetElement(coords, ofmVal);
                }
            }
        }
    }
}

 inline int CastTest::runTest()
 {

    /**********************Test for cast i8 to f32************************/
    // Initalize input size
    const int ifm_height = 10;
    const int ifm_width  = 8;
    const int ofmifm_depth = 300;
    const int batch = 1;

    // Initalize inputs
    unsigned int ifmofmInitializer[] = {ofmifm_depth,ifm_width,ifm_height,batch};
    int8_4DTensor ifm(ifmofmInitializer);
    ifm.FillWithData();
    float_4DTensor ofm(ifmofmInitializer);
    float_4DTensor ofm_ref(ifmofmInitializer);

    IndexSpace indexSpace = {{0}};
    int depthIS = (ofmifm_depth + 255) / 256 ;
    indexSpace.size[0] = depthIS;
    indexSpace.size[1] = ifm_width;
    indexSpace.size[2] = ifm_height;
    indexSpace.size[3] = batch;

    // Define input and output scale for quantization
    float scale = m_in_defs.inputTensors[0].quantizationParam.scale = 1.0;
    Cast::CastParams def;
    def.scale = scale;

    // Define cast datatype to load respective kernels
    Cast::CastDataType_t m_castType = Cast::i8_to_f32;

    // execute reference implementation of the kernel.
    this->cast_int8_to_f32_ref(ifm,
                               ofm_ref,
                               indexSpace,
                               scale);

    // generate input for query call
    m_in_defs.NodeParams = &def;
    m_in_defs.inputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]),ifm );

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]),ofm );

    Cast kernelClass(m_castType);

    // make the call into the glue code.
    gcapi::GlueCodeReturn_t result = kernelClass.GetGcDefinitions(&m_in_defs,&m_out_defs);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "glue test failed!! " << result << std::endl;
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDescriptor> vec;
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
            std::cout << "Cast I8_2_F32 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "Cast I8_2_F32 test pass!!" << std::endl;





    /**********************Test for cast f32 to i16************************/
    float_4DTensor input(ifmofmInitializer);
    input.InitRand(0, 100);

    int16_4DTensor out(ifmofmInitializer);
    int16_4DTensor out_ref(ifmofmInitializer);

    depthIS = (ofmifm_depth + 63) / 64 ;
    indexSpace.size[0] = depthIS;

    // execute reference implementation of the kernel.
    this->cast_f32_to_int16_ref(input,
                              out_ref,
                              indexSpace,
                              scale);

    // generate input for query call
    m_in_defs.NodeParams = &def;
    m_in_defs.inputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]),input );

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]),out );

    m_castType = Cast::f32_to_i16;
    Cast kernelClass2(m_castType);

    // make the call into the glue code.
    result = kernelClass2.GetGcDefinitions(&m_in_defs,&m_out_defs);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "glue test failed!! " << result << std::endl;
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDescriptor> vec1;
    vec1.push_back(input.GetTensorDescriptor());
    vec1.push_back(out.GetTensorDescriptor());

    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec1, m_in_defs, m_out_defs);
    out.Print(0);
    out_ref.Print(0);
    for (int element = 0 ; element <  out_ref.ElementCount() ; element++)
    {
        if (out.Data()[element] != out_ref.Data()[element])
        {
            std::cout << "Cast F32_2_I16 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "Cast F32_2_I16 test pass!!" << std::endl;
    return 0;
 }

#endif /* CAST_TEST_HPP */
