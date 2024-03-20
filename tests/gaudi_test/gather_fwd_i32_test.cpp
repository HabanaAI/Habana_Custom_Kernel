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

#include "gather_fwd_i32_test.hpp"
#include "entry_points.hpp"
#include "gather_fwd_i32.hpp"

void GatherFwdI32Test::GatherElementsOnnxRef(
    int32_5DTensor &ifm,
    int32_5DTensor &indices,
    int32_5DTensor &ofm,
    int axis)
{
    const int depth    = 0;
    const int width    = 1;
    const int height   = 2;
    const int batch    = 3;
    const int fifthDim = 4;

    int newCoords[5] = {0};
    int inCoords[5]  = {0};

    for (unsigned int f = 0; f < ofm.Size(4); f++)
    {
        inCoords[fifthDim]  = f;
        newCoords[fifthDim] = f;

        for (unsigned int b = 0; b < ofm.Size(3); b++)
        {
            inCoords[batch]  = b;
            newCoords[batch] = b;

            for (unsigned int h = 0; h < ofm.Size(2); h++)
            {
                inCoords[height]  = h;
                newCoords[height] = h;
 
                 for (unsigned int w = 0; w < ofm.Size(1); w++)
                {
                    unsigned int depthSize    = ofm.Size(0);
                    inCoords[width]  = w;
                    newCoords[width] = w;
 
                    for (unsigned int d = 0; d < depthSize; d++)
                    {
                        inCoords[depth]  = d;
                        newCoords[depth] = d;
                        int index        = indices.ElementAt(inCoords);

                        // Set index along axis
                        newCoords[axis] = index;
                        int32_t value      = ifm.ElementAt(newCoords);

                        ofm.SetElement(inCoords, value);
                    }
                }
            }
        }
    }
}

void GatherFwdI32Test::GatherFwdRefImplementation(
        int32_5DTensor &ifm,
        int32_5DTensor &index,
        int32_5DTensor &ofm,
        int inputDims, int indexDims,
        int axis)
{

   int ofmcoords[5] = {0};
   if (axis == 0)
   {
        for (unsigned f = 0; f < ofm.Size(4); f += 1)
        {
                ofmcoords[4] = f;
                for (unsigned b = 0; b < ofm.Size(3); b += 1)
                {
                    ofmcoords[3] = b;
                    for (unsigned h = 0; h < ofm.Size(2); h += 1)
                    {
                        ofmcoords[2] = h;
                        for (unsigned w = 0; w < ofm.Size(1); w += 1)
                        {
                            ofmcoords[1] = w;
                            for (unsigned d = 0; d < ofm.Size(0); d += 1)
                            {
                                ofmcoords[0] = d;
                                int tmp = index.ElementAt(ofmcoords);
                                ofmcoords[0] = tmp;
                                float x = ifm.ElementAt(ofmcoords);
                                int ofmcord[] = {(int)d, (int)w, (int)h, (int)b, (int)f};
                                ofm.SetElement(ofmcord, x);
                            }
                        }
                    }
                }   
        } 
   }
   else if (axis == 1)
   {
        for (unsigned f = 0; f < ofm.Size(4); f += 1)
        {
            ofmcoords[4] = f;
            for (unsigned b = 0; b < ofm.Size(3); b += 1)
            {
                ofmcoords[3] = b;
                for (unsigned h = 0; h < ofm.Size(2); h += 1)
                {
                    ofmcoords[2] = h;
                    for (unsigned d = 0; d < ofm.Size(0); d += 1)
                    {
                        ofmcoords[0] = d;
                        for (unsigned w = 0; w < ofm.Size(1); w += 1)
                        {
                            ofmcoords[1] = w;
                            int tmp = index.ElementAt(ofmcoords);
                            ofmcoords[1] = tmp;
                            float x = ifm.ElementAt(ofmcoords);
                            int ofmcord[] = {(int)d, (int)w, (int)h, (int)b, (int)f};
                            ofm.SetElement(ofmcord, x);
                        }
                    }
                }
            }   
        } 
   }
}


int GatherFwdI32Test::runTest(Gaudi_Kernel_Name_e NameofKernel)
{

    const int depth  = 4;
    const int width  = 4;
    const int height = 1;
    const int batch  = 1;
    const int fifthdim  = 1;

    uint64_t fmInitializer[] = {depth, width, height, batch, fifthdim};
    uint64_t indexInitializer[] = {depth, width, height, batch, fifthdim};

    int32_5DTensor input_tensor(fmInitializer);
    int32_5DTensor index_tensor(indexInitializer);
    int32_5DTensor out_tensor(indexInitializer);
    int32_5DTensor out_tensor_ref(indexInitializer);

    // initializing tensors with the sizes
    input_tensor.FillWithData(1,16);
    //index_tensor.FillWithData(0,3);
    index_tensor.FillWithSpecificData(GATHER_INDEX);

    GatherFwdI32::GatherFwdParam param;
    if(NameofKernel == GAUDI_KERNEL_GATHER_FWD_DIM0_I32)
        param.axis = 0;  
    else
        param.axis = 1;

    // generate input for query call
    m_in_defs.inputTensorNr = 2;
    m_in_defs.deviceId = tpc_lib_api::DEVICE_ID_GAUDI;
    m_in_defs.nodeParams.nodeParams = &param;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), input_tensor);
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), index_tensor);

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), out_tensor);


    // filling scale-bias input tensor

    //GatherFwdRefImplementation(input_tensor, index_tensor, out_tensor_ref, 5,5,0);
    GatherElementsOnnxRef(input_tensor, index_tensor, out_tensor_ref, param.axis);

    char**   kernelNames = nullptr;
    unsigned kernelCount = 0;
    tpc_lib_api::GlueCodeReturn result = GetKernelGuids(kernelNames, &kernelCount, tpc_lib_api::DEVICE_ID_GAUDI);
    kernelNames = new char*[kernelCount];
    for (unsigned i = 0; i < kernelCount; i++)
    {
        kernelNames[i] = new char[tpc_lib_api::MAX_NODE_NAME];
    }    
    result = GetKernelGuids(kernelNames, &kernelCount, tpc_lib_api::DEVICE_ID_GAUDI);
    if (result != tpc_lib_api::GLUE_SUCCESS)
    {
        std::cout << "Can't get kernel name!! " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    strcpy(m_in_defs.guid.name, kernelNames[NameofKernel]);
    result  = InstantiateTpcKernel(&m_in_defs,&m_out_defs);
    if (result != tpc_lib_api::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel!! " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc2> vec;
    vec.push_back(input_tensor.GetTensorDescriptor());
    vec.push_back(index_tensor.GetTensorDescriptor());
    vec.push_back(out_tensor.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount); 
    out_tensor.Print(0);
    out_tensor.Print(1);
    out_tensor.Print(2);
    out_tensor.Print(3);

    printf("Break \n");
    out_tensor_ref.Print(0);
    out_tensor_ref.Print(1);
    out_tensor_ref.Print(2);
    out_tensor_ref.Print(3);
    for (int element = 0 ; element <  out_tensor_ref.ElementCount() ; element++)
    {
        if (out_tensor.Data()[element] != out_tensor_ref.Data()[element])
        {
            std::cout << "Gather forward I32 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "Gather fforward I32 test pass!!" << std::endl;
    return 0;
}
