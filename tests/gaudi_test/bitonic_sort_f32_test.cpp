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

#include "bitonic_sort_f32_test.hpp"
#include "entry_points.hpp"
template<class vecType> void BitonicSortF32Test:: SortVec(vecType& curVec, intVec_t& lut, intVec_t& pred, bool sortDir)
{
    vecType tmpVec(curVec.size());
    for (unsigned int i = 0; i < tmpVec.size(); i++)
    {
        tmpVec[i].first  = curVec[i].first;
        tmpVec[i].second = curVec[i].second;
    }

    // Generate shuffled vector according to shuffle pattern
    vecType shuffVec(tmpVec.size());
    for (unsigned int i = 0; i < tmpVec.size(); i++)
    {
        shuffVec[i].first  = tmpVec[lut[i]].first;
        shuffVec[i].second = tmpVec[lut[i]].second;
    }

    // Store inVec to outVec if it is <= shuffVec, else store shuffVec
    for (unsigned int i = 0; i < tmpVec.size(); i++)
    {
        if (tmpVec[i].second <= shuffVec[i].second)
        {
            curVec[i].first  = tmpVec[i].first;
            curVec[i].second = tmpVec[i].second;
        }
        else
        {
            curVec[i].first  = shuffVec[i].first;
            curVec[i].second = shuffVec[i].second;
        }
    }

    // Wherever predicate is set, overwrite outVec with shuffVec if it is greater than inVec
    for (unsigned int i = 0; i < tmpVec.size(); i++)
    {
        if (pred[i] == 1)
        {
            if (tmpVec[i].second >= shuffVec[i].second)
            {
                curVec[i].first  = tmpVec[i].first;
                curVec[i].second = tmpVec[i].second;
            }
            else
            {
                curVec[i].first  = shuffVec[i].first;
                curVec[i].second = shuffVec[i].second;
            }
        }
    }

}

intVec_t BitonicSortF32Test::GenerateShufflePattern(int vecSize, int stageNo, int subStageNo)
{
    intVec_t tmpLut(vecSize);
    intVec_t shuffLut(vecSize);

    int curBlockSize = pow(2, stageNo);
    int curGroupSize = curBlockSize / pow(2, subStageNo - 1);

    // Intialize lut with 0 -> vecSize - 1
    std::iota(tmpLut.begin(), tmpLut.end(), 0);

    // Shuffle groups
    for (unsigned int i = 0; i < tmpLut.size(); i += curGroupSize)
    {
        for (int j = 0; j < curGroupSize / 2; j++)
        {
            shuffLut[i + j] = tmpLut[i + curGroupSize / 2 + j];
        }
        for (int j = 0; j < curGroupSize / 2; j++)
        {
            shuffLut[i + curGroupSize / 2 + j] = tmpLut[i + j];
        }
    }
    return shuffLut;
}

intVec_t BitonicSortF32Test::GeneratePredPattern(int32_t vecSize, bool curSign, int32_t stageNo, int32_t subStageNo)
{
    intVec_t pred;
    int      curBlockSize  = pow(2, stageNo);
    int      curGroupSize  = curBlockSize / pow(2, subStageNo - 1);
    int      nBlocks       = vecSize / curBlockSize;
    int      nGroupsPerBlk = curBlockSize / curGroupSize;
    // Iterate over all blocks in vector
    for (int b = 0; b < nBlocks; b++)
    {
        // Iterate over all groups within the block
        for (int m = 0; m < nGroupsPerBlk; m++)
        {
            // Fill the first half of group
            for (int j = 0; j < curGroupSize / 2; j++)
            {
                pred.push_back(curSign == true ? 0 : 1);
            }
            // Fill the second half of group
            for (int j = 0; j < curGroupSize / 2; j++)
            {
                pred.push_back(curSign == true ? 1 : 0);
            }
        }
        // Alternate sign between blocks
        curSign = !curSign;
    }
    return pred;
}

void BitonicSortF32Test::OneVectorSort(indDataVec_t& curVec, SortType curDir, bool sortDir)
{
    bool curSign = (curDir == SORT_ASCENDING);
    int  nStages = log2(m_vecSize);
    // Iterate over each stage
    for (int i = 1; i <= nStages; i++)
    {
        // Iterate over each sub-stage
        for (int j = 1; j <= i; j++)
        {
            intVec_t pred(m_vecSize);
            intVec_t lut(m_vecSize);
            pred = GeneratePredPattern(m_vecSize, curSign, i, j);
            lut  = GenerateShufflePattern(m_vecSize, i, j);
            SortVec(curVec, lut, pred, sortDir);
        }
    }
    return;
}

    
void BitonicSortF32Test::bitonic_sort_reference_implementation(float_5DTensor& ifmData, float_5DTensor& ofmRef,
                   int32_5DTensor& ofmIdxRef, int chunkSize, SortType dir)
{
    const int32_t W = ifmData.Size(1);
    const int32_t H = ifmData.Size(2);
    const int32_t B = ifmData.Size(3);
    const int32_t F = ifmData.Size(4);
    int              vecSize   = 64;
    const int c_nVec = ceil(chunkSize / vecSize);       // num vectors in one chunk
    int nChunks      = ceil(ifmData.Size(0) / chunkSize); // num chunks along full sort length
    /* data index pairs that would be stored in kernel's local memory */
    indDataVec_t vecListVlm[c_nVec];
    // Set default/pad value to be used when fcd is not a multiple of m_chunkSize
    float defVal = 0.0f;
    if (dir == SORT_ASCENDING)
    {
        // In ascending sort, last values should be maximum to avoid swap
        defVal = std::numeric_limits<float>::max();
    }
    else
    {
        // In descending sort, last values should be minimum to avoid swap
        defVal = std::numeric_limits<float>::min();
    }
    int depthStart = 0;
    for(int fif = 0;fif < F; fif++)
    {
        for (int b = 0; b < B; b++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                {
                    int ind = 0; // Generated indices start from 0 for each line
                    /* Divide the total FCD into chunks and sort each chunk separately */
                    for (int iChunk = 0; iChunk < nChunks; iChunk++)
                    {
                        int chunkStart = depthStart + iChunk * chunkSize;
                        // Set direction for each chunk
                        SortType curDir  = dir;
                        bool            sortDir = dir == SORT_ASCENDING ? 1 : 0;
                        /* Stage 1: 1-vector Sort (BS64) on all individual vectors and store
                        into array */
                        for (int iVec = 0; iVec < c_nVec; iVec++)
                        {
                            // Initialize current data vector
                            floatVec_t curData(m_vecSize, defVal);
                            int        vecStart = chunkStart + iVec * m_vecSize;
                            int        vecEnd   = vecStart + m_vecSize;
                            // Load data
                            std::copy(ifmData.Data() + vecStart, ifmData.Data() + vecEnd,
                                    curData.begin());
                            // Initialize original index locations
                            intVec_t curIdx(m_vecSize);
                            // Generate default indices
                            std::iota(curIdx.begin(), curIdx.end(), ind);
                            ind += vecSize;
                            // Create vector of index-data pair
                            indDataVec_t curVec;
                            for (int i = 0; i < vecSize; i++)
                            {
                                curVec.push_back(std::make_pair(curIdx[i], curData[i]));
                            }
                            // Do 1-vector sort (BS64)
                            OneVectorSort(curVec, curDir, sortDir);
                            vecListVlm[iVec] = curVec;
                            // Switch direction
                            if (curDir == SORT_ASCENDING)
                            {
                                curDir = SORT_DESCENDING;
                            }
                            else
                            {
                                curDir = SORT_ASCENDING;
                            }
                        }
                        /* Stage 2 : Arrange vectors into block of i vectors and
                        do sort on each block */
                        for (int i = 2; i <= c_nVec; i *= 2)
                        {
                            // Iterate over all possible group sizes within a block
                            for (int j = i; j > 1; j /= 2)
                            {
                                bool sign = (dir == SORT_ASCENDING);
                                // Iterate over all j-vector groups
                                for (int m = 0; m < c_nVec / j; m++)
                                {
                                    // Iterate over all vectors in current group
                                    for (int l = m * j; l < m * j + j / 2; l++)
                                    {
                                        for (int iEle = 0; iEle < vecSize; iEle++)
                                        {
                                            bool predV1 = vecListVlm[l][iEle].second >
                                                        vecListVlm[j / 2 + l][iEle].second;
                                            bool predV2 = vecListVlm[l][iEle].second <
                                                        vecListVlm[j / 2 + l][iEle].second;
                                            bool swap = (predV1 && sign) || (predV2 && !sign);
                                            if (swap)
                                            {
                                                indDataPair_t tmp = vecListVlm[j / 2 + l][iEle];
                                                vecListVlm[j / 2 + l][iEle] = vecListVlm[l][iEle];
                                                vecListVlm[l][iEle]         = tmp;
                                            }
                                        }
                                        if ((j / 2 + l + 1) % i == 0)
                                        {
                                            // Alternate sign between blocks
                                            sign = !sign;
                                        }
                                    }
                                }
                            }
                            // BM-64 on each vector (Steps 6.1 to 6.6)
                            bool sign = (dir == SORT_ASCENDING);
                            for (int iVec = 0; iVec < c_nVec; iVec++)
                            {
                                int nStages = log2(vecSize);
                                // Iterate over all substages in n-th stage
                                for (int j = 1; j <= nStages; j++)
                                {
                                    intVec_t pred(vecSize);
                                    intVec_t lut(vecSize);
                                    pred = GeneratePredPattern(vecSize, sign, nStages, j);
                                    lut  = GenerateShufflePattern(vecSize, nStages, j);
                                    SortVec(vecListVlm[iVec], lut, pred, sortDir);
                                }
                                if ((iVec + 1) % i == 0)
                                {
                                    // Alternate sign between blocks
                                    sign = !sign;
                                }
                            }
                        }
                        // Copy sorted chunk to output tensors
                        int i = chunkStart;
                        for (int iVec = 0; iVec < c_nVec; iVec++)
                        {
                            for (int ind = 0; ind < vecSize; ind++)
                            {
                                ofmRef.Data()[i]    = vecListVlm[iVec][ind].second;
                                ofmIdxRef.Data()[i] = vecListVlm[iVec][ind].first;
                                i++;
                            }
                        }
                    }
                    depthStart += ifmData.Size(0);
                }
            }
        }
    }
}

int BitonicSortF32Test::runTest()
{
    const int height = 4;
    const int width  = 4;
    const int depth  = 1024;    // up to 8K
    const int batch  = 1;
    const int fifdim  = 1;
    int chunkSize   = depth;

    BitonicSortF32::BitonicSortF32Param def;
    def.chunkSize = chunkSize;
    def.sortDir = SORT_ASCENDING;
    def.expChunkSize = log2(chunkSize);
    m_vecSize = 64;

    unsigned int fmInitializer[] = {depth, width, height, batch, fifdim};

    float_5DTensor input(fmInitializer);
    input.InitRand(10, 100, 96);

    int32_5DTensor indexOut(fmInitializer);
    int32_5DTensor indexOut_ref(fmInitializer);
    float_5DTensor output(fmInitializer);
    float_5DTensor output_ref(fmInitializer);

    // execute reference implementation of the kernel.
    bitonic_sort_reference_implementation(input, output_ref, indexOut_ref, def.chunkSize, (SortType)def.sortDir);

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI;
    m_in_defs.NodeParams = &def;

    m_in_defs.inputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), input);

    m_in_defs.outputTensorNr = 2;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), output);
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[1]), indexOut);

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

    strcpy(m_in_defs.nodeName, kernelNames[GAUDI_KERNEL_BITONIC_SORT_F32]);
    result  = HabanaKernel(&m_in_defs,&m_out_defs);
    // Declaration of auxiliary tensor
    int8_1DTensor aux_tensor({6144});
    // Allocate memory for aux tensor if not allocated
    if (result == gcapi::GLUE_INSUFICIENT_AUX_BUFFER_SIZE)
    {
        if (m_out_defs.auxiliaryTensors[0].pData)
        {
            delete [] (int8_t*)m_out_defs.auxiliaryTensors[0].pData;
            m_out_defs.auxiliaryTensors[0].pData = NULL;
        }

        m_out_defs.auxiliaryTensors[0].pData =
                                    new int8_t[m_out_defs.auxiliaryTensors[0].bufferSize / sizeof(int8_t)];
        // second call of glue-code to load Auxiliary data.
        result  = HabanaKernel(&m_in_defs,&m_out_defs);
        // AUXILIARY TENSOR init based on parameters got from glue code
        aux_tensor.Init(m_out_defs.auxiliaryTensors[0].geometry.sizes,
                                    (int8_t*)m_out_defs.auxiliaryTensors[0].pData);
    }


    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDescriptor> vec;
    vec.push_back(input.GetTensorDescriptor());
    vec.push_back(output.GetTensorDescriptor());
    vec.push_back(indexOut.GetTensorDescriptor());
    vec.push_back(aux_tensor.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount);
    printf("Output data \n");
    output.Print(0);
    output.Print(depth-1);
    printf("\nOutput ref data \n");
    output_ref.Print(0);
    output_ref.Print(depth-1);
    for (int element = 0 ; element <  output_ref.ElementCount() ; element++)
    {
        if (abs(output.Data()[element] - output_ref.Data()[element]) > 1e-6)
        {
            std::cout << "Bitonic Sort F32 test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "Bitonic Sort F32 test pass!!" << std::endl;
    return 0;
}
