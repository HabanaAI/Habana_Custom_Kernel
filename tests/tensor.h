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

#ifndef HELP_TENSOR_HPP
#define HELP_TENSOR_HPP

#include <string.h>
#include <stdlib.h>

#include <vector>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>

#include "bfloat16.h"
#include "float16.h"
#include "tpc_test_core_types.h"
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

#define GATHER_INDEX 1
using namespace tpc_tests;

#define TENSOR_DESCRIPTOR_CONFIGURATION(elementSizeType, validDimMask, lastDim)                                        \
    (((elementSizeType)&0x3) | (((validDimMask) & ((1 << (DEF_NUM_DIMS_IN_IRF)) - 1)) << 8) | (((lastDim)&0x7) << 16))

inline uint32_t VpeTensorDescriptorElementSizeTypeFromElementSizeInBytes(uint32_t elementSizeInBytes)
{
    switch (elementSizeInBytes) {
        case 1: return 0;
        case 2: return 1;
        case 4: return 2;
        case 8: return 11; // TensorDT_INT64
        default: break;
    }
    return 0;
}
namespace test
{
// This is helper class for reference implementation of tensors.
template <class T, int DIM>
class Tensor
{
public:
    Tensor() :
            m_pdata(nullptr)
    {

    }
    Tensor(const uint64_t sizes [DIM], T* data = NULL, int32_t padValue = 0)
    :
            m_pdata(nullptr)
    {
        std::vector<uint64_t> stdSizes ;
        for (int i = 0 ; i < DIM; i++)
        {
            stdSizes.push_back(sizes[i]);
        }
        Init(stdSizes,data,padValue);
    }

    Tensor(const std::vector<uint64_t>& sizes, T* data = NULL, int32_t padValue = 0)
    :
            m_pdata(nullptr)
    {
        Init(sizes,data,padValue);
    }

    void Init(const uint64_t sizes[DIM], T* data = NULL, int32_t padValue = 0)
    {
        std::vector<uint64_t> stdSizes;
        for (int i = 0; i < DIM; i++)
        {
            stdSizes.push_back(sizes[i]);
        }
        Init(stdSizes, data, padValue);
    }
    void Init(const std::vector<uint64_t>& sizes, T* data = NULL,  int32_t padValue = 0)
    {
        if (m_pdata)
        {
            delete [] m_pdata;
            m_pdata= NULL;
        }

        m_dim_array[0].stride = 1;
        m_dim_array[0].size = sizes[0];
        m_element_count = sizes[0];

        for (int i = 1 ; i < DIM; i++)
        {
            m_element_count *= sizes[i];
            m_dim_array[i].size = sizes[i];
            m_dim_array[i].stride = m_dim_array[i-1].stride * m_dim_array[i-1].size;
        }

        m_pdata = new T[m_element_count](); // NB: array is zero-initialized here
        if (data != NULL)
        {
            memcpy(m_pdata, data, sizeof(T)*m_element_count);
        }
        m_pad_value = padValue;
    }

    ~Tensor()
    {
        delete [] m_pdata;
    }

    T ElementAt(int coords[DIM]) const
    {
        int offset = 0;
        bool outOfBounds = false;
        for (int i = 0 ; i < DIM; i++)
        {
            if (coords[i] < 0 || coords[i] >= ((int)m_dim_array[i].size))
            {
                outOfBounds = true;
                break;
            }
            offset += coords[i] * m_dim_array[i].stride;
        }
        if (outOfBounds)
        {
            return m_pad_value;
        }
        return m_pdata[offset];
    }

    void SetElement(int coords[DIM], T value)
    {
         int offset = 0;
        bool outOfBounds = false;
        for (int i = 0 ; i < DIM; i++)
        {
            if (coords[i] < 0 || coords[i] >= (int)m_dim_array[i].size)
            {
                outOfBounds = true;
                break;
            }
            offset += coords[i] * m_dim_array[i].stride;
        }
        if (outOfBounds)
        {
            return;
        }
        m_pdata[offset] = value;
    }

    void Print(int plane = 0, std::ostream& os = std::cout) const
    {
        os << std::endl;
        if (DIM >= 3 && DIM <= 5)
        {
            for (int d1 = 0; d1 < ((int) m_dim_array[2].size); d1++)
            {
                for (int d2 = 0; d2 < ((int) m_dim_array[1].size); d2++)
                {
                    int coords[] = { plane, d2, d1, 0, 0 };
                    os << std::fixed << std::setw(7) << std::setprecision(4)
                       << + (float)(T)(ElementAt(coords)) << ",";
                }
                os << std::endl;
            }
        }
        else if (DIM == 2)
        {
            for (int d2 = 0; d2 < ((int) m_dim_array[1].size); d2++)
            {
                for ( int d1 = 0; d1 < ((int)m_dim_array[0].size); d1++ )
                {
                    int coords[] = { d1, d2, 0, 0, 0 };

                    os << std::fixed << std::setw(7) << std::setprecision(4)
                       << (float)(T)ElementAt(coords) << ",";
                }
                os << std::endl;
            }
        }
        else if (DIM == 1)
        {
            for (int d = 0; d < ((int) m_dim_array[0].size); d++)
            {
                int coords[] = { d, 0, 0, 0, 0 };
                os << std::fixed << std::setw(7) << std::setprecision(4)
                   << (float)(T)ElementAt(coords) << ",";

                if (((d + 1) % 32) == 0)
                {
                    os << std::endl;
                }
            }
        }
    }

    void FillWithData(int modulo = 9)
    {
        int num = 0;
        for (int i = 0 ; i < m_element_count; i++)
        {
            m_pdata[i] = num;
            num += 1;
            if (num >= modulo)
            {
                num = 0;
            }
        }
    }

    void FillWithData_f16(int modulo = 0x4C00)
    {
        int num = 0x3C00;
        for (int i = 0 ; i < m_element_count; i++)
        {
            m_pdata[i] = num;
            num += 300;
            if (num >= modulo)
            {
                num = 0x3C00;
            }
        }
    }

    void FillWithData(int min_val, int max_val)
    {
        int num = min_val;
        for ( int i = 0; i < m_element_count; i++ )
        {
            m_pdata[i] = num;
            num += 1;
            if ( num > max_val )
            {
                num = min_val;
            }
        }
    }

    void FillWithValue(int val = 0)
    {
        for (int i = 0 ; i < m_element_count; i++)
        {
            m_pdata[i] = val;
        }
    }

    void FillWithSortedValue(bool isSeq = 1)
    {
        if(isSeq) {
            m_pdata[0] = 1.0; m_pdata[3] = 3.0; m_pdata[6] = 5.0; m_pdata[9] = 7.0; m_pdata[12] = 9.0;
            m_pdata[1] = 2.0; m_pdata[4] = 4.0; m_pdata[7] = 6.0; m_pdata[10] = 8.0; m_pdata[13] = 10.0;
            m_pdata[2] = 4.0; m_pdata[5] = 8.0; m_pdata[8] = 12.0; m_pdata[11] = 15.0; m_pdata[14] = 20.0;
        }
        else{
            m_pdata[0] = 3.0; m_pdata[3] = 6.0; m_pdata[6] = 9.0; 
            m_pdata[1] = 3.0; m_pdata[4] = 6.0; m_pdata[7] = 9.0; 
            m_pdata[2] = 12.0; m_pdata[5] = 13.0; m_pdata[8] = 14.0; 

        }
    }
        
    void FillWithSpecificData(int type)
    {
        if(type == GATHER_INDEX) {
            m_pdata[0] = 0; m_pdata[1] = 1; m_pdata[2] = 2; m_pdata[3] = 3; 
            m_pdata[4] = 3; m_pdata[5] = 2; m_pdata[6] = 1; m_pdata[7] = 0; 
            m_pdata[8] = 2; m_pdata[9] = 3; m_pdata[10] = 0; m_pdata[11] = 1; 
            m_pdata[12] = 1; m_pdata[13] = 2; m_pdata[14] = 1; m_pdata[15] = 0; 
        }

    }

    void InitRand(const T rangemin, const T rangemax, unsigned seed = 0)
    {
        if ( seed != 0 )
        {
            std::cout << "Test Seed:" << seed << std::endl;
            srand(seed);
        }

        for ( int i = 0; i < m_element_count; i++ )
        {
            m_pdata[i] = (T)((T)rangemin + (T)rand() / ((T)RAND_MAX / ((T)rangemax - rangemin + (T)1) + (T)1));
        }
    }

    Tensor& operator*=(const T & value)
    {
        for (int i = 0 ; i < m_element_count; i++)
        {
            m_pdata[i] *= value;
        }
        return *this;
    }

    Tensor& operator+=(const T & value)
    {
        for (int i = 0 ; i < m_element_count; i++)
        {
            m_pdata[i] += value;
        }
        return *this;
    }

    Tensor& operator=(const T & value)
    {
        for (int i = 0; i < m_element_count; i++)
        {
            m_pdata[i] = value;
        }
        return *this;
    }

    unsigned int Size(int dimension) const
    {
        return m_dim_array[dimension].size;
    }

    // @brief returns tensor descriptor for TPC simulator
    TensorDesc2 GetTensorDescriptor() const
    {
        TensorDesc2 tensorDesc = {};
        tensorDesc.baseAddrUnion.baseAddr = (uint64_t)m_pdata;

        uint32_t validDimMask = 0;
        for (int dim = 0 ;  dim <  DIM; dim++)
        {
            tensorDesc.dimDescriptors[dim].size = m_dim_array[dim].size;
            tensorDesc.dimDescriptors[dim].stride = m_dim_array[dim].stride;
            validDimMask |= (1 << dim);
        }
        // Due to HW consideration, the padd value needs to be replicated across
        // all lane if the data type is smaller than dword.
        // for example if pad value is 0x80 , pad dword should be 0x80808080.
        const uint32_t* pVal = reinterpret_cast<const uint32_t*>(&m_pad_value);
        unsigned int padValueInt = *pVal;
        switch (sizeof(T))
        {
            case 1:
            {
                memset(&(tensorDesc.paddingValue),padValueInt,4);
            }
            break;
            case 2:
            {
                tensorDesc.paddingValue = padValueInt | (padValueInt << 16);
            }
            break;
            case 4:
            {
                tensorDesc.paddingValue = padValueInt;
            }
            break;
            default:
            {
                break;
            }
        }

        uint32_t elementSizeType = VpeTensorDescriptorElementSizeTypeFromElementSizeInBytes(sizeof(T));
        uint32_t lastDim = DIM - 1;
        tensorDesc.configuration = TENSOR_DESCRIPTOR_CONFIGURATION(elementSizeType, validDimMask, lastDim);
        return tensorDesc;
    }

    T* Data()
    {
        return m_pdata;
    }


    int GetDimensionality() const {return DIM;}
    int ElementCount() const { return m_element_count;}
    void SetPadValue(T value) { m_pad_value = value;}
    T    GetPadValue() const { return m_pad_value;}

private:

    struct Dim
    {
        unsigned int size;
        unsigned int stride;
    };

    // FCD is slot 0.
    Dim       m_dim_array[DIM];
    int32_t   m_pad_value;
    T*        m_pdata;
    int       m_element_count;

    Tensor(const Tensor& other) = delete;
    Tensor& operator=(const Tensor& other) = delete;
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper operator
// == usefull for GTEST
template <class T, int DIM>
inline bool operator==(const Tensor<T,DIM>& lhs, const Tensor<T,DIM>& rhs){ return lhs.IsEquivalentTo(rhs); }
template <class T, int DIM>
inline bool operator!=(const Tensor<T,DIM>& lhs, const Tensor<T,DIM>& rhs){ return !(lhs == rhs); }

template<int NUM>
tpc_lib_api::TensorDataType getGcDataType(const Tensor<signed char,NUM>& a)
{
   return tpc_lib_api::DATA_I8;
}

template<int NUM>
tpc_lib_api::TensorDataType getGcDataType(const Tensor<signed int,NUM>& a)
{
   return tpc_lib_api::DATA_I32;
}

template<int NUM>
tpc_lib_api::TensorDataType getGcDataType(const Tensor<signed short,NUM>& a)
{
   return tpc_lib_api::DATA_I16;
}

template<int NUM>
tpc_lib_api::TensorDataType getGcDataType(const Tensor<float,NUM>& a)
{
   return tpc_lib_api::DATA_F32;
}

template<int NUM>
tpc_lib_api::TensorDataType getGcDataType(const Tensor<bfloat16,NUM>& a)
{
   return tpc_lib_api::DATA_BF16;
}

template<int NUM>
tpc_lib_api::TensorDataType getGcDataType(const Tensor<float16,NUM>& a)
{
   return tpc_lib_api::DATA_F16;
}

static const unsigned MAX_TENSOR_DIM = 5;


} // name space test



typedef test::Tensor<int8_t,1>  int8_1DTensor;
typedef test::Tensor<int8_t,2>  int8_2DTensor;
typedef test::Tensor<int8_t,3>  int8_3DTensor;
typedef test::Tensor<int8_t,4>  int8_4DTensor;
typedef test::Tensor<int16_t,1> int16_1DTensor;
typedef test::Tensor<int16_t,2> int16_2DTensor;
typedef test::Tensor<int16_t,3> int16_3DTensor;
typedef test::Tensor<int16_t,4> int16_4DTensor;
typedef test::Tensor<int16_t,5> int16_5DTensor;
typedef test::Tensor<int32_t,1> int32_1DTensor;
typedef test::Tensor<int32_t,2> int32_2DTensor;
typedef test::Tensor<int32_t,3> int32_3DTensor;
typedef test::Tensor<int32_t,4> int32_4DTensor;
typedef test::Tensor<int32_t,5> int32_5DTensor;
typedef test::Tensor<float,1>   float_1DTensor;
typedef test::Tensor<float,2>   float_2DTensor;
typedef test::Tensor<float,3>   float_3DTensor;
typedef test::Tensor<float,4>   float_4DTensor;
typedef test::Tensor<float,5>   float_5DTensor;
typedef test::Tensor<bfloat16,2>   bfloat16_2DTensor;
typedef test::Tensor<bfloat16,3>   bfloat16_3DTensor;
typedef test::Tensor<bfloat16,4>   bfloat16_4DTensor;
typedef test::Tensor<bfloat16,5>   bfloat16_5DTensor;
typedef test::Tensor<float16,2>    float16_2DTensor;
typedef test::Tensor<float16,3>    float16_3DTensor;
typedef test::Tensor<float16,4>    float16_4DTensor;
typedef test::Tensor<float16,5>    float16_5DTensor;


struct IndexSpace
{
    int offset [test::MAX_TENSOR_DIM];
    int size   [test::MAX_TENSOR_DIM];
};

#endif // HELPLER_TENSOR_HPP
