/**********************************************************************
Copyright (c) 2020 Habana Labs.

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

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <stdint.h>
#include <cfenv>

#define FLOAT_BF16_MIN_VAL        (0x0080)
#define FLOAT_BF16_MAX_VAL        (0x7f7f)
#define EXPONENT_OFFSET_FP32      (23)

inline float bf16ToFloat(uint16_t a)
{
    uint32_t val_32b = ((uint32_t)a) << 16;
    float* val_fp32 = reinterpret_cast<float*>(&val_32b);
    return *val_fp32;
}

inline uint16_t floatToBf16(float input)
{
    uint32_t* val_32b = reinterpret_cast<uint32_t*>(&input);
    uint32_t inputUint = *val_32b;
    uint16_t res;

    if (std::isnan(input) || std::isinf(input))
    {
        return *val_32b >> 16;
    }
    else
    {
        uint32_t inputSign = (inputUint & (1UL << 31)) >> 31;
        bool roundedMSB = ((inputUint & (1<<15)) != 0);

        int32_t inputExponent = (inputUint >> EXPONENT_OFFSET_FP32) & 0xFF;

        int32_t outputExponent = inputExponent;

        uint32_t inputMantissa = inputUint & ((1 << (EXPONENT_OFFSET_FP32+1)) - 1);
        inputMantissa |= (1 << EXPONENT_OFFSET_FP32);

        int32_t outputMantissa = inputMantissa >> 16;

        if (roundedMSB)
        {
            outputMantissa++;
        }
        if (outputMantissa & (1 << 8))
        {
            outputExponent++;
        }
        res = (inputSign << 15) | (outputExponent << 7) | (outputMantissa & 0x7F);
    }
    return res;
}

inline float floatTobf16ToFloat(float a)
{
    return bf16ToFloat(floatToBf16(a));
}

/*
Multiply two Bfloat parameters
*/
inline uint16_t bf16Mult(uint16_t a, uint16_t b)
{
    float dst = bf16ToFloat(a);
    float src = bf16ToFloat(b);
    float res = dst * src;
    uint16_t res_bf16 = floatToBf16(res);
    return res_bf16;
}

class bfloat16
{
public:
    bfloat16(float v = 0) {this->val = floatToBf16(v);}

    float operator-(float rhs) {return bf16ToFloat(val) - rhs;}
    float operator+(float rhs) {return bf16ToFloat(val) + rhs;}
    bool operator<(float rhs) const {return bf16ToFloat(val) < rhs;}
    bool operator>(float rhs) const{return bf16ToFloat(val) > rhs;}
    bool operator!=(float rhs) const{return bf16ToFloat(val) != rhs;}

    bfloat16 operator+(bfloat16 rhs) 
    {
        bfloat16 out;
        out.val = floatToBf16(bf16ToFloat(val) + bf16ToFloat(rhs.val));
        return out;
    }

    bfloat16 operator-(bfloat16 rhs) 
    {
        bfloat16 out;
        out.val = floatToBf16(bf16ToFloat(val) - bf16ToFloat(rhs.val));
        return out;
    }

    bfloat16 operator/(bfloat16 rhs) 
    {
        bfloat16 out;
        out.val = floatToBf16(bf16ToFloat(val) / bf16ToFloat(rhs.val));
        return out;
    }

    float abs(bfloat16 in)
    {
        return std::abs(bf16ToFloat(in.val));
    
    }

    operator double() const {return bf16ToFloat(val);}
    operator float()  const {return bf16ToFloat(val);}

    uint16_t val;
};

static_assert(sizeof(bfloat16) == sizeof(uint16_t), "reinterpret casting to bfloat16 won't work");
