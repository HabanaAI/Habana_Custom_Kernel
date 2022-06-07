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

#pragma once

#ifndef FLOAT16_H
#define FLOAT16_H

#include <cmath>
#include <cstdint>

#define RND_TO_NE   0
#define RND_TO_0    1
#define RND_TO_PINF 2
#define RND_TO_NINF 3
#define RND_SR      4
#define RND_CSR     5
#define RND_HALF_AZ 6

#define SIGN_OFFSET_FP32        31
#define EXPONENT_BIAS_FP32      127
#define SIGN_MASK_FP32 0x80000000
#define EXPONENT_MASK_FP32 0x7F800000
#define SIGNIFICAND_MASK_FP32 0x007FFFFF

#define SIGN_MASK_FP16 0x8000
#define EXPONENT_MASK_FP16 0x7C00
#define SIGNIFICAND_MASK_FP16 0x03FF
#define EXPONENT_OFFSET_FP16 10
#define SIGN_OFFSET_FP16     15
#define EXPONENT_BIAS_FP16   15

// sbs implements select bits x[high:low]
inline uint32_t sbs(uint32_t x, uint8_t high, uint8_t low)
{
    return (high == 31) ? (x >> low) : ((x & ((1U << (high + 1)) - 1)) >> low);
}

inline bool is_nan_fp32(uint32_t x)
{
    return ((sbs(x, 30, 23) == 0xFF) && (sbs(x, 22, 0) != 0));
}

inline bool is_inf_fp32(uint32_t x)
{
    return ((sbs(x, 30, 23) == 0xFF) && (sbs(x, 22, 0) == 0));
}

inline bool is_zero_fp32(uint32_t x)
{
    return ((sbs(x, 30, 23) == 0x00) && (sbs(x, 22, 0) == 0));
}

int fp_accommodate_rounding(uint32_t     intValuePreRounding,
                            bool         roundedMSB,
                            bool         roundedLSBs,
                            unsigned int sign,
                            int          roundingMode,
                            uint32_t     lfsrVal,
                            uint32_t     discardedAlignedLeft);

uint16_t fp32_to_fp16(float input, int roundingMode);

inline bool fp16_is_zero(uint16_t val)
{
    return (val & (~SIGN_MASK_FP16)) ? 0 : 1;
}

inline bool fp16_is_infinity(uint16_t val)
{
    return (val & 0x7FFF) == EXPONENT_MASK_FP16 ? 1 : 0;
}

inline bool fp16_is_nan(uint16_t val)
{
    bool isAllExponentBitsSet = ((val & EXPONENT_MASK_FP16) == EXPONENT_MASK_FP16);
    bool isAnyMantissaBitSet  = ((val & SIGNIFICAND_MASK_FP16) != 0);
    return (isAllExponentBitsSet & isAnyMantissaBitSet);
}

inline bool fp16_is_denormal(uint16_t val)
{ // Do not consider zero as denormal
    return (((val & EXPONENT_MASK_FP16) == 0) && ((val & SIGNIFICAND_MASK_FP16) != 0));
}

// Count the Number of Leading Zero Bits
int lzcnt(uint32_t bits, uint32_t int_num);

class float16;

float16 fp32ToFloat16(float v);
float   float16ToFP32(float16 v);

class float16
{
public:
    float16(float v=0);

    friend bool operator<(float16 a, float16 b);

    friend bool operator>(float16 a, float16 b);

    friend bool operator<=(float16 a, float16 b);

    friend bool operator>=(float16 a, float16 b);

    friend bool operator==(float16 a, float16 b);

    friend bool operator!=(float16 a, float16 b);

    friend float operator+(float a, float16 b);
    friend float operator-(float a, float16 b);
    friend float operator*(float a, float16 b);
    friend float operator/(float a, float16 b);

    friend float   float16ToFP32(float16 v);

    friend float16 fp32ToFloat16(float v);

    operator double() const;

    operator float()  const;

    uint16_t get_val() { return val; }

    void set_val(uint16_t a) { this->val = a; }

    uint16_t val;
};

inline float abs(float16 val)
{
    return std::abs(float16ToFP32(val));
}

inline float sqrt(float16 val)
{
    return std::sqrt(float16ToFP32(val));
}

float float16ToFP32(float16 v);
void fp16_to_fp32(uint16_t inputUint, float *output);
static_assert(sizeof(float16) == sizeof(uint16_t), "reinterpret casting to float16 won't work");

#endif