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

#include "bfloat16.h"
#include "float16.h"

int fp_accommodate_rounding(uint32_t     intValuePreRounding,
                            bool         roundedMSB,
                            bool         roundedLSBs,
                            unsigned int sign,
                            int          roundingMode,
                            uint32_t     lfsrVal,
                            uint32_t     discardedAlignedLeft)
{
    uint32_t result = 0;
    result          = intValuePreRounding;
    switch (roundingMode) {
        case RND_TO_0: result = intValuePreRounding; break;
        case RND_TO_PINF:
            if ((sign == 0) && ((roundedMSB == 1) || (roundedLSBs == 1))) {
                result = intValuePreRounding + 1;
            }
            break;
        case RND_TO_NINF:
            if ((sign == 1) && ((roundedMSB == 1) || (roundedLSBs == 1))) {
                result = intValuePreRounding + 1;
            }
            break;
        case RND_HALF_AZ:
            if (roundedMSB == 1) // half or above half will be rounded away from zero
            {
                result = intValuePreRounding + 1;
            }
            break;
        case RND_SR:
            if (discardedAlignedLeft >= lfsrVal) {
                result = intValuePreRounding + 1;
            }
            break;
        case RND_TO_NE:
        default:
            if ((((intValuePreRounding & 0x1) == 1) && (roundedMSB == 1)) ||
                (((intValuePreRounding & 0x1) == 0) && (roundedMSB == 1) && (roundedLSBs == 1))) {
                result = intValuePreRounding + 1;
            }
            break;
    }
    return result;
}

uint16_t fp32_to_fp16(float input, int roundingMode)
{
    int      inputExponent, inputSign, unbiasedExp = 0;
    uint32_t inputMantissa;
    bool     roundedMSB = 0, roundedLSBs = 0;
    int      minExp     = -25;
    int      minNormExp = -14;
    int      maxExp     = 15;
    uint16_t output;

    uint32_t inputUint = *(uint32_t*)&input;

    inputMantissa = (inputUint & SIGNIFICAND_MASK_FP32);
    inputExponent = (inputUint & EXPONENT_MASK_FP32) >> EXPONENT_OFFSET_FP32;
    inputSign     = (inputUint & SIGN_MASK_FP32) >> SIGN_OFFSET_FP32;

    int32_t outputExponent;
    int32_t outputMantissa;
    int32_t outputSign = inputSign;
    if (is_nan_fp32(inputUint)) {
        // return the same NAN always (0x7FFF), as NVDA does
        outputSign     = 0x0;
        outputExponent = 0x1F;
        outputMantissa = 0x3FF;
    } else if (is_zero_fp32(inputUint)) {
        // return +-0
        outputExponent = 0x0;
        outputMantissa = 0x0;
    } else if (is_inf_fp32(inputUint)) {
        // return +-infinity
        outputExponent = 0x1F;
        outputMantissa = 0x0;
    } else {
        // Valid number
        unbiasedExp = inputExponent - EXPONENT_BIAS_FP32;
        inputMantissa |= (1 << EXPONENT_OFFSET_FP32);

        if (unbiasedExp > maxExp) {

            if ((roundingMode == RND_TO_0) || (inputSign && (roundingMode == RND_TO_PINF)) ||
                (!inputSign && (roundingMode == RND_TO_NINF)))

            { // +- 65504.0 - that's what NVDA does
                outputMantissa = 0x3FF;
                outputExponent = maxExp + EXPONENT_BIAS_FP16;
            } else { // +-infinity
                outputExponent = 0x1F;
                outputMantissa = 0x0;
            }
        } else if (unbiasedExp < minExp) {
            // The result will be either 0 or 0x1
            roundedMSB     = 0;
            roundedLSBs    = 1;
            outputMantissa = fp_accommodate_rounding(0, roundedMSB, roundedLSBs, inputSign, roundingMode, 0, 0);
            outputExponent = 0x0;
        } else { // minExp <= unbiasedExp <= maxExp
            outputExponent = unbiasedExp;
            int rc_bit_idx =
                (unbiasedExp < minNormExp) ? -(unbiasedExp + 2) : (EXPONENT_OFFSET_FP32 - EXPONENT_OFFSET_FP16 - 1);
            roundedMSB                    = (inputMantissa >> rc_bit_idx) & 0x1;
            roundedLSBs                   = (inputMantissa & ((1 << rc_bit_idx) - 1)) != 0;
            uint32_t discardedAlignedLeft = inputMantissa << (31 - rc_bit_idx);
            outputMantissa                = inputMantissa >> (rc_bit_idx + 1);
            outputMantissa                = fp_accommodate_rounding(
                outputMantissa, roundedMSB, roundedLSBs, inputSign, roundingMode, 0, discardedAlignedLeft);
            if (((unbiasedExp < minNormExp) && (outputMantissa & (1 << EXPONENT_OFFSET_FP16))) ||
                (outputMantissa & (1 << (EXPONENT_OFFSET_FP16 + 1)))) { // Should handle two cases:
                // 1. The number was denormal, and after rounding became normal
                // 2. The number was rounded to the 1.0 * 2^(next exponent)
                outputExponent = outputExponent + 1;
            }
            if (outputExponent > maxExp) {
                // return infinity
                outputExponent = 0x1F;
                outputMantissa = 0x0;
            } else {
                if (outputExponent < minNormExp) {
                    outputExponent = 0x0;
                } else {
                    outputExponent += EXPONENT_BIAS_FP16;
                }
                // normalize - leave 10 bits
                outputMantissa &= SIGNIFICAND_MASK_FP16;
            }
        }
    }
    output = outputMantissa | (outputExponent << EXPONENT_OFFSET_FP16) | (outputSign << SIGN_OFFSET_FP16);

    return output;
}

// Count the Number of Leading Zero Bits
int lzcnt(uint32_t bits, uint32_t int_num)
{
    int i;
    int msb = bits - 1;
    int lsb = 0;
    for (i = msb; i >= lsb; i--) {
        if ((int_num & (1 << i)) != 0) {
            break;
        }
    }
    return bits - i - 1;
}

void fp16_to_fp32(uint16_t inputUint, float *output)
{

    int32_t inputMantissa = (inputUint & SIGNIFICAND_MASK_FP16);
    int32_t inputExponent = (inputUint & EXPONENT_MASK_FP16) >> EXPONENT_OFFSET_FP16;
    int32_t inputSign     = (inputUint & SIGN_MASK_FP16) >> SIGN_OFFSET_FP16;
    uint32_t&       outputUint = *(uint32_t*)output;    

    int32_t outputExponent;
    int32_t outputMantissa;
    int32_t outputSign = inputSign;

    if (fp16_is_zero(inputUint)) {
        outputExponent = 0x0;
        outputMantissa = 0x0;
    } else if (fp16_is_nan(inputUint)) {
        outputExponent = 0xFF;
        outputMantissa = 0x007FFFFF;
        outputSign     = 0;
    } else if (fp16_is_infinity(inputUint)) {
        outputExponent = 0xFF;
        outputMantissa = 0x0;
    } else {
        outputExponent                = inputExponent - EXPONENT_BIAS_FP16 + EXPONENT_BIAS_FP32;
        int32_t mantissaForAdjustment = inputMantissa;
        if (fp16_is_denormal(inputUint)) {
            int shift = lzcnt(EXPONENT_OFFSET_FP16, inputMantissa);
            // Shift leading 1 to bit 10 (normalize) and fixup the exponent accordingly
            mantissaForAdjustment = (inputMantissa << (shift + 1)) & SIGNIFICAND_MASK_FP16;
            outputExponent -= shift;
        }
        // Normal case
        outputMantissa = mantissaForAdjustment << (EXPONENT_OFFSET_FP32 - EXPONENT_OFFSET_FP16);
    }

    outputUint = outputMantissa | outputExponent << EXPONENT_OFFSET_FP32 | outputSign << SIGN_OFFSET_FP32;

}

float float16ToFP32(float16 v)
{
    float retVal = 0.0;
    fp16_to_fp32(v.get_val(), &retVal);
    return retVal;
}

float16 fp32ToFloat16(float v)
{
    uint16_t val = fp32_to_fp16(v, RND_TO_NE);
    return float16(val);
}

float16::float16(float v)
{
    uint16_t value = fp32_to_fp16(v, RND_TO_NE);
    this->val = value;
}

float16::operator float() const
{
    return float16ToFP32(*this);
}

float16::operator double() const
{
    return double(float16ToFP32(*this));
}
