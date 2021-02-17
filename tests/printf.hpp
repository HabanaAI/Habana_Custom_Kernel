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

#ifndef PRINTF_HPP
#define PRINTF_HPP

#include <vector>
#include <string>
#include <boost/algorithm/string/replace.hpp>
#include "tensor.h"


namespace test
{


class PrintfTensor  : public test::Tensor<int32_t, 2>
{

public:

    PrintfTensor() : test::Tensor<int32_t, 2>()
    {
        m_parsedData.clear();
    };

    PrintfTensor(const unsigned sizeBytes) : test::Tensor<int32_t, 2>()
    {
        m_size32 = sizeBytes / sizeof(int32_t);
        Init({1, m_size32}, /*data*/ NULL , /*padValue*/ 0);
        m_parsedData.clear();
        PreparePrintVec();
    };

    virtual ~PrintfTensor(){};

    int GetNumMessages();
    std::string GetMessageWithNumber(int num);
    std::string GetAllMessages();
    unsigned int GetImprint(int num);
    bool ValueIs(int num);

private:

    struct ParsedData
    {
        std::string m_fullString;     // message with value
        std::string m_string;         // message only
        unsigned int m_imprint;       // value only
        bool m_val_is;                // false: message is sole text, true: message has a numeric value
    };

    const unsigned int c_endOffStreamMagic = 0xffffffff; // 4294967295
    const unsigned int c_recordBeginMagic  = 0xcdcdcdcd; // 3452816845
    std::vector<ParsedData> m_parsedData = {};
    unsigned int m_size32 = 0;         // size of the buffer allocated for printing in 32-bit words

    void PreparePrintVec();
    std::string CreateRawMessage(unsigned int** ppRawData, unsigned int* pEndRawData);
    int ParsingEntireBuffer(void);

};

} // name space test

#endif //PRINTF_HPP
