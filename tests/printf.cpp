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


#include "printf.hpp"

namespace test
{

///////////////////////////////////////////////////////////////////////////////////////////

int PrintfTensor::GetNumMessages()
{
    return ParsingEntireBuffer();
}

///////////////////////////////////////////////////////////////////////////////////////////

std::string PrintfTensor::GetMessageWithNumber(int num)
{
    int really_num_mess = ParsingEntireBuffer();

    num = std::min(num, really_num_mess - 1);
    if(num < 0)
    {
        std::string empty = "";
        return empty;
    }
    else
    {
        return m_parsedData[num].m_fullString;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////

std::string PrintfTensor::GetAllMessages()
{
    int really_num_mess = ParsingEntireBuffer();

    std::string all_messages = "";

    for(int n = 0; n < really_num_mess; n++)
    {
        all_messages = all_messages + m_parsedData[n].m_fullString;
    }

    return all_messages;
}

///////////////////////////////////////////////////////////////////////////////////////////

unsigned int PrintfTensor::GetImprint(int num)
{
    int really_num_mess = ParsingEntireBuffer();

    num = std::min(num, really_num_mess - 1);
    if(num < 0)
    {
        return 0;
    }
    else
    {
        return m_parsedData[num].m_imprint;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////

bool PrintfTensor::ValueIs(int num)
{
    int really_num_mess = ParsingEntireBuffer();

    num = std::min(num, really_num_mess - 1);
    if(num < 0)
    {
        return false;
    }
    else
    {
        return m_parsedData[num].m_val_is;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////

void PrintfTensor::PreparePrintVec()
{
    ((uint32_t*)Data())[0] = c_endOffStreamMagic;
}

///////////////////////////////////////////////////////////////////////////////////////////

std::string PrintfTensor::CreateRawMessage(unsigned int** ppRawData, unsigned int* pEndRawData)
{
    unsigned int * pRawData = * ppRawData;
    std::string message = "";
    while (*pRawData != c_endOffStreamMagic &&
           *pRawData != c_recordBeginMagic  &&
            pRawData < pEndRawData)
    {
        char* fourCharacters = reinterpret_cast<char*>(pRawData++);
        message += fourCharacters[0];
        message += fourCharacters[1];
        message += fourCharacters[2];
        message += fourCharacters[3];
    }

    *ppRawData  =  pRawData;  // modify external pointer

    return message;
}

///////////////////////////////////////////////////////////////////////////////////////////

int PrintfTensor::ParsingEntireBuffer(void)
{

    if((m_size32 == 0) || (Data() == nullptr))
    {
        assert(false && "An incorrect attempt to use the tensor without initializing it. ");
    }

    uint32_t * pRawData = (uint32_t*)Data();
    uint32_t * const pEndRawData = (uint32_t*)Data() + m_size32;
    m_parsedData.clear();

    while ((*pRawData != c_endOffStreamMagic) &&
           (*pRawData == c_recordBeginMagic)  &&
           ( pRawData < pEndRawData - 1))
    {

        int n;
        char buf[1024];
        bool val_is = false;
        unsigned int value = *++pRawData;

        std::string message = CreateRawMessage(&pRawData, pEndRawData); // in this place the pointer
                                                                        // pRawData is modified
        // determine whether there is a value in the message or only text
        if(message.find("%") != std::string::npos)
        {
            message = message.erase (0,4);
            val_is = true;
        }

        if (message.find("%f") != std::string::npos)
        {
            n = sprintf(buf, message.data(), *(float*)&value);
        }
        else
        {
            n = sprintf(buf, message.data(), value);
        }

        std::string fullString(buf, n);
        m_parsedData.push_back({fullString, message, value, val_is});
    }

    return (int)m_parsedData.size();
}

///////////////////////////////////////////////////////////////////////////////////////////


} // namespace test
