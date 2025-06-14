#pragma once


#include "Tools.h"
#include <iostream>
#include <filesystem>
#include <cstring>


class Utf8FileWriter
{
    const size_t    bufferSize = 65535;
    std::ofstream   _stream;
    char* const     _buffer;
    char*           _pos;

    std::ofstream CreateUtf8File(const std::string& filename)
    {
        // Open file.
        std::ofstream stream(filename,std::ios_base::out|std::ios_base::binary);
        if(!stream.is_open())
            throw "Can't open file '"+filename+"'.";

        // Write UTF-8 BOM.
        const unsigned char utf8bom[] = {0xef,0xbb,0xbf};
	    stream.write((char*)utf8bom,3);

        return stream;
    }

    void Flush()
    {
        if(_pos!=_buffer)
        {
            *_pos = '\0';
            _stream<<_buffer;
            _pos = _buffer;
        }
    }

    inline size_t BufferSpace() const
    {
        return bufferSize-(_pos-_buffer);
    }

    void Write(const char* str,const size_t length)
    {
        if(length<=BufferSpace())
        {
            // Copy to buffer.
            memcpy(_pos,str,length);
            _pos += length;
        }
        else
        {
            if(_pos!=_buffer)
            {
                // Flush and retry.
                Flush();
                Write(str,length);
            }
            else
            {
                // Too long for buffer.
                _stream<<str;
            }
        }
    }

    void Write(const std::string& str)
    {
        Write(str.data(),str.length());
    }

    void Write(const std::wstring& str)
    {
        Write(to_mbcstring(str));
    }

public:
    Utf8FileWriter(const std::string& filename) :
        _stream(CreateUtf8File(filename)),
        _buffer(new char[bufferSize+1]),
        _pos(_buffer)
    {
    }

    ~Utf8FileWriter()
    {
        if(_stream.is_open())
            Close();
        delete [] _buffer;
    }

    Utf8FileWriter& operator << (const std::string& text)
    {
        Write(text.data(),text.length());
        return *this;
    }

    Utf8FileWriter& operator << (const std::wstring& text)
    {
        Write(text);
        return *this;
    }

    Utf8FileWriter& operator << (const char* text)
    {
        Write(text,strlen(text));
        return *this;
    }

    Utf8FileWriter& operator << (const char c)
    {
        if(BufferSpace()==0)
            Flush();
        *_pos++ = c;
        return *this;
    }

    Utf8FileWriter& operator << (const int v)
    {
        (*this)<<std::to_string(v);
        return *this;
    }

    Utf8FileWriter& operator << (const int64_t v)
    {
        (*this)<<std::to_string(v);
        return *this;
    }

    Utf8FileWriter& operator << (const size_t v)
    {
        (*this)<<std::to_string(v);
        return *this;
    }

    Utf8FileWriter& operator << (const double v)
    {
        (*this)<<std::to_string(v);
        return *this;
    }

    void Close()
    {
        Flush();
        _stream.close();
    }
};


class Utf8FileReader
{
    std::ifstream               _stream;
    std::string                 _lineByteBuffer;
    int                         _lineBufferSize;
    std::unique_ptr<wchar_t[]>  _lineBuffer;

    std::ifstream OpenUtf8File(const std::string& filename)
    {
        // Open file.
        std::ifstream stream(filename,std::ios_base::in|std::ios_base::binary);
        if(!stream.is_open())
            throw "Can't open file '"+filename+"'.";

        // Read and validate UTF-8 BOM.
        const unsigned char utf8bom[] = {0xef,0xbb,0xbf};
        char bom[4];
	    stream.get(bom,4);
        if(memcmp(bom,utf8bom,sizeof(utf8bom)))
            throw "File '"+filename+"' has no UTF-8 BOM.";

        return stream;
    }

public:
    Utf8FileReader(const std::string& filename) :
        _stream(OpenUtf8File(filename)),
        _lineBufferSize(0)
    {
    }

    void operator >> (std::wstring& line)
    {
        _lineByteBuffer.clear();
        while(!_stream.eof())
        {
            const char c = (char)_stream.get();
            if(c==L'\n'||_stream.eof())
                break;
            if(c!=L'\r')
                _lineByteBuffer.push_back(c);
        }        
        line = to_wcstring(_lineByteBuffer);
    }

    void close()
    {
        _stream.close();
    }

    bool eof()
    {
        return _stream.eof();
    }
};


class Utf16FileReader
{
    std::ifstream _stream;

    std::ifstream OpenUtf16File(const std::string& filename)
    {
        // Open file.
        std::ifstream stream(filename,std::ios_base::in|std::ios_base::binary);
        if(!stream.is_open())
            throw "Can't open file '"+filename+"'.";

        // Read and validate UTF-16 BOM.
        const unsigned char utf16bom[] = {0xff,0xfe};
        char bom[3];
	    stream.get(bom,3);
        if(memcmp(bom,utf16bom,sizeof(utf16bom)))
            throw "File '"+filename+"' has no UTF-16 BOM.";

        return stream;
    }

public:
    Utf16FileReader(const std::string& filename) :
        _stream(OpenUtf16File(filename))
    {
    }

    bool IsOpen() const
    {
        return _stream.is_open();
    }

    bool eof() const
    {
        return _stream.eof();
    }

    void operator >> (std::wstring& line)
    {
        line.clear();
        while(!_stream.eof())
        {
            wchar_t c = 0;  // 4 bytes on Linux.
            ((char*)&c)[0] = (char)_stream.get();
            ((char*)&c)[1] = (char)_stream.get();
            if(c==L'\n'||_stream.eof())
                break;
            if(c!=L'\r')
                line.push_back(c);
        }
    }

    void Close()
    {
        _stream.close();
    }
};


class AsciiFileReader
{
    std::ifstream   _stream;
    std::string     _buffer;

    std::ifstream OpenFile(const std::string& filename)
    {
        // Open file.
        std::ifstream stream(filename,std::ios_base::in|std::ios_base::binary);
        if(!stream.is_open())
            throw "Can't open file '"+filename+"'.";

        // No BOM.

        return stream;
    }

public:
    AsciiFileReader(const std::string& filename) :
        _stream(OpenFile(filename))
    {
    }

    void operator >> (std::string& line)
    {
        _buffer.clear();
        while(!_stream.eof())
        {
            const char c = (char)_stream.get();
            if(c==L'\n'||_stream.eof())
                break;
            if(c!=L'\r')
                _buffer.push_back(c);
        }        
        line = _buffer;
    }

    operator std::string ()
    {
        _buffer.clear();
        while(!_stream.eof())
        {
            const char c = (char)_stream.get();
            if(_stream.eof())
                break;
            _buffer.push_back(c);
        }
        return _buffer;
    }

    void close()
    {
        _stream.close();
    }

    bool eof()
    {
        return _stream.eof();
    }
};
