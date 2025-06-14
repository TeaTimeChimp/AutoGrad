#pragma once

#include <string>
#include <fstream>


std::string to_string(const std::string& str);
std::string to_string(const std::wstring& str);
std::wstring to_wstring(const std::string& str);
std::string to_lower(const std::string& str);
std::wstring to_lower(const std::wstring& str);
std::string to_mbcstring(const std::wstring& wcstring);
std::wstring to_wcstring(const std::string& mbcstring);

std::ostream& operator << (std::ostream&,const std::wstring& str);

std::wstring Next(const std::wstring& line,size_t& start);

std::wstring Escape(std::wstringstream& ss,const std::wstring& str,const wchar_t qualifier);

std::string AbsPath(const std::string& path);

#ifndef _WIN32

#define UNREFERENCED_PARAMETER(x) do { (void)(x); } while (0)
#define _ASSERT(x) if(!(x)) throw "Assert!";
#define _MAX_U64TOSTR_BASE16_COUNT (16 + 1)
#define BYTE unsigned char

class _com_error
{
public:
	const std::wstring ErrorMessage() const
	{
		return L"COM Exception.";
	}
};

#endif


#define BITSET(mem,offset)	(*(((BYTE*)(mem))+((offset)>>3)) & (BYTE)(1<<((offset)&7)))
#define SETBIT(mem,offset)	(*(((BYTE*)(mem))+((offset)>>3)) |= (BYTE)(1<<((offset)&7)))
