#include "Tools.h"
#include <algorithm>
#include <sstream>
#include <filesystem>


using namespace std;


std::string to_string(const std::string& str)
{
	return str;
}


string to_string(const wstring& str)
{
	string out(str.length(),' ');
	transform(str.begin(),str.end(),out.begin(),[](wchar_t c){return c>=32&&c<=127?(unsigned char)c:' ';});
	return out;
}


wstring to_wstring(const string& str)
{
	wstring out(str.length(),' ');
	transform(str.begin(),str.end(),out.begin(),[](char c){return (wchar_t)c;});
	return out;
}


string to_lower(const string& str)
{
	string out(str);
	transform(str.begin(),str.end(),out.begin(),[](char c){return tolower(c);});
	return out;
}


wstring to_lower(const wstring& str)
{
	wstring out(str);
	transform(str.begin(),str.end(),out.begin(),[](wchar_t c){return tolower(c);});
	return out;
}

ostream& operator << (ostream& os,const wstring& str)
{
	os<<to_string(str);
	return os;
}


wstring Escape(wstringstream& ss,const wstring& str,const wchar_t qualifier)
{
	ss.str(L"");
	for(wchar_t c:str)
	{
		if(c==qualifier)
			ss<<qualifier;
		ss<<c;
	}
	return ss.str();
}


wstring Next(const wstring& line,size_t& start)
{
    const size_t end = line.find('\t',start);
    const wstring part = end==wstring::npos?line.substr(start):line.substr(start,end-start);
    start = end+1;
    return part;
}


string to_mbcstring(const wstring& wcstring)
{
	string mbcstring;
	mbcstring.reserve(wcstring.length());
	const wchar_t* data = wcstring.data();
	for(size_t i=0;i<wcstring.length();++i)
	{
		const unsigned int c = data[i];
		if(c<0x80)
			mbcstring.push_back((char)c);
		else if(c<0x800)
		{
			mbcstring.push_back((char)(0xc0|(c>>6)));
			mbcstring.push_back((char)(0x80|(c&0x3f)));

		}
		else if(c<0x10000)
		{
			mbcstring.push_back((char)(0xe0|(c>>12)));
			mbcstring.push_back((char)(0x80|((c>>6)&0x3f)));
			mbcstring.push_back((char)(0x80|(c&0x3f)));
		}
		else if(c<0x110000)
		{
			mbcstring.push_back((char)(0xf0|(c>>18)));
			mbcstring.push_back((char)(0x80|((c>>12)&0x3f)));
			mbcstring.push_back((char)(0x80|((c>>6)&0x3f)));
			mbcstring.push_back((char)(0x80|(c&0x3f)));
		}
		else
			throw "Character cannot by converted to UTF-8.";
	}
	return mbcstring;
}


wstring to_wcstring(const string& mbcstring)
{
	wstring wcstring;
	wcstring.reserve(mbcstring.length());
	size_t i = 0;
	const unsigned char* data = (const unsigned char*)mbcstring.data();
	while(i<mbcstring.length())
	{
		if((data[i]&0x80)==0x00)
		{
			// 1 byte [0xxxxxxx].
			wcstring.push_back(data[i]);
			++i;
		}
		else if((data[i]&0xe0)==0xc0)
		{
			// 2 byte [110xxxxx][10xxxxxx].
			wcstring.push_back(((data[i]&0x1f)<<6)|(data[i+1]&0x3f));
			i+=2;
		}
		else if((data[i]&0xf0)==0xe0)
		{
			// 3 bytes [1110xxxx][10xxxxxx][10xxxxxx].
			wcstring.push_back(((data[i]&0x0f)<<12)|((data[i+1]&0x3f)<<6)|(data[i+2]&0x3f));
			i+=3;
		}
		else if(sizeof(wchar_t)>2&&(data[i]&0xf8)==0xf0)	// Only supported when wchar_t is more than 2 bytes.
		{
			// 4 bytes [11110xxx][10xxxxxx][10xxxxxx][10xxxxxx].
			wcstring.push_back(((data[i]&0x07)<<18)|((data[i+1]&0x3f)<<12)|((data[i+2]&0x3f)<<6)|(data[i+3]&0x3f));
			i+=4;
		}
		else
			throw "Invalid UTF-8 encoding.";
	}
	return wcstring;
}


string AbsPath(const string& path)
{
	if(!path.empty())
		return filesystem::path(path).root_path().empty()?
			(filesystem::current_path()/filesystem::path(path)).string():
			path;
	else
		return path;
}