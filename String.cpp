#include "String.h"
#include <sstream>


using namespace std;


string to_string(const NDShape& v)
{
	stringstream ss;

	ss<<"[";

	bool first = true;
	for(auto i:v)
	{
		if(!first)
			ss<<",";
		ss<<i;
		first = false;
	}

	ss<<"]";

	return ss.str();
}


vector<string> Split(const string& str,const char sep)
{
	vector<string> splits;

	size_t offset = 0;
	while(offset<str.length())
	{
		size_t pos = str.find_first_of(sep,offset);
		if(pos==string::npos)
		{
			splits.emplace_back(str.substr(offset));
			offset = str.length();
		}
		else
		{
			splits.emplace_back(str.substr(offset,pos-offset));
			offset = pos+1;
		}
	}

	return splits;
}