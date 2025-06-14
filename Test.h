#pragma once

#include "Tensor.h"


class TestFailed
{
	const char* _msg;

public:
	TestFailed(const char* msg) :
		_msg(msg)
	{
	}
};


void Assert(bool pass,const char* msg=nullptr);
void print(const std::string& str);
void print(const NDArray& data,const char* label=nullptr);
void print(const NDShape& v);
void print(const TensorPtr& data,const char* label=nullptr);

extern void Test();