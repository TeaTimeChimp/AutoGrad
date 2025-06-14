#pragma once

#include "String.h"
#include "NDShape.h"
#include <iostream>


class Exception
{
	std::string _msg;

public:
	Exception()
	{
	}
	Exception(const std::string& msg) :
		_msg(msg)
	{
		std::cout<<"Exception! "<<_msg<<std::endl;
	}
};

class FileNotFound
{
};


class IndexOutOfBounds
{
};


class InvalidDimension
{
};


class NotImplemented : public Exception
{
public:
	NotImplemented()
	{
	}
	NotImplemented(const std::string& msg) :
		Exception(msg)
	{
	}
};
class ProbabilitiesMustSumToOne
{
};

class RaggedVector
{
};

class UnexpectedCharacter
{
};

class InvalidRange
{
};

class InvalidSlice
{
};

class InvalidProbabilities
{
};

class IteratorAtEnd
{
};


class InvalidOperator : public Exception
{
public:
	InvalidOperator(const std::string& op) :
		Exception("Unknown creation operator '"+op+"'.")
	{
	}
};

class AlreadyPropagated
{
};


class IncompatibleShape : public Exception
{
public:
	IncompatibleShape()
	{
	}

	IncompatibleShape(const std::string& msg) :
		Exception(msg)
	{
	}

	IncompatibleShape(const NDShape& shape1,const NDShape& shape2) :
		Exception("Shape "+to_string(shape1)+" incompatible with "+to_string(shape2)+".")
	{
	}
};
class InvalidAxis
{
};
class MissingChild
{
};
class ChildNotSpecified
{
};


class InvalidArgument
{
};
