#pragma once

#include "NDArray.h"


typedef std::shared_ptr<class Kernel> KernelPtr;

class Kernel
{
public:
	virtual NDArray		Forward(const NDArrays& input) = 0;
	virtual NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) = 0;
};