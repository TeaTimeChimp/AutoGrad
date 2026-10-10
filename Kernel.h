#pragma once

#include "NDArray.h"


typedef std::shared_ptr<class Kernel> KernelPtr;

class Kernel
{
public:
	virtual NDArray		Forward(const NDArrays& input) = 0;
	virtual NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) = 0;

	// True if the kernel can work on device-resident arrays. Other kernels get their inputs synced to the host first.
	virtual bool		DeviceAware() const { return false; }
};