#pragma once

#include "Kernel.h"


class KoDot : public Kernel
{
public:
	NDArray		Forward(const NDArrays& inputs) override;
	NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) override;
};