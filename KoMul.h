#pragma once

#include "Kernel.h"


class KoMul : public Kernel
{
public:
	NDArray		Forward(const NDArrays& inputs) override;
	NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) override;
};