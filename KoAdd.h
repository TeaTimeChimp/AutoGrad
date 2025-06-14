#pragma once

#include "Kernel.h"


class KoAdd : public Kernel
{
public:
	NDArray		Forward(const NDArrays& input) override;
	NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) override;
};