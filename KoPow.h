#pragma once

#include "Kernel.h"


class KoPow : public Kernel
{
	const FP	_exponent;

public:
				KoPow(const FP exponent);
	NDArray		Forward(const NDArrays& input) override;
	NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) override;
};