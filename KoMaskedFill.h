#pragma once

#include "Kernel.h"


class KoMaskedFill : public Kernel
{
	const NDArray	_mask;
	const FP		_value;

public:
				KoMaskedFill(const NDArray& mask,const FP value);
	NDArray		Forward(const NDArrays& input) override;
	NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) override;
};