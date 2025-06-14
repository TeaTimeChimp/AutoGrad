#pragma once

#include "Kernel.h"


class KoIndexSelect : public Kernel
{
	const NDArray	_indices;

public:
				KoIndexSelect(const NDArray& indices);
	NDArray		Forward(const NDArrays& input) override;
	NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) override;
};