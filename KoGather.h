#pragma once

#include "Kernel.h"


class KoGather : public Kernel
{
	const int		_dim;
	const NDArray	_indices;

public:
				KoGather(const int dim,const NDArray& indices);
	NDArray		Forward(const NDArrays& input) override;
	NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) override;
};