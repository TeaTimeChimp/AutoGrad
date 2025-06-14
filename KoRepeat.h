#pragma once

#include "Kernel.h"


class KoRepeat : public Kernel
{
	const int	_dim;
	const int	_copies;

public:
				KoRepeat(const int dim,const int copies);
	NDArray		Forward(const NDArrays& input) override;
	NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) override;
};