#pragma once

#include "Kernel.h"


class KoSum : public Kernel
{
	const int	_dim;
	const bool	_keepDims;

public:
				KoSum(const int dim,const bool keepDims);
	NDArray		Forward(const NDArrays& input) override;
	NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) override;
};