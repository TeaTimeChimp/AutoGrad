#pragma once

#include "Kernel.h"


class KoMax : public Kernel
{
	const int	_dim;		// Dimension to take the maximum over.
	NDArray		_argmax;	// Store the argmax for use in Backward.

public:
				KoMax(const int dim);
	NDArray		Forward(const NDArrays& inputs) override;
	NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) override;
};