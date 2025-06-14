#pragma once

#include "Kernel.h"


class KoArgMax : public Kernel
{
	const int _dim;

public:
				KoArgMax(const int dim);
	NDArray		Forward(const NDArrays& input) override;
	NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) override;
};