#pragma once

#include "Kernel.h"


class KoCat : public Kernel
{
	const int _dim;

public:
				KoCat(const int dim);
	NDArray		Forward(const NDArrays& inputs) override;
	NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) override;
};