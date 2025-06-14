#pragma once

#include "Kernel.h"


class KoUnsqueeze : public Kernel
{
	const int _dim;

public:
				KoUnsqueeze(const int dim);
	NDArray		Forward(const NDArrays& input) override;
	NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) override;
};