#pragma once

#include "Kernel.h"


class KoSqueeze : public Kernel
{
	const int _dim;

public:
				KoSqueeze(const int dim);
	NDArray		Forward(const NDArrays& input) override;
	NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) override;
};