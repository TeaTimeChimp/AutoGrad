#pragma once

#include "Kernel.h"


class KoReshape : public Kernel
{
	const NDShape _shape;

public:
				KoReshape(const NDShape& shape);
	NDArray		Forward(const NDArrays& input) override;
	NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) override;
};