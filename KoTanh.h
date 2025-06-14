#pragma once

#include "Kernel.h"


class KoTanh : public Kernel
{
public:
	NDArray		Forward(const NDArrays& input) override;
	NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) override;
};