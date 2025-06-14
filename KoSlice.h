#pragma once

#include "Kernel.h"


class KoSlice : public Kernel
{
	const std::initializer_list<std::initializer_list<int>>	_slicer;

public:
				KoSlice(const std::initializer_list<std::initializer_list<int>>& slicer);
	NDArray		Forward(const NDArrays& input) override;
	NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) override;
};