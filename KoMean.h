#pragma once

#include "Kernel.h"


class KoMean : public Kernel
{
	const int	_dim;
	const bool	_keepDims;

	NDArray		Scalar_Forward(const NDArrays& inputs);
	NDArrays	Scalar_Backward(const NDArray& gradient,const NDArrays& inputs);

	NDArray		OneDim_Forward(const NDArrays& inputs);
	NDArrays	OneDim_Backward(const NDArray& gradient,const NDArrays& inputs);

public:
				KoMean(const bool keepDims);
				KoMean(const int dim,const bool keepDims);
	NDArray		Forward(const NDArrays& inputs) override;
	NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) override;
};