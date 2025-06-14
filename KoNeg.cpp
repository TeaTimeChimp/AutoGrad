#include "KoNeg.h"


NDArray KoNeg::Forward(const NDArrays& inputs)
{
	return -inputs[0];
}


NDArrays KoNeg::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	return
	{
		// Negate the gradient and backprop.
		-gradient
	};
}