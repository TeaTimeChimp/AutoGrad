#include "KoTanh.h"


NDArray KoTanh::Forward(const NDArrays& inputs)
{
	return inputs[0].Tanh();
}


NDArrays KoTanh::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	return
	{
		// Differental of tanh(x) WRT x is 1-tanh(x)^2, multiply this by the gradient and backprop.
		(gradient.Ones()-(inputs[0]*inputs[0]))*gradient
	};
}