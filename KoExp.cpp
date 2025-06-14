#include "KoExp.h"


NDArray KoExp::Forward(const NDArrays& inputs)
{
	return inputs[0].Exp();
}


NDArrays KoExp::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	return
	{
		// Derivative of y=e^x is y'=e^x.
		gradient*inputs[0].Exp()
	};
}