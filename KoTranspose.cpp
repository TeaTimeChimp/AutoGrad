#include "KoTranspose.h"


NDArray KoTranspose::Forward(const NDArrays& inputs)
{
	return inputs[0].Transpose();
}


NDArrays KoTranspose::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	return
	{
		// No differentiation required, last two dimensions must be transposed.
		gradient.Transpose()
	};
}