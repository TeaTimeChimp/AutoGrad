#include "KoRepeat.h"


KoRepeat::KoRepeat(const int dim,const int copies) :
	_dim(dim),
	_copies(copies)
{
}


NDArray KoRepeat::Forward(const NDArrays& inputs)
{
	return inputs[0].Repeat_Numpy(_dim,_copies);
}


NDArrays KoRepeat::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	return
	{
		// No differentiation required, sum the gradient along the dimension that was expanded and backprop.
		gradient.Sum(_dim,true)		// KeepDim because Expand does not add a dimension.
	};
}