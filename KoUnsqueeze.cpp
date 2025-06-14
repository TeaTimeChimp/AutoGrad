#include "KoUnsqueeze.h"


KoUnsqueeze::KoUnsqueeze(const int dim) :
	_dim(dim)
{
}


NDArray KoUnsqueeze::Forward(const NDArrays& inputs)
{
	return inputs[0].Unsqueeze(_dim);
}


NDArrays KoUnsqueeze::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	// No differentiation required, remove the extra dimension that was added to the shape.
	NDShape shape(gradient.Shape());
	shape.erase(shape.begin()+_dim);	// Should now be the same shape as the creator, so could have just used as a short	
	return
	{
		gradient.Reshape(shape)
	};
}