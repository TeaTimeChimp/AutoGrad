#include "KoMax.h"


KoMax::KoMax(const int dim) :
	_dim(dim)
{
}


NDArray KoMax::Forward(const NDArrays& inputs)
{
	_argmax._Attach(inputs[0].ArgMax(_dim));		// Store the argmax for use in Backward.
	NDArray max_a = inputs[0].Gather(_dim,_argmax);	// Gather the maximum values along the specified dimension.
	NDArray max_b = inputs[0].Max(_dim);
	if(!max_a.IsEqualTo(max_b))
	{
		// If the max values are not equal, this is a bug.
		throw IncompatibleShape(max_a.Shape(),max_b.Shape());
	}
	return max_a;
}


NDArrays KoMax::Backward(const NDArray& gradient,const NDArrays& inputs)
{	
	_ASSERT_EXPR(inputs.size()==1,"KoMax::Backward: inputs.size() must be 1");	
	return
	{
		// No differentiation required, pass gradient where the input was the maximum.
		inputs[0].Scatter(_dim,_argmax,gradient)
	};
}