#include "KoGather.h"


KoGather::KoGather(const int dim,const NDArray& indices) :
	_dim(dim),
	_indices(indices)

{
	// Only tested for dim==1.
	if(_dim!=1)
		throw NotImplemented();
}


NDArray KoGather::Forward(const NDArrays& inputs)
{
	return inputs[0].Gather(_dim,_indices);			// Gather indexed value along dimension.
}


NDArrays KoGather::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	return
	{
		inputs[0].Scatter(_dim,_indices,gradient)	// Scatter gradients over input.
	};
}