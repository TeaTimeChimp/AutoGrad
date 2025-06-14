#include "KoSum.h"


KoSum::KoSum(const int dim,const bool keepDims) :
	_dim(dim),
	_keepDims(keepDims)
{
}


NDArray KoSum::Forward(const NDArrays& inputs)
{
	return inputs[0].Sum(_dim,_keepDims);
}


NDArrays KoSum::Backward(const NDArray& gradient,const NDArrays& inputs)
{	
	// Sum is just like multiple additions. The gradient WRT each term is 1, so just pass the gradient back using chain rule i.e. 1xg.
	return
	{
		(_keepDims?gradient:gradient.Unsqueeze(_dim))		// Optionally reinstate dimension that was removed.
			.Repeat_Numpy(_dim,inputs[0].Shape()[_dim])		// Expand gradient along the dimension the sum was performed.
	};
}