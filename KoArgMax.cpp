#include "KoArgMax.h"


KoArgMax::KoArgMax(const int dim) :
	_dim(dim)
{
}


NDArray KoArgMax::Forward(const NDArrays& inputs)
{
	return inputs[0].ArgMax(_dim);
}


NDArrays KoArgMax::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	throw NotImplemented();
}