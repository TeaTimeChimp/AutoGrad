#include "KoReshape.h"


KoReshape::KoReshape(const NDShape& shape) :
	_shape(shape)
{
}


NDArray KoReshape::Forward(const NDArrays& inputs)
{
	return inputs[0].Reshape(_shape);
}


NDArrays KoReshape::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	const NDShape& shape = inputs[0].Shape();
	return
	{
		// No differentiation required, restore original shape.
		gradient.Reshape(std::initializer_list(shape.data(),shape.data()+shape.size()))
	};
}