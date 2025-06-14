#include "KoSqueeze.h"


KoSqueeze::KoSqueeze(const int dim) :
	_dim(dim)
{
}


NDArray KoSqueeze::Forward(const NDArrays& inputs)
{
	// Remove dimension.
	NDShape shape = inputs[0].Shape();
	shape.erase(shape.begin()+_dim);
	return inputs[0].Reshape(shape);
}


NDArrays KoSqueeze::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	// No differentiation equired, add dimension that was removed from the shape.
	NDShape newShape(gradient.Shape());
	newShape.insert(newShape.begin()+_dim,1);
	return
	{
		gradient.Reshape(newShape)
	};
}