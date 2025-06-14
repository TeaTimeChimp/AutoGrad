#include "KoDot.h"


NDArray KoDot::Forward(const NDArrays& inputs)
{
	return inputs[0].Dot(inputs[1]);
}


NDArrays KoDot::Backward(const NDArray& gradient,const NDArrays& input)
{
	// Differential WRT each term is the other term, backprop [1].gradient to branch [0] and [0].gradient to branch [1].
	return
	{
		// Gradient WRT _creator[0] is _creator[1] chained with the incoming gradient.
		NDData::ReverseBroadcast(gradient.Dot(input[1].Transpose()),input[0].Shape()),

		// Gradient WRT _creator[1] is _creator[0] chained with the incoming gradient.
		NDData::ReverseBroadcast(input[0].Transpose().Dot(gradient),input[1].Shape())
	};
}