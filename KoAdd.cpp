#include "KoAdd.h"


NDArray KoAdd::Forward(const NDArrays& inputs)
{
	return inputs[0]+inputs[1];
}


NDArrays KoAdd::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	return
	{
		// Differential WRT each term is 1, backprop 1*gradient to both branches.
		NDData::ReverseBroadcast(gradient,inputs[0].Shape()),
		NDData::ReverseBroadcast(gradient,inputs[1].Shape())
	};
}