#include "KoMul.h"


NDArray KoMul::Forward(const NDArrays& inputs)
{
	return inputs[0]*inputs[1];
}


NDArrays KoMul::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	// Differentiate WRT to each creator.
	return
	{
		NDData::ReverseBroadcast(gradient*inputs[1],inputs[0].Shape()),
		NDData::ReverseBroadcast(gradient*inputs[0],inputs[1].Shape())
	};
}