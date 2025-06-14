#include "KoDiv.h"


NDArray KoDiv::Forward(const NDArrays& inputs)
{
	return inputs[0]/inputs[1];
}


NDArrays KoDiv::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	return
	{
		// Differentiate WRT to each input and pass back using chain rule (division rewritten as multiplication by reciprocal or power of -1).
		NDData::ReverseBroadcast(gradient*inputs[1].Pow(-1),inputs[0].Shape()),					// y=x/c =x.c^-1, y'=c^-1
		NDData::ReverseBroadcast(gradient*-1.0*inputs[0]*inputs[1].Pow(-2),inputs[1].Shape())	// y=c/x =c.x^-1, y'=-1.c.x^-2
	};
}