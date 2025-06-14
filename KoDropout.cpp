#include "KoDropout.h"


KoDropout::KoDropout(const FP p) :
	_p(p)
{
}


NDArray KoDropout::Forward(const NDArrays& inputs)
{
	_dropout._Attach(inputs[0].Ones().Dropout(_p));	// Dropout on ones gives the scaled multiplier for each element or zero.
	return inputs[0]*_dropout;						// Multiply by dropout to zero dropped value and amplify remaining values.
}


NDArrays KoDropout::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	return
	{
		// No differentiation required, pass gradient scaled by dropout.
		gradient*_dropout
	};
}