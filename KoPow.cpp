#include "KoPow.h"


KoPow::KoPow(const FP exponent) :
	_exponent(exponent)
{
}


NDArray KoPow::Forward(const NDArrays& inputs)
{
	return inputs[0].Pow(_exponent);
}


NDArrays KoPow::Backward(const NDArray& gradient,const NDArrays& inputs)
{	
	return
	{
		// Power rule WRT creator n.x^y --> y.n.x^(y-1).
		gradient*(inputs[0].Pow(_exponent-1)*_exponent)
	};
}