#include "KoLog.h"


NDArray KoLog::Forward(const NDArrays& inputs)
{
	return inputs[0].Log();
}


NDArrays KoLog::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	return
	{
		// Derivative of ln(x) is 1/x. Differentiate WRT creator and pass backward using chain rule.
		gradient*(inputs[0].Ones()/inputs[0])
	};
}