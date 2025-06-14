#include "KoRelu.h"


NDArray KoRelu::Forward(const NDArrays& inputs)
{
	return (inputs[0]>0)*inputs[0];
}


NDArrays KoRelu::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	return
	{
		(inputs[0]>0)*gradient
	};
}