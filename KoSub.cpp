#include "KoSub.h"


NDArray KoSub::Forward(const NDArrays& inputs)
{
	return inputs[0]-inputs[1];
}


NDArrays KoSub::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	return
	{
		// Differentiate WRT each creator.
		NDData::ReverseBroadcast(gradient,inputs[0].Shape()),
		NDData::ReverseBroadcast(-gradient,inputs[1].Shape())
	};
}