#include "KoIndexSelect.h"


KoIndexSelect::KoIndexSelect(const NDArray& indices) :
	_indices(indices)
{
}


NDArray KoIndexSelect::Forward(const NDArrays& inputs)
{
	return inputs[0][_indices];
}


NDArrays KoIndexSelect::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	return
	{
		inputs[0].UnindexSelect(_indices,gradient)
	};
}