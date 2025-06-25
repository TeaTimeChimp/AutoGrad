#include "KoMean.h"


KoMean::KoMean(const bool keepDims) :
	_dim(-1),
	_keepDims(keepDims)
{
}


KoMean::KoMean(const int dim,const bool keepDims) :
	_dim(dim),
	_keepDims(keepDims)
{
}


NDArray	KoMean::Scalar_Forward(const NDArrays& inputs)
{
	return inputs[0].Mean(_keepDims);
}


NDArrays KoMean::Scalar_Backward(const NDArray& gradient,const NDArrays& inputs)
{
	return
	{
		// Gradient split equally.
		(inputs[0].Ones()/FP(inputs[0].Size()))*gradient
	};
}


NDArray	KoMean::OneDim_Forward(const NDArrays& inputs)
{
	return inputs[0].Mean(_dim,_keepDims);
}


NDArrays KoMean::OneDim_Backward(const NDArray& gradient,const NDArrays& inputs)
{
	return 
	{
		// Gradient split equally along dimension of aggregation.
		(inputs[0].Ones()/FP(inputs[0].Shape()[_dim]))*		// (1/count) *
			(_keepDims?gradient:gradient.Unsqueeze(_dim))	//		gradient.
	};
}


NDArray KoMean::Forward(const NDArrays& inputs)
{
	if(_dim==-1)
		return Scalar_Forward(inputs);
	else
		return OneDim_Forward(inputs);
}


NDArrays KoMean::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	if(_dim==-1)
		return Scalar_Backward(gradient,inputs);
	else
		return OneDim_Backward(gradient,inputs);
}