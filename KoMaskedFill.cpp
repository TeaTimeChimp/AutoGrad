#include "KoMaskedFill.h"


KoMaskedFill::KoMaskedFill(const NDArray& mask,const FP value) :
	_mask(mask),
	_value(value)
{
}


NDArray KoMaskedFill::Forward(const NDArrays& inputs)
{
	return inputs[0].MaskedFill(_mask,_value);
}


NDArrays KoMaskedFill::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	return
	{
		// No differation required, masked gradients must be zeroed (achieved by multiplying by the inverse of the mask).
		gradient*(_mask==_mask.Zeros())
	};
}