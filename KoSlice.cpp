#include "KoSlice.h"


KoSlice::KoSlice(const std::initializer_list<std::initializer_list<int>>& slicer) :
	_slicer(slicer)
{
}


NDArray KoSlice::Forward(const NDArrays& inputs)
{
	return inputs[0].Slice(_slicer);
}


NDArrays KoSlice::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	UNREFERENCED_PARAMETER(gradient);
	UNREFERENCED_PARAMETER(inputs);
	throw NotImplemented();
}