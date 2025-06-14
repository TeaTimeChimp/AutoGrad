#include "KoCat.h"


KoCat::KoCat(const int dim) :
	_dim(dim)
{
	_ASSERT(dim>=0);
}


NDArray KoCat::Forward(const NDArrays& inputs)
{
	// Concatenate arrays.
	return NDData::Cat(inputs,_dim);
}


NDArrays KoCat::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	// Split the gradient sending backward the corrensponding values.
	const int dimCount = inputs[0].Shape().size();

	// Build n-dimensional slicer with variable range in concatenation dimension.
	std::vector<int> catSlice(2);	// Range (begin,end).
	std::vector<std::initializer_list<int>> slices;
	for(int i=0;i<dimCount;++i)
	{
		if(i==_dim)
			slices.emplace_back(catSlice.data(),catSlice.data()+catSlice.size());
		else
			slices.emplace_back();
	}							
	const std::initializer_list<std::initializer_list<int>> slicer(slices.data(),slices.data()+slices.size());

	// Backprop slice of gradients to each creator.
	NDArrays gradients;
	for(auto& input:inputs)
	{
		catSlice[1] = catSlice[0]+input.Shape()[_dim];		// End of slice for gradient passed to creator.
		gradients.emplace_back(gradient.Slice(slicer));		// Add gradient slice to return vector.
		catSlice[0] = catSlice[1];							// Next creator slice after this slice.
	}
	return gradients;
}