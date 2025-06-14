#pragma once

#include "Tensor.h"


class Tensors
{
	int			_size;
	TensorPtr	_tensors[2];

public:
	Tensors(const TensorPtr& tensor) :
		_size(1)
	{
		_tensors[0] = tensor;
	}

	Tensors(const std::initializer_list<const TensorPtr> tensors) :
		_size(0)
	{
		if(tensors.size()>2)
			throw InvalidArgument();
		for(auto& tensor:tensors)
			_tensors[_size++] = tensor;
	}

	operator const TensorPtr& () const
	{
		if(_size>1)
			throw InvalidArgument();
		return _tensors[0];
	}

	const TensorPtr& operator [] (const int i) const
	{
		if(i>=_size)
			throw InvalidArgument();
		return _tensors[i];
	}

	// Pretend to be a single tensor for syntactic simplicity.
	//
	const TensorPtr& operator -> () const
	{
		if(_size>1)
			throw InvalidArgument();
		return _tensors[0];
	}

	// Get indexer used by auto[...] binding.
	//
	template <std::size_t Index>
	std::tuple_element_t<Index,Tensors>& get()
	{
		return _tensors[Index];
	}
};


// Class used by compiler to get the number of elements to bind to using auto[...]
// in the template class. In this case 'Tensors' is fix at 2 elements.
namespace std
{
  template<>
  struct tuple_size<::Tensors>
  {
    static constexpr size_t value = 2;
  };
}


// Class to return the 'type' of the element at the specified 'Index' position.
// See https://devblogs.microsoft.com/oldnewthing/20201015-00/?p=104369 for more details.
// Here all elements are the same type so whatever 'Index' is we define 'type' as TensorPtr.
//
namespace std
{
  template<size_t Index>
  struct tuple_element<Index,::Tensors>
  {
	  using type = TensorPtr;
  };
}
