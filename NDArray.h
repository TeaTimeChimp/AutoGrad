#pragma once

#include "Random.h"
#include "Exceptions.h"
#include "NDAllocator.h"
#include "NDThreadPool.h"
#include "TiledMatMul.h"
#include <ppl.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <numeric>
#include "String.h"
#include "Files.h"
#include <ranges>
#include <algorithm>
#include <execution>


// Terminology
// ===========
// 
// Scalar			=	 1		= {}
//
//						|1|
// Column Vector	=	|2|		= {j}
//						|3|
//
// Row Vector		=	|1 2 3|	= {1,j}
//
//
//						|1 4|
// Matrix			=	|2 5|	= {i,j}
//						|3 6|
//

// Shape
// =====
// Shape is a list of dimension sizes {i,j} where dimension significance is left-to-right/most-to-least, the same as digits in a number.
// The '_shape' vector is big-endian because an initialiser-list pushes values in left-to-right onto the vector.
//
// Memory Layout
// =============
// Elements of the least significant dimension are placed next to one another. Further dimensions repeat less significant dimensions.
// More significant dimensions can be used to address self-contained instances of lower dimensional shapes.
// 
//  E.g.	In a 3 dimensional array of shape (2,3,4) a raw pointer can be generated for a Batch 'n' with shape (3,4), or a Timestep 'm' within Batch 'n' 
//			with shape (4).
//			The array is placed in memory from left-to-right and top-to-bottom. In this case, the values shown are also the address offsets.
// 
//		[						Batch 1
//			[  0,  1,  2,  4 ]		Timestep 1
//			[  5,  6,  7,  8 ]		Timestep 2
//			[  9, 10, 11, 12 ]		Timestep 3
//		]
//		[						Batch 2
//			[ 13, 14, 15, 16 ]		Timestep 1
//			[ 17, 18, 19, 20 ]		Timestep 2
// 			[ 21, 22, 23, 24 ]		Timestep 3
//		]
//


class NDData;
typedef std::shared_ptr<NDData>			NDDataPtr;
typedef std::shared_ptr<const NDData>	NDDataPtrC;


// Reference class that can be used with overloaded operators so that the implementation of Tensor looks cleaner.
//
class NDArray
{
	friend class NDData;	// Allows NDData to take NDArray as parameter and have full access.

	NDDataPtr _data;

	NDData* operator->() const
	{
		return _data.get();
	}

public:
	NDArray()
	{
	}

	inline NDArray(const FP v);

	NDArray(const NDDataPtr& data):
		_data(data)
	{
	}

	NDData& operator*() const
	{
		return *_data;
	}

	inline void				operator=(const NDArray& v);
	inline NDArray			operator==(const NDArray& v) const;
	inline NDArray			operator!=(const NDArray& v) const;
	inline NDArray			operator+(const NDArray& v) const;
	inline void				operator+=(const NDArray& v);
	inline NDArray			operator-() const;
	inline NDArray			operator-(const NDArray& v) const;
	inline void				operator-=(const NDArray& v) const;
	inline NDArray			operator*(const NDArray& v) const;
	inline void				operator*=(const NDArray& v);
	inline NDArray			operator/(const NDArray& v) const;
	inline NDArray			operator>(const FP v) const;
	inline FP&				operator[](const std::initializer_list<int>& indices);
	inline const FP&		operator[](const std::initializer_list<int>& indices) const;
	inline NDArray			operator[](const NDArray& indices) const;

	inline void				_Attach(const NDArray& v);
	inline void				_Reset();

	inline int				ArgMax() const;
	inline NDArray			ArgMax(const int dim) const;
	inline void				_ClipNorm(const FP clipNorm) const;
	inline NDArray			Dot(const NDArray& v) const;
	inline NDArray			Dropout(const FP p) const;
	inline NDArray			Entropy() const;
	inline NDArray			Exp() const;
	inline NDArray			Flatten() const;
	inline NDArray			Gather(const int dim,const NDArray& indices) const;
	inline bool				IsEqualTo(const NDArray& other) const;
	inline bool				IsScalar() const;
	inline NDArray			Log() const;
	inline NDArray			LogSumExp() const;
	inline NDArray			MaskedFill(const NDArray& mask,const FP value) const;
	inline NDArray			Max(const int dim) const;
	inline NDArray			Mean(const bool keepdims) const;
	inline NDArray			Mean(const int dim,const bool keepDims) const;
	inline NDArray			Ones() const;
	inline NDArray			Pow(const FP v) const;
	inline NDArray			Repeat_Numpy(const int dim,const int copies) const;
	inline NDArray			Repeat_Torch(const std::initializer_list<int>& sizes) const;
	inline NDArray			Reshape(const std::initializer_list<int>& shape) const;
	inline NDArray			Reshape(const NDShape& shape) const;
	inline void				Save(const std::string& filename) const;
	inline NDArray			Scatter(const int dim,const NDArray& indices,const NDArray& source) const;
	inline int				Size() const;
	inline const NDShape&	Shape() const;
	inline NDArray			Slice(const std::initializer_list<std::initializer_list<int>>& slices);
	inline const NDArray	Slice(const std::initializer_list<std::initializer_list<int>>& slices) const;
	inline NDArray			Softmax(const int dim) const;
	inline NDArray			Sqrt() const;
	inline NDArray			StdDev() const;
	inline NDArray			Sum() const;
	inline NDArray			Sum(const int dim,const bool keepDims) const;
	inline NDArray			Tanh() const;
	inline NDArray			Transpose() const;
	inline NDArray			Tril() const;
	inline NDArray			UnindexSelect(const NDArray& indices,const NDArray& source) const;
	inline NDArray			Unsqueeze(const int dim) const;
	inline NDArray			Var() const;
	inline NDArray			Var(const int dim,const bool keepDims) const;
	inline NDArray			Zeros() const;

	inline void				Print(std::ostream& out) const;
};
typedef std::vector<NDArray> NDArrays;


class NDData : public std::enable_shared_from_this<NDData>
{
	NDShape				_shape;					// Shape {i,j,k} most significant dimension at position 0 due to initialisation-list construction.
	NDShape				_stride;				// Pointer increment required for next value in each dimension.
	int					_size;					// Total number of elements in the array.
	FP*					_data;					// Pointer to memory containing array elements - see InitialiseSizeAndStride for formatting.
	
	const NDDataPtrC	_parent;				// Ponter to parent if data is not owned (i.e. this is a view).

	// Shape iterator - iterates every dimension value combination.
	//
	class NDIterator
	{
	protected:
		const NDShape&				_shape;
		NDShape						_offsets;
		bool						_end;
		std::initializer_list<int>	_slices[NDShape::MaxDims];	// Slice for each dimension (wraps _offset).

	public:
		NDIterator(const NDShape& shape) :
			_shape(shape),
			_offsets(shape.size()),
			_end(false)
		{
			for(int i=0;i<_offsets.size();++i)
				_slices[i] = std::initializer_list<int>(_offsets.data()+i,_offsets.data()+i+1);
		}

		// Increments the position of the iterator.
		//
		inline void operator++()
		{
			if(_end)
				throw IteratorAtEnd();

			int dim = _offsets.size()-1;
			while(dim>=0)
			{
				// Increment offset dimension.
				++_offsets[dim];

				// Exit if there's no overflow.
				if(_offsets[dim]<_shape[dim])
					return;

				// Carry.
				_offsets[dim] = 0;
				--dim;
			}

			// Reached end.
			_end = true;
		}

		inline bool end() const
		{
			return _end;
		}

		// Returns the iterator position in a form compatible with the [] operator.
		//
		inline operator std::initializer_list<int>() const
		{
			return std::initializer_list<int>(_offsets.begin(),_offsets.end());
		}

		// Returns the iterator position in a form compatible with the Slice method.
		//
		inline operator const std::initializer_list<std::initializer_list<int>> () const
		{
			return std::initializer_list<std::initializer_list<int>>(_slices,_slices+_offsets.size());
		}
	};

	// Data iterator - iterates every data value.
	//
	class NDIterator2
	{
		const NDShape&				_shape;
		const NDShape&				_stride;
		NDShape						_offsets;
		std::initializer_list<int>	_slices[NDShape::MaxDims];	// Slice for each dimension (wraps _offset).
		FP*							_data;
		int							_remaining;

		int ComputeSize() const
		{
			// Compute end pointer based on shape and stride.
			int size = 1;
			for(int i=0;i<_shape.size();++i)
				size *= _shape[i];
			return size;
		}

	public:
		NDIterator2(const NDData& data) :
			_shape(data._shape),
			_stride(data._stride),
			_offsets(_shape.size()),
			_data(data._data),
			_remaining(ComputeSize())
		{
			for(int i=0;i<_offsets.size();++i)
				_slices[i] = std::initializer_list<int>(_offsets.data()+i,_offsets.data()+i+1);
		}

		NDIterator2(const NDShape& shape,const NDShape& stride,FP* const data) :
			_shape(shape),
			_stride(stride),
			_offsets(_shape.size()),
			_data(data),
			_remaining(ComputeSize())
		{
			for(int i=0;i<_offsets.size();++i)
				_slices[i] = std::initializer_list<int>(_offsets.data()+i,_offsets.data()+i+1);
		}

		// Increments the position of the iterator.
		//
		inline void operator++()
		{
			// When remaining hits zero, there is no more valid data so exit (else the offsets would carry on each dimension, not harmful, just no point).
			if(--_remaining==0)
				return;

			int dim = _offsets.size()-1;
			do
			{
				// Increment offset dimension.
				++_offsets[dim];
				_data += _stride[dim];

				// Exit if there's no overflow.
				if(_offsets[dim]<_shape[dim])
					return;

				// Carry.
				_offsets[dim] = 0;
				_data -= _shape[dim]*_stride[dim];
				--dim;
			} while(dim>=0);

			// Will only get here if _remaining has gone negative.
			throw IteratorAtEnd();
		}

		constexpr bool end() const
		{
			return 0;
		}

		inline bool operator != (const int pos) const
		{
			return _remaining!=pos;
		}

		inline FP& operator* ()
		{
			return *_data;
		}

		// Returns the iterator position in a form compatible with the [] operator.
		//
		inline operator std::initializer_list<int>() const
		{
			return std::initializer_list<int>(_offsets.begin(),_offsets.end());
		}

		// Returns the iterator position in a form compatible with the Slice method.
		//
		inline operator const std::initializer_list<std::initializer_list<int>> () const
		{
			return std::initializer_list<std::initializer_list<int>>(_slices,_slices+_offsets.size());
		}
	};

	template<typename F>
	inline void Aggregate(NDArray& dst,const int dim,const F& foo) const
	{		
		// Isolate shapes and strides of non-aggregate dimensions.
		NDShape srcShape(_shape.size()-1);
		NDShape srcStride(_stride.size()-1);
		NDShape dstShape(dst->_shape.size()-1);
		NDShape dstStride(dst->_stride.size()-1);
		for(int i=0,j=0;i<_shape.size();++i)
		{
			_ASSERT(_shape[i]==dst->_shape[i]||i==dim&&dst->_shape[i]==1);
			if(i!=dim)
			{
				srcShape[j] = _shape[i];
				srcStride[j] = _stride[i];
				dstShape[j] = dst->_shape[i];
				dstStride[j] = dst->_stride[i];
				++j;
			}
		}

		// Iterate over non-fixed dimensions.
		const int srcDimStride = _stride[dim];
		const int dstDimStride = dst->_stride[dim];
		const int dimLength = _shape[dim];
		for(NDIterator2 srcIter(srcShape,srcStride,_data),dstIter(dstShape,dstStride,dst->_data);srcIter!=srcIter.end();++srcIter,++dstIter)
		{
			// Iterate the aggregate dimension.
			const FP* srcData = &*srcIter;
			FP acc = 0.0;
			for(int i=0;i<dimLength;++i)
			{
				foo(acc,*srcData);
				srcData += srcDimStride;
			}
			*dstIter = acc;
		}
	}

	// Execute the lambda for every data value.
	//
	template<typename F>
	inline void Iterate(const NDData& data,F& foo)
	{
		const NDShape&	shape(data.Shape());
		const NDShape&	stride(data._stride);
		NDShape			offsets;
		FP*				data(data._data);

		// Iterate over the dimensions.
		int dim = offsets.size()-1;
		while(dim>=0)
		{
			// Iterate over the current dimension.
			while(offsets[dim]<shape[dim])
			{
				// Execute on data.				
				foo(data);

				// Increment offset dimension.
				++offsets[dim];
				data += stride[dim];
			}

			// Carry.
			offsets[dim] = 0;
			data -= shape[dim]*stride[dim];
			--dim;
		}
	}

	NDIterator2 Iter() const
	{
		return NDIterator2(*this);
	}

	// Computes overall 'size' and 'place value' of each dimension in the multi dimensional array.
	//
	// Memory format:
	// Values are organised such that values next to each other in the least significant dimension (_shape[-1]) are next to each other in memory.
	// This is the most logical layout for specifying lists in code using {...}, for debugging, reshaping, and appears to match Pytorch.
	// 
	int InitialiseSizeAndStride()
	{
		_ASSERT(_stride.size()==0);	// Stride gets wiped so would loose any repetition and transposition.

		// Iterate dimensions computing the place value (cummulative multiple of previous dimensions).
		// NOTE: Left-most dimension values are contiguous so in a 2D matrix row values are contiguous - grouped by sample and same as debug window layout.
		if(_shape.size()==0)
		{
			// Scalar.
			_size = 1;
		}
		else
		{
			// n-dimensional.

			// Reserve 1 place value per dimension.
			_stride.resize(_shape.size());

			// Last dimension is least significant and values are contiguous.
			int dim = _shape.size()-1;
			_size = _shape[dim];
			_stride[dim] = 1;
			while(dim>0)
			{
				--dim;
				_size *= _shape[dim];
				_stride[dim] = _stride[dim+1]*_shape[dim+1];
			}
		}
		return _size;
	}

	// Returns 'true' if there are no repeated dimensions where the stride is zero.
	//
	bool HasRepeatedDimensions() const
	{
		for(int i=0;i<_stride.size()-1;++i)
		{
			if(_stride[i]==0)
				return true;
		}
		return false;
	}

	// Returns 'true' if the stride is in natural order (more significant dimensions have a larger stride).
	//
	bool HasNaturalStride() const
	{
		// Only if this is a view is it possible the stride is unnatural.
		if(_parent)
		{
			int stride = 1;
			for(int i=_stride.size()-1;i>=0;--i)
			{
				if(_stride[i]!=stride)
					return false;
				stride *= _shape[i];
			}
		}
		return true;
	}

	// Check for divergence.
	void DebugRangeCheck() const
	{
#ifdef _DEBUG
		static bool thrown = false;
		if(thrown)
			return;
		const FP threshold = (FP)1.0e20;

		try
		{
			if(_parent)
			{
				for(NDIterator iter=_shape;!iter.end();++iter)
				{
					if(isnan((*this)[iter]))
						throw "NAN";
					if((*this)[iter]<-threshold||(*this)[iter]>threshold)
						throw "Too big?";
				}
			}
			else
			{
				const FP* const data = _data;
				for(int i=0;i<_size;++i)
				{
					if(isinf(data[i]))
						continue;
					if(isnan(data[i]))
						throw "NAN";
					if(data[i]<-threshold||data[i]>threshold)
						throw "Out of range.";
				}
			}
		}
		catch(...)
		{
			thrown = true;
			throw;
		}
#endif
	}

	void _Print(std::ostream& out,const NDShape& outerIndices) const
	{
		if(_shape.size()==0)
		{
			// Scalar.
			out<<operator[]({});
		}
		else
		{
			NDShape indices(outerIndices);
			indices.emplace_back(0);

			if(_shape.size()-indices.size()==0)
			{
				// Inner-most dimension.
				out<<"[";
				for(int i=0;i<_shape[outerIndices.size()];++i)
				{
					if(i>0)
						out<<",";
					*indices.rbegin()=i;
					out<<operator[](std::initializer_list<int>(indices.data(),indices.data()+indices.size()));
				}
				out<<"]";
			}
			else
			{
				// Outer dimension.
				out<<"[";
				for(int i=0;i<_shape[outerIndices.size()];++i)
				{
					if(i>0)
						out<<","<<std::endl;
					*indices.rbegin()=i;
					_Print(out,indices);
				}
				out<<"]";
			}
		}
	}

	// Template method for applying the inplace operation 'op' to each element of the arrays.
	//
	template<typename OP>
	inline void Elementwise(const NDDataPtrC& v,const OP& op)
	{
		const NDArray v_ = v->Broadcast(Self());
		if(v_->_shape!=_shape)
			throw IncompatibleShape();

		if(HasNaturalStride())
		{
			if(v_->HasNaturalStride())
			{
				// No iterator.
				FP* const dst = _data;
				const FP* src = v_->_data;
				for(int i=0;i<_size;++i)
					op(dst[i],src[i]);
			}
			else
			{
				// Source iterator.
				FP* dst = _data;
				for(NDIterator2 src=v_->Iter();src!=src.end();++dst,++src)
					op(*dst,*src);
			}
		}
		else
		{
			if(v_->HasNaturalStride())
			{
				// Destination iterator.
				const FP* src = v_->_data;
				for(NDIterator2 dst=Iter();dst!=dst.end();++dst,++src)
					op(*dst,*src);
			}
			else
			{
				// Source and destination iterator.
				for(NDIterator2 dst=Iter(),src=v_->Iter();dst!=dst.end();++dst,++src)
					op(*dst,*src);
			}
		}
		DebugRangeCheck();
	}

	static void print(const NDArray& a)
	{
		a.Print(std::cout);
		std::cout<<std::endl;
	}

	static void print(const char* a)
	{
		std::cout<<a<<std::endl;
	}

	// Private class required by constructors.
	//
	class P
	{
	};

public:	// Constructors must be public to use make_shared.

	// Deep-copy-constructor.
	//
	NDData(const P&,const NDData& data) :
		_shape(data._shape),
		_data(mem.Alloc<FP>(InitialiseSizeAndStride()))
	{
		if(data.HasNaturalStride())
		{
			// Use fast memory copy.
			memcpy(_data,data._data,data._size*sizeof(FP));
		}
		else
		{
			// Use iterator for source.
			FP* cell = _data;
			for(NDIterator2 iter=data.Iter();iter!=iter.end();++iter)
				*cell++ = *iter;
		}
	}

	// View constructor.
	//
	NDData(const P&,const NDDataPtrC& parent) :
		_shape(parent->_shape),
		_stride(parent->_stride),
		_size(parent->_size),
		_data(parent->_data),
		_parent(parent->DataOwner())
	{
		_ASSERT(!_parent->_parent);
	}

	// Iterator shape and uninitialised data.
	//
	template<class iter>
	NDData(const P&,const iter& begin,const iter& end) :
		_shape(begin,end),
		_data(mem.Alloc<FP>(InitialiseSizeAndStride()))
	{
	}

	// Iterator shape and default value.
	//
	template<class iter>
	NDData(const P&,const iter& begin,const iter& end,const FP data) :
		_shape(begin,end),
		_data(mem.Alloc<FP>(InitialiseSizeAndStride()))
	{
		for(int i=0;i<_size;++i)
			_data[i] = data;
	}

	// Iterator shape and data list.
	//
	template<class siter,class diter>
	NDData(const P&,const siter& sbegin,const siter& send,const diter& dbegin,const diter& dend) :
		_shape(sbegin,send),
		_data(mem.Alloc<FP>(InitialiseSizeAndStride()))
	{	
		if(_size!=dend-dbegin)
			throw IncompatibleShape();

		// Cast/copy data.
		FP* p = _data;
		for(diter d=dbegin;d!=dend;++d)
			*p++ = FP(*d);
	}

	// List shape and data vector.
	//
	NDData(const P&,const std::initializer_list<int>& shape,const std::vector<FP>& data) :
		_shape(shape),
		_data(mem.Alloc<FP>(InitialiseSizeAndStride()))
	{
		if(_size!=data.size())
			throw IncompatibleShape();

		// Cast/copy data.
		FP* p = _data;
		for(auto d:data)
			*p++ = d;
	}

	NDDataPtrC DataOwner() const
	{
		if(_parent)
			return _parent->DataOwner();
		else
			return Self();
	}

	// Slice constructor.
	//
	NDData(const P&,const NDDataPtr& parent,const std::initializer_list<std::initializer_list<int>>& slicer) :
		_parent(parent->DataOwner()),
		_size(1),
		_data(parent->_data)
	{
		_ASSERT(!_parent->_parent);

		// Slicer cannot have more dimensions than the array being sliced.
		if(slicer.size()>parent->Shape().size())
			throw InvalidSlice();

		// Each slice of the slicer describes what to take from a dimension, starting with the most significant dimension.
		auto slice = slicer.begin();
		for(int i=0;i<slicer.size();++i)
		{
			auto j = slice->begin();
			switch(slice->size())
			{
				case 0:
				{
					// Full range.
					_shape.emplace_back(parent->_shape[i]);
					_stride.emplace_back(parent->_stride[i]);
					break;
				}
				case 1:
				{
					// 'begin' only, single value, dimension removed.
					int begin = *j;
					if(begin<0)
						begin += parent->_shape[i];
					_data += begin*parent->_stride[i];
					break;
				}
				case 2:
				{
					// 'begin' to 'end' range.
					int begin = *j;
					if(begin<0) // Relative to end of dimension.
					{
						begin += parent->_shape[i];
						if(begin<0)
							begin = 0;	// Slice before beginning, bound to beginning.
					}
					++j;
					int end = *j;
					if(end<=0)
						end += parent->_shape[i];
					_shape.emplace_back(end-begin);
					_stride.emplace_back(parent->_stride[i]);
					_data += begin*parent->_stride[i];
					break;
				}
				case 3:
				{
					// 'begin' to 'end' with 'step'.
					int begin = *j;
					if(begin<0)
						begin += parent->_shape[i];
					++i;
					int end = *j;
					if(end<=0)
						end += parent->_shape[i];
					_shape.emplace_back(end-begin);
					++i;
					int step = *j;
					_stride.emplace_back(parent->_stride[i]*step);
					_data += begin*parent->_stride[i];
					break;
				}
				default:
				{
					throw InvalidSlice();
				}
			}

			// Next dimension slice.
			++slice;
		}

		// Add full range of any remaining unspecified trailing parent dimensions.
		for(int i=(int)slicer.size();i<(int)parent->_shape.size();++i)
		{
			_shape.emplace_back(parent->_shape[i]);
			_stride.emplace_back(parent->_stride[i]);
		}

		// Compute '_size'.
		for(auto length:_shape)
			_size *= length;
	}


	// Reshape constructor.
	//
	NDData(const P&,const NDDataPtr& parent,const NDShape& shape) :
		_shape(shape),
		_size(0),
		_data(parent->_data),
		_parent(parent)
	{
		InitialiseSizeAndStride();
		if(_size!=parent->_size)
			throw InvalidSlice();
	}

	~NDData()
	{
		if(!_parent)
		{
			mem.Free(_data);
			_data = nullptr;
		}
	}

public:
	// List shape and uninitialized data.
	//
	static NDArray New(const std::initializer_list<int>& shape)
	{
		return std::make_shared<NDData>(P(),shape.begin(),shape.end());
	}

	// Vector shape and uninitialised data - is this needed?
	//
	static NDArray New(const NDShape& shape)
	{
		return std::make_shared<NDData>(P(),shape.begin(),shape.end());
	}

	// Vector shape and default value.
	//
	static NDArray New(const NDShape& shape,const FP data)
	{
		return std::make_shared<NDData>(P(),shape.begin(),shape.end(),data);
	}

	// List shape and default value.
	//
	static NDArray New(const std::initializer_list<int>& shape,const FP data)
	{
		return std::make_shared<NDData>(P(),shape.begin(),shape.end(),data);
	}

	// List shape, vector of values.
	//
	static NDArray New(const std::initializer_list<int>& shape,const std::vector<FP>& data)
	{
		return std::make_shared<NDData>(P(),shape.begin(),shape.end(),data.begin(),data.end());
	}

	// Vector shape, vector of values.
	//
	static NDArray New(const std::vector<int>& shape,const std::vector<FP>& data)
	{
		return std::make_shared<NDData>(P(),shape.begin(),shape.end(),data.begin(),data.end());
	}

	// Deep-copy constructor.
	//
	static NDArray New(const NDData& data)
	{
		return std::make_shared<NDData>(P(),data);
	}

	// View constructor.
	//
	static NDArray New(const NDDataPtrC& data)
	{
		return std::make_shared<NDData>(P(),data);
	}

	// Slice constructor.
	//
	static NDArray New(const NDData& data,const std::initializer_list<std::initializer_list<int>>& slicer)
	{
		// TODO: Eliminate const-cast.
		return std::make_shared<NDData>(P(),const_cast<NDData&>(data).shared_from_this(),slicer);
	}

	// Reshape constructor.
	//
	static NDArray New(const NDData& data,const NDShape& shape)
	{
		if(data.HasNaturalStride())
		{
			// TODO: Eliminate const-cast.
			return std::make_shared<NDData>(P(),const_cast<NDData&>(data).shared_from_this(),shape);
		}
		else
		{
			// Deep-copy and reshape.
			return New(data).Reshape(shape);
		}
	}

	// Constructor for a sequence of integers.
	static NDArray Arrange(const int size)
	{
		std::vector<FP> sequence(size);
		std::iota(sequence.begin(),sequence.end(),FP(0));
		return NDData::New({size},sequence);
	}

	// Returns a new array initialised with random values sampled from a standard normal distribution.
	//
	static NDArray RandN(const std::initializer_list<int>& shape)
	{
		NDArray r = NDData::New(shape);

		std::normal_distribution<FP> distribution(0,1);	// Mean=0, Standard Deviation=1.
		FP* const rData = r->_data;
		for(int i=0;i<r->_size;++i)
			rData[i] = distribution(rnd.Generator());

		return r;
	}

	// Constructor for arrays filled with zeros.
	static NDArray Zeros(const std::initializer_list<int>& shape)
	{
		return NDData::New(shape,0.0);
	}

	// Constructor for arrays filled with ones.
	static NDArray Ones(const std::initializer_list<int>& shape)
	{
		return NDData::New(shape,1.0);
	}

	// Constructor for an array created by concatenating multiple arrays.
	//
	static NDArray Cat(const NDArrays& arrays,int dim)
	{
		// All tensors must be the same shape (except in the concatenation dimension) or be empty.
		const NDShape shape = arrays[0]->_shape;
		if(dim<0)
			dim = (int)shape.size()+dim;
		if(dim>=shape.size())
			throw InvalidDimension();

		// Find length of concatenation dimension and check array compatibility.
		int catDimLength = 0;
		for(auto& array:arrays)
		{
			// All arrays must have the same number of dimensions.
			if(array->_shape.size()!=shape.size())
				throw IncompatibleShape();

			// Each array must have the same dimension lengths - except for the concatenation dimension.
			for(int i=0;i<shape.size();++i)
			{
				if(i!=dim&&array->_shape[i]!=shape[i])
					throw IncompatibleShape();
			}

			// Sum length.
			catDimLength += array->_shape[dim];
		}

		// Create result array.
		NDShape newShape(shape);
		newShape[dim] = catDimLength;
		NDArray r = NDData::New(newShape);

		// Create a initialiser_list with an initialiser_list for each dimension, and where the dimension of along concatenation is addressable.
		std::vector<std::initializer_list<int>> slices;
		std::vector<int> catSlice(2);
		for(int i=0;i<shape.size();++i)
		{
			if(i==dim)
				slices.emplace_back(std::initializer_list<int>(catSlice.data(),catSlice.data()+catSlice.size()));
			else
				slices.emplace_back(std::initializer_list<int>());
		}
		const std::initializer_list<std::initializer_list<int>> slicer(slices.data(),slices.data()+slices.size());
		for(auto& array:arrays)
		{
			catSlice[1] = catSlice[0]+array->_shape[dim];	// Set end of slice to assign.
			r->Slice(slicer) = array;
			catSlice[0] = catSlice[1];						// Next slice begins after the slice just assigned.
		}

		return r;
	}

	// Attempts to return 'this' array with the same shape as the 'target' array.
	// 
	// If the two arrays are different shapes then the following steps are performed.
	// 
	//  1) If 'this' has fewer dimensions, dimensions of size 1 are added to the left of _shape.
	// 
	//  2) From right-to-left dimension of size 1 are replicated to match 'target'.
	// 
	// If 'target' has less dimensions, or any dimension post expansion does not match then broadcasting fails
	// and a version of 'this' returned with the shape at the point of failure.
	// 
	// NOTE: At present copies of 'this' array are created when the shape is modified.
	//
	const NDArray Broadcast(const NDDataPtrC& target)
	{
		// If the arrays are the same shape nothing needs doing.
		const NDShape& shape = Shape();
		const NDShape& targetShape = target->Shape();
		if(targetShape==shape)
			return Self();

		// If this array is higher dimensional than the other then broadcasting is not possible.
		if(shape.size()>targetShape.size())
			return Self();

		// Create a new view of the shared read-only data for broadcasting.
		const NDDataPtr data = NDData::New(NDDataPtrC(shared_from_this()))._data;

		// Add missing dimension(s) to the left.
		while(data->_shape.size()<targetShape.size())
		{
			data->_shape.insert(data->_shape.begin(),1);	// New dimension with length 1.
			data->_stride.insert(data->_stride.begin(),0);	// Could be 1 or 0, but use 0 to indicate this is a fabricated dimension.
		}

		// Expand unit length dimensions to match the target.
		for(int dim=(int)data->_shape.size()-1;dim>=0;--dim)
		{
			if(data->_shape[dim]==1&&targetShape[dim]>1)
			{
				data->_shape[dim] = targetShape[dim];		// Force length to required expanded length.
				data->_stride[dim] = 0;						// Force stride to zero so values are repeated for each element of this dimension.
			}
		}

		// Return broadcast data (even if it does not match the target because this may be the first step).
		return data;
	}

	const NDArray Broadcast(const NDDataPtrC& target) const
	{
		return const_cast<NDData*>(this)->Broadcast(target);
	}

	// Broadcasts a and b together.
	// 
	// Returns a copy of 'a' with a shape compatible with the returned 'b'.
	// 'b' is copied only if 'b' was broadcast.
	//
	static std::tuple<NDArray,NDArray> Broadcast(const NDDataPtrC& a,const NDArray& b)
	{
		// Create a copy of 'a' with a shape compatible with 'b'.
		const NDArray a_ = a->Broadcast(b._data);
		const NDArray ab = (a->_data==a_->_data)?NDData::New(*a_):NDArray(const_cast<NDData&>(*a_).shared_from_this());

		// Return 'b' compatible with 'ab'.
		const NDArray ba = b->Broadcast(ab._data);

		// If successfully broadcast, shape of ab and ba will be identical.
		if(ab->Shape()!=ba->Shape())
			throw IncompatibleShape();

		// Return copy of 'a' and copy/reference to 'b' with identical shape.
		return std::tuple(ab,ba);
	}

	static const NDArray ReverseBroadcast(const NDArray& gradient,const NDShape& creatorShape)
	{
		NDDataPtr shapedGradient = gradient._data;
		if(shapedGradient->_shape!=creatorShape)
		{
			// Sum and remove added dimnsions.
			while(shapedGradient->_shape.size()>creatorShape.size())
				shapedGradient = shapedGradient->Sum(0,false)._data;

			// Sum expanded dimensions.
			for(int i=0;i<creatorShape.size();++i)
			{
				if(creatorShape[i]<shapedGradient->_shape[i])
				{
					_ASSERT(creatorShape[i]==1);
					shapedGradient = shapedGradient->Sum(i,true)._data;
				}
			}
		}
		return shapedGradient;
	}

	// Elementwise assignment to 'this' from 'v'.
	//
	void Assign(const NDArray& v)
	{
		if(_shape!=v->_shape)
			throw IncompatibleShape();

		if(_stride==v->_stride)
		{
			// Memory layout is identical.
			memcpy(_data,v->_data,_size*sizeof(FP));
		}
		else
		{
			// Assign each value in the shape - there's in implicit transposition.
			for(NDIterator2 i=Iter(),j=v->Iter();i!=i.end();++i,++j)
				*i = *j;
		}
	}

	// Returns a non-const shared pointer this.
	//
	inline NDDataPtr Self()
	{
		return shared_from_this();
	}

	// Returns a const shared pointer this.
	//
	inline NDDataPtrC Self() const
	{
		return shared_from_this();
	}

	// Returns shape of array. Most significant dimension [0] to least significant dimension [n-1].
	//
	inline const NDShape& Shape() const
	{
		return _shape;
	}

	// Returns number of values in the array.
	//
	inline int Size() const
	{
		return _size;
	}

	// Returns a non-const pointer to the data element specified by indices.
	//
	FP* const DataPtr(const std::initializer_list<int> indices)
	{
		// Missing trailing indices are assumed to be zero - this makes it easier when working with a batch iterator that only has batch dimensions.
		if(indices.size()>_shape.size())
			throw IncompatibleShape();

		size_t offset = 0;
		int d = 0;
		for(auto i:indices)
		{
			_ASSERT(i<_shape[d]);
			offset += i*_stride[d++];
		}

		return _data+offset;
	}

	// Returns a const pointer to the data element specified by indices.
	//
	const FP* const DataPtr(const std::initializer_list<int> indices) const
	{
		return const_cast<NDData*>(this)->DataPtr(indices);
	}

	// Arbitrary dimension tuple accessor.
	//
	FP& operator [] (const std::initializer_list<int>& indices)
	{
		return *const_cast<FP*>(DataPtr(indices));
	}

	// Arbitrary dimension tuple accessor.
	//
	FP operator [] (const std::initializer_list<int>& indices) const
	{
		return *DataPtr(indices);
	}

	// Returns a slice of the array.
	//
	NDArray Slice(const std::initializer_list<std::initializer_list<int>>& sliceIndicies)
	{
		// Return slice.
		return NDData::New(*this,sliceIndicies);
	}

	// Returns a slice of the array.
	//
	const NDArray Slice(const std::initializer_list<std::initializer_list<int>>& sliceIndicies) const
	{
		// Return slice.
		return NDData::New(*this,sliceIndicies);
	}

	// Returns index of the largest value in a 1-by-n array.
	//
	int ArgMax() const
	{
		_ASSERT(HasNaturalStride());

		// Only implemented for 1-by-n vector.
		if(_shape[0]!=1)
			throw IncompatibleShape();

		// Scan 1 dimensional data for largest value and latch index.
		int mj = 0;
		for(int j=1;j<_size;++j)
			if(_data[j]>_data[mj])
				mj = j;

		return mj;
	}

	// Returns the index of the maximum value along the specified dimension.
	// 
	NDArray ArgMax(const int dim) const
	{
		// NOTE: The following code was copied from Mean as a starting point for ArgMax.
		// Added because I needed Max for adding a correction bias to Softmax that was numerically unstable when using fp32.
		_ASSERT(dim>=0&&dim<_shape.size());

		// Create the new shape and result data where the aggregation dimension is 1.
		NDShape newShape(_shape);
		newShape[dim] = 1;
		NDArray r = NDData::New(newShape);

		// Iterate reduced shape.
		const int dimSize = _shape[dim];
		for(NDIterator2 iter=r->Iter();iter!=iter.end();++iter)
		{
			// ArgMax along the aggregation dimension.
			NDShape indices(iter);
			const std::initializer_list<int> index(indices.data(),indices.data()+indices.size());
					
			// Assume first value.
			int max_i = 0;
			FP max_v = (*this)[index];

			// Iterate remaining values.
			for(int i=0;i<dimSize;++i)
			{
				indices[dim] = i;
				const FP v = (*this)[index];
				if(v>max_v)
				{
					max_v = v;
					max_i = i;
				}
			}

			// Store index of max value.
			*iter = FP(max_i);
		}

		const bool keepDims = true;
		if(!keepDims)
		{
			newShape.erase(newShape.begin()+dim);
			return r.Reshape(newShape);
		}
		else
			return r;
	}

	// Elementwise inplace addition.
	//
	void _Add(const NDArray& v)
	{
		Elementwise(v._data,[](FP& a,const FP b)
		{
			a += b;
		});
	}

	// Elementwise addition.
	//
	NDArray Add(const NDArray& v) const
	{
		// Broadcast to create result.
		auto [r,v_] = Broadcast(Self(),v);

		// Inplace operation on result.
		r._data->_Add(v_);

		return r;
	}

	// Clips values relative to the norm (vector length).
	void _ClipNorm(const FP clipNorm)
	{
		_ASSERT(HasNaturalStride());

		// Scale values if the array exceeds the clipping value.
		const FP norm = Norm();
		if(clipNorm<norm)
		{
			FP* const rData = _data;
			for(int i=0;i<_size;++i)
				rData[i] = rData[i]*clipNorm/norm;
		}
	}

	// Returns an array with random values set to zero with probability p.
	//
	NDArray Dropout(const FP p)
	{
		_ASSERT(HasNaturalStride());

		NDArray r = NDData::New(_shape,0.0);

		const FP scale = 1/(1-p);

		std::uniform_real_distribution<FP> distribution(0.0,1.0);
		const FP* const data = _data;
		FP* const rData = r->_data;
		for(int i=0;i<r->_size;++i)
		{
			if(distribution(rnd.Generator())>=p)
				rData[i] = data[i]*scale;
		}

		return r;
	}

	// Softmax over dimension 'dim'.
	// 
	//	softmax(x) = e^x/sum(e^x).
	//
	NDArray Softmax(int dim) const
	{
		// Need code here to subtract max for better numerical stability - don't want large positive numbers going in.
		NDArray temp = Exp();				// e^z for each value.
		return temp/temp.Sum(dim,true);		// e^z/sum(e^z) for each row/sample.		
	}

	// Elementwise inplace division.
	//
	void _Div(const NDArray& v)
	{
		Elementwise(v._data,[](FP& a,const FP b)
		{
			a /= b;
		});
	}

	// Elementwise division.
	//
	NDArray Div(const NDArray& v) const
	{
		NDArray r = NDData::New(*this);
		r->_Div(v);		
		return r;
	}


	// Matrix multiplication.
	//
	// Sum of row*column a(ij).b(xy)=c(iy) where j=x...
	//
	//  a11 a12   b11 b12   a11*b11+a12*b21 a11*b12+a12*b22
	//  a21 a22 . b21 b22 = a21*b11+a22*b21 a12*b12+a22*b22
	//
	NDArray MatMul(const NDArray& v) const
	{
		// Both arrays must be 2D.
		if(_shape.size()!=2||v->_shape.size()!=2)
			throw IncompatibleShape();

		// Cache matrix dimensions.
		const bool aRowMajor = _stride[1]==1;	// Make a copy if not row major, so copying tiles from matrix is cache friendly.
		const NDArray a = aRowMajor?NDArray(const_cast<NDData*>(this)->shared_from_this()):NDArray(NDData::New(*this));
		const int aRows = a->_shape[0];
		const int aCols = a->_shape[1];
		const int aRowStride = a->_stride[0];
		const int aColStride = a->_stride[1];
		_ASSERT(aColStride==1);
		const int bRows = v->_shape[0];
		const int bCols = v->_shape[1];
		const int bRowStride = v->_stride[0];
		const int bColStride = v->_stride[1];

		// LHS columns must equal RHS rows.
		if(aCols!=bRows)
			throw IncompatibleShape();

		// Expect at least one stride to be 1.
		if(aRowStride!=1&&aColStride!=1||bRowStride!=1&&bColStride!=1)
			throw IncompatibleShape();

		// Create output shape.
		NDShape shape(_shape);
		shape[1] = bCols;
		NDArray c = NDData::New(shape,0.0);
		const int cRows = c->_shape[0];
		const int cCols = c->_shape[1];
		const int cRowStride = c->_stride[0];
		const int cColStride = c->_stride[1];
		_ASSERT(cColStride==1);

		// Dimensions of the output matrix in tile
		const int tile_rows = ((aRows-1)/tile_size)+1;
		const int tile_cols = ((bCols-1)/tile_size)+1;
		const int inner_tiles = ((aCols-1)/tile_size)+1;

		// Transpose the RHS matrix so that the tiles are aligned for multiplication.
		const bool bColMajor = bRowStride==1;	// Make a copy if not column major, so copying tiles from matrix is cache friendly.
		const NDArray bT = bColMajor?v->Transpose():NDArray(NDData::New(*(v->Transpose())));
		const int bTRows = bT->_shape[0];
		const int bTCols = bT->_shape[1];
		const int bTRowStride = bT->_stride[0];
		const int bTColStride = bT->_stride[1];
		_ASSERT(bTColStride==1);

		{// Scoped TaskGroup.
			NDThreadPool::TaskGroup taskGroup;

			// Iterate over the tiles of the output matrix.		
			for(int tile_row=0;tile_row<tile_rows;++tile_row)
			{
				for(int tile_col=0;tile_col<tile_cols;++tile_col)
				{
					// Compute the output tile - run independent output tile as task - better parallelism than parallel for.
					taskGroup.Run([
						&a,aRows,aCols,aRowStride,
						&bT,bTRows,bTCols,bTRowStride,
						&c,cRows,cCols,cRowStride,
						tile_row,tile_col,inner_tiles
					]()
					{
						// Reserve tile buffers per thread - these are held in L1 cache.
						alignas(64) float x[tile_size][tile_size];
						alignas(64) float y[tile_size][tile_size];
						alignas(64) float z[tile_size][tile_size];
						for(int p=0;p<inner_tiles;++p)
						{
							// Copy p-th inner tile from tile row a.
							copy_to_tile(
								&x[0][0],											// Destination tile matrix.
								a->_data,aRows,aCols,aRowStride,						// Source matrix description.
								tile_row*tile_size,p*tile_size);					// Source tile row and column.

							// Copy p-th inner tile from tile row bT.
							copy_to_tile(
								&y[0][0],											// Destination tile matrix.
								bT->_data,bTRows,bTCols,bTRowStride,				// Source matrix description.
								tile_col*tile_size,p*tile_size);					// Source tile row and column.

							// Multiply the two tiles a@bT.
							matmul_tile(&x[0][0],&y[0][0],&z[0][0]);

							// Add the result tile to the output matrix.
							add_from_tile(
								&z[0][0],											// Source tile matrix (row major).
								c->_data,cRows,cCols,cRowStride,					// Destination matrix decription (row major).
								tile_row*tile_size,tile_col*tile_size);				// Destination tile row and column.
						}
					});
				}
			}
		} // Waits for TaskGroup.

		c->DebugRangeCheck();
		return c;
	}


	// Dot product for n-dimensional arrays.
	// 
	//		A(i,...,k) @ B(k,j) = C(i,...,j)
	// 
	//	Thus, a unique copy of the matrix (i,j) exist for each internal 'batch' dimension. For example, (batch,timesteps,channels) @ (channels,outputs) = (batch,timesteps,outputs).
	//  Each sample in the batch, and each timestep is treated as an dependant matrix multiplication with the weights.
	//
	NDArray Dot(const NDArray& v) const
	{
		if(_shape.size()<2||v->_shape.size()<2)
			throw NotImplemented();

		// If the RHS has more than 2 dimensions then perform a batch matrix multiply.
		// This does a 2D matrix multiply for each batch dimension.
		if(v->_shape.size()>2)
		{
			// Only implemented for 1 batch dimension.
			if(_shape.size()>3||v->_shape.size()>3)
				throw NotImplemented();

			// Broadcast 'this' to v - ensuring the shape has the same size and batch dimensions match (batch,i,k)@(batch,k,j).
			const NDArray self = Broadcast(v._data);
			//self->Print(std::cout);
			
			// Create result shape - all dimensions from self and last dimension from v.
			NDShape shape(self->_shape);
			shape[shape.size()-1] = v->_shape[v->_shape.size()-1];

			// Allocate uninitialised result array.
			NDArray r(NDData::New(shape));

			// Multiply the corresponding pair of 2D arrays in the batch.
			//for(int b=0;b<v->_shape[0];++b)
			//const auto rng = std::views::iota(0,v->_shape[0]);
			//std::for_each(std::execution::par_unseq,rng.begin(),rng.end(),[&v,&self,&r](const int b)
			//concurrency::affinity_partitioner ap;
			//concurrency::parallel_for(0,v->_shape[0],[&v,&self,&r](const int b)
			NDThreadPool::ForEach(0,v->_shape[0],[&v,&self,&r](const int b)
			{
				/*
				const NDArrayC vb = v.Slice({{b}});			// 2D slice of v.
				print("vb:");
				print(vb);
				const NDArrayC sb = self->Slice({{b}});		// 2D slice of self.
				print("sb:");
				print(sb);
				const NDArrayC v = sb->MatMul(vb);			// 2D @ 2D.
				print("=");
				print(v);
				r->Slice({{b}}) = v;						// Assign 2D result.
				*/
				r->Slice({{b}}) = self->Slice({{b}})->MatMul(v.Slice({{b}}));
			});/*,concurrency::static_partitioner());*/
			return r;
		}
		else
		{
			// RHS is 2D.

			// If LHS is >2D then batch dimensions can be combined to create a 2D matrix.
			// E.g. (B,T,C) @ (C,H) reshape to (B*T,C) @ (C,H) = (B*T,H) reshape to result (B,T,H).
			if(_shape.size()>2)
			{
				// Compute length of new row dimension.
				int i = _shape[0];
				for(int n=1;n<_shape.size()-1;++n)
					i *= _shape[n];

				// Reshape as 2D array.
				NDArray lhs = Reshape({i,_shape[_shape.size()-1]});

				// 2D @ 2D multiply.
				NDArray r = lhs->MatMul(v);

				// Reshape result.
				NDShape shape(_shape);
				shape[shape.size()-1] = v->_shape[v->_shape.size()-1];
				return r->Reshape(shape);
			}
			else
			{
				// 2D @ 2D matrix multiply.
				return MatMul(v);
			}
		}
	}

	// Entropy.
	//
	NDArray Entropy() const
	{
		_ASSERT(HasNaturalStride());

		switch(_shape.size())
		{
			case 2:
			{
				const int rows = _shape[0];
				const int cols = _shape[1];
				NDArray r = NDData::New({rows});
				for(int i=0;i<rows;++i)
				{
					FP entropy = 0.0;
					for(int j=0;j<cols;++j)
					{
						const FP p = (*this)[{i,j}];
						entropy += -p*std::log(p);
					}
					r[{i}] = entropy;
				}
				return r;
			}
		}
		throw IncompatibleShape();
	}

	// Elementwise equality, returns boolean array of same shape.
	//
	NDArray Equal(const NDArray& v) const
	{
		_ASSERT(HasNaturalStride());

		if(_shape!=v.Shape())
			throw IncompatibleShape();

		NDArray r = NDData::New(_shape);
		const FP* const data = _data;
		const FP* const vData = v->_data;
		FP* const rData = r->_data;
		for(int i=0;i<_size;++i)
			rData[i] = data[i]==vData[i];
		return r;
	}

	// Inplace elementwise exponential.
	//
	void _Exp()
	{
		_ASSERT(HasNaturalStride());

		FP* const data = _data;
		for(int i=0;i<_size;++i)
			data[i] = exp(data[i]);

		DebugRangeCheck();
	}

	// Elementwise exponential.
	//
	NDArray Exp() const
	{
		NDArray r = NDData::New(*this);
		r->_Exp();
		return r;
	}

	// Identity matrix factory.
	//
	static NDArray Eye(const int size)
	{
		NDArray r = NDData::New({size,size},0.0);
		for(int i=0;i<size;++i)
			r[{i,i}] = 1.0;
		return r;
	}

	// Flatten to n rows and 1 column.
	//
	NDArray Flatten() const
	{
		_ASSERT(HasNaturalStride());

		NDArray r = NDData::New({_size});
		memcpy(r->_data,_data,_size*sizeof(FP));
		return r;
	}

	// Gather elements along the specified dimension.
	//
	NDArray Gather(const int dim,const NDArray& indices) const
	{
		// Must be the same dimensionality.
		if(indices->_shape.size()!=_shape.size())
			throw IncompatibleShape();

		// Uninitialised output the shape of indices.
		NDArray r = NDData::New(indices->_shape);

		// Pick each output value as specified by indices.
		for(NDIterator2 iter=indices->Iter();iter!=iter.end();++iter)
		{
			// Index source value.
			NDShape index(iter);
			index[dim] = (int)*iter;

			// Write source value to result.
			r[iter] = (*this)[index];
		}
		return r;
	}

	// Elementwise greater, returns boolean array of same shape.
	//
	NDArray Greater(const FP v) const
	{
		_ASSERT(HasNaturalStride());

		NDArray r = NDData::New(*this);
		FP* const rData = r->_data;
		for(int i=0;i<_size;++i)
			rData[i] = rData[i]>v;
		return r;
	}

	// Returns true if v is dimensionally and numerically the same.
	//
	bool IsEqualTo(const NDArray& v) const
	{
		if(_shape!=v->_shape)
			throw IncompatibleShape();

		bool r = true;
		const_cast<NDData*>(this)->Elementwise(v._data,[&r](FP& a,const FP b)
		{
			const FP d = abs(a-b);
			if(d>=0.0001||isnan(d))
				r = false;
		});				
		return r;
	}

	// Returns true if the array contains exactly 1 value.
	//
	bool IsScalar() const
	{
		_ASSERT(HasNaturalStride());

		for(int i=0;i<_shape.size();++i)
			if(_shape[i]!=1)
				return false;
		return true;
	}

	// Elementwise log.
	//
	NDArray Log() const
	{
		_ASSERT(HasNaturalStride());

		NDArray r = NDData::New(_shape);
		for(int i=0;i<_size;++i)
			r->_data[i] = log(_data[i]);
		return r;
	}

	// Returns Y values on the log curve which are evently spaced along the X axis.
	//
	static NDArray LogSpace(const FP start,const FP stop,const int size,const bool endPoint,const FP base)
	{
		NDArray y = NDData::New({size});
		if(size>0)
		{
			// Size of interval, depends on if the 'stop' value should be included or not.
			const FP interval = (stop-start)/(endPoint?size-1:size);

			// Compute the Y value on the log curve for each X value evenly spaced by the interval.
			FP* const yData = y->_data;
			for(int i=0;i<size;++i)
			{
				const FP x = start+(i*interval);
				yData[i] = pow(base,x);
			}
		}

		return y;
	}

	// Fills this array with 'value' where 'mask' is non-zero.
	//
	void _MaskedFill(const NDArray& mask,const FP value)
	{
		const NDArray mask_ = mask->Broadcast(Self());

		if(_shape!=mask_->Shape())
			throw IncompatibleShape();

		Elementwise(mask_._data,[value](FP& a,const FP b)
		{
			if(b)
				a = value;
		});
	}

	// Returns an array filled with 'value' where 'mask' is non-zero.
	//
	NDArray MaskedFill(const NDArray& mask,const FP value) const
	{
		NDArray r = NDData::New(*this);
		r->_MaskedFill(mask,value);
		return r;
	}

	// Max value along dimension.
	//
	NDArray Max(const int dim) const
	{
		_ASSERT(dim>=0&&dim<_shape.size());

		// Create the new shape and result data where the aggregation dimension is 1.
		NDShape newShape(_shape);
		newShape[dim] = 1;
		NDArray r = NDData::New(newShape);

		// Iterate reduced shape.
		const int dimSize = _shape[dim];
		for(NDIterator2 iter=r->Iter();iter!=iter.end();++iter)
		{
			// Max along the aggregation dimension.
			NDShape indices(iter);
			const std::initializer_list<int> index(indices.data(),indices.data()+indices.size());
					
			// Assume first value, and iterate over the rest.
			FP max_v = (*this)[index];
			for(int i=0;i<dimSize;++i)
			{
				indices[dim] = i;
				const FP v = (*this)[index];
				if(v>max_v)
					max_v = v;
			}

			// Store max value.
			*iter = max_v;
		}

		const bool keepDims = true;
		if(!keepDims)
		{
			newShape.erase(newShape.begin()+dim);
			return r.Reshape(newShape);
		}
		else
			return r;
	}

	// Compute mean for dimension, where that dimension collapses to 1 value.
	//
	NDArray Mean(const int dim,const bool keepDims) const
	{
		_ASSERT(dim>=0&&dim<_shape.size());

		// Create the new shape and result data where the aggregation dimension is 1.
		NDShape newShape(_shape);
		newShape[dim] = 1;
		NDArray r = NDData::New(newShape);

		// Iterate reduced shape.
		const int dimSize = _shape[dim];
		for(NDIterator2 iter=r->Iter();iter!=iter.end();++iter)
		{
			// Sum along the aggregation dimension.
			NDShape indices(iter);
			const std::initializer_list<int> index(indices.data(),indices.data()+indices.size());
			FP sum = 0.0;
			for(int i=0;i<dimSize;++i)
			{
				indices[dim] = i;
				sum += (*this)[index];
			}

			// Store output mean.
			*iter = sum/dimSize;
		}

		if(!keepDims)
		{
			newShape.erase(newShape.begin()+dim);
			return r.Reshape(newShape);
		}
		else
			return r;
	}

	// Scalar mean value - should be deprecated when other 'Mean' can accept a list of 'dims'.
	//
	NDArray Mean(const bool keepDims) const
	{
		// Result either same dimensionality as this or a scalar.
		NDShape shape;
		if(keepDims)
		{
			shape.reserve(_shape.size());
			for(size_t i=0;i<_shape.size();++i)
				shape.emplace_back(1);
		}
		NDArray r = NDData::New(shape);

		FP sum = 0.0;
		if(HasNaturalStride())
		{
			for(int i=0;i<_size;++i)
				sum += _data[i];
		}
		else
		{
			for(NDIterator2 iter=Iter();iter!=iter.end();++iter)
				sum += *iter;
		}
		r->_data[0] = sum/_size;

		return r;
	}

	// Elementwise inplace multiplication.
	//
	void _Mul(const NDArray& v)
	{
		Elementwise(v._data,[](FP& a,const FP b)
		{
			a *= b;
		});
	}

	// Elementwise multiply by nd-array.
	//
	NDArray Mul(const NDArray& v) const
	{
		// Broadcast to create result.
		auto [r,v_] = Broadcast(Self(),v);

		// Inplace operation on result.
		r->_Mul(v_);

		return r;
	}

	// Elementwise negation.
	//
	NDArray Negate() const
	{
		NDArray r = NDData::New(_shape);
		r->Elementwise(Self(),[](FP& a,const FP b)
		{
			a = -b;
		});
		return r;
	}

	// Returns the Euclidean norm - the length of the vector/distance of the point from the origin. Extendion of Pythagorean theorem to n dimensions, square root of sum of squares.
	//
	FP Norm() const
	{
		_ASSERT(HasNaturalStride());

		FP* const rData = _data;
		FP ss = 0.0;
		for(int i=0;i<_size;++i)
			ss += pow(rData[i],FP(2));
		return sqrt(ss);
	}

	// Elementwise not equal, returns boolean array of same shape.
	//
	NDArray NotEqual(const NDArray& v) const
	{
		_ASSERT(HasNaturalStride());

		NDArray r = NDData::New(_shape);
		const FP* const data = _data;
		const FP* const vData = v->_data;
		FP* const rData = r->_data;
		for(int i=0;i<_size;++i)
			rData[i] = data[i]!=vData[i];
		return r;
	}

	// Returns a new array of the same shape initialised with ones.
	//
	NDArray Ones() const
	{
		return NDData::New(Shape(),1.0);
	}

	// Elementwise power.
	//
	void _Pow(const FP exponent)
	{
		_ASSERT(HasNaturalStride());

		FP* const rData = _data;
		if(exponent==2.0)
		{
			for(int i=0;i<_size;++i)
				rData[i] *= rData[i];
		}
		else
		{
			for(int i=0;i<_size;++i)
				rData[i] = pow(rData[i],exponent);
		}
	}

	NDArray Pow(const FP exponent) const
	{
		NDArray r = NDData::New(*this);
		r->_Pow(exponent);
		return r;
	}

	// Repeat values in dimension 'dim' 'copies' times.
	//
	NDArray Repeat_Numpy(const int dim,const int repeats) const
	{
		// If the repeat dimension of the source is 1 then copy can be avoided by setting the stride to 0 - thus the dimension never increments.
		if(_shape[dim]==1)
		{
			// Make a copy of this and apply non-copying repeat.
			NDArray r = NDData::New(NDDataPtrC(shared_from_this()));
			r->_shape[dim] = repeats;
			r->_stride[dim] = 0;
			return r;
		}
		else
		{
			// Make a result with each item of the repeat dimension 'repeats' times.
			NDShape newShape(_shape);
			newShape[dim] *= repeats;
			NDArray r = NDData::New(newShape);

			// Build slicer for output.
			int outDimSlice = 0;
			std::vector<std::initializer_list<int>> outSlices;
			for(size_t i=0;i<newShape.size();++i)
			{
				if(i==dim)
					outSlices.emplace_back(&outDimSlice,&outDimSlice+1);
				else
					outSlices.emplace_back();
			}
			std::initializer_list<std::initializer_list<int>> outSlicer(outSlices.data(),outSlices.data()+outSlices.size());

			// Build slicer for source.
			int srcDimSlice = 0;
			std::vector<std::initializer_list<int>> srcSlices;
			for(size_t i=0;i<newShape.size();++i)
			{
				if(i==dim)
					srcSlices.emplace_back(&srcDimSlice,&srcDimSlice+1);
				else
					srcSlices.emplace_back();
			}
			std::initializer_list<std::initializer_list<int>> srcSlicer(srcSlices.data(),srcSlices.data()+srcSlices.size());

			// Iterate each element in the repeat dimension any copy to the output array repeat times.
			for(int i=0;i<_shape[dim];++i)
			{
				srcDimSlice = i;
				for(int j=0;j<repeats;++j)
				{
					outDimSlice = (i*repeats)+j;
					r.Slice(outSlicer) = Slice(srcSlicer);
				}
			}

			return r;
		}
	}

	NDArray Repeat_Torch(const std::initializer_list<int>& sizes) const
	{
		NDShape shape(sizes);
		NDIterator iter(shape);

		for(auto i:_shape)
			shape.emplace_back(i);

		NDArray r = NDData::New(shape);

		for(;!iter.end();++iter)
		{
			r.Slice(iter) = Slice({});
		}

		return r;
	}

	// Reshape (without moving elements).
	// 
	// Templated for use with initializer_list or vector interators.
	//
	template<class iter>
	NDArray Reshape(const iter& begin,const iter& end) const
	{
		_ASSERT(!HasRepeatedDimensions());
		NDShape shape;

		// Compute size.
		int inferredDimensionIndex = -1;
		int size = 1;
		for(auto i=begin;i!=end;++i)
		{
			int dimensionSize = *i;

			// Exactly one value can be -1, in which case the value is inferred from the length.
			if(dimensionSize<0)
			{
				if(inferredDimensionIndex>=0)
					throw IncompatibleShape("Only one dimension can be inferred.");
				if(dimensionSize!=-1)
					throw IncompatibleShape("Invalid dimension value.");
				inferredDimensionIndex = (int)shape.size();
				dimensionSize = 1;
			}
			shape.emplace_back(dimensionSize);
			size *= dimensionSize;
		}

		// Infer missing dimension if required.
		if(inferredDimensionIndex>=0)
		{
			if(_size%size)
				throw IncompatibleShape("Explicit shape is not a multiple of _size.");
			const int dimensionSize = _size/size;
			shape[inferredDimensionIndex] = dimensionSize;
			size *= dimensionSize;
		}

		// Sizes must match.
		if(size!=_size)
			throw IncompatibleShape();

		return NDData::New(*this,shape);
	}

	NDArray Reshape(const std::initializer_list<int>& shape) const
	{
		return Reshape(shape.begin(),shape.end());
	}

	NDArray Reshape(const NDShape& shape) const
	{
		return Reshape(shape.begin(),shape.end());
	}

	// Scatter elements from 'source' into 'result' positions specified by 'indices', reciprocal of 'Gather'.
	// The data in this object does not participate in the operation, only the shape.
	// 
	// NOTE! This differs from torch whereby this accumulates values as it is used for gradients.
	//
	NDArray Scatter(const int dim,const NDArray& indices,const NDArray& source) const
	{
		// Shapes of indices and source must be the same.
		if(indices->_shape!=source->_shape)
			throw IncompatibleShape();

		// Zeroed result same shape as this.
		NDArray r = Zeros();										

		// Iterate indices.
		for(NDIterator2 iter=indices->Iter();iter!=iter.end();++iter)
		{
			// Index result value.
			NDShape index(iter);
			index[dim] = (int)*iter;

			// Add source value to result.
			r[index] += source[iter];
		}
		return r;
	}

	NDArray Sqrt() const
	{
		_ASSERT(HasNaturalStride());

		NDArray r = NDData::New(*this);
		FP* const rData = r->_data;
		for(int i=0;i<_size;++i)
			rData[i] = sqrt(rData[i]);
		return r;
	}

	NDArray Softmax() const
	{
		_ASSERT(HasNaturalStride());

		NDArray r = NDData::New(_shape);
		FP* const data = _data;
		FP* const rData = r->_data;
		FP sum = 0.0;
		for(int i=0;i<_size;++i)
		{
			rData[i] = exp(data[i]);
			sum += rData[i];
		}
		for(int i=0;i<_size;++i)
			rData[i] /= sum;
		return r;
	}

	NDArray StdDev() const
	{
		_ASSERT(HasNaturalStride());

		const FP mean = Mean(false)[{}];
		FP se = 0.0;
		for(int i=0;i<_size;++i)
			se += pow(_data[i]-mean,FP(2));
		const FP mse = se/_size;

		NDArray stddev = NDData::New({1},sqrt(mse));
		return stddev;
	}

	void _Sub(const NDArray& v)
	{
		Elementwise(v._data,[](FP& a,const FP b)
		{
			a -= b;
		});
	}

	NDArray Sub(const NDArray& v) const
	{
		NDArray r = NDData::New(*this);
		r->_Sub(v);
		return r;
	}

	NDArray Sum() const
	{
		_ASSERT(HasNaturalStride());

		NDArray r = NDData::New({1},0.0);
		FP& rData = r->_data[0];
		const FP* vData = _data;
		for(int i=0;i<_size;++i)
			rData += vData[i];
		return r;
	}

	// Compute sum for dimension, where that dimension collapses to 1 value.
	//
	NDArray Sum(int dim,const bool keepDims) const
	{
		if(dim<0)
			dim += _shape.size();

		// Create the new shape and result data where the aggregation dimension is 1.
		NDShape newShape(_shape);
		newShape[dim] = 1;
		NDArray r = NDData::New(newShape);

		// Aggreate along the dimension.
		Aggregate(r,dim,[](FP& acc,const FP& value)
		{
			acc += value;
		});

		// Remove aggregation dimension if not kept.
		if(!keepDims)
		{
			newShape.erase(newShape.begin()+dim);
			return r.Reshape(newShape);
		}

		return r;
	}

	// Elementwise tanh.
	//
	NDArray Tanh() const
	{
		_ASSERT(HasNaturalStride());

		NDArray r = NDData::New(_shape);
		for(int i=0;i<_size;++i)
			r->_data[i] = tanh(_data[i]);
		return r;
	}

	void _Transpose()
	{
		if(_shape.size()<2)
			throw IncompatibleShape();

		// Swap last 2 dimensions.
		std::swap(_shape[_shape.size()-1],_shape[_shape.size()-2]);
		std::swap(_stride[_stride.size()-1],_stride[_stride.size()-2]);
	}

	NDArray Transpose() const
	{
		NDArray r = NDData::New(Self());

		if(_shape.size()>=2)
			r->_Transpose();

		return r;
	}

	// Returns the lower trianglar matrix.
	//
	NDArray Tril() const
	{
		_ASSERT(HasNaturalStride());

		// Only implemented for 2D array.
		if(_shape.size()!=2)
			throw IncompatibleShape();

		NDArray r = NDData::New(*this);
		FP* rData = r->_data;
		const FP* vData = _data;
		const int rows = _shape[0];
		const int cols = _shape[1];
		for(int i=0;i<rows;++i)
		{
			int j=0;
			for(;j<=i;++j)
				*rData++ = vData[i+j*rows];
			for(;j<cols;++j)
				*rData++ = 0.0;
		}
		return r;
	}

	// Returns a new array with a dimension inserted at the specified position.
	//
	NDArray Unsqueeze(int dim) const
	{
		// Negative dim implies position from end.
		if(dim<0)
			dim += (int)_shape.size()+1;	// +1 is needed here because 'vector::insert' inserts to the left of the index.

		NDArray r = NDData::New(*this);
		r->_shape.insert(r->_shape.begin()+dim,1);
		r->_stride.insert(r->_stride.begin()+dim,r->_stride.size()>0?r->_stride[0]:1);

		return r;
	}

	// Returns a new array where each element of the indices array is replaced with the slice from the source array where the most significant dimension is set
	// by the indices value.
	//
	NDArray IndexSelect(const NDArray& indices) const
	{
		// Cannot index a scalar.
		if(_shape.size()==0)
			throw IncompatibleShape();

		// Indices values are the indexes into the most significant dimension of the source array, and the remaining source dimension shape returned.
		// Thus, the most significant dimension of the source is removed.
		NDShape shape(_shape);
		shape.erase(shape.begin());

		// Each selected source slice is copied to the position corresponding of indices which becomes the most significant dimension(s) in the result.
		for(int i=0;i<indices->_shape.size();++i)
			shape.insert(shape.begin()+i,indices->_shape[i]);
		NDArray r = NDData::New(shape);

		// Iterate indices.
		for(NDIterator2 iter=indices->Iter();iter!=iter.end();++iter)
			r.Slice(iter) = Slice({{(int)*iter}});

		return r;
	}

	NDArray UnindexSelect(const NDArray indices,const NDArray& source) const
	{
		// Source expected to have been created from indices by IndexSelect.
		for(int i=0;i<indices->_shape.size();++i)
		{
			if(source->_shape[i]!=indices->_shape[i])
				throw IncompatibleShape();	
		}

		// Create the ouput gradient from rows from the input gradient selected by the indices.
		NDArray r = Zeros();
		for(NDIterator iter=indices.Shape();!iter.end();++iter)
			r.Slice({{(int)indices[iter]}}) += source.Slice(iter);	// Add gradients because indices aren't unique, which indicates multiple calculation paths.

		return r;
	}

	NDArray Var() const
	{
		_ASSERT(HasNaturalStride());

		const FP mean = Mean(false)[{}];
		FP se = 0.0;
		for(int i=0;i<_size;++i)
			se += pow(_data[i]-mean,FP(2));
		return NDData::New({},se/_size);
	}

	NDArray Var(const int dim,const bool keepDim,const int correction=1/*Bessel's correction*/) const
	{		
		_ASSERT(dim>=0&&dim<_shape.size());
		NDArray tmp = this->Sub(Mean(dim,true));
		tmp._data->_Pow(2.0);
		NDArray r = tmp.Sum(dim,keepDim);
		r._data->_Div(NDData::New({},_shape[dim]-FP(correction)));
		return r;
	}


	// Returns a new array of the same shape initialised with zeros.
	//
	NDArray Zeros() const
	{
		return NDData::New(Shape(),0.0);
	}

	// Print to output stream.
	//
	void Print(std::ostream& out) const
	{
		NDShape indices;
		_Print(out,indices);
	}

	static char NextChar(std::fstream& file)
	{
		char c;
		do
		{
			file>>c;
			if(file.eof())
			{
				c = '\0';
				break;
			}
		} while(c<=' ');
		return c;
	}

	static void ReadExponentPart(std::fstream& file,std::stringstream& ss,char& c)
	{
		// Exponent sign.
		if(c=='-'||c=='+')
		{
			ss<<c;
			file>>c;

			// Exponent value.
			while(c>='0'&&c<='9')
			{
				ss<<c;
				file>>c;
			}
		}
		else
			throw UnexpectedCharacter();
	}

	static void ReadFractionalPart(std::fstream& file,std::stringstream& ss,char& c)
	{
		// Fractional digits.
		if(c>='0'&&c<='9')
		{
			while(c>='0'&&c<='9')
			{
				ss<<c;
				file>>c;
			}

			// Exponent.
			if(c=='e')
			{
				ss<<c;
				file>>c;
				ReadExponentPart(file,ss,c);
			}
		}
		else
			throw UnexpectedCharacter();
	}

	static void ReadIntegerPart(std::fstream& file,std::stringstream& ss,char& c)
	{
		// Integer digits.
		if(c>='0'&&c<='9')
		{
			while(c>='0'&&c<='9')
			{
				ss<<c;
				file>>c;
			}

			// Decimal separator.
			if(c=='.')
			{
				ss<<c;
				file>>c;
				ReadFractionalPart(file,ss,c);
			}
		}
		else
			throw UnexpectedCharacter();
	}

	static void ReadNumber(std::fstream& file,std::stringstream& ss,char& c)
	{
		switch(c)
		{
			case '-':
			{
				// Signed number.
				ss<<c;
				file>>c;
				ReadIntegerPart(file,ss,c);
				break;
			}
			case '0':
			case '1':
			case '2':
			case '3':
			case '4':
			case '5':
			case '6':
			case '7':
			case '8':
			case '9':
			{
				// Unsigned number.
				ReadIntegerPart(file,ss,c);
				break;
			}
		}
	}

	static void ReadValue(std::fstream& file,std::vector<FP>& data,char& c)
	{
		std::stringstream ss;
		ReadNumber(file,ss,c);
		data.emplace_back(FP(atof(ss.str().c_str())));
	}

	static int ReadDelimitedValueOrEnd(std::fstream& file,std::vector<int>& shape,std::vector<FP>& data,char& c)
	{
		switch(c)
		{
			case ',':
			{
				// Consume ','.
				c = NextChar(file);

				// Expect value.
				ReadValue(file,data,c);

				// Expect delimited value or end.
				return ReadDelimitedValueOrEnd(file,shape,data,c)+1;
			}
			case ']':
			{
				// End of array - leave ']' to match with '['.
				return 0;
			}
			default:
			{
				throw UnexpectedCharacter();
			}
		}
	}

	static int ReadArray(std::fstream& file,std::vector<int>& shape,std::vector<FP>& data,const int dim,char& c)
	{
		switch(c)
		{
			case '[':
			{
				// Start of array.

				// Extend size the first time the dimension is seen.
				if(shape.size()==dim)
					shape.emplace_back(-1);

				// Consume '[',
				c = NextChar(file);

				// Expect values for this array, or a nested array.
				const int length = ReadValuesOrNestedArray(file,shape,data,dim,c);

				// Should unwind with matching ']'.
				if(c!=']')
					throw UnexpectedCharacter();

				// Consume ']'.
				c = NextChar(file);

				// Set dimension length if not set, otherwise length must match.
				if(shape[dim]==-1)
					shape[dim] = length;
				else if(shape[dim]!=length)
					throw InvalidDimension();

				// Expect delimited array or end.
				return ReadDelimitedArrayOrEnd(file,shape,data,dim,c)+1;	// 1+ because array read.
			}
			default:
			{
				throw UnexpectedCharacter();
			}
		}
	}

	static int ReadDelimitedArrayOrEnd(std::fstream& file,std::vector<int>& shape,std::vector<FP>& data,const int dim,char& c)
	{
		switch(c)
		{
			case ',':
			{
				// Delimiter followed by array.
				c = NextChar(file);
				return ReadArray(file,shape,data,dim,c);
			}
			case ']':
			{
				// End of array - leave ']' to match with '['.
				return 0;
			}
			case '\0':
			{
				// End of file.
				return 0;
			}
			default:
			{
				throw UnexpectedCharacter();
			}
		}
	}

	static int ReadValuesOrNestedArray(std::fstream& file,std::vector<int>& shape,std::vector<FP>& data,const int dim,char& c)
	{
		switch(c)
		{
			case '[':
			{
				// Nested array.
				return ReadArray(file,shape,data,dim+1,c);
			}
			case '-':
			case '0':
			case '1':
			case '2':
			case '3':
			case '4':
			case '5':
			case '6':
			case '7':
			case '8':
			case '9':
			{
				// First value.
				ReadValue(file,data,c);

				// Expect delimited value or end.
				return ReadDelimitedValueOrEnd(file,shape,data,c)+1;	// 1+ because of first value.
			}
			default:
			{
				throw UnexpectedCharacter();
			}
		}
	}

	static void ReadWhitespace(std::fstream& file,char& c)
	{
		while(c<=' '&&c!='\0')
			c = NextChar(file);
	}

	static void ReadArrayOrScalar(std::fstream& file,std::vector<int>& shape,std::vector<FP>& data,char& c)
	{
		switch(c)
		{
			case '[':
			{
				// N-dimensional array.
				ReadArray(file,shape,data,0,c);
				break;
			}
			case '-':
			case '0':
			case '1':
			case '2':
			case '3':
			case '4':
			case '5':
			case '6':
			case '7':
			case '8':
			case '9':
			{
				// Scalar.
				ReadValue(file,data,c);
				ReadWhitespace(file,c);
				break;
			}
			case '\0':
			{
				// Empty file.
				break;
			}
			default:
			{
				throw UnexpectedCharacter();
			}
		}

		// Check end-of-file was reached.
		if(c!='\0')
			throw UnexpectedCharacter();
	}

	// Load from file with implicit shape using nested square brackets.
	//
	static NDArray LoadWithImplicitShape(const std::string& filename)
	{		
		std::fstream file;
		file.open(filename,std::ios::in);
		if(file.is_open())
		{
			file>>std::noskipws;

			std::vector<int> shape;
			std::vector<FP> data;
			
			char c = NextChar(file);
			ReadArrayOrScalar(file,shape,data,c);
			file.close();

			return NDData::New(shape,data);
		}
		else
			throw FileNotFound();
	}

	void Save(const std::string& filename) const
	{
		Utf8FileWriter file(filename);	// Need Ascii version.

		bool first = true;
		for(auto i:_shape)
		{
			if(first)
				first = false;
			else
				file<<",";
			file<<std::to_string(i);
		}
		file<<"\n";

		NDArray flat = NDData::New(*this).Flatten();
		const FP* const data = flat->_data;
		for(int i=0;i<_size;++i)
		{
			file<<std::to_string(data[i]);
			file<<"\n";
		}

		file.Close();
	}

	// Load from file where the explicit shape is the first line.
	//
	static NDArray LoadWithExplicitShape(const std::string& filename)
	{		
		AsciiFileReader file(filename);

		// Read comma delimited shape.
		std::string line;
		file>>line;
		std::vector<std::string> dims = Split(line,',');

		std::vector<int> shape;
		int values = 1;
		for(auto& dim:dims)
		{
			int length = atoi(dim.c_str());
			values *= length;
			shape.emplace_back(length);
		}

		std::vector<FP> data;
		for(int i=0;i<values;++i)
		{
			file>>line;
			data.emplace_back(FP(atof(line.c_str())));
		}

		file.close();

		return NDData::New(shape,data);
	}

	// Load from a file created with NumPy 'savetxt'.
	//
	static NDArray LoadNP(const std::string& filename)
	{
		AsciiFileReader file(filename);

		std::vector<FP> values;
		int i = 0;
		int j = 0;

		std::string line;
		do
		{
			file>>line;
			if(line.length()>0)
				++i;
			size_t pos = std::string::npos;
			do
			{
				pos = line.find_first_of(' ');
				if(pos!=std::string::npos||line.length()>0)
				{
					std::string value = line.substr(0,pos);
					values.push_back(FP(atof(value.c_str())));
					if(i==1)
						++j;
					line = line.substr(pos+1);
				}
			} while(pos!=std::string::npos);
		}while(!file.eof());

		NDArray r = j==1?
			NDData::New({i},values):	// 1-D column vector.
			NDData::New({i,j},values);	// 2-D matrix.

		return r;
	}


	static NDArray Load(const std::string& filename)
	{
		// Default to explicit shape - easiest to implement with any number of dimensions.
		return LoadWithExplicitShape(filename);
	}

};


// NDArray

NDArray::NDArray(const FP v) :
	_data(NDData::New({},v)._data)
{
}

void NDArray::_Attach(const NDArray& v)
{
	_data = v._data;
}

void NDArray::_Reset()
{
	_data.reset();
}

void NDArray::operator=(const NDArray& v)
{
	_data->Assign(v);
}

NDArray NDArray::operator==(const NDArray& v) const
{
	return _data->Equal(v);
}

NDArray NDArray::operator!=(const NDArray& v) const
{
	return _data->NotEqual(v);
}

NDArray NDArray::operator+(const NDArray& v) const
{
	return _data->Add(v);
}

void NDArray::operator+=(const NDArray& v)
{
	_data->_Add(v);
}

NDArray NDArray::operator-() const
{
	return _data->Negate();
}

NDArray NDArray::operator-(const NDArray& v) const
{
	return _data->Sub(v);
}

void NDArray::operator-=(const NDArray& v) const
{
	_data->_Sub(v);
}

NDArray NDArray::operator*(const NDArray& v) const
{
	return _data->Mul(v);
}

void NDArray::operator*=(const NDArray& v)
{
	return _data->_Mul(v);
}

NDArray NDArray::operator/(const NDArray& v) const
{
	return _data->Div(v);
}

NDArray NDArray::operator>(const FP v) const
{
	return _data->Greater(v);
}

inline FP& NDArray::operator[](const std::initializer_list<int>& indices)
{
	return _data->operator[](indices);
}

inline const FP& NDArray::operator[](const std::initializer_list<int>& indices) const
{		
	return _data->operator[](indices);
}

inline NDArray NDArray::operator[](const NDArray& indices) const
{
	return _data->IndexSelect(indices);
}

int NDArray::ArgMax() const
{
	return _data->ArgMax();
}

NDArray NDArray::ArgMax(const int dim) const
{
	return _data->ArgMax(dim);
}

void NDArray::_ClipNorm(const FP v) const
{
	return _data->_ClipNorm(v);
}

NDArray NDArray::Dot(const NDArray& v) const
{
	return _data->Dot(v);
}

NDArray NDArray::Dropout(const FP p) const
{
	return _data->Dropout(p);
}

NDArray NDArray::Entropy() const
{
	return _data->Entropy();
}

NDArray NDArray::Exp() const
{
	return _data->Exp();
}

NDArray NDArray::Flatten() const
{
	return _data->Flatten();
}

NDArray NDArray::Gather(const int dim,const NDArray& indices) const
{
	return _data->Gather(dim,indices);
}

bool NDArray::IsEqualTo(const NDArray& v) const
{
	return _data->IsEqualTo(v);
}

bool NDArray::IsScalar() const
{
	return _data->IsScalar();
}

NDArray NDArray::Log() const
{
	return _data->Log();
}

NDArray NDArray::MaskedFill(const NDArray& mask,const FP value) const
{
	return _data->MaskedFill(mask,value);
}

NDArray NDArray::Max(const int dim) const
{
	return _data->Max(dim);
}

NDArray NDArray::Mean(const bool keepDims) const
{
	return _data->Mean(keepDims);
}

NDArray NDArray::Mean(const int dim,const bool keepDims) const
{
	return _data->Mean(dim,keepDims);
}

NDArray NDArray::Ones() const
{
	return _data->Ones();
}

NDArray NDArray::Pow(const FP v) const
{
	return _data->Pow(v);
}

NDArray NDArray::Repeat_Numpy(const int dim,const int copies) const
{
	return _data->Repeat_Numpy(dim,copies);
}

NDArray NDArray::Repeat_Torch(const std::initializer_list<int>& sizes) const
{
	return _data->Repeat_Torch(sizes);
}

NDArray NDArray::Reshape(const std::initializer_list<int>& range) const
{
	return _data->Reshape(range.begin(),range.end());
}

NDArray NDArray::Reshape(const NDShape& range) const
{
	return _data->Reshape(range.begin(),range.end());
}

void NDArray::Save(const std::string& filename) const
{
	_data->Save(filename);
}

NDArray NDArray::Scatter(const int dim,const NDArray& indices,const NDArray& source) const
{
	return _data->Scatter(dim,indices,source);
}

const NDShape& NDArray::Shape() const
{
	return _data->Shape();
}

NDArray NDArray::Slice(const std::initializer_list<std::initializer_list<int>>& indices)
{
	return _data->Slice(indices);
}

const NDArray NDArray::Slice(const std::initializer_list<std::initializer_list<int>>& indices) const
{
	return _data->Slice(indices);
}

int NDArray::Size() const
{
	return _data->Size();
}

NDArray NDArray::Softmax(const int dim) const
{
	return _data->Softmax(dim);
}

NDArray NDArray::Sqrt() const
{
	return _data->Sqrt();
}

NDArray NDArray::StdDev() const
{
	return _data->StdDev();
}

NDArray NDArray::Sum() const
{
	return _data->Sum();
}

NDArray NDArray::Sum(const int dim,const bool keepDims) const
{
	return _data->Sum(dim,keepDims);
}

NDArray NDArray::Tanh() const
{
	return _data->Tanh();
}

NDArray NDArray::Transpose() const
{
	return _data->Transpose();
}

NDArray NDArray::Tril() const
{
	return _data->Tril();
}

NDArray NDArray::UnindexSelect(const NDArray& indices,const NDArray& source) const
{
	return _data->UnindexSelect(indices,source);
}

NDArray NDArray::Unsqueeze(const int dim) const
{
	return _data->Unsqueeze(dim);
}

NDArray NDArray::Var() const
{
	return _data->Var();
}

NDArray NDArray::Var(const int dim,const bool keepDim) const
{
	return _data->Var(dim,keepDim);
}

NDArray NDArray::Zeros() const
{
	return _data->Zeros();
}

void NDArray::Print(std::ostream& out) const
{
	_data->Print(out);
}
