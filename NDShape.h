#pragma once

#include <initializer_list>
#include <memory>


class NDShape
{
public:
	static const int MaxDims = 3;

private:
	int _size;
	int _dims[MaxDims];

public:
	inline NDShape() :
		_size(0)
	{
	}

	inline NDShape(const int size) :
		_size(size)
	{
		_ASSERT(size<=MaxDims);
		memset(_dims,0,size*sizeof(int));
	}

	template<typename T>
	inline NDShape(const T& begin,const T& end)
	{
		_ASSERT(end-begin<=MaxDims);
		_size = (int)(end-begin);
		int* d = _dims;
		for(T i=begin;i<end;++i)
			*d++ = *i;
	}

	inline NDShape(const std::initializer_list<int> shape)
	{
		_ASSERT(shape.size()<=MaxDims);
		_size = (int)shape.size();
		int* d = _dims;
		for(auto i:shape)
			*d++ = i;
	}

	inline const int size() const
	{
		return _size;
	}

	inline const int operator [] (const int i) const
	{
		_ASSERT(i>=0&&i<_size);
		return _dims[i];
	}

	inline int& operator [] (const int i)
	{
		_ASSERT(i>=0&&i<_size);
		return _dims[i];
	}

	inline const bool operator == (const NDShape& v) const
	{
		return _size==v._size&&memcmp(_dims,v._dims,_size*sizeof(int))==0;
	}

	inline const bool operator != (const NDShape& v) const
	{
		return _size!=v._size||memcmp(_dims,v._dims,_size*sizeof(int))!=0;
	}

	inline void emplace_back(const int i)
	{
		_ASSERT(_size<MaxDims);
		_dims[_size] = i;
		++_size;
	}

	inline int* rbegin()
	{
		return _dims+_size-1;
	}

	inline const int* data() const
	{
		return _dims;
	}

	inline const int* begin() const
	{
		return _dims;
	}

	inline int* begin()
	{
		return _dims;
	}

	inline const int* end() const
	{
		return _dims+_size;
	}

	inline int* end()
	{
		return _dims+_size;
	}

	inline void insert(const int* pos,const int v)
	{
		_ASSERT(pos-_dims<MaxDims);
		_ASSERT(_size<MaxDims);
		const size_t offset = pos-_dims;
		const size_t count = _size-offset;
		memcpy(_dims+offset+1,_dims+offset,count*sizeof(int));
		++_size;
		_dims[offset] = v;
	}

	inline void erase(const int* pos)
	{
		_ASSERT(pos-_dims<MaxDims);
		_ASSERT(_size>0);
		--_size;
		const size_t offset = pos-_dims;
		const size_t count = _size-offset;
		memcpy(_dims+offset,_dims+offset+1,count*sizeof(int));
	}

	inline void reserve(int)
	{
	}

	inline void resize(const int size)
	{
		_ASSERT(size<=MaxDims);
		_size = size;
	}

	inline operator std::initializer_list<int>() const
	{
		return std::initializer_list<int>(_dims,_dims+_size);
	}
};
