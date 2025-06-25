#pragma once

#include <concurrent_queue.h>


class NDAllocator
{
	static const size_t MinAllocationBit	= 4;
	static const size_t MinAllocation		= 1ULL<<MinAllocationBit;
	static const size_t MaxAllocators		= 32;

	class Allocator
	{
		size_t									_size;
		concurrency::concurrent_queue<void*>	_allocations;

	public:
		Allocator() :
			_size(0)
		{
		}

		void Initialise(const size_t size)
		{
			_size = size;
		}

		void Uninitialise()
		{
			void* allocation = nullptr;
			while(_allocations.try_pop(allocation))
				free(allocation);
			_size = 0;
		}

		void* Allocate()
		{
			void* allocation = nullptr;
			if(!_allocations.try_pop(allocation))
				allocation = malloc(_size);
			return allocation;
		}

		void Free(void* allocation)
		{
			_allocations.push(allocation);
		}

		size_t Size() const
		{
			return _size;
		}
	};

	Allocator _allocators[MaxAllocators];
	
	Allocator& GetAllocator(const size_t size)
	{
		_ASSERT(size<=(1ULL<<(MaxAllocators+MinAllocationBit-1)));
		size_t msb = MinAllocationBit;
		while((1ULL<<msb)<size)
			++msb;
		return _allocators[msb-MinAllocationBit];
	}

	struct BlockHeader
	{
		size_t			_size;
		#pragma warning(suppress:4200)
		unsigned char	_data[];
	};

public:
	NDAllocator()
	{
		for(int i=0;i<MaxAllocators;++i)
			_allocators[i].Initialise(1ULL<<(MinAllocationBit+i));
	}

	~NDAllocator()
	{
		for(int i=0;i<MaxAllocators;++i)
			_allocators[i].Uninitialise();
	}

	template<typename T>
	T* Alloc(const size_t size)
	{
		Allocator& allocator = GetAllocator(sizeof(BlockHeader)+(size*sizeof(T)));
		BlockHeader* block = (BlockHeader*)allocator.Allocate();
		block->_size = allocator.Size();
		return (T*)block->_data;
	}

	void Free(void* const data)
	{
		BlockHeader* blockHeader = ((BlockHeader*)data)-1;
		GetAllocator(blockHeader->_size).Free(blockHeader);
	}
};


extern NDAllocator mem;