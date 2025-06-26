#pragma once

#include <concurrent_queue.h>
#include <malloc.h>


class NDAllocator
{
	static const size_t MaxAllocators = 32;

	class Allocator
	{
		size_t									_size;
		size_t									_allocationCount;
		concurrency::concurrent_queue<void*>	_allocations;

	public:
		Allocator() :
			_size(0),
			_allocationCount(0)
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
			{
				_aligned_free(allocation);
				--_allocationCount;
			}
			_size = 0;
		}

        void* Allocate()
        {
            void* allocation = nullptr;
            if(!_allocations.try_pop(allocation))
			{
                allocation = _aligned_malloc(_size,64);
				++_allocationCount;
			}
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
		_ASSERT(size<=(1ULL<<MaxAllocators));
		size_t msb = 0;
		while((1ULL<<msb)<size)
			++msb;
		_ASSERT(size<=_allocators[msb].Size());
		return _allocators[msb];
	}

	struct BlockHeader
	{
		Allocator*		_allocator;		// Pointer to the allocator that allocated this block.
		char			_padding[56];	// Padding to ensure _data is 64-byte aligned.
		#pragma warning(suppress:4200)
		unsigned char	_data[];
	};
	static_assert(sizeof(BlockHeader) % 64 == 0,"BlockHeader must be 64-byte aligned");

public:
	NDAllocator()
	{
		for(int i=0;i<MaxAllocators;++i)
			_allocators[i].Initialise(1ULL<<i);
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
		BlockHeader* const block = (BlockHeader*)allocator.Allocate();
		block->_allocator = &allocator;
		return (T*)block->_data;
	}

	void Free(void* const data)
	{
		BlockHeader* const blockHeader = ((BlockHeader*)data)-1;
		blockHeader->_allocator->Free(blockHeader);
	}
};


extern NDAllocator mem;