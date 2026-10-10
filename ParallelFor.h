#pragma once

// Low-latency parallel-for.
//
// Persistent worker threads spin briefly waiting for work, then park with WaitOnAddress, so dispatching a job costs
// microseconds rather than the milliseconds of a condition-variable queue. Only one job runs at a time; a call made
// while the pool is busy (e.g. from inside another parallel job) simply runs on the calling thread.

#include <Windows.h>
#include <synchapi.h>
#include <atomic>
#include <thread>
#include <vector>
#include <mutex>
#include <algorithm>
#include <immintrin.h>

#pragma comment(lib,"Synchronization.lib")


class ParallelPool
{
	typedef void (*Trampoline)(const void* fn,int begin,int end);

	std::vector<std::thread>	_threads;
	std::mutex					_submit;			// Held while a job is running.
	std::atomic<unsigned>		_generation = 0;	// Incremented to publish a new job.
	std::atomic<bool>			_stop = false;
	std::atomic<int>			_next = 0;			// Next index to claim.
	std::atomic<int>			_active = 0;		// Workers still inside the current job.
	int							_end = 0;
	int							_chunk = 1;
	const void*					_fn = nullptr;
	Trampoline					_trampoline = nullptr;

	static inline thread_local bool _isWorker = false;

	void RunChunks()
	{
		for(;;)
		{
			const int i = _next.fetch_add(_chunk,std::memory_order_relaxed);
			if(i>=_end)
				break;
			_trampoline(_fn,i,(std::min)(i+_chunk,_end));
		}
	}

	void Worker()
	{
		_isWorker = true;
		unsigned seen = 0;
		for(;;)
		{
			// Spin for a while, then sleep until the generation changes.
			int spins = 0;
			unsigned g;
			while((g = _generation.load(std::memory_order_acquire))==seen)
			{
				if(++spins<20000)
					_mm_pause();
				else
					WaitOnAddress(&_generation,&seen,sizeof(seen),INFINITE);
			}
			seen = g;
			if(_stop)
				return;
			RunChunks();
			_active.fetch_sub(1,std::memory_order_acq_rel);
		}
	}

public:
	ParallelPool()
	{
		const int n = (int)(std::max)(1u,std::thread::hardware_concurrency());
		for(int i=1;i<n;++i)
			_threads.emplace_back([this](){Worker();});
	}

	~ParallelPool()
	{
		_stop = true;
		_generation.fetch_add(1,std::memory_order_release);
		WakeByAddressAll(&_generation);
		for(auto& t:_threads)
			t.join();
	}

	int Threads() const
	{
		return (int)_threads.size()+1;
	}

	// Calls fn(begin,end) on sub-ranges of [begin,end) of roughly 'grain' indices, in parallel.
	template<typename F>
	void ForRange(const int begin,const int end,const int grain,const F& fn)
	{
		if(end<=begin)
			return;
		const int n = end-begin;
		if(_isWorker||n<=grain||_threads.empty()||!_submit.try_lock())
		{
			fn(begin,end);
			return;
		}

		_fn = &fn;
		_trampoline = [](const void* f,int b,int e){(*(const F*)f)(b,e);};
		_end = end;
		_chunk = (std::max)(grain,n/(Threads()*4));
		_next.store(begin,std::memory_order_relaxed);
		_active.store((int)_threads.size(),std::memory_order_relaxed);
		_generation.fetch_add(1,std::memory_order_release);
		WakeByAddressAll(&_generation);

		// The caller works too, then waits for the stragglers.
		const bool wasWorker = _isWorker;
		_isWorker = true;
		RunChunks();
		_isWorker = wasWorker;
		while(_active.load(std::memory_order_acquire)>0)
			_mm_pause();
		_submit.unlock();
	}

	// Calls fn(i) for each i in [begin,end) in parallel.
	template<typename F>
	void For(const int begin,const int end,const F& fn)
	{
		ForRange(begin,end,1,[&fn](int b,int e){for(int i=b;i<e;++i) fn(i);});
	}

	static ParallelPool& Get()
	{
		static ParallelPool pool;
		return pool;
	}
};


template<typename F>
inline void ParallelFor(const int begin,const int end,const F& fn)
{
	ParallelPool::Get().For(begin,end,fn);
}

template<typename F>
inline void ParallelForRange(const int begin,const int end,const int grain,const F& fn)
{
	ParallelPool::Get().ForRange(begin,end,grain,fn);
}
