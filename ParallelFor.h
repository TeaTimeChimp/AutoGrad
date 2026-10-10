#pragma once

// Low-latency parallel-for.
//
// Persistent worker threads spin briefly waiting for work, then park with WaitOnAddress, so dispatching a job costs
// microseconds rather than the milliseconds of a condition-variable queue. A job finishes as soon as all of its
// indices are done; it does not wait for every worker to notice it, so a worker that the OS has descheduled cannot
// stall the caller. Only one job runs at a time; a call made while the pool is busy (e.g. from inside another
// parallel job) simply runs on the calling thread.
//
// Job publication uses a sequence number: odd while the caller is rewriting the job description, even once it is
// published. A worker registers in _inflight and then re-checks the sequence, so it either sees a complete job or
// backs off; the caller only rewrites the job once no worker is registered.

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
	std::atomic<unsigned>		_sequence = 0;		// Odd: job being written. Even: job published.
	std::atomic<int>			_inflight = 0;		// Workers currently reading the job description.
	std::atomic<bool>			_stop = false;
	std::atomic<int>			_next = 0;			// Next index to claim.
	std::atomic<int>			_remaining = 0;		// Indices not yet completed.
	int							_end = 0;
	int							_chunk = 1;
	const void*					_fn = nullptr;
	Trampoline					_trampoline = nullptr;

	static inline thread_local bool _isWorker = false;

	void RunChunks()
	{
		for(;;)
		{
			const int i = _next.fetch_add(_chunk);
			if(i>=_end)
				break;
			const int e = (std::min)(i+_chunk,_end);
			_trampoline(_fn,i,e);
			_remaining.fetch_sub(e-i);
		}
	}

	void Worker()
	{
		_isWorker = true;
		unsigned seen = 0;
		for(;;)
		{
			// Wait for a newly published (even) sequence number.
			int spins = 0;
			unsigned s;
			while((s = _sequence.load())==seen||(s&1))
			{
				if(++spins<20000)
					_mm_pause();
				else
				{
					WaitOnAddress(&_sequence,&s,sizeof(s),10);
					spins = 0;
				}
				if(_stop)
					return;
			}
			if(_stop)
				return;

			_inflight.fetch_add(1);
			if(_sequence.load()==s)
				RunChunks();				// Job description is stable while we're registered.
			_inflight.fetch_sub(1);
			seen = s;
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
		_sequence.fetch_add(2);
		WakeByAddressAll(&_sequence);
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

		// Unpublish, wait for stragglers from the previous job to leave, then write the new job.
		_sequence.fetch_add(1);
		while(_inflight.load()>0)
			_mm_pause();
		_fn = &fn;
		_trampoline = [](const void* f,int b,int e){(*(const F*)f)(b,e);};
		_end = end;
		_chunk = (std::max)(grain,n/(Threads()*4));
		_remaining.store(n);
		_next.store(begin);
		_sequence.fetch_add(1);
		WakeByAddressAll(&_sequence);

		// The caller works too, then waits until every index is done.
		_isWorker = true;
		RunChunks();
		_isWorker = false;
		while(_remaining.load()>0)
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
