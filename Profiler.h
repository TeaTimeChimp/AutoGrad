#pragma once

#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <typeinfo>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>


// Lightweight per-kernel profiler. Disabled by default; when enabled, Tensor records the
// wall-clock time spent in each kernel's Forward and Backward, keyed by the kernel's type.
//
class Profiler
{
	struct Entry
	{
		double	seconds = 0.0;
		long	calls = 0;
	};

	static inline bool							_enabled = false;
	static inline std::mutex					_lock;
	static inline std::map<std::string,Entry>	_entries;

public:
	static bool Enabled()
	{
		return _enabled;
	}

	static void Enable(const bool enabled)
	{
		_enabled = enabled;
	}

	static void Reset()
	{
		std::unique_lock<std::mutex> lock(_lock);
		_entries.clear();
	}

	static void Record(const std::string& name,const double seconds)
	{
		std::unique_lock<std::mutex> lock(_lock);
		Entry& e = _entries[name];
		e.seconds += seconds;
		e.calls++;
	}

	static void Print(std::ostream& os=std::cout)
	{
		std::unique_lock<std::mutex> lock(_lock);
		std::vector<std::pair<std::string,Entry>> sorted(_entries.begin(),_entries.end());
		std::sort(sorted.begin(),sorted.end(),[](const auto& a,const auto& b){return a.second.seconds>b.second.seconds;});
		double total = 0.0;
		for(auto& e:sorted)
			total += e.second.seconds;
		os<<std::left<<std::setw(40)<<"kernel"<<std::right<<std::setw(12)<<"seconds"<<std::setw(10)<<"calls"<<std::setw(8)<<"%"<<std::endl;
		for(auto& e:sorted)
			os<<std::left<<std::setw(40)<<e.first<<std::right<<std::setw(12)<<std::fixed<<std::setprecision(4)<<e.second.seconds
				<<std::setw(10)<<e.second.calls<<std::setw(8)<<std::setprecision(1)<<(100.0*e.second.seconds/total)<<std::endl;
		os<<std::defaultfloat;
	}

	// RAII timer.
	class Scope
	{
		const char*								_name;
		const std::chrono::steady_clock::time_point	_start;
	public:
		Scope(const char* name) :
			_name(name),
			_start(std::chrono::steady_clock::now())
		{
		}
		~Scope()
		{
			if(_name)
				Record(_name,std::chrono::duration<double>(std::chrono::steady_clock::now()-_start).count());
		}
	};
};
