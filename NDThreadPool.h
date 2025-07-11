#pragma once

#include <Windows.h>
#include <mutex>
#include <vector>
#include <concurrent_queue.h>
#include <iostream>


class NDThreadPool
{
    const int _numHardwareCores = 32;
    static NDThreadPool _pool;

public:
    class TaskGroup;

private:
    struct Task
    {
		TaskGroup&  _taskGroup;

        Task(TaskGroup& taskGroup):
			_taskGroup(taskGroup)
        {
			_taskGroup.AddTask();
        }

        virtual ~Task() = default;

        void Run() const
        {
            Invoke();
            _taskGroup.ReleaseTask();
        }

        virtual void Invoke() const = 0;
    };

public:
    class TaskGroup
    {
		// Task to run a lambda function.
        template<typename T>
        class LambdaTask : public Task
        {
		    const T _lambda;    // Thread-safe copy of lambda function.

        public:
            LambdaTask(TaskGroup& taskGroup,const T& lambda) :
                Task(taskGroup),
                _lambda(lambda)
            {
            }

            void Invoke() const override
            {
                _lambda();
            }
	    };

		std::atomic<int> _taskCount;    // Number of tasks left to run.
    
    public:
        TaskGroup() :
			_taskCount(0)
        {
		}

		// Destructor, waits for all tasks in this group to complete.
        ~TaskGroup()
        {
            // Process work from the queue until all tasks in this group are complete.
            while(_taskCount>0)
                _pool.ServiceWorkQueue(false);
        }

        void AddTask()
        {
            ++_taskCount;
        }

		void ReleaseTask()
        {
            --_taskCount;
        }

		// Run a task in this group.
        template<typename T>
        void Run(const T& lambda)
        {
            _pool.Run(new LambdaTask<T>(*this,lambda));
		}
    };

private:
    std::mutex                              _workersMutex;
    std::condition_variable                 _workersCondition;
    std::vector<std::thread>                _workers;
    std::mutex                              _workMutex;
    std::condition_variable                 _workCondition;
    concurrency::concurrent_queue<Task*>	_work;
    bool                                    _stop;

	// Task to run a lambda function for each value in a range.
	template<typename T>
    class ForEachTask : public Task
    {
        const int   _begin;
        const int   _end;
		const T&    _lambda;    // Lambda reference is thread-safe because it's scoped to the for loop.

    public:
        ForEachTask(TaskGroup& taskGroup,const int begin,const int end,const T& lambda) :
            Task(taskGroup),
            _begin(begin),
            _end(end),
            _lambda(lambda)
        {
        }

        void Invoke() const override
        {
            // Run user task for each value in range.
            for(int i=_begin;i<_end;++i)
                _lambda(i);
		}
    };


    bool ServiceWorkQueue(const bool waitForWork)
    {
        Task* task = nullptr;

		// Look for work on the queue - without blocking.
		if(!_work.try_pop(task)&&waitForWork)
        {
            // Wait for work to be added to the queue - blocking.
            std::unique_lock<std::mutex> lk(_workMutex);       // Take work lock before waiting on the queue.
            _workCondition.wait(lk,[this,waitForWork]()        // Release lock and suspend this thread until notified. When notified this thread will own the lock again.
            {
                return !_work.empty()||_stop;                   // Don't enter wait if either of these conditions are true.
            });     

            // Check queue for work - this thread now owns '_workMutex'.
			_work.try_pop(task);
        } // Release '_workMutex'.

        // Run work popped from the queue.
        if(task)
        {
            // Run user task.
			task->Run();

			// Delete the task - task complete and no longer needed.
			delete task;
            return true;
        }
        else
            return false;
    }


    struct WorkerThreadArgs
    {
        NDThreadPool*   _threadPool;
        char            _coreNumber;
    };


    static void WorkerLoop(WorkerThreadArgs* args)
    {
        // Copy args - input is temporary.
        const int coreNumber = args->_coreNumber;
        NDThreadPool* threadPool = args->_threadPool;
        args = nullptr;

        {
            // Initialise this thread and then the notify controller.
			std::unique_lock<std::mutex> lk(threadPool->_workersMutex);     // Wait for main thread to release this lock.
            SetThreadAffinityMask(GetCurrentThread(),0x1ULL<<coreNumber);
            lk.unlock();
			threadPool->_workersCondition.notify_one();                     // Inform main thread that this worker is running.
        }        
    
        // Process work until stopped.
        while(!threadPool->_stop)
            threadPool->ServiceWorkQueue(true);
    }

    template<typename T>
    void _ForEach(const int begin,const int end,const T& lambda)
    {
        // Split range.
        const int subRange = (std::max)((end-begin+_numHardwareCores-1)/_numHardwareCores,1);
        const int taskCount = (std::min)((end-begin)+subRange-1/subRange,_numHardwareCores);
        if(subRange*taskCount<(end-begin))
            throw "Error!";
		
        {// Scoped TaskGroup.
		    TaskGroup taskGroup;

            // Enqueue tasks.
            for(int i=0;i<taskCount;++i)
			    Run(new ForEachTask<T>(taskGroup,i*subRange,(std::min)((i+1)*subRange,end),lambda));
		}// Waits for TaskGroup.
    }

public:
    NDThreadPool() :
        _stop(false)
    {
        // Place this thread on core 0.
        SetThreadAffinityMask(GetCurrentThread(),0x1ULL);

        // Start n threads, one on each core.
        WorkerThreadArgs args {this,0};
        for(int i=1;i<_numHardwareCores;++i)
        {
            std::unique_lock<std::mutex> lk(_workersMutex);
            args._coreNumber = i;
            _workers.emplace_back(WorkerLoop,&args);    // Create a new thread running the WorkerLoop.
            _workersCondition.wait(lk);                 // Wait for the new worker to signal it's running.
        }
    }

    ~NDThreadPool()
    {
        {
            std::unique_lock<std::mutex> lk(_workMutex);
            _stop = true;
            lk.unlock();
            _workCondition.notify_all();
        }

        for(auto& worker:_workers)
            worker.join();
    }

	void Run(Task* task)
    {
		_work.push(task);               // Add task to the queue.
        _workCondition.notify_one();    // Notify a worker that work is ready.
    }

    template<typename T>
    static void ForEach(const int begin,const int end,const T& lambda)
    {
        // Call instance method.
		_pool._ForEach(begin,end,lambda);
    }
};

