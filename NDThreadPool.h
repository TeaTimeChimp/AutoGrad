#pragma once

#include <Windows.h>
#include <mutex>
#include <vector>
#include <queue>
#include <iostream>



class NDThreadPool
{
    static const int    _numHardwareCores = 32;
    static NDThreadPool _pool;

public:
    // Base interface class for a queueable task.
    class Task
    {
    protected:
        virtual void Invoke() const = 0;

    public:
        virtual ~Task() = default;

        virtual void Run() const
        {
            Invoke();
        }
    };

    class TaskGroup
    {
	public:
        // TaskGroup task that counts against the number of tasks in the group (i.e some real useful work).
        class CountedTask : public Task
        {
            TaskGroup&  _taskGroup;
        public:
            CountedTask(TaskGroup& taskGroup):
			    _taskGroup(taskGroup)
            {
			    _taskGroup.AddTask();
            }

            void Run() const override
            {
                Task::Run();
                _taskGroup.ReleaseTask();
                delete this;
            }
        };

    private:
        // Uncounted task to signal/wake the potentially waiting creation thread.
        class CompletionTask : public Task
        {
            bool& _taskGroupComplete;

            void Invoke() const override
            {
                _taskGroupComplete = true;
            }

        public:
            CompletionTask(bool& taskGroupComplete) :
			    _taskGroupComplete(taskGroupComplete)
            {
		    }
        };

		// Task to run a lambda function.
        template<typename T>
        class LambdaTask : public CountedTask
        {
		    const T _lambda;    // Thread-safe copy of lambda function.

            void Invoke() const override
            {
                _lambda();
            }

        public:
            LambdaTask(TaskGroup& taskGroup,const T& lambda) :
                CountedTask(taskGroup),
                _lambda(lambda)
            {
            }
	    };

		const int           _creatorWorkerNumber;   // Worker thread that created this group.
		std::atomic<int>    _taskCounter;           // Number of tasks remaining.
		bool    			_taskGroupComplete;     // Flag to indicate all tasks have completed and the group can be destroyed.
		CompletionTask      _completionTask;        // Task to signal completion of the group.
    
    public:
        TaskGroup() :
            _creatorWorkerNumber(NDThreadPool::_pool.GetCurrentWorkerNumber()),
			_taskCounter(1),    // Count creation task to prevent counter dipping to 0 before the taskgroup wait.
			_taskGroupComplete(false),
			_completionTask(_taskGroupComplete)
        {            
		}

		// Destructor, waits for all tasks in this group to complete.
        ~TaskGroup()
        {
            // Release creation task to allow task counter to reach 0.
            ReleaseTask();
            
            // Process work from the queue until all tasks in this group are complete.
            while(!_taskGroupComplete)
                _pool.ServiceWorkQueue();
        }

		// Add a counted task to this group - work queued.
        void AddTask()
        {
            ++_taskCounter;
        }

		// Release a counted task in this group - work completed.
		void ReleaseTask()
        {
			// When all tasks have complete post a completion task to wake the creator thread (this thread can be the creator).
            if(--_taskCounter==0)
                _pool.GetWorkerContext(_creatorWorkerNumber).Enqueue(&_completionTask);
        }

		// Run a task in this group.
        template<typename T>
        void Run(const T& lambda)
        {
            _pool.Run(new LambdaTask<T>(*this,lambda));
		}
    };

    class WorkerThreadContext
    {
		const int				_threadNumber;      // Thread number associated with this context (debugging).
		int                     _nextWorkerNumber;  // Next worker to queue work with (round-robin).

        // Thread local work queue.
        std::mutex              _workMutex;
        std::condition_variable _workCondition;
        std::queue<Task*>	    _workQueue;

    public:
        WorkerThreadContext(const int threadNumber) :
            _threadNumber(threadNumber),
            _nextWorkerNumber(threadNumber)
        {
            // Wait for main thread to release before signalling.
			std::unique_lock<std::mutex> lk(_pool._workersMutex);           
            lk.unlock();
			_pool._workersCondition.notify_one();
        }

        int GetNextWorkerNumber()
        {
            _nextWorkerNumber = (_nextWorkerNumber+1)%_numHardwareCores;
			return _nextWorkerNumber;
        }

        bool ServiceWorkQueue()
        {
            Task* task = nullptr;

            // Wait for work to be added to the queue - blocking.
            {
                std::unique_lock<std::mutex> lk(_workMutex);                // Take work lock before waiting on the queue.
                _workCondition.wait(lk,[this]()                             // Release lock and suspend this thread until notified. When notified this thread will own the lock again.
                {
                    return !_workQueue.empty()||NDThreadPool::_pool._stop;  // Don't enter wait if either of these conditions are true.
                });     
                if(!_workQueue.empty()&&!NDThreadPool::_pool._stop)
                {
                    task = _workQueue.front();
                    _workQueue.pop();
                }
            } // Release '_workMutex'.

            // Run work popped from the queue.
            if(task)
            {
                // Run user task.
			    task->Run();
                return true;
            }
            else
                return false;
        }


        // Process work until stopped.
        //
        void ServiceQueueUntilStopped()
        {
            while(!_pool._stop)
                _pool.GetCurrentWorkerContext().ServiceWorkQueue();
        }

        void Wake()
        {
            std::unique_lock<std::mutex> lk(_workMutex);
            _workCondition.notify_one();
        }

        void Enqueue(Task* const task)
        {
            std::unique_lock<std::mutex> lk(_workMutex);
		    _workQueue.push(task);
            _workCondition.notify_one();
        }
    };

    static thread_local int _currentThreadNumber;
    void SetCurrentThreadNumber(const int threadNumber)
    {
		// Set per-thread thread number and thread affinity.
		_currentThreadNumber = threadNumber;
        SetThreadAffinityMask(GetCurrentThread(),0x1ULL<<threadNumber);
    }

    int GetCurrentWorkerNumber() const
    {
        return _currentThreadNumber;
    }

    WorkerThreadContext& GetWorkerContext(const int threadNumber) const
    {
        return *_workerContexts[threadNumber];
	}

    WorkerThreadContext& GetCurrentWorkerContext() const
    {
        return GetWorkerContext(_currentThreadNumber);
	}

private:
    std::mutex                          _workersMutex;
    std::condition_variable             _workersCondition;
    std::vector<std::thread>            _workerThreads;
    std::vector<WorkerThreadContext*>   _workerContexts;
    bool                                _stop;

	// Task to run a lambda function for each value in a range.
	template<typename T>
    class ForEachTask : public TaskGroup::CountedTask
    {
        const int   _begin;
        const int   _end;
		const T&    _lambda;    // Lambda reference is thread-safe because it's scoped to the for loop.

    public:
        ForEachTask(TaskGroup& taskGroup,const int begin,const int end,const T& lambda) :
            CountedTask(taskGroup),
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

    bool ServiceWorkQueue()
    {
		// Service thread work queue.
		return GetCurrentWorkerContext().ServiceWorkQueue();
    }

	// Create a new worker context for the calling thread.
    //
    void CreateThreadWorkerContext()
    {
		const int threadNumber = (int)_workerContexts.size();
        _pool.SetCurrentThreadNumber(threadNumber);
        _workerContexts.emplace_back(new WorkerThreadContext(threadNumber));
    }

	// Entry point for a new worker thread.
    //
    static void WorkerThreadEntryPoint(void*)
    {
		// Associate this thread with a worker context.
		_pool.CreateThreadWorkerContext();
    
        // Process work until stopped.
        _pool.GetCurrentWorkerContext().ServiceQueueUntilStopped();
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
        // Create the first worker context for this thread.
        CreateThreadWorkerContext();

        // Start n more worker threads.
        for(int i=1;i<_numHardwareCores;++i)
        {
            std::unique_lock<std::mutex> lk(_workersMutex);
            _workerThreads.emplace_back(WorkerThreadEntryPoint,nullptr);    // Create a new thread running the WorkerLoop.
            _workersCondition.wait(lk);                                     // Wait for the new worker to signal it's running.
        }
    }

    ~NDThreadPool()
    {
        // Wake all worker threads.
        for(auto& worker:_workerContexts)
            worker->Wake();

        // Wait for all worker threads to stop.
        for(auto& workerThread:_workerThreads)
            workerThread.join();

		// Delete all worker contexts.
        for(auto& worker:_workerContexts)
            delete worker;
    }

	void Run(Task* const task)
    {
		// Queue the task with the next worker.
		const int threadNumber = GetCurrentWorkerContext().GetNextWorkerNumber();
        GetWorkerContext(threadNumber).Enqueue(task);
    }

    template<typename T>
    static void ForEach(const int begin,const int end,const T& lambda)
    {
        // Call instance method.
		_pool._ForEach(begin,end,lambda);
    }
};

