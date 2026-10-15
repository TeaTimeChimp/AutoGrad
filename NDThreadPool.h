#pragma once

#include <Windows.h>
#include <mutex>
#include <vector>
#include <queue>
#include <iostream>


// Alignment friendly number of bytes a Task instance should be.
//  - Must be large enough to copy lambda (64 was too small).
//  - 64 byte aligned in case lambda needs maximum alignment.
//  - Size must be a multiple of alignment so arrays are space efficient.
constexpr size_t TASK_ALIGNMENT=64;
constexpr size_t TASK_SIZEOF=TASK_ALIGNMENT*2;

class NDThreadPool
{
    const int _numHardwareCores;

public:
    // Type erased task for dispatching anonymous type lambdas.
    //  - No virtial functions.
    //  - Task objects can be stored in a homogenous array.
    class Task
    {
        alignas(TASK_ALIGNMENT) char _storage[TASK_SIZEOF-sizeof(void(*)())*2]; // Lambda storage - the 2 function pointers.
        void (*_lambdaInvoke)(const void*);                                     // Pointer to lambda to type dispatch invoke.
        void (*_lambdaDestructor)(void*);                                       // Pointer to lambda to type dispatch destructor.

        Task(const Task&) = delete;
        void operator=(const Task&) = delete;

        inline void Run() const
        {
            _lambdaInvoke(_storage);
        }

        inline void Reset()
        {
            if(_lambdaDestructor)
            {
                _lambdaDestructor(_storage);
                _lambdaInvoke = nullptr;
                _lambdaDestructor = nullptr;
            }
        }

    public:
        inline Task():
            _lambdaInvoke(nullptr),
            _lambdaDestructor(nullptr)
        {
            static_assert(sizeof(Task)==TASK_SIZEOF,"Unexpected Task size.");
        }

        inline ~Task()
        {
            Reset();
        }

        template<typename T>
        inline void Set(T&& lambda)
        {
            static_assert(sizeof(T)<=sizeof(_storage),"Lambda is too large for Task storage buffer!");
            _ASSERT(!_lambdaInvoke&&!_lambdaDestructor);

            new (_storage) T(std::forward<T>(lambda));
            _lambdaInvoke = [](const void* storage)
            {
                (*(T*)storage)();
            };
            _lambdaDestructor = [](void* storage)
            {
                ((T*)storage)->~T();
            };
        }

        inline void RunAndReset()
        {
            Run();
            Reset();
        }
    };

    class TaskGroup
    {
    public:
        static const int MAX_TASKS = 32;            // Need a static value for array allocation.

    private:
        NDThreadPool&       _pool;
        const int           _creatorWorkerNumber;   // Worker thread that created this group.
        std::atomic<int>    _taskCounter;           // Number of tasks running.
        bool    			_taskGroupComplete;     // Flag to indicate all tasks have completed and the group can be destroyed.
        
        Task                _completionTask;        // Task to signal completion of the group.
        
        std::atomic<int>    _nextNewTaskIndex=0;
        Task                _tasks[MAX_TASKS];      // Work will only be split into N tasks depending on CPU.

        TaskGroup(const TaskGroup&) = delete;       // Copying a TaskGroup would break the guarantee that
        void operator=(const TaskGroup&) = delete;  // Task lambda instances are in scope.

    public:
        TaskGroup():
            _pool(NDThreadPool::Get()),
            _creatorWorkerNumber(_pool.GetCurrentWorkerNumber()),
            _taskCounter(1),                    // Count creation as a Task to prevent the counter dipping to 0 before the TaskGroup wait.
            _taskGroupComplete(false)
        {
            _completionTask.Set([this]()
            {
                _taskGroupComplete=true;    // Singal all Tasks in TaskGroup complete, breaks destructors ServiceWorkQueue loop.
            });
        }

        // Destructor, waits for all tasks in this group to complete.
        ~TaskGroup()
        {
            // Release creation pseudo-Task to allow task counter to reach 0.
            DecrementTaskCounter();
            
            // Process work from the queue until all Tasks in this TaskGroup are complete.
            //  - When '_taskCounter' reaches zero the 'CompletionTask' runs and sets '_taskGroupComplete' to 'true'.
            while(!_taskGroupComplete)
                _pool.ServiceWorkQueue();   // Do useful work, don't just wait.
        }

        // Increment the group's task counter - work queued.
        int IncrementTaskCounter()
        {
            return _taskCounter++;
        }

        // Decrement the group's task counter - work completed.
        void DecrementTaskCounter()
        {
            // When all tasks have complete post a completion task to the creator's work queue.
            //  - This thread can be the creator, so could be queuing work for itself.
            if(--_taskCounter==0)
                _pool.GetWorkerContext(_creatorWorkerNumber).Enqueue(&_completionTask);
        }

        // Run a task in this group.
        template<typename T>
        void Run(const T& lambda)
        {
            const int taskIndex = _nextNewTaskIndex++;
            IncrementTaskCounter();
            _tasks[taskIndex].Set([this,lambda]()   // Capture 'lambda' by copy, the lambda passed in is temporary.
            {
                lambda();
                DecrementTaskCounter();
                // No access to Task or TaskGroup after task counter decrement.
            });
            _pool.Run(&_tasks[taskIndex]);
        }
    };

    class WorkerThreadContext
    {
		NDThreadPool&           _pool;              // Reference to the thread pool that owns this context.
        const int				_threadNumber;      // Thread number associated with this context (debugging).
        int                     _nextWorkerNumber;  // Next worker to queue work with (round-robin).

        // Thread local work queue.
        std::mutex              _workMutex;
        std::condition_variable _workCondition;
        std::queue<Task*>	    _workQueue;

    public:
        WorkerThreadContext(NDThreadPool& pool,const int threadNumber):
			_pool(pool),
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
            _nextWorkerNumber = (_nextWorkerNumber+1)%_pool._numHardwareCores;
            return _nextWorkerNumber;
        }

        bool ServiceWorkQueue()
        {
            Task* task = nullptr;

            // Wait for work to be added to the queue - blocking.
            {
                std::unique_lock<std::mutex> lk(_workMutex);    // Take work lock before waiting on the queue.
                _workCondition.wait(lk,[this]()                 // Release lock and suspend this thread until notified. When notified this thread will own the lock again.
                {
                    return !_workQueue.empty()||_pool._stop;    // Don't enter wait if either of these conditions are true.
                });
                if(!_workQueue.empty()&&!_pool._stop)
                {
                    task = _workQueue.front();
                    _workQueue.pop();
                }
            } // Release '_workMutex'.

            // Run work popped from the queue.
            if(task)
            {
                // Run user task.
                task->RunAndReset();
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
    std::atomic<bool>                   _stop;

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
        SetCurrentThreadNumber(threadNumber);
        _workerContexts.emplace_back(new WorkerThreadContext(*this,threadNumber));
    }

    // Entry point for a new worker thread.
    //
    static void WorkerThreadEntryPoint(void* arg)
    {
        NDThreadPool& pool = *(NDThreadPool*)arg;

        // Associate this thread with a worker context.
        pool.CreateThreadWorkerContext();

        // Process work until stopped.
        pool.GetCurrentWorkerContext().ServiceQueueUntilStopped();
    }

    template<typename T>
    void _ForEach(const int begin,const int end,const T& lambda)
    {
        // Split range.
        // TODO: Need a lower limit on viable subRange size.
        const int subRange = (std::max)((end-begin+TaskGroup::MAX_TASKS-1)/TaskGroup::MAX_TASKS,1);
        const int taskCount = (std::min)((end-begin+subRange-1)/subRange,TaskGroup::MAX_TASKS);
        if(subRange*taskCount<(end-begin))
            throw "Error!";
		
        {// Scoped TaskGroup.
		    TaskGroup taskGroup;

            // Enqueue tasks.
            for(int i=0;i<taskCount;++i)
            {
                const int rangeBegin = begin+(i*subRange);
                const int rangeEnd = (std::min)(begin+((i+1)*subRange),end);
			    taskGroup.Run([rangeBegin,rangeEnd,&lambda]()   // See comment inside 'Run', the lambda passed here (not the captured parameter '&lambda') is temporary.
                {
                    for(int i=rangeBegin;i<rangeEnd;++i)
                        lambda(i);
                });
            }
		}// Waits for TaskGroup.
    }

public:
    NDThreadPool():
		_numHardwareCores((int)(std::max)(1u,std::thread::hardware_concurrency())),
        _stop(false)
    {
        // Create the first worker context for this thread.
        CreateThreadWorkerContext();

        // Start n more worker threads.
        for(int i=1;i<_numHardwareCores;++i)
        {
            std::unique_lock<std::mutex> lk(_workersMutex);
            _workerThreads.emplace_back(WorkerThreadEntryPoint,this);    // Create a new thread running the WorkerLoop.
            _workersCondition.wait(lk);                                  // Wait for the new worker to signal it's running.
        }
    }

    ~NDThreadPool()
    {
        // Signal workers to stop, otherwise their wait predicate never becomes true and join() blocks forever.
        _stop = true;

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

    static NDThreadPool& Get()
    {
		static NDThreadPool singleton;
        return singleton;
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
		Get()._ForEach(begin,end,lambda);
    }
};

