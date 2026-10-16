#pragma once

#include <Windows.h>
#include <vector>
#include <stdexcept>
#include <thread>


// Alignment friendly number of bytes a Task instance should be.
//  - Must be large enough to copy lambda (64 was too small).
//  - 64 byte aligned in case lambda needs maximum alignment.
//  - Size must be a multiple of alignment so arrays are space efficient.
constexpr size_t TASK_ALIGNMENT=64;
constexpr size_t TASK_SIZEOF=TASK_ALIGNMENT*2;

class NDThreadPool
{
    const int _numHardwareThreads;

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
            _taskCounter(1),                        // Count creation as a Task to prevent the counter dipping to 0 before the TaskGroup wait.
            _taskGroupComplete(false)
        {
            _completionTask.Set([this]()
            {
                _taskGroupComplete=true;            // Signal all Tasks in TaskGroup complete, breaks destructors ServiceWorkQueue loop.
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

    // Static queue. 
    // TODO: Template this on type and size.
    class StaticQueue
    {
        static const size_t     SIZE=8;             // Static sizeof queue.
        static const size_t     SLOT_MASK=SIZE-1;   // Masks slot/index from read/write indexes.

        Task*	                _slots[SIZE];       // Static array of items.
        size_t                  _readIndex;         // Read index, increment-only.
        size_t                  _writeIndex;        // Write index, increment-only.

    public:
        StaticQueue():
            _readIndex(0),
            _writeIndex(0)
        {
        }

        inline bool empty() const
        {
            _ASSERT(_readIndex<=_writeIndex);
            return _readIndex==_writeIndex;
        }

        inline bool full() const
        {
            return _writeIndex-_readIndex==SIZE;
        }

        inline Task* front() const
        {
            if(empty())
                throw std::runtime_error("Queue empty.");

            return _slots[_readIndex&SLOT_MASK];
        }

        inline void pop()
        {
            if(empty())
                throw std::runtime_error("Queue empty.");

            // Erase and increment read index, queue cannot now be full.
            _slots[_readIndex&SLOT_MASK] = nullptr;
            ++_readIndex;
        }

        inline void push(Task* const task)
        {
            // Throw if queue full - rework client or increase static size.
            if(full())
                throw std::runtime_error("Queue full.");

            // Store and increment write index.
            _slots[_writeIndex&SLOT_MASK] = task;
            ++_writeIndex;
        }
    };


    class CasLock
    {
    public:
        class Locker
        {
            CasLock&    _lock;
            bool        _locked;
        public:
            inline Locker(CasLock& lock):
                _lock(lock),
                _locked(false)
            {
                Acquire();
            }

            inline ~Locker()
            {
                Release();
            }
    
            inline void Acquire()
            {
                if(!_locked)
                {
                    _lock.Acquire();
                    _locked = true;
                }
            }

            inline void Release()
            {
                if(_locked)
                {
                    _lock.Release();
                    _locked = false;
                }
            }
        };

    private:
        static const int NotOwned = -1;
        std::atomic<int> _owner;
    
    public:
        CasLock():
            _owner(NotOwned)
        {
        }

        void Acquire()
        {
            const int myWorkerNumber = NDThreadPool::GetCurrentWorkerNumber();

            int owningWorkerNumber;
            owningWorkerNumber = NotOwned;
            for(;;)
            {
                // Try to take ownership.
                if(_owner.compare_exchange_strong(owningWorkerNumber,myWorkerNumber))
                    break;

                // Contention, wait until the lock appears free.
                while(_owner.load(std::memory_order_relaxed)!=NotOwned)
                    _mm_pause();
            }
        }
    
        void Release()
        {
            const int myWorkerNumber = NDThreadPool::GetCurrentWorkerNumber();
            if(_owner!=myWorkerNumber)
                throw std::runtime_error("Not owned.");
            _owner = NotOwned;
        }
    };


    class CasCondition
    {
        std::atomic<bool>   _signal;
    public:
        CasCondition():
            _signal(false)
        {
        }

        void Wait(CasLock::Locker& locker)
        {
            // Release lock and sleep until notified.
            locker.Release();

            // Sleep if contition is false.
            bool notSignalled = false;
            do
            {
                WaitOnAddress(&_signal,&notSignalled,sizeof(bool),INFINITE);    // Waits if _signal==notSignalled.
            } while(!_signal);  // Loop on spurious wakes.

            // Re-aquire lock.
            locker.Acquire();

            // Reset signal.
            _signal = false;
        }

        template<typename T>
        void Wait(CasLock::Locker& locker,const T& noWait)
        {
            while(!noWait())
                Wait(locker);
        }

        void NotifyOne()
        {
            // Set signal and wake waiting thread.
            _signal = true;
            WakeByAddressSingle(&_signal);
        }
    };


    class WorkerThreadContext
    {
		NDThreadPool&   _pool;              // Reference to the thread pool that owns this context.
        const int		_workerNumber;      // Worker number associated with this context.
        int             _nextWorkerNumber;  // Next worker to queue work with (round-robin).

        // Thread local work queue.
        CasLock         _workMutex;
        CasCondition    _workCondition;
        StaticQueue     _workQueue;         // Static queue, work queue should be small.

    public:
        WorkerThreadContext(NDThreadPool& pool,const int workerNumber):
			_pool(pool),
            _workerNumber(workerNumber),
            _nextWorkerNumber(workerNumber)
        {
            // Wait for main thread to release before signalling.
            if(workerNumber!=0)
            {
                CasLock::Locker lk(_pool._workersMutex);
                lk.Release();
                _pool._workersCondition.NotifyOne();
            }
        }

        int GetNextWorkerNumber()
        {
            _nextWorkerNumber = (_nextWorkerNumber+1)%_pool._numHardwareThreads;
            return _nextWorkerNumber;
        }

        bool ServiceWorkQueue()
        {
            Task* task = nullptr;

            // Wait for work to be added to the queue - blocking.
            {
                CasLock::Locker lk(_workMutex);                 // Take work lock before waiting on the queue.
                _workCondition.Wait(lk,[this]()                 // Release lock and suspend this thread until notified. When notified this thread will own the lock again.
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
            CasLock::Locker lk(_workMutex);
            _workCondition.NotifyOne();
        }

        void Enqueue(Task* const task)
        {
            CasLock::Locker lk(_workMutex);
            const bool wasEmpty = _workQueue.empty();
            _workQueue.push(task);
            if(wasEmpty)
                _workCondition.NotifyOne();
        }
    };

    static thread_local int _currentWorkerNumber;
    void SetCurrentWorkerNumber(const int workerNumber)
    {
        // Set per-thread thread number and thread affinity.
        _currentWorkerNumber = workerNumber;
        SetThreadAffinityMask(GetCurrentThread(),0x1ULL<<workerNumber);
    }

    // Returns the callers 'worker number' stored in TLS.
    //
    static int GetCurrentWorkerNumber()
    {
        return _currentWorkerNumber;
    }

    WorkerThreadContext& GetWorkerContext(const int threadNumber) const
    {
        return *_workerContexts[threadNumber];
    }

    WorkerThreadContext& GetCurrentWorkerContext() const
    {
        return GetWorkerContext(_currentWorkerNumber);
    }

private:
    CasLock                             _workersMutex;
    CasCondition                        _workersCondition;
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
        SetCurrentWorkerNumber(threadNumber);
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
		_numHardwareThreads((int)(std::max)(1u,std::thread::hardware_concurrency())),
        _stop(false)
    {
        // Create the first worker context for this thread (main).
        CreateThreadWorkerContext();

        // Start n more worker threads.
        for(int i=1;i<_numHardwareThreads;++i)
        {
            CasLock::Locker lk(_workersMutex);
            _workerThreads.emplace_back(WorkerThreadEntryPoint,this);    // Create a new thread running the WorkerLoop.
            _workersCondition.Wait(lk);                                  // Wait for the new worker to signal it's running.
        }
        _ASSERT(_workerContexts.size()==_numHardwareThreads);
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
