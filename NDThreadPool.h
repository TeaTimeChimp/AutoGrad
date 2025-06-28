#pragma once

#include <Windows.h>
#include <mutex>
#include <vector>
#include <queue>
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
			_taskGroup.IncrementRemainingTasks();
        }

        void Run()
        {
            Invoke();
            _taskGroup.DecrementRemainingTasks();
        }

        virtual void Invoke() = 0;
    };

public:
    class TaskGroup
    {
        std::mutex              _remainingTasksMutex;
        std::condition_variable _remainingTasksCondition;
        volatile int            _remainingTasks;

    public:
        TaskGroup() :
            _remainingTasks(0)
        {
		}

        void IncrementRemainingTasks()
        {
            std::unique_lock<std::mutex> lk(_remainingTasksMutex);
			++_remainingTasks;
        }

		void DecrementRemainingTasks()
        {
            // Decrement task group remaining task counter and notify control thread when complete.
            std::unique_lock<std::mutex> lk(_remainingTasksMutex);
            if(--_remainingTasks==0)
            {
                lk.unlock();    // Release control lock (before notify).
                _remainingTasksCondition.notify_one();
            }
        }
        

        template<typename T>
        struct LambdaTask : Task
        {
			const T _lambda;    // Thread-safe copy of lambda function.

            LambdaTask(TaskGroup& taskGroup,const T& lambda) :
                Task(taskGroup),
                _lambda(lambda)
            {
            }

            void Invoke() override
            {
                _lambda();
            }
		};


        template<typename T>
        void Run(const T& lambda)
        {
            _pool.Run(new LambdaTask<T>(*this,lambda));
		}

        void Wait()
        {
            // Process work from the queue to avoid deadlock (if all threads are waiting for work to complete then there are no threads available to do the work).
            while(_pool.ServiceWorkQueue(false));

            std::unique_lock<std::mutex> lk(_remainingTasksMutex);
            _remainingTasksCondition.wait(lk,[this]()
            {   
                return _remainingTasks==0;       // Don't enter wait if task is already complete.
            });
            if(_remainingTasks!=0)
                throw "Error!";
        }
    };

private:
    std::mutex                  _workersMutex;
    std::condition_variable     _workersCondition;
    std::vector<std::thread>    _workers;
    std::mutex                  _workMutex;
    std::condition_variable     _workCondition;
    std::queue<Task*>           _work;
    bool                        _stop;


	template<typename T>
    struct ForEachTask : Task
    {
        int      _begin;
        int      _end;
		const T& _lambda;

        ForEachTask(TaskGroup& taskGroup,const int begin,const int end,const T& lambda) :
            Task(taskGroup),
            _begin(begin),
            _end(end),
            _lambda(lambda)
        {
        }

        void Invoke() override
        {
            // Run user task for each value in range.
            for(int i=_begin;i<_end;++i)
                _lambda(i);
		}
    };


    bool ServiceWorkQueue(const bool waitForWork)
    {
        Task* task = nullptr;

        {
            // Wait for work to be added to the queue.
            std::unique_lock<std::mutex> lk(_workMutex);       // Take work lock before waiting on the queue.
            _workCondition.wait(lk,[this,waitForWork]()        // Release lock and suspend this thread until notified. When notified this thread will own the lock again.
            {
                return !waitForWork||!_work.empty()||_stop;     // Don't enter wait if any of these conditions are true.
            });     

            // Check queue for work - this thread now owns '_queueMutex'.
            if(!_work.empty())
            {
                task = _work.front();
                _work.pop();
            }
        } // Release '_queueMutex'.

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
        int subRange = (std::max)((end-begin+_numHardwareCores-1)/_numHardwareCores,1);
        int taskCount = (std::min)((end-begin)+subRange-1/subRange,_numHardwareCores);
        if(subRange*taskCount<(end-begin))
            throw "Error!";

		// Create a task group to track remaining tasks.
		TaskGroup taskGroup;

        // Enqueue tasks.
        for(int i=0;i<taskCount;++i)
			Run(new ForEachTask<T>(taskGroup,i*subRange,(std::min)((i+1)*subRange,end),lambda));

        // Wait for completion.
        taskGroup.Wait();
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
        std::unique_lock<std::mutex> lk(_workMutex);    // Lock work queue.
		_work.emplace(task);                            // Add task to the queue.
		lk.unlock();                                    // Unlock work queue.
        _workCondition.notify_one();                    // Notify a worker that work is ready.
    }

    template<typename T>
    static void ForEach(const int begin,const int end,const T& lambda)
    {
        // Call instance method.
		_pool._ForEach(begin,end,lambda);
    }
};

