#pragma once

#include <Windows.h>
#include <mutex>
#include <vector>
#include <queue>
#include <iostream>


class NDThreadPool
{
    const int _numHardwareCores = 32;

    std::mutex                  _controlMutex;
    std::condition_variable     _controlCondition;
    std::mutex                  _queueMutex;
    std::condition_variable     _queueCondition;
    std::vector<std::thread>    _workers;
    bool                        _stop;

    class ITaskHandle
    {
    public:
        virtual void Invoke(const int i) = 0;
    };

    struct Task
    {
        int*            _counter;
        int             _begin;
        int             _end;
        ITaskHandle*    _taskHandle;

        Task() :
            _counter(0),
            _begin(0),
            _end(0),
            _taskHandle(nullptr)
        {
        }

        Task(int* counter,const int begin,const int end,ITaskHandle* const taskHandle) :
            _counter(counter),
            _begin(begin),
            _end(end),
            _taskHandle(taskHandle)
        {
        }
    };
    std::queue<Task> _work;


    bool ServiceWorkQueue(const bool waitForWork)
    {
        Task task;

        {
            // Wait for work to be added to the queue.
            std::unique_lock<std::mutex> lk(_queueMutex);       // Take work lock.
            _queueCondition.wait(lk,[this,waitForWork]()        // Release lock and suspend this thread until notified. When notified this thread will own the lock again.
            {
                return !waitForWork||!_work.empty()||_stop;}    // Don't enter wait if any of these conditions are true.
            );     

            // Check queue for work.
            if(!_work.empty())
            {
                task = _work.front();
                _work.pop();
            }
        }

        // Run work popped from the queue.
        if(task._taskHandle)
        {
            // Run user task.
            for(int i=task._begin;i<task._end;++i)
                task._taskHandle->Invoke(i);
            delete task._taskHandle;
            task._taskHandle = nullptr;

            // Decrement task group counter and notify control thread when complete.
            std::unique_lock<std::mutex> lk(_controlMutex);    // Take control lock.
            if(--*(task._counter)==0)
            {
                lk.unlock();                                   // Release control lock (before notify).
                _controlCondition.notify_all();                // Notify all controller(s) - there may be multiple waiting.
            }
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
            std::unique_lock<std::mutex> lk(threadPool->_controlMutex);
            SetThreadAffinityMask(GetCurrentThread(),0x1ULL<<coreNumber);
            lk.unlock();
            threadPool->_controlCondition.notify_one();
        }        
    
        // Process work until stopped.
        while(!threadPool->_stop)
            threadPool->ServiceWorkQueue(true);
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
            std::unique_lock<std::mutex> lk(_controlMutex);
            args._coreNumber = i;
            _workers.emplace_back(WorkerLoop,&args);
            _controlCondition.wait(lk);
        }
    }

    ~NDThreadPool()
    {
        {
            std::unique_lock<std::mutex> lk(_queueMutex);
            _stop = true;
            lk.unlock();
            _queueCondition.notify_all();
        }

        for(auto& worker:_workers)
            worker.join();
    }

    template<typename T>
    class TaskHandle : public ITaskHandle
    {
        const T& _task;

    public:
        TaskHandle<T>(const T& task) :
            _task(task)
        {
        }

        inline void Invoke(const int i) override
        {
            _task(i);
        }
    };

    template<typename T>
    void foreach(const int begin,const int end,const T& task)
    {
        // Split range.
        int subRange = (std::max)((end-begin+_numHardwareCores-1)/_numHardwareCores,1);
        int taskCount = (std::min)((end-begin)+subRange-1/subRange,_numHardwareCores);
        if(subRange*taskCount<(end-begin))
            throw "Error!";

        // Enqueue tasks.
        int remainingTaskCount = taskCount;
        for(int i=0;i<taskCount-1;++i)
        {
            std::unique_lock<std::mutex> lk(_queueMutex);
            _work.emplace(&remainingTaskCount,i*subRange,(std::min)((i+1)*subRange,end),new TaskHandle(task));
            lk.unlock();
            _queueCondition.notify_one();
        }

        // Run last task on this thread.
        const int taskEnd = (std::min)(taskCount*subRange,end);
        for(int i=(taskCount-1)*subRange;i<taskEnd;++i)
            task(i);
        {
            // Take control lock and decrement task counter.
            std::unique_lock<std::mutex> lk(_controlMutex);
            --remainingTaskCount;
        }

        // Process work from the queue to avoid deadlock (if all threads are waiting for work to complete and there are no threads available to do the work).
        while(ServiceWorkQueue(false));

        // Wait for completion.
        std::unique_lock<std::mutex> lk(_controlMutex);
        _controlCondition.wait(lk,[&remainingTaskCount]()
        {   
            return remainingTaskCount==0;       // Don't enter wait if task is already complete.
        });
        if(remainingTaskCount!=0)
            throw "Error!";
    }
};
extern NDThreadPool pool;
