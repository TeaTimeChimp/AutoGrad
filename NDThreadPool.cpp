#include "NDThreadPool.h"


NDThreadPool NDThreadPool::_pool;
thread_local int NDThreadPool::_currentThreadNumber;