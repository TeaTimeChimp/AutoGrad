#include "Test_NDThreadPool.h"
#include "NDThreadPool.h"


void Test_NDThreadPool()
{
	// Outer 'foreach' runs 2 nested 'foreach' loops.
	// The inner 'foreach' loops take different times to run.
	// The notification of completion for the first must be ignored by the second.
	NDThreadPool::ForEach(0,2,[](const int i)
	{
		NDThreadPool::ForEach(0,i,[](const int i)
		{
			Sleep(1000*(i+1));
		});
	});
}