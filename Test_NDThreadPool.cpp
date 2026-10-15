#include "Test_NDThreadPool.h"
#include "NDThreadPool.h"


void foo(const int i)
{
	Sleep(10000*(i+1));	// Add a long wait for debugging.
}


void Test_NDThreadPool()
{
	// Zero tasks, construct/destruct TaskGroup.
	NDThreadPool::ForEach(0,0,foo);

	NDThreadPool::ForEach(0,1,foo);

	NDThreadPool::ForEach(0,2,[](const int i)
	{
		Sleep(1000*(i+1));
	});


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
