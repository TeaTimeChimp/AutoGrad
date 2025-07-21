#include "NDArray.h"
#include "Test.h"


using namespace std;


void Test_Broadcast()
{
	{
		NDArray x = NDData::New({2},
			{
				1,2
			});
		NDArray y = NDData::New({3,2});

		NDArray z = (*x).Broadcast(NDDataPtrC((*y).shared_from_this()));

		Assert(z.IsEqualTo(NDData::New({3,2},
		{
			1,2,
			1,2,
			1,2
		})),"Broadcast(scalar,matrix).");
	}
}