#include "Matrix.h"


void Test_Matrix()
{
	{
		// Transpose.
		Matrix d(2,3,{1,2,3,4,5,6});
		Matrix a = d.Transpose();

		Matrix e(3,2,{1,3,5,2,4,6});
		if(a!=e)
			throw "Transpose";
	}

	/* Norm removed
	{
		// Norm.
		Matrix x(1,5,{1,2,3,4,5});
		Matrix y = x.Norm(1);
		y.Print();
	}
	*/

	{
		// ClipNorm.
		Matrix x(1,5,{1,2,3,4,5});
		x.ClipNorm(2.0);
		x.Print();
	}
}