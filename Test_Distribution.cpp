#include "CategoricalDistribution.h"
#include <iostream>

using namespace std;


void Test_Distribution()
{
	{
		NormalDistribution p(0,1);
		double z = p.Sample();
	}

	{
		TensorPtr probs = Tensor::New(NDData::New({2,3},{0.0,1.0,0.0,0.0,1.0,0.0}));
		probs->Print();

		CategoricalDistribution dist(probs,false);
		const TensorPtr sample = dist.Sample();
		sample->Print();

		TensorPtr lp = dist.LogProb(sample);
		lp->Print();
	}

	{
		TensorPtr probs = Tensor::New(NDData::New({2,4},{0.1,0.4,0.2,0.3,0.3,0.2,0.4,0.1}));
		probs->Print();

		CategoricalDistribution dist(probs,false);
		const TensorPtr sample = dist.Sample();
		sample->Print();

		TensorPtr lp = dist.LogProb(sample);
		lp->Print();
	}

}