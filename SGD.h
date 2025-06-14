#pragma once

#include <vector>
#include "Tensor.h"
#include "Optimiser.h"


class SGD : public Optimiser
{
	const FP _alpha;

public:
	SGD(const std::vector<TensorPtr>& parameters,FP alpha) :
		Optimiser(parameters),
		_alpha(alpha)
	{
	}

	void Step(bool zero=true)
	{
		for(auto p:_parameters)
		{
			// Reduce data by fraction of gradient to steer loss toward 0.
			p->Data() -= p->Gradient()->Data()*_alpha;
			if(zero)
				p->Gradient()->Data() *= 0.0;
		}
	}
};