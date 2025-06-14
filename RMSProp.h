#pragma once

#include <vector>
#include "Tensor.h"
#include "Optimiser.h"


class RMSProp : public Optimiser
{
	const double _alpha;
	const double _beta;

public:
	RMSProp(const std::vector<TensorPtr>& parameters,double alpha) :
		Optimiser(parameters),
		_alpha(alpha),
		_beta(0.99)
	{
	}

	void Step(bool zero=true)
	{
		for(auto& p:_parameters)
		{
			// Reduce data by fraction of gradient to steer loss toward 0.
			p->Momentum()->Data() = p->Momentum()->Data()*_beta + (p->Gradient()->Data()*p->Gradient()->Data())*(1.0-_beta);
			//p->Momentum()->Print();
			p->Data() -= (p->Gradient()->Data()/(p->Momentum()->Data().Sqrt()+1e-08))*_alpha;
			if(zero)
				p->Gradient()->Data() *= 0.0;
		}
	}
};