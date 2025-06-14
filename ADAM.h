#pragma once

#include <vector>
#include "Tensor.h"
#include "Optimiser.h"


class ADAM : public Optimiser
{
	const FP		_eps = 1e-8;
	const double	_alpha;
	const double	_beta1;
	const double	_beta2;
	int				_t;

public:
	ADAM(const std::vector<TensorPtr>& parameters,double alpha=0.001) :
		Optimiser(parameters),
		_alpha(alpha),
		_beta1(0.9),
		_beta2(0.999),
		_t(0)
	{
	}

	void Step(bool zero=true)
	{
		for(auto& p:_parameters)
		{
			++_t;

			p->Momentum()->Data() = p->Momentum()->Data()*_beta1 + p->Gradient()->Data()*(1-_beta1);
			p->Momentum2()->Data() = p->Momentum2()->Data()*_beta2 + ((p->Gradient()->Data()*p->Gradient()->Data())*(1-_beta2));

			FP beta1t = pow(_beta1,_t);
			FP beta2t = pow(_beta2,_t);

			NDArray mhat = p->Momentum()->Data()/(1-beta1t);
			NDArray vhat = p->Momentum2()->Data()/(1-beta2t);

			p->Data() -= (mhat*_alpha)/(vhat.Sqrt()+_eps);
			
			if(zero)
				p->Gradient()->Data() *= 0.0;
		}
	}
};