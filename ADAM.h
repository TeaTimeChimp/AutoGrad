#pragma once

#include <vector>
#include "Tensor.h"
#include "Optimiser.h"
#include "ParallelFor.h"


class ADAM : public Optimiser
{
	const FP	_eps = FP(1e-8);
	FP			_lr;
	const FP	_beta1;
	const FP	_beta2;
	int			_t;

public:
	ADAM(const std::vector<TensorPtr>& parameters,const FP alpha=0.001) :
		Optimiser(parameters),
		_lr(alpha),
		_beta1(FP(0.9)),
		_beta2(FP(0.999)),
		_t(0)
	{
	}

	// Decoupled weight decay (AdamW). Applied only to parameters with 2 or more dimensions (weight matrices),
	// not to biases, LayerNorm gains or 1D vectors.
	FP weightDecay = FP(0);

	void SetLearningRate(const FP alpha)
	{
		_lr = alpha;
	}

	void Step(bool zero=true)
	{
		// The time step counts optimiser steps, not parameters. (Incrementing it inside the parameter loop made the
		// bias corrections below decay N times too fast, where N is the number of parameter tensors.)
		++_t;
		const FP c1 = FP(1)/(FP(1)-pow(_beta1,FP(_t)));
		const FP c2 = FP(1)/(FP(1)-pow(_beta2,FP(_t)));

		for(auto& p:_parameters)
		{
			if(!p->Gradient())
				continue;

#ifdef AUTOGRAD_CUDA
			if(NDConfig::useGpu)
			{
				const FP decay = p->Data().Shape().size()>=2?FP(1)-_lr*weightDecay:FP(1);
				dev::Adam(p->Data().DevModify(),p->Gradient()->Data().DevModify(),p->Momentum()->Data().DevModify(),
					p->Momentum2()->Data().DevModify(),p->Data().Size(),_lr,_beta1,_beta2,_eps,c1,c2,decay);
				p->ClearChildren();
				continue;
			}
#endif
			// Fused update over raw memory (parameters, gradients and moments are all contiguous).
			const int n = p->Data().Size();
			FP* const w = p->Data().Raw();
			FP* const g = p->Gradient()->Data().Raw();
			FP* const m = p->Momentum()->Data().Raw();
			FP* const v = p->Momentum2()->Data().Raw();
			const FP lr = _lr;
			const FP b1 = _beta1, b2 = _beta2, eps = _eps;
			const FP decay = p->Data().Shape().size()>=2?FP(1)-lr*weightDecay:FP(1);
			ParallelForRange(0,n,16384,[=](int begin,int end)
			{
				for(int i=begin;i<end;++i)
				{
					const FP gi = g[i];
					m[i] = b1*m[i]+(FP(1)-b1)*gi;
					v[i] = b2*v[i]+(FP(1)-b2)*gi*gi;
					w[i] = w[i]*decay-lr*(m[i]*c1)/(sqrt(v[i]*c2)+eps);
					if(zero)
						g[i] = FP(0);
				}
			});

			// Parameters record every tensor computed from them (including inference-only passes, which never
			// backpropagate), so clear the record each step to stop it growing forever.
			p->ClearChildren();
		}
	}
};