#pragma once


class Optimiser
{
protected:
	const std::vector<TensorPtr>	_parameters;

public:
	Optimiser(const std::vector<TensorPtr>& parameters) :
		_parameters(parameters)
	{
	}

	// Clip gradients components to norm (scaled to norm unit vector).
	//
	void ClipGrad(const FP clipNorm)
	{
		for(auto& p:_parameters)
		{
			const TensorPtr& gradient = p->Gradient();
			if(gradient)
				gradient->Data()._ClipNorm(clipNorm);
		}
	}

	void ZeroGrad()
	{
		for(auto& p:_parameters)
		{
			const TensorPtr& gradient = p->Gradient();
			if(gradient)
				gradient->Data() *= 0.0;
		}
	}

	virtual void Step(bool zero=true) = 0;
};