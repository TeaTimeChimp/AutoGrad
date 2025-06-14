#pragma once

#include "Layer.h"


class MSELoss : public Layer
{
public:
	MSELoss()
	{
	}

	const Tensors Forward(const TensorPtr& x) const
	{
		return x;
	}

	TensorPtr Forward(const TensorPtr& pred,const TensorPtr& target) const
	{
		TensorPtr error = pred->Sub(target);
		return error->Pow(2)->Mul(Tensor::New(NDData::New({1,1},0.5),true))->Mean(true);
	}

	const std::vector<TensorPtr> GetParameters() const override
	{
		return {};
	}
};
