#pragma once

#include "Layer.h"


typedef std::shared_ptr<class Tanh> TanhPtr;

class Tanh : public Layer
{
public:
	static TanhPtr New()
	{
		return std::make_shared<Tanh>();
	}

	const Tensors Forward(const TensorPtr& input) const override
	{
		return input->Tanh();
	}

	const std::vector<TensorPtr> GetParameters() const override
	{
		return {};
	}
};