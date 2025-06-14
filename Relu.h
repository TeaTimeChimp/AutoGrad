#pragma once


#include "Layer.h"


typedef std::shared_ptr<class Relu> ReluPtr;
class Relu : public Layer
{
	class P
	{
	};
public:
	static ReluPtr New()
	{
		return std::make_shared<Relu>(P());
	}

	Relu(const P&)
	{
	}

	const Tensors Forward(const TensorPtr& input) const override
	{
		return input->Relu();
	}

	const std::vector<TensorPtr> GetParameters() const override
	{
		return {};
	}
};