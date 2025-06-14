#pragma once

#include "Model.h"


typedef std::shared_ptr<class Sequential> SequentialPtr;
class Sequential : public Model
{
	std::vector<LayerPtr> _layers;

	class P
	{
	};
public:
	Sequential(const std::initializer_list<LayerPtr>& layers)
	{
		_layers.insert(_layers.end(),layers);
	}

	const std::vector<LayerPtr> GetLayers() const override
	{
		return _layers;
	}

	const Tensors Forward(const TensorPtr& input) const override
	{
		TensorPtr x = input;
		for(auto& layer:_layers)
			x = layer->Forward(x);
		return x;
	}

	static SequentialPtr New(const std::initializer_list<LayerPtr>& layers)
	{
		return std::make_shared<Sequential>(layers);
	}
};