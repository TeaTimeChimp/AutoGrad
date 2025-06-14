#pragma once

#include <initializer_list>
#include <vector>
#include "Layer.h"


class Model : public Layer
{
protected:
	Model()
	{
	}

public:
	virtual const std::vector<LayerPtr> GetLayers() const = 0;

	const std::vector<TensorPtr> GetParameters() const override
	{
		std::vector<TensorPtr> parameters;
		for(auto& layer:GetLayers())
		{
			for(auto& p:layer->GetParameters())
				parameters.emplace_back(p);
		}
		return parameters;
	}

	void SetMode(const Layer::Mode mode) override
	{
		for(auto& layer:GetLayers())
			layer->SetMode(mode);
	}

	virtual const Tensors Forward(const TensorPtr& x,const TensorPtr& y=nullptr)
	{
		return x;
	}

	Tensors operator() (const TensorPtr& x,const TensorPtr& y=nullptr)
	{
		return {x,y};
	}
};