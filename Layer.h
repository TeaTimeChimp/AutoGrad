#pragma once

#include <vector>
#include "Tensors.h"


typedef std::shared_ptr<class Layer> LayerPtr;
class Layer
{
public:
	enum class Mode
	{
		Training,
		Inference
	};

	// Forward pass of x through the layer.
	//
	virtual const Tensors Forward(const TensorPtr& x) const = 0;

	// Returns the trainable parameters in the layer.
	//
	virtual const std::vector<TensorPtr> GetParameters() const
	{
		return {};
	}

	// Signals the layer should adopt training or inference behaviour.
	//
	virtual void SetMode(const Mode)
	{
	}
};

