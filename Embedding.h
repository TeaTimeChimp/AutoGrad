#pragma once

#include "Layer.h"


typedef std::shared_ptr<class Embedding> EmbeddingPtr;

class Embedding : public Layer
{
	const TensorPtr _weight;

public:
	static EmbeddingPtr New(const int vocabSize,const int embeddingSize,const std::string& name="Embedding")
	{
		return std::make_shared<Embedding>(vocabSize,embeddingSize,name);
	}

	Embedding(const int vocabSize,const int embeddingSize,const std::string& name) :
		_weight(Tensor::New(NDData::RandN({vocabSize,embeddingSize})*sqrt(2.0/vocabSize),true))	// He initialization.
	{
		_weight->Name(name+".weight");
	}

	const Tensors Forward(const TensorPtr& indices) const override
	{
		// Select the weight rows (embeddings) indicated by the input indices.
		return _weight->IndexSelect(indices);
	}

	const std::vector<TensorPtr> GetParameters() const override
	{
		return {_weight};
	}

	void Load(const std::string& weight)
	{
		_weight->Data() = NDData::Load(weight);
	}

	const TensorPtr& Weight() const
	{
		return _weight;
	}
};
