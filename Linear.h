#pragma once


#include "Layer.h"
#include "Relu.h"


typedef std::shared_ptr<class Linear> LinearPtr;
class Linear : public Layer
{
public:
	TensorPtr	_weight;
	TensorPtr	_bias;
	ReluPtr		_activation;

	// Private class to prevent constructor being called directly.
	class P
	{
	};

public:
	static LinearPtr New(const int nInputs,const int nOutputs,const std::string& name="Linear",const bool bias=true,const ReluPtr& activation=nullptr)
	{
		return std::make_shared<Linear>(P(),nInputs,nOutputs,bias,activation,name);
	}

	Linear(const P&,const int nInputs,const int nOutputs,const bool bias,const ReluPtr& activation,const std::string& name) :
		_activation(activation)
	{
		// He initialisation.
		const FP stdev = sqrt(2.0/nInputs);
		NDArray weight = NDData::RandN({nInputs,nOutputs})*stdev;
		_weight = Tensor::New(weight,true);
		_weight->Name(name+".weight");

		if(bias)
		{
			_bias = Tensor::New(NDData::Zeros({nOutputs}),true);
			_bias->Name(name+".bias");
		}
	}

	const Tensors Forward(const TensorPtr& input) const override
	{
		// Multiple input by weights.
		TensorPtr z = input->Dot(_weight);
		
		// Add optional bias.
		if(_bias)
			z = z->Add(_bias);

		// Apply optional activation function.
		if(_activation)
			return _activation->Forward(z);
		else
			return z;
	}

	void Print(bool grad) const
	{
		if(grad)
			if(_weight->Gradient())
				_weight->Gradient()->Print();
		else
			_weight->Print();
	}

	void Load(const std::string& weight)
	{
		_weight->Data() = NDData::Load(weight).Transpose();	// Transpose because PyTorch uses transposed version.
	}

	void Load(const std::string& weight,const std::string& bias)
	{
		Load(weight);
		_bias->Data() = NDData::Load(bias);
	}

	const std::vector<TensorPtr> GetParameters() const override
	{
		if(_bias)
			return {_weight,_bias};
		else
			return {_weight};
	}
};