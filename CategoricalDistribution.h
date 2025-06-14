#pragma once

#include "Random.h"
#include "Tensor.h"


class CategoricalDistribution
{
	TensorPtr	_probabilities;

public:
	CategoricalDistribution(const TensorPtr& probabilitiesOrLogits,const bool inputLogits)
	{
		//probabilitiesOrLogits->Print();
		if(inputLogits)
		{
			// Values are logits and can assume any value. 
			// Use logits-LogSumExp(logits) to normalise the logits, i.e. after taking logs the divide is replaced by subtraction.
			// Exponentiate to convert logits to probabilities this is identical to p_i = Exp(x_i)/Sum(Exp(x_j)) just rewritten to avoid divide.
			// See https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
			const TensorPtr logSumExp = probabilitiesOrLogits->Exp()->Sum(-1,true)->Log();	// KeepDim=true else backprop would need to add it back.
			const TensorPtr logits = probabilitiesOrLogits->Sub(logSumExp);					// This normalizes the logits, torch stores this internally.
			_probabilities = logits->Exp();													// Torch then calls logits_to_probs, which uses softmax, but this is equivalent.
		}
		else
		{
			// Probabilities:
			// The list of probabilities of each category being sampled - the probabilities are normalised and must sum to 1.
			// The last dimension is assumed to be the probabilities and any other dimensions separate samples.
			const NDArray sum = probabilitiesOrLogits->Data().Sum(-1,true);
			_probabilities = Tensor::New(probabilitiesOrLogits->Data()/sum);
		}
		//_probabilities->Print();
	}

	// Draws a sample from the distribution.
	//
	TensorPtr Sample()
	{
		// The resulting tensor is the same shape as the 'probabilities' tensor minus the last dimension (the probabilites). 
		// Each value in the result tensor is the chosen category.
		const int categoryCount = *(_probabilities->Shape().end()-1);
		const NDShape shape(_probabilities->Shape().begin(),_probabilities->Shape().end()-1);
		const TensorPtr result = Tensor::New(shape);
		for(int sample=0;sample<shape[0];++sample)
		{
			bool chosen = false;
			const FP targetProbability = rnd.Sample();
			FP probability = 0.0;
			size_t category = 0;
			for(int category=0;category<categoryCount;++category)
			{
				probability += _probabilities->Data()[{sample,category}];
				if(probability>=targetProbability)
				{
					result->Data()[{sample}]=category;
					chosen = true;
					break;
				}
			}
			_ASSERT(chosen);
		}
		return result;
	}

	// Returns the log probability of the given categories.
	TensorPtr LogProb(const TensorPtr& categories)
	{
		return _probabilities->Gather(1,
			categories->Unsqueeze(-1)	// Gather requires dimensions are the same.
			)->Log();
	}

	// Returns the amount of entropy/equal probabilities for each row/independent categorical distribution.
	// This is the sum of -p_i*log(p_i) for each category.
	// The more equal or uniform the probabilities are, the higher the entropy.
	TensorPtr Entropy()
	{
		//return Tensor::New(NDData::New(_probabilities->Data().Entropy()),true);
		//NDArray a = NDData::New(_probabilities->Data().Entropy());
		//TensorPtr b = _probabilities->Mul(_probabilities->Log())->Neg()->Sum(-1,false);
		//if(!a.IsEqualTo(b->Data()))
			//throw "bad";
		return _probabilities->Mul(_probabilities->Log())->Neg()->Sum(-1,false);
	}
};
