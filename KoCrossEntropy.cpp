#include "KoCrossEntropy.h"


KoCrossEntropy::KoCrossEntropy(const NDArray& targets) :
	_targets(targets)
{
}


NDArray KoCrossEntropy::Forward(const NDArrays& inputs)
{
	const NDArray& logits = inputs[0];

	// 1-hot encode the target indices (the correct labels).
	_targetsOH._Attach(NDData::Eye(logits.Shape()[1])[_targets]);

	// Normalize logits - so they can be interpreted as probabilities.
	_softmax._Attach(logits.Softmax(-1));

	// Compute cross-entropy per sample and then average.
	return -(_softmax.Log()*_targetsOH).Sum(1,false).Mean(false);
}


NDArrays KoCrossEntropy::Backward(const NDArray& gradient,const NDArrays& inputs)
{
	return
	{
		// Differential WRT input is the activation of correct output -1 (see LSTM document you wrote for full math explanation!).
		// Here softmaxOutput is the predicted p-dist over the outputs and targetDist is the identity matrix row for the correct output i.e. 1.
		// The derivative is softmax(x)-target(x) and then divided by the number of samples to account for taking the mean.
		gradient*((_softmax-_targetsOH)/_targetsOH.Shape()[0])
	};
}