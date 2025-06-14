#pragma once

#include "Kernel.h"


class KoCrossEntropy : public Kernel
{
	NDArray	_targets;

	NDArray	_targetsOH;		// One hot encoded targets.
	NDArray	_softmax;

public:
				KoCrossEntropy(const NDArray& targets);
	NDArray		Forward(const NDArrays& input) override;
	NDArrays	Backward(const NDArray& gradient,const NDArrays& inputs) override;
};