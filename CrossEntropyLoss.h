#pragma once

#include "Tensor.h"


class CrossEntropyLoss
{
public:
	TensorPtr Forward(const TensorPtr& input,const TensorPtr& target)
	{
		return input->CrossEntropy(target);
	}
};