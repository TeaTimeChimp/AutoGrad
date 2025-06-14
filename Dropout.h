#pragma once

#include "Layer.h"


typedef std::shared_ptr<class Dropout> DropoutPtr;
class Dropout : public Layer
{
	const FP	_p;
	Layer::Mode _mode;

	class P
	{
	};
public:
	static DropoutPtr New(const FP p)
	{
		return std::make_shared<Dropout>(P(),p);
	}

	Dropout(const P&,const FP p) :
		_p(p),
		_mode(Layer::Mode::Inference)
	{
	}

	const Tensors Forward(const TensorPtr& x) const override
	{
		if(_mode==Layer::Mode::Training)
			return x->Dropout(_p);
		else
			return x;
	}

	void SetMode(const Layer::Mode mode) override
	{
		_mode = mode;
	}
};
