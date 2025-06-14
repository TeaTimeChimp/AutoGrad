// AutoGrad.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "Tensor.h"
#include "Sequential.h"
#include "Embedding.h"
#include "Tanh.h"
#include "Linear.h"
#include "CrossEntropyLoss.h"
#include "MSELoss.h"
#include "SGD.h"
#include "Test.h"

using namespace std;


int main()
{
	Test();

	{
		// MSE
		cout<<"MSE"<<endl;
		TensorPtr p = Tensor::New(NDData::New({1,1},3),true);
		TensorPtr t = Tensor::New(NDData::New({1,1},5),true);
		MSELoss loss;
		TensorPtr l = loss.Forward(p,t);
		l->Backward();
		p->Print();
		SGD optim({p},1);
		optim.Step();
		p->Print();
	}

	{
		// x*y=10.
		TensorPtr x = Tensor::New(NDData::New({1,1},2),true);
		TensorPtr y = Tensor::New(NDData::New({1,1},3),true);
		MSELoss loss;
		SGD optim({x,y},0.01);
		for(int i=0;i<10;++i)
		{
			TensorPtr p = x->Mul(y);
			p->Print();
			TensorPtr l = loss.Forward(p,Tensor::New(NDData::New({1,1},10),true));
			l->Print();
			l->Backward();
			optim.Step();
			x->Print();
			y->Print();
		}
	}

	TensorPtr data = Tensor::New(NDData::New({4},		// 4 samples, 1 dictionary index each.
		{
			1,
			2,
			1,
			2
		}),true);		
	TensorPtr target = Tensor::New(NDData::New({4},		// 4 samples, 1 label index each.
		{
			0,
			1,
			0,
			1
		}),true);
	Sequential model(
		{
			Embedding::New(3,3),
			Tanh::New(),
			Linear::New(3,4,"",true)
		});

	/*
	TensorPtr data = Tensor::New({{1,2},1,2},true);	// 1 Input values, 1 for each input of the linear layer.
	TensorPtr target = Tensor::New({1},true);		// 1 Output index for the 1 sample, one of 0,1,2 indicating the output that should fire.
	Sequential model({Linear::New(2,3)});
	*/

	CrossEntropyLoss criterion;
	SGD optim(model.GetParameters(),0.1);

	for(int i=0;i<10;++i)
	{
		TensorPtr pred = model.Forward(data);

		TensorPtr loss = criterion.Forward(pred,target);

		loss->Backward();
	}
}
