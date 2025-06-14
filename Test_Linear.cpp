#include "Test.h"
#include "Linear.h"


using namespace std;


void Test_Linear()
{
	{
		// This tests backprop though the 'lm_head' linear layer on the multihead attention exaple.

		TensorPtr x = Tensor::New(NDData::Load("E:\\Temp\\MHA_sa_heads(x).txt"),true);		// Output x from previous layer, sa_heads.

		LinearPtr lm_head = Linear::New(32,65);												// Create layer, lm_head.
		lm_head->Load("E:\\Temp\\MHA_LmHeadW.txt","E:\\Temp\\MHA_LmHeadB.txt");				// Load initial weight and bias.

		TensorPtr y = lm_head->Forward(x);													// Pass x through test layer.
			
		TensorPtr g = Tensor::New(NDData::Load("E:\\Temp\\MHA_lm_head(x).gradient.txt"));	// Gradient created by calling backward on the previous layer.
		y->Backward(g,nullptr);																// Pass gradient backward through test layer.

		// Load expected gradients produced by PyTorch.
		TensorPtr expected_dw = Tensor::New(NDData::Load("E:\\Temp\\MHA_LmHeadW.gradient.txt").Transpose());
		TensorPtr expected_db = Tensor::New(NDData::Load("E:\\Temp\\MHA_LmHeadB.gradient.txt"));
		TensorPtr expected_dx = Tensor::New(NDData::Load("E:\\Temp\\MHA_sa_heads(x).gradient.txt"));

		// Test gradient of lm_head are as expected.
		Assert(lm_head->_weight->Gradient()->IsEqualTo(expected_dw));
		Assert(lm_head->_bias->Gradient()->IsEqualTo(expected_db));
		Assert(x->Gradient()->IsEqualTo(expected_dx));
	}

	{
		// Backward through 'v = self.value(x)' in each head.
		const int num_heads = 4;
		for(int i=0;i<num_heads;++i)
		{
			TensorPtr x = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_x_"+to_string(i)+".txt"),true);									// Input value x head[0].

			LinearPtr value = Linear::New(32,8,"",false);
			value->Load("E:\\Temp\\MHA_head_value_weight_"+to_string(i)+".txt");

			TensorPtr y = value->Forward(x);																							// Pass x though layer (to get a 'y').

			TensorPtr g = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_value(x)_"+to_string(i)+".gradient.txt"));						// Gradient created by previous layer.
			y->Backward(g,nullptr);																										// Pass gradient backward through test layer.

			// Load expected gradients produced by PyTorch.
			TensorPtr vwg = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_value_weight_"+to_string(i)+".gradient.txt").Transpose());		// dL WRT value weight.
			TensorPtr xg = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_x_"+to_string(i)+".gradient_value.txt"));						// dL WRT head x.

			//print(value->_weight->Gradient());
			//print(x->Gradient());

			// Test gradient of lm_head are as expected.
			Assert(value->_weight->Gradient()->IsEqualTo(vwg));
			Assert(x->Gradient()->IsEqualTo(xg));
		}
	}

	{
		// Backward through 'q=query(x)' in each head.
		const int num_heads = 4;
		for(int i=0;i<num_heads;++i)
		{
			TensorPtr x = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_x_"+to_string(i)+".txt"),true);									// Input value x head[i].

			LinearPtr query = Linear::New(32,8,"",false);
			query->Load("E:\\Temp\\MHA_head_query_weight_"+to_string(i)+".txt");														// Initial query weight, head [i].

			TensorPtr y = query->Forward(x);																							// Pass x though layer (to get a 'y').

			TensorPtr g = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_q_"+to_string(i)+".gradient.txt"));								// Gradient created by previous layer.
			y->Backward(g,nullptr);																										// Pass gradient backward through test layer.

			TensorPtr qwg = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_query_weight_"+to_string(i)+".gradient.txt").Transpose());		// dL WRT query weight, head[i].
			TensorPtr xg = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_x_"+to_string(i)+".gradient_query.txt"));						// dL WRT head x, head[i].

			//print(query->_weight->Gradient());															
			//print(x->Gradient());																		

			// Test gradient of lm_head are as expected.
			Assert(query->_weight->Gradient()->IsEqualTo(qwg));
			Assert(x->Gradient()->IsEqualTo(xg));
		}
	}


	{
		// Backward through 'k=key(x)' in each head.
		const int num_heads = 4;
		for(int i=0;i<num_heads;++i)
		{
			TensorPtr x = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_x_"+to_string(i)+".txt"),true);									// Input value x head[i].

			LinearPtr key = Linear::New(32,8,"",false);
			key->Load("E:\\Temp\\MHA_head_key_weight_"+to_string(i)+".txt");															// Initial key weight, head [i].

			TensorPtr y = key->Forward(x);																							// Pass x though layer (to get a 'y').

			TensorPtr g = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_k_"+to_string(i)+".gradient.txt"));								// Gradient created by previous layer.
			y->Backward(g,nullptr);																										// Pass gradient backward through test layer.

			TensorPtr kwg = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_key_weight_"+to_string(i)+".gradient.txt").Transpose());		// dL WRT key weight, head[i].
			TensorPtr xg = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_x_"+to_string(i)+".gradient_key.txt"));							// dL WRT head x, head[i].

			//print(key->_weight->Gradient());
			//print(x->Gradient());

			// Test gradient of lm_head are as expected.
			Assert(key->_weight->Gradient()->IsEqualTo(kwg));
			Assert(x->Gradient()->IsEqualTo(xg));
		}
	}
}