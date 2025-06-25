#include "Tensor.h"
#include "Test.h"


using namespace std;


namespace
{
	void Test_Add()
	{
		{	// scalar+scalar
			TensorPtr x = Tensor::New(NDData::New({},{1}),true); print(x);			
			TensorPtr y = Tensor::New(NDData::New({},{2}),true); print(y);			
			TensorPtr z = x->Add(y); print(z);			
			z->Backward(); print(z->Gradient()); print(y->Gradient()); print(z->Gradient());			
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({},{1}))),"scalar+scalar: z");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({},{1}))),"scalar+scalar: y");
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({},{1}))),"scalar+scalar: x");
		}

		{	// scalar+column
			TensorPtr x = Tensor::New(NDData::New({},{2}),true); print(x);			
			TensorPtr y = Tensor::New(NDData::New({2},{11,12}),true); print(y);			
			TensorPtr z = x->Add(y); print(z);			
			z->Backward(); print(z->Gradient()); print(y->Gradient()); print(x->Gradient());			
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({2},{1,1}))),"scalar+column: z");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({2},{1,1}))),"scalar+column: y");
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({},{2}))),"scalar+column: x");
		}

		{	// scalar+row
			TensorPtr x = Tensor::New(NDData::New({},{2}),true); print(x);
			TensorPtr y = Tensor::New(NDData::New({1,2},
				{
					11,12
				}),true); print(y);
			TensorPtr z = x->Add(y); print(z);
			z->Backward(); print(z->Gradient()); print(y->Gradient()); print(x->Gradient());
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2},{1,1}))),"scalar+row: z");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2},{1,1}))),"scalar+row: y");
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({},{2}))),"scalar+row: x");
		}

		{	// scalar+cube
			TensorPtr x = Tensor::New(NDData::New({},{2}),true); print(x);
			TensorPtr y = Tensor::New(NDData::New({1,2,3},
				{
					11,12,13,
					21,22,23
				}),true); print(y);
			TensorPtr z = x->Add(y); print(z);
			z->Backward(); print(z->Gradient()); print(y->Gradient()); print(x->Gradient());
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2,3},
				{
					1,1,1,
					1,1,1
				}))),"scalar+cube: z");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2,3},
				{
					1,1,1,
					1,1,1
				}))),"scalar+cube: y");
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({},{6}))),"scalar+row: x");
		}

		{	// column+scalar
			TensorPtr x = Tensor::New(NDData::New({2},{11,12}),true); print(x);
			TensorPtr y = Tensor::New(NDData::New({},{2}),true); print(y);
			TensorPtr z = x->Add(y); print(z);
			z->Backward(); print(z->Gradient()); print(y->Gradient()); print(x->Gradient());
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({2},{1,1}))),"column+scalar: z");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({},{2}))),"column+scalar: y");
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({2},{1,1}))),"column+scalar: x");
		}

		{	// column+column
			TensorPtr x = Tensor::New(NDData::New({1},{11}),true); print(x);
			TensorPtr y = Tensor::New(NDData::New({2},{21,22}),true); print(y);
			TensorPtr z = x->Add(y); print(z);
			z->Backward();
			print(z->Gradient()); print(y->Gradient()); print(x->Gradient());
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({2},{1,1}))),"column+column");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({2},{1,1}))),"column+column");
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({1},{2}))),"column+column");
		}

		{	// column+column
			TensorPtr x = Tensor::New(NDData::New({2},{11,12}),true); print(x);
			TensorPtr y = Tensor::New(NDData::New({1},{21}),true); print(y);
			TensorPtr z = x->Add(y); print(z);
			z->Backward();
			print(z->Gradient()); print(y->Gradient()); print(x->Gradient());
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({2},{1,1}))),"column+column");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({1},{2}))),"column+column");
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({2},{1,1}))),"column+column");
		}

		{	// column+column
			TensorPtr x = Tensor::New(NDData::New({2},{11,12}),true); print(x);
			TensorPtr y = Tensor::New(NDData::New({2},{21,22}),true); print(y);
			TensorPtr z = x->Add(y); print(z);
			z->Backward();
			print(z->Gradient()); print(y->Gradient()); print(x->Gradient());
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({2},{1,1}))),"column+column");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({2},{1,1}))),"column+column");
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({2},{1,1}))),"column+column");
		}

		{	// column+row
			TensorPtr x = Tensor::New(NDData::New({2},{11,12}),true); print(x);
			TensorPtr y = Tensor::New(NDData::New({1,2},{21,22}),true);	print(y);
			TensorPtr z = x->Add(y); print(z);
			z->Backward(); print(z->Gradient()); print(y->Gradient()); print(x->Gradient());
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2},{1,1}))),"column+row");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2},{1,1}))),"column+row");
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({2},{1,1}))),"column+row");
		}

		{	// row+scalar
			TensorPtr x = Tensor::New(NDData::New({1,2},
				{
					11,12
				}),true); print(x);
			TensorPtr y = Tensor::New(NDData::New({},{2}),true); print(y);
			TensorPtr z = x->Add(y); print(z);
			z->Backward(); print(z->Gradient()); print(y->Gradient()); print(x->Gradient());
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2},
				{
					1,1
				}))),"row+scalar");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({},{2}))),"row+scalar");
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2},
				{
					1,1
				}))),"row+scalar");
		}

		{	// row+row
			TensorPtr x = Tensor::New(NDData::New({1,2},
				{
					11,12
				}),true); print(x);
			TensorPtr y = Tensor::New(NDData::New({1,2},
				{
					21,22
				}),true); print(y);
			TensorPtr z = x->Add(y); print(z);
			z->Backward(); print(z->Gradient()); print(y->Gradient()); print(x->Gradient());
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2},
				{
					1,1
				}))),"row+row");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2},
				{
					1,1
				}))),"row+row");
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2},
				{
					1,1
				}))),"row+row");
		}

		{	// row+rows
			TensorPtr x = Tensor::New(NDData::New({1,2},
				{
					11,12
				}),true); print(x);
			TensorPtr y = Tensor::New(NDData::New({2,2},
				{
					21,22,
					31,32
				}),true); print(y);
			TensorPtr z = x->Add(y); print(z);
			z->Backward(); print(z->Gradient()); print(y->Gradient()); print(x->Gradient());
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({2,2},
				{
					1,1,
					1,1
				}))),"row+rows");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({2,2},
				{
					1,1,
					1,1
				}))),"row+rows");
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2},
				{
					2,2
				}))),"row+rows");
		}

		{	// rows+row
			TensorPtr x = Tensor::New(NDData::New({2,2},
				{
					11,12,
					21,22
				}),true); print(x);
			TensorPtr y = Tensor::New(NDData::New({1,2},
				{
					31,32
				}),true); print(y);
			TensorPtr z = x->Add(y); print(z);
			z->Backward(); print(z->Gradient()); print(y->Gradient()); print(x->Gradient());
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({2,2},
				{
					1,1,
					1,1
				}))),"rows+row");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2},
				{
					2,2
				}))),"rows+row");
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({2,2},
				{
					1,1,
					1,1
				}))),"rows+row");
		}

		{	// rows+rows
			TensorPtr x = Tensor::New(NDData::New({2,2},
				{
					11,12,
					21,22
				}),true); print(x);
			TensorPtr y = Tensor::New(NDData::New({2,2},
				{
					31,32,
					41,42
				}),true); print(y);
			TensorPtr z = x->Add(y); print(z);
			z->Backward(); print(z->Gradient()); print(y->Gradient()); print(x->Gradient());
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({2,2},
				{
					1,1,
					1,1
				}))),"rows+rows");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({2,2},
				{
					1,1,
					1,1
				}))),"rows+rows");
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({2,2},
				{
					1,1,
					1,1
				}))),"rows+rows");
		}

		{	// row+cube
			TensorPtr x = Tensor::New(NDData::New({1,2},
				{
					11,12
				}),true); print(x);
			TensorPtr y = Tensor::New(NDData::New({1,2,3},
				{
					21,22,23,
					31,32,33
				}),true); print(y);
			bool exception = false;
			try
			{
				TensorPtr z = x->Add(y); print(z);
			}
			catch(const IncompatibleShape&)
			{
				exception = true;
			}
			Assert(exception,"row+cube");
		}

		{	// row+cube
			TensorPtr x = Tensor::New(NDData::New({1,3},
				{
					11,12,13
				}),true); print(x);
			TensorPtr y = Tensor::New(NDData::New({1,2,3},
				{
					21,22,23,
					31,32,33
				}),true); print(y);
			TensorPtr z = x->Add(y); print(z);
			z->Backward(); print(z->Gradient()); print(y->Gradient()); print(x->Gradient());
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2,3},
				{
					1,1,1,
					1,1,1
				}))),"row+cube");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2,3},
				{
					1,1,1,
					1,1,1
				}))),"row+cube");
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,3},
				{
					2,2,2
				}))),"row+cube");
		}

		{
			// cube+scalar.
			TensorPtr x = Tensor::New(NDData::New({1,2,3},
				{
					12,12,13,
					21,22,23
				}),true); print(x);
			TensorPtr y = Tensor::New(NDData::New({},1),true); print(y);
			TensorPtr z = x->Add(y); print(z);
			z->Backward(); print(z->Gradient()); print(y->Gradient()); print(x->Gradient());
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2,3},
				{
					1,1,1,
					1,1,1
				}))),"cube+scalar");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({},{6}))),"cube+scalar");
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2,3},
				{
					1,1,1,
					1,1,1
				}))),"cube+scalar");
		}

		{
			// cube+column.
			TensorPtr x = Tensor::New(NDData::New({1,2,3},
				{
					12,12,13,
					21,22,23
				}),true); print(x);
			TensorPtr y = Tensor::New(NDData::New({3},{1,2,3}),true); print(y);
			TensorPtr z = x->Add(y); print(z);
			z->Backward(); print(z->Gradient()); print(y->Gradient()); print(x->Gradient());
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2,3},
				{
					1,1,1,
					1,1,1
				}))),"cube+column");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({3},{2,2,2}))),"cube+column");
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2,3},
				{
					1,1,1,
					1,1,1
				}))),"cube+column");
		}
		{
			// cube+row.
			TensorPtr x = Tensor::New(NDData::New({1,2,3},
				{
					12,12,13,
					21,22,23
				}),true); print(x);
			TensorPtr y = Tensor::New(NDData::New({1,3},
				{
					1,2,3
				}),true); print(y);
			TensorPtr z = x->Add(y); print(z);
			z->Backward(); print(z->Gradient()); print(y->Gradient()); print(x->Gradient());
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2,3},
				{
					1,1,1,
					1,1,1
				}))),"cube+row");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,3},
				{
					2,2,2
				}))),"cube+row");
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2,3},
				{
					1,1,1,
					1,1,1
				}))),"cube+row");
		}
		{
			// cube+cube.
			TensorPtr x = Tensor::New(NDData::New({1,2,3},
				{
					11,12,13,
					21,22,23
				}),true); print(x);
			TensorPtr y = Tensor::New(NDData::New({1,2,3},
				{
					31,32,33,
					41,42,43
				}),true); print(y);
			TensorPtr z = x->Add(y); print(z);
			z->Backward(); print(z->Gradient()); print(y->Gradient()); print(x->Gradient());
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2,3},
				{
					1,1,1,
					1,1,1
				}))),"cube+cube");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2,3},
				{
					1,1,1,
					1,1,1
				}))),"cube+cube");
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2,3},
				{
					1,1,1,
					1,1,1
				}))),"cube+cube");
		}

		{
			// cube+cube.
			TensorPtr x = Tensor::New(NDData::New({1,2,3},
				{
					11,12,13,
					21,22,23
				}),true); print(x);
			TensorPtr y = Tensor::New(NDData::New({2,2,3},
				{
					131,132,133,
					141,142,143,

					231,232,233,
					241,242,243
				}),true); print(y);
			TensorPtr z = x->Add(y); print(z);
			z->Backward(); print(z->Gradient()); print(y->Gradient()); print(x->Gradient());
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({2,2,3},
				{
					1,1,1,
					1,1,1,

					1,1,1,
					1,1,1
				}))),"cube+cube");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({2,2,3},
				{
					1,1,1,
					1,1,1,

					1,1,1,
					1,1,1
				}))),"cube+cube");
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2,3},
				{
					2,2,2,
					2,2,2
				}))),"cube+cube");
		}

		{
			// cube+cube.
			TensorPtr x = Tensor::New(NDData::New({2,2,3},
				{
					111,112,113,
					121,122,123,

					211,212,113,
					221,222,123
				}),true); print(x);
			TensorPtr y = Tensor::New(NDData::New({1,2,3},
				{
					131,132,133,
					141,142,143
				}),true); print(y);
			TensorPtr z = x->Add(y); print(z);
			z->Backward(); print(z->Gradient()); print(y->Gradient()); print(x->Gradient());
			Assert(z->Gradient()->IsEqualTo(Tensor::New(NDData::New({2,2,3},
				{
					1,1,1,
					1,1,1,

					1,1,1,
					1,1,1
				}))),"cube+cube");
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2,3},
				{
					2,2,2,
					2,2,2
				}))),"cube+cube");
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({2,2,3},
				{
					1,1,1,
					1,1,1,

					1,1,1,
					1,1,1
				}))),"cube+cube");
		}

		if(false)	// Lost test data file.
		{
			// Each head has produced a contribution to the final gradient of x.
			// These contributions must be summed, there for 3 for each each 1 from each of key, query, and value.
			const int num_heads = 4;
			TensorPtr dx = Tensor::Zeros({32,8,32});
			for(int i=0;i<num_heads;++i)
			{
				TensorPtr k = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_x_"+to_string(i)+".gradient_key.txt"));
				TensorPtr q = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_x_"+to_string(i)+".gradient_query.txt"));
				TensorPtr v = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_x_"+to_string(i)+".gradient_value.txt"));
				dx = dx->Add(k);
				dx = dx->Add(q);
				dx = dx->Add(v);
			}
			TensorPtr gx = Tensor::New(NDData::Load("E:\\Temp\\MHA_tok_emb_pos_emb.gradient.txt"));
			Assert(dx->IsEqualTo(gx));
		}

		if(false)	// Lost test data file.
		{
			// Backward through 'x=tok_emb+pos_emb'.
			TensorPtr tok_emb = Tensor::New(NDData::Load("E:\\Temp\\MHA_tok_emb.txt"),true);										// Input tok_emb.
			TensorPtr pos_emb = Tensor::New(NDData::Load("E:\\Temp\\MHA_pos_emb.txt"),true);										// Input tok_emb.

			TensorPtr y = tok_emb->Add(pos_emb);																					// Test operation.

			TensorPtr g = Tensor::New(NDData::Load("E:\\Temp\\MHA_tok_emb_pos_emb.gradient.txt"));									// Gradient created by previous layer.	
			y->Backward(g,nullptr);

			TensorPtr gt = Tensor::New(NDData::Load("E:\\Temp\\MHA_tok_emb.gradient.txt"));											// dL WRT tok_emb.
			TensorPtr gp = Tensor::New(NDData::Load("E:\\Temp\\MHA_pos_emb.gradient.txt"));											// dL WRT pos_emb.

			print(tok_emb->Gradient());
			print(pos_emb->Gradient());
			Assert(tok_emb->Gradient()->IsEqualTo(gt));
			Assert(pos_emb->Gradient()->IsEqualTo(gp));
		}
	}

	void Test_Cat()
	{
		{
			TensorPtr x = Tensor::New(NDData::New({1,2},
				{
					11,12
				}),true); print(x);
			TensorPtr xm = x->Mul(Tensor::New(NDData::New({},2)));

			TensorPtr y = Tensor::New(NDData::New({1,2},
				{
					21,22
				}),true); print(y);
			TensorPtr ym = y->Mul(Tensor::New(NDData::New({},3)));

			TensorPtr z = Tensor::Cat(vector<TensorPtr>{xm,ym},1); print(z);
			z->Backward();
			print(y->Gradient());
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2},
				{
					3,3
				}))),"cat_0");
			print(x->Gradient());
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2},
				{
					2,2
				}))),"cat_1");
		}

		{
			// Test for backprop through multihead attention.
			// Backward through 'out = torch.cat(hxs,dim=-1)'.
			TensorPtr x0 = Tensor::New(NDData::Load("TestData\\MHA_heads(x)_0.txt"),true);		// Output x from heads[0].
			TensorPtr x1 = Tensor::New(NDData::Load("TestData\\MHA_heads(x)_1.txt"),true);		// Output x from heads[1].
			TensorPtr x2 = Tensor::New(NDData::Load("TestData\\MHA_heads(x)_2.txt"),true);		// Output x from heads[2].
			TensorPtr x3 = Tensor::New(NDData::Load("TestData\\MHA_heads(x)_3.txt"),true);		// Output x from heads[3].

			TensorPtr y = Tensor::Cat({x0,x1,x2,x3},-1);										// Test operation.

			TensorPtr g = Tensor::New(NDData::Load("TestData\\MHA_sa_heads(x).gradient.txt"));	// Gradient created by previous layer.
			y->Backward(g,nullptr);

			TensorPtr xg0 = Tensor::New(NDData::Load("TestData\\MHA_heads(x)_0.gradient.txt"));	// dL WRT heads[0].
			TensorPtr xg1 = Tensor::New(NDData::Load("TestData\\MHA_heads(x)_1.gradient.txt"));	// dL WRT heads[1].
			TensorPtr xg2 = Tensor::New(NDData::Load("TestData\\MHA_heads(x)_2.gradient.txt"));	// dL WRT heads[2].
			TensorPtr xg3 = Tensor::New(NDData::Load("TestData\\MHA_heads(x)_3.gradient.txt"));	// dL WRT heads[3].

			//print(x0->Gradient());
			//print(x1->Gradient());
			//print(x2->Gradient());
			//print(x3->Gradient());

			Assert(x0->Gradient()->IsEqualTo(xg0));			
			Assert(x1->Gradient()->IsEqualTo(xg1));			
			Assert(x2->Gradient()->IsEqualTo(xg2));			
			Assert(x3->Gradient()->IsEqualTo(xg3));
		}
	}

	void Test_CrossEntropy()
	{
		{
			TensorPtr x = Tensor::New(NDData::New({1,4},
				{
					0.25,0.25,0.25,0.25
				}));

			TensorPtr y = Tensor::New(NDData::New({1},{3}));
			TensorPtr z = x->CrossEntropy(y);
			print(z);
			Assert(z->IsEqualTo(Tensor::New(NDData::New({},{1.38629f}))),"4x1");
		}

		{
 			TensorPtr x = Tensor::New(NDData::LoadNP("TestData\\XeLogits.txt"),true);		// (BT,C)
			print(x->Shape());
			TensorPtr y = Tensor::New(NDData::LoadNP("TestData\\XeTargets.txt"),true);		// (BT)
			print(y->Shape());
			TensorPtr z = x->CrossEntropy(y);
			print(z);
			Assert(z->IsEqualTo(Tensor::New(NDData::New({},{4.22416f}))),"256x65");
			z->Backward();
			print(x->Gradient()->Shape());
			print(x->Gradient());
			TensorPtr g = Tensor::New(NDData::LoadNP("TestData\\XeGradients.txt"),true);	// (BT)
			print(g);
			Assert(x->Gradient()->IsEqualTo(g),"gradient");
		}
	}

	void Test_Dot()
	{
		{
			TensorPtr c0 = Tensor::New(NDData::New({2,3,4},
				{
						 1, 2, 3, 4,
						 5, 6, 7, 8,
						 9,10,11,12,

						13,14,15,16,
						17,18,19,20,
						21,22,23,24
				}),true);

			TensorPtr c1 = Tensor::New(NDData::New({4,5},
				{
					 1, 2, 3, 4, 5,
					 6, 7, 8, 9,10,
					11,12,13,14,15,
					16,17,18,19,20
				}),true);

			TensorPtr y = c0->Dot(c1);
			print("y:");
			print(y);

			TensorPtr g = Tensor::New(NDData::New({2,3,5},
				{
						 1, 2, 3, 4, 5,
						 6, 7, 8, 9,10,
						11,12,13,14,15,
					
						16,17,18,19,20,
						21,22,23,24,25,
						26,27,28,29,30
				}));
			y->Backward(g,nullptr);
			Assert(y->IsEqualTo(Tensor::New(NDData::New({2,3,5},
				{
					110,120,130,140,150,
					246,272,298,324,350,
					382,424,466,508,550,

					518,576,634,692,750,
					654,728,802,876,950,
					790,880,970,1060,1150
				}))));

			print("dc0:");
			print(c0->Gradient());
			Assert(c0->Gradient()->IsEqualTo(Tensor::New(NDData::New({2,3,4},
				{
					  55, 130, 205, 280,
					 130, 330, 530, 730,
					 205, 530, 855,1180,

					 280, 730,1180,1630,
					 355, 930,1505,2080,
					 430,1130,1830,2530
				}))));

			print("dc1:");
			print(c1->Gradient());
			Assert(c1->Gradient()->IsEqualTo(Tensor::New(NDData::New({4,5},
				{
					1241,1307,1373,1439,1505,
					1322,1394,1466,1538,1610,
					1403,1481,1559,1637,1715,
					1484,1568,1652,1736,1820
				}))));
		}

		if(false)	// Lost test data files.
		{
			// Backward through 'out = wei@v' in each head.
			const int num_heads = 4;
			for(int i=0;i<num_heads;++i)
			{
				TensorPtr wei = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_softmax(wei)_"+to_string(i)+".txt"),true);			// Input wei head[i].
				TensorPtr v = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_value(x)_"+to_string(i)+".txt"),true);				// Input v head[i].

				TensorPtr y = wei->Dot(v);																						// Test operation.

				TensorPtr g = Tensor::New(NDData::Load("E:\\Temp\\MHA_heads(x)_"+to_string(i)+".gradient.txt"));				// Gradient created by previous layer, head[i].
				y->Backward(g,nullptr);

				// Load expected gradients.
				TensorPtr weig = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_softmax(wei)_"+to_string(i)+".gradient.txt"));	// dL WRT softmax(wei) heads[i].
				TensorPtr vg = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_value(x)_"+to_string(i)+".gradient.txt"));			// dL WRT value(x) heads[i].

				//print(wei->Gradient());
				//print(v->Gradient());

				// Compare actual gradients with expected.
				Assert(wei->Gradient()->IsEqualTo(weig));
				Assert(v->Gradient()->IsEqualTo(vg));
			}
		}
	}

	void Test_Dropout()
	{
		{
			TensorPtr x = Tensor::New(NDData::RandN({5,5}),true);
			print(x,"x");
			TensorPtr y = x->Dropout(0.5);
			print(y,"y");
			y->Backward();
			TensorPtr yg = y->Gradient();
			print(yg,"yg");
			TensorPtr xg = x->Gradient();
			print(xg,"xg");
			Assert(xg->IsEqualTo(Tensor::New(NDData::New({5,5},
				{
					0,0,0,2,0,
					0,0,0,0,0,
					2,0,2,2,0,
					2,2,0,0,2,
					2,0,0,2,2
				}))));
		}
	}

	void Test_Gather()
	{
		{
			TensorPtr x = Tensor::New(NDData::New({2,2},
				{
					-0.5,1.0,
					 0.5,2.0
				}),true);
			print(x,"x");
			TensorPtr y = x->Gather(1,Tensor::New(NDData::New({2,1},
				{
					0,
					1
				}),true));
			print(y,"y");
			y->Backward();
			TensorPtr dx = x->Gradient();
			print(dx,"dx");
			Assert(dx->IsEqualTo(Tensor::New(NDData::New({2,2},
				{
					1,0,
					0,1
				}))));
		}
		{
			TensorPtr x = Tensor::New(NDData::New({2,2},
				{
					-0.5,1.0,
					 0.5,2.0
				}),true);
			print(x,"x");
			TensorPtr y = x->Gather(1,Tensor::New(NDData::New({2,2},
				{
					0,0,
					1,0
				}),true));
			print(y,"y");
			y->Backward();
			TensorPtr dx = x->Gradient();
			print(dx,"dx");
			Assert(dx->IsEqualTo(Tensor::New(NDData::New({2,2},
				{
					2,0,
					1,1
				}))));
		}
		{
			TensorPtr x = Tensor::New(NDData::New({2,2},
				{
					-0.5,1.0,
					 0.5,2.0
				}),true);
			print(x,"x");
			TensorPtr y = x->Gather(1,Tensor::New(NDData::New({2,3},
				{
					0,1,0,
					1,0,1
				}),true));
			print(y,"y");
			y->Backward();
			TensorPtr dx = x->Gradient();
			print(dx,"dx");
			Assert(dx->IsEqualTo(Tensor::New(NDData::New({2,2},
				{
					2,1,
					1,2
				}))));
		}
	}

	void Test_MaskedFill()
	{
		{
			// Backward through 'wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))'.
			TensorPtr wei = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_wei_k_0.txt"),true);				// Input wei head[0]
			const int T = 8;
			TensorPtr itril = Tensor::New(NDData::Ones({T,T}).Tril())->Equal(Tensor::Zeros({T,T}));

			TensorPtr y = wei->MaskedFill(itril,-numeric_limits<FP>::infinity());							// Test operation.

			TensorPtr g = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_masked_fill(wei)_0.gradient.txt"));	// Gradient created by previous layer.
			y->Backward(g,nullptr);

			// Load expected gradient.
			TensorPtr weig = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_wei_k_0.gradient.txt"));			// dL WRT operation.

			//print(wei->Gradient());

			// Compare actual gradients with expected.
			Assert(wei->Gradient()->IsEqualTo(weig));
		}
	}

	void Test_Mean()
	{
		{
			// Scalar mean.
			TensorPtr x = Tensor::New(NDData::New({1,3},{1,2,3}),true);
			print(x,"x");
			TensorPtr y = x->Mean(true);
			print(y,"y");
			y->Backward();
			TensorPtr xg = x->Gradient();
			print(xg,"xg");
			Assert(xg->IsEqualTo(Tensor::New(NDData::New({1,3},{0.333333f,0.333333f,0.333333f}))));
		}

		{
			// Scalar mean.
			TensorPtr x = Tensor::New(NDData::New({1,3},{1,2,3}),true);
			print(x,"x");
			TensorPtr y = x->Mean(false);
			print(y,"y");
			y->Backward();
			TensorPtr xg = x->Gradient();
			print(xg,"xg");
			Assert(xg->IsEqualTo(Tensor::New(NDData::New({1,3},{0.333333f,0.333333f,0.333333f}))));
		}

		{
			// One dimension mean - keepdims.
			TensorPtr x = Tensor::New(NDData::New({3,4},
				{
					 1, 2, 3, 4,
					 5, 6, 7, 8,
					 9,10,11,12
				}),true);
			print(x,"x:");
			TensorPtr y = x->Mean(1,true);	// Mean of each row.
			print(y,"y:");
			y->Backward(Tensor::New(NDData::New({3,1},
				{
					1,
					2,
					4
				})),nullptr);
			TensorPtr xg = x->Gradient();
			print(xg,"xg");
			Assert(xg->IsEqualTo(Tensor::New(NDData::New({3,4},
				{
					0.25,0.25,0.25,0.25,
					0.50,0.50,0.50,0.50,
					1.00,1.00,1.00,1.00
				}))));
		}

		{
			// One dimension mean - !keepdims.
			TensorPtr x = Tensor::New(NDData::New({3,4},
				{
					 1, 2, 3, 4,
					 5, 6, 7, 8,
					 9,10,11,12
				}),true);
			print(x,"x:");
			TensorPtr y = x->Mean(1,false);	// Mean of each row.
			print(y,"y:");
			y->Backward(Tensor::New(NDData::New({3},
				{
					1,2,4
				})),nullptr);
			TensorPtr xg = x->Gradient();
			print(xg,"xg");
			Assert(xg->IsEqualTo(Tensor::New(NDData::New({3,4},
				{
					0.25,0.25,0.25,0.25,
					0.50,0.50,0.50,0.50,
					1.00,1.00,1.00,1.00
				}))));
		}
	}

	void Test_Mul()
	{
		{
			TensorPtr x = Tensor::New(NDData::New({1,2},
				{
					1,2
				}),true); print(x);
			TensorPtr y = x->Mul(Tensor::New(NDData::New({},3))); print(y);
			y->Backward(); print(y->Gradient()); print(x->Gradient());
			Assert(y->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2},
				{
					1,1
				}))),"mul_0");			
			Assert(x->Gradient()->IsEqualTo(Tensor::New(NDData::New({1,2},
				{
					3,3
				}))),"mul_1");
		}

		{
			// Backward through 'wei = wei * k.shape[-1]**-0.5'.
			const int num_heads = 4;
			for(int i=0;i<num_heads;++i)
			{
				TensorPtr wei = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_q_k_T_"+to_string(i)+".txt"),true);			// Input wei head[0]
				const FP k_shape = 8.0;

				TensorPtr y = wei->Mul(Tensor::New(NDData::New({},{pow(k_shape,-0.5f)})));									// Test operation.

				TensorPtr g = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_wei_k_"+to_string(i)+".gradient.txt"));			// Gradient created by previous layer.
				y->Backward(g,nullptr);

				TensorPtr weig = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_q_k_T_"+to_string(i)+".gradient.txt"));		// dL WRT operation.

				print(wei->Gradient());
				Assert(wei->Gradient()->IsEqualTo(weig));
			}
		}
	}

	void Test_Pow()
	{
	}

	void Test_Reshape()
	{
		{
			TensorPtr x = Tensor::New(NDData::New({32,8,65},0.0),true);										// (B,T,C)
			print(x->Shape());
			TensorPtr y = x->Reshape({32*8,65});															// (BT,C)
			print(y->Shape());
			TensorPtr z = Tensor::New(NDData::LoadNP("TestData\\XeGradients.txt"),true);					// (BT,C)
			print(z->Shape());
			y->Backward(z,nullptr);
			print(x->Gradient()->Shape());
			
			TensorPtr g = Tensor::New(NDData::LoadWithImplicitShape("TestData\\ViewGradients.txt"),true);	// (B,T,C)
			Assert(x->Gradient()->IsEqualTo(g),"gradient");
		}
	}

	void Test_Softmax()
	{
		{
			// Backward through 'wei = F.softmax(wei, dim=-1)' in each head.
			const int num_heads = 4;
			for(int i=0;i<num_heads;++i)
			{
				TensorPtr wei = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_masked_fill(wei)_"+to_string(i)+".txt"),true);			// Input wei head[0]

				TensorPtr y = wei->Softmax(-1);																						// Test operation.

				// Load reference gradient to pass back.
				TensorPtr g = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_softmax(wei)_"+to_string(i)+".gradient.txt"));			// Gradient created by previous layer.
				y->Backward(g,nullptr);

				// Load expected gradient.
				TensorPtr weig = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_masked_fill(wei)_"+to_string(i)+".gradient.txt"));	// dL WRT head 'softmax(wei, dim=-1)'.

				//print(wei->Gradient());

				// Check acutal gradient is expected.
				Assert(wei->Gradient()->IsEqualTo(weig));
			}
		}
	}

	void Test_Transpose()
	{
		{
			// Backward through 'wei = q @ k.transpose(-2,-1)' for each head.	
			const int num_heads = 4;
			for(int i=0;i<num_heads;++i)
			{
				TensorPtr q = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_q_"+to_string(i)+".txt"),true);					// Input q head[i]
				TensorPtr k = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_k_"+to_string(i)+".txt"),true);					// Input k head[i]

				TensorPtr y = q->Dot(k->Transpose());																		// Test operation.

				TensorPtr g = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_q_k_T_"+to_string(i)+".gradient.txt"));			// Gradient created by previous layer.
				y->Backward(g,nullptr);

				TensorPtr qg = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_q_"+to_string(i)+".gradient.txt"));				// dL WRT q.
				TensorPtr kg = Tensor::New(NDData::Load("E:\\Temp\\MHA_head_k_"+to_string(i)+".gradient.txt"));				// dL WRT k.

				//print(q->Gradient());
				//print(k->Gradient());

				// Check acutal gradient is expected.
				Assert(q->Gradient()->IsEqualTo(qg));
				Assert(k->Gradient()->IsEqualTo(kg));
			}
		}
	}
}


void Test_Var()
{
	{
		// column vector.
		TensorPtr x = Tensor::New({5},{1,2,3,4,5});
		TensorPtr y = x->Var(0,true);
		print(y);
	}

	{
		// row vector.
		TensorPtr x = Tensor::New({1,5},{1,2,3,4,5});
		TensorPtr y = x->Var(1,true);
		print(y);
	}

	{
		// matrix 0.
		TensorPtr x = Tensor::New({5,2},
			{
				 1, 1,
				 2, 3,
				 3, 5,
				 4, 8,
				 5,11
			});
		TensorPtr y = x->Var(0,true);
		print(y);
	}

	{
		// matrix 1.
		TensorPtr x = Tensor::New({2,5},
			{
				 1, 2, 3, 4, 5,
				 1, 3, 5, 8,11
			});
		TensorPtr y = x->Var(1,true);
		print(y);
	}

	{
		TensorPtr x = Tensor::New(NDData::RandN({32,100}));
		print(x->Slice({{},{0}})->Mean(true));
	}
}


void Test_Tensor()
{
	Test_Add();
	Test_Cat();
	Test_CrossEntropy();
	Test_Dot();
	Test_Dropout();
	Test_Gather();
	Test_MaskedFill();
	Test_Mean();
	Test_Mul();
	Test_Pow();
	Test_Reshape();
	Test_Softmax();
	Test_Transpose();
	Test_Var();

	{
		cout<<"repeat_0: input."<<endl;
		TensorPtr x = Tensor::New(NDData::New({4,1},{0.0,0.1,0.2,0.3}),true);	// Column vector.
		x->Print();
		cout<<"repeat_0: output."<<endl;
		TensorPtr y = x->Repeat(0,3);	// Make column 3 times longer.
		y->Print();
		Assert(y->Data().IsEqualTo(NDData::New({12,1},{0.0,0.0,0.0,0.1,0.1,0.1,0.2,0.2,0.2,0.3,0.3,0.3})),"expand");
	}

	{
		cout<<"repeat_1"<<endl;
		TensorPtr x = Tensor::New(NDData::New({1,1},{0.1}),true);	// Scalar.
		x->Print();
		TensorPtr y = x->Repeat(1,3);	// Copy 3 times as columns.
		y->Print();
	}

	{
		TensorPtr a = Tensor::New(NDData::New({1,2},{1,2}),true);
		TensorPtr b = Tensor::New(NDData::New({1,2},{1,2}),true);
		TensorPtr c = a->Add(b);
		TensorPtr d1 = c->Neg();
		TensorPtr d2 = c->Neg();
		TensorPtr e = d1->Add(d2);
		e->Backward();
	}

	{
		// relu
		cout<<"relu"<<endl;
		TensorPtr x = Tensor::New(NDData::New({1,3},{-0.5,0.5,0}),true);
		x->Print();
		TensorPtr y = x->Relu();
		y->Print();
		y->Backward();
		x->Gradient()->Print();
	}

	{
		// mul (by scalar)
		cout<<"mul(by scalar)"<<endl;
		TensorPtr x = Tensor::New(NDData::New({1,2},{-0.5,0.5}),true);
		x->Print();
		TensorPtr y = x->Mul(Tensor::New(NDData::New({1,1},3),true));
		y->Print();
		y->Backward();
		x->Gradient()->Print();
	}

	{
		// mul (by scalar reverse)
		cout<<"mul(by scalar reverse)"<<endl;
		TensorPtr x = Tensor::New(NDData::New({1,2},{-0.5,0.5}),true);
		x->Print();
		TensorPtr y = Tensor::New(NDData::New({1,1},3),true)->Mul(x);
		y->Print();
		y->Backward();
		x->Gradient()->Print();
	}

	{
		// mul
		cout<<"mul"<<endl;
		TensorPtr x = Tensor::New(NDData::New({1,2},{1,2}),true);
		x->Print();
		TensorPtr y = x->Mul(Tensor::New(NDData::New({1,2},{3,4}),true));
		y->Print();
		y->Backward();
		x->Gradient()->Print();
	}

	{
		// mul (reverse)
		cout<<"mul(reverse)"<<endl;
		TensorPtr x = Tensor::New(NDData::New({1,2},{1,2}),true);
		x->Print();
		TensorPtr y = Tensor::New(NDData::New({1,2},{3,4}),true)->Mul(x);
		y->Print();
		y->Backward();
		x->Gradient()->Print();
	}

	{
		// sub
		cout<<"sub"<<endl;
		TensorPtr x = Tensor::New(NDData::New({1,2},{1,2}),true);
		x->Print();
		TensorPtr y = x->Sub(Tensor::New(NDData::New({1,2},{4,3}),true));
		y->Print();
		y->Backward();
		x->Gradient()->Print();
	}

	{
		// sub (reverse)
		cout<<"sub(reverse)"<<endl;
		TensorPtr x = Tensor::New(NDData::New({1,2},{1,2}),true);
		x->Print();
		TensorPtr y = Tensor::New(NDData::New({1,2},{4,3}),true)->Sub(x);
		y->Print();
		y->Backward();
		x->Gradient()->Print();
	}

	{
		// dot (aka mm)
		cout<<"dot"<<endl;
		TensorPtr x = Tensor::New(NDData::New({1,2},{1,2}),true);
		x->Print();
		TensorPtr y = x->Dot(Tensor::New(NDData::New({2,1},{3,4}),true));
		y->Print();
		y->Backward();
		x->Gradient()->Print();
	}
}


