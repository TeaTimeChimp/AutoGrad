#include "Test.h"
#include "Embedding.h"


using namespace std;


void Test_Embedding()
{
	{
		// Backward through 'position_embedding_table' layer.
		const int T = 8;
		TensorPtr x = Tensor::New(NDData::Arrange(T));																			// Input.

		EmbeddingPtr position_embedding_table = Embedding::New(8,32);															// Test layer.
		position_embedding_table->Load("E:\\Temp\\MHA_PosEmbLayer_Weight.txt");													// Initial weights.

		TensorPtr y = position_embedding_table->Forward(x);																		// Pass x through layer to get a 'y'.

		TensorPtr g = Tensor::New(NDData::Load("E:\\Temp\\MHA_pos_emb.gradient.txt"));											// Gradient from previous operation.
		y->Backward(g,nullptr);

		TensorPtr ge = Tensor::New(NDData::Load("E:\\Temp\\MHA_PosEmbLayer_Weight.gradient.txt"));								// dL WRT position_embedding_table weight.

		//print(position_embedding_table->Weight()->Gradient());

		Assert(position_embedding_table->Weight()->Gradient()->IsEqualTo(ge));
	}
	{
		// Backward through 'token_embedding_table' layer.
		TensorPtr x = Tensor::New(NDData::Load("E:\\Temp\\MHA_idx.txt"));															// Input idx.

		EmbeddingPtr token_embedding_table = Embedding::New(65,32);																	// Test layer.
		token_embedding_table->Load("E:\\Temp\\MHA_TokEmbLayer_Weight.txt");														// Initial weights.

		TensorPtr y = token_embedding_table->Forward(x);																			// Pass x through layer to get a 'y'.

		TensorPtr g = Tensor::New(NDData::Load("E:\\Temp\\MHA_tok_emb.gradient.txt"));												// Gradient from previous operation.
		y->Backward(g,nullptr);

		print(token_embedding_table->Weight()->Gradient());

		TensorPtr eg = Tensor::New(NDData::Load("E:\\Temp\\MHA_TokEmbLayer_Weight.gradient.txt"));									// dL WRT token_embedding_table weight.

		Assert(token_embedding_table->Weight()->Gradient()->IsEqualTo(eg));
	}
}