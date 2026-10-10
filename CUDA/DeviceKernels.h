#pragma once

// Plain-C++ interface to the CUDA kernels in DeviceKernels.cu (float only). All pointers are device pointers unless
// named host*. Every call is asynchronous on the default stream, except the host transfers which synchronise.

#include <cstddef>

namespace dev
{
	// Memory (caching allocator: freed blocks are kept per power-of-two size for reuse).
	void*	Alloc(size_t bytes);
	void	Free(void* p);
	void	Upload(float* dst,const float* host,size_t n);
	void	Download(float* host,const float* src,size_t n);
	void	Copy(float* dst,const float* src,size_t n);
	void	Zero(float* dst,size_t n);
	void	Synchronize();
	bool	Available();

	// C = alpha*op(A)@op(B) + beta*C for 'batches' independent matrices. Batch z uses offsets
	// (z/inner)*outer + (z%inner)*innerStride for each operand, so per-head views of (B,T,H*hs) tensors work in place.
	struct Batch
	{
		int		count = 1;
		int		inner = 1;
		long long aOuter = 0,aInner = 0;
		long long bOuter = 0,bInner = 0;
		long long cOuter = 0,cInner = 0;
	};
	void Gemm(bool transA,bool transB,int M,int N,int K,float alpha,const float* A,int lda,const float* B,int ldb,
		float beta,float* C,int ldc,const Batch& batch=Batch());

	void FillRows(float* y,const float* row,int M,int N);					// y[i,:] = row
	void ColumnSum(const float* a,int M,int N,float* out);					// out[j] = sum_i a[i,j]
	void Add(const float* a,const float* b,float* y,size_t n);				// y = a+b
	void AddInPlace(float* y,const float* a,size_t n);						// y += a
	void Scale(float* y,float s,size_t n);									// y *= s
	double SumSquares(const float* a,size_t n);								// synchronises

	void LayerNormForward(const float* x,const float* g,const float* b,float* y,float* mean,float* rstd,int M,int C,float eps);
	void LayerNormBackward(const float* x,const float* g,const float* dy,const float* mean,const float* rstd,
		float* dx,float* dg,float* db,int M,int C);

	// type: 0 gelu, 1 relu, 2 relu^2
	void ActivationForward(int type,const float* x,float* y,size_t n);
	void ActivationBackward(int type,const float* x,const float* dy,float* dx,size_t n);

	// Copies the per-head Q or K slices of qkv (B,T,3C) into (B*H,T,hs), optionally rotating (RoPE).
	void GatherHeads(const float* qkv,float* dst,int B,int T,int C,int H,int which,bool rope);
	// Inverse of GatherHeads for gradients: writes dst (B*H,T,hs) back into dqkv's Q or K slice, un-rotating.
	void ScatterHeads(const float* src,float* dqkv,int B,int T,int C,int H,int which,bool rope);
	void CausalSoftmax(float* P,int rows,int T);							// in place, rows of length T, row r is query r%T
	void CausalSoftmaxBackward(const float* P,float* dP,int rows,int T,float scale);	// dP -> dS in place

	// Mean cross entropy. Writes probabilities and returns nothing; loss (device scalar) is accumulated into *loss.
	void SoftmaxCrossEntropyForward(const float* logits,const float* targets,float* probs,float* loss,int N,int V);
	void SoftmaxCrossEntropyBackward(const float* probs,const float* targets,const float* dloss,float* dlogits,int N,int V);

	void EmbeddingForward(const float* tok,const float* pos,const float* idx,float* y,int B,int T,int C);
	void EmbeddingBackward(const float* dy,const float* idx,float* dtok,float* dpos,int B,int T,int C,int V);

	// One LSTM timestep. z: pre-activations rows (stride ldz); activates in place; writes c and h rows.
	void LstmStepForward(float* z,int ldz,const float* cprev,int ldc,float* c,float* h,int ldh,int B,int H);
	void LstmStepBackward(const float* a,int lda,const float* c,const float* cprev,int ldc,const float* g,int ldg,
		float* dh,float* dc,float* dz,int ldz,int B,int H);
	void LstmPreviousH(const float* a,const float* c,float* hprev,int B,int T,int H);

	void Adam(float* w,float* g,float* m,float* v,size_t n,float lr,float b1,float b2,float eps,float c1,float c2,float decay);
	void NesterovMomentum(const float* g,float* m,float* u,size_t n,float mu);
	void Transpose(const float* src,float* dst,int rows,int cols);
	void Axpy(float* y,const float* x,float a,size_t n);					// y += a*x
}

namespace dev
{
	// y = x * keep(i)/(1-p) where keep(i) = hash(seed,i) >= p (same hash as the CPU path).
	void Dropout(const float* x,float* y,size_t n,float p,unsigned seed);
}

namespace dev
{
	// For each of 'count' contiguous matrices of n elements: x /= (||x|| + eps). One block per matrix, no host sync.
	void NormalizeBatched(float* x,int count,size_t n,float eps);
}
