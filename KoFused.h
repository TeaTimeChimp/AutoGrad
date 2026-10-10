#pragma once

// Fused kernels.
//
// The generic kernels (KoAdd, KoMul, KoDot, ...) compose into any model, but every op allocates, makes a graph node
// and walks memory once, and the batched KoDot runs one small matrix multiply per batch item. These kernels do the
// same maths for the common building blocks in a single node each, with hand-written backward passes, multithreaded
// loops and the packed GEMM. They plug into Tensor exactly like the generic kernels.
//
// Every kernel here is checked against finite differences by "Speedrun gradcheck" (build with AUTOGRAD_DOUBLE).

#include <vector>
#include <mutex>
#include <cmath>
#include "Kernel.h"
#include "Gemm.h"
#include "ParallelFor.h"


namespace fused
{
	// Number of rows when the last dimension is treated as the row length.
	inline int Rows(const NDArray& a)
	{
		const NDShape& s = a.Shape();
		return a.Size()/s[s.size()-1];
	}

	inline int Cols(const NDArray& a)
	{
		const NDShape& s = a.Shape();
		return s[s.size()-1];
	}

	// out[j] = sum_i a[i*N+j], multithreaded over row chunks.
	inline void ColumnSum(const FP* a,const int M,const int N,FP* out)
	{
		std::fill(out,out+N,FP(0));
		std::mutex lock;
		ParallelForRange(0,M,64,[&](int r0,int r1)
		{
			std::vector<FP> part(N,FP(0));
			for(int i=r0;i<r1;++i)
			{
				const FP* row = a+(size_t)i*N;
				for(int j=0;j<N;++j)
					part[j] += row[j];
			}
			std::unique_lock<std::mutex> l(lock);
			for(int j=0;j<N;++j)
				out[j] += part[j];
		});
	}
}


// y = x @ W (+ b) for x of shape (...,K), W (K,N), b (N).
class KoLinear : public Kernel
{
	const bool _bias;

public:
	KoLinear(const bool bias) :
		_bias(bias)
	{
	}

	NDArray Forward(const NDArrays& in) override
	{
		const NDArray x = in[0].Contiguous();
		const NDArray W = in[1].Contiguous();
		const int K = fused::Cols(x);
		const int M = fused::Rows(x);
		const int N = W.Shape()[1];
		NDShape shape = x.Shape();
		shape[shape.size()-1] = N;
		NDArray y = NDData::New(shape);
		FP* yr = y.Raw();
		if(_bias)
		{
			const FP* b = in[2].Contiguous().Raw();
			ParallelForRange(0,M,256,[&](int r0,int r1)
			{
				for(int i=r0;i<r1;++i)
					std::copy(b,b+N,yr+(size_t)i*N);
			});
		}
		gemm::Gemm<FP>(false,false,M,N,K,FP(1),x.Raw(),K,W.Raw(),N,_bias?FP(1):FP(0),yr,N);
		return y;
	}

	NDArrays Backward(const NDArray& gradient,const NDArrays& in) override
	{
		const NDArray g = gradient.Contiguous();
		const NDArray x = in[0].Contiguous();
		const NDArray W = in[1].Contiguous();
		const int K = fused::Cols(x);
		const int M = fused::Rows(x);
		const int N = W.Shape()[1];

		NDArray dx = NDData::New(x.Shape());
		gemm::Gemm<FP>(false,true,M,K,N,FP(1),g.Raw(),N,W.Raw(),N,FP(0),dx.Raw(),K);		// dx = g @ W^T
		NDArray dW = NDData::New(W.Shape());
		gemm::Gemm<FP>(true,false,K,N,M,FP(1),x.Raw(),K,g.Raw(),N,FP(0),dW.Raw(),N);		// dW = x^T @ g
		if(!_bias)
			return {dx,dW};
		NDArray db = NDData::New(in[2].Shape());
		fused::ColumnSum(g.Raw(),M,N,db.Raw());											// db = sum over rows of g
		return {dx,dW,db};
	}
};


// LayerNorm over the last dimension: y = gamma * (x-mean)/sqrt(var+eps) + beta.
class KoLayerNorm : public Kernel
{
	const FP			_eps;
	std::vector<FP>		_mean;
	std::vector<FP>		_rstd;

public:
	KoLayerNorm(const FP eps=FP(1e-5)) :
		_eps(eps)
	{
	}

	NDArray Forward(const NDArrays& in) override
	{
		const NDArray x = in[0].Contiguous();
		const FP* gamma = in[1].Contiguous().Raw();
		const FP* beta = in[2].Contiguous().Raw();
		const int C = fused::Cols(x);
		const int M = fused::Rows(x);
		_mean.resize(M);
		_rstd.resize(M);
		NDArray y = NDData::New(x.Shape());
		const FP* xr = x.Raw();
		FP* yr = y.Raw();
		ParallelForRange(0,M,32,[&](int r0,int r1)
		{
			for(int i=r0;i<r1;++i)
			{
				const FP* xi = xr+(size_t)i*C;
				FP* yi = yr+(size_t)i*C;
				double mean = 0.0;
				for(int j=0;j<C;++j)
					mean += xi[j];
				mean /= C;
				double var = 0.0;
				for(int j=0;j<C;++j)
				{
					const double d = xi[j]-mean;
					var += d*d;
				}
				var /= C;
				const FP rstd = FP(1.0/std::sqrt(var+_eps));
				_mean[i] = FP(mean);
				_rstd[i] = rstd;
				for(int j=0;j<C;++j)
					yi[j] = gamma[j]*((xi[j]-FP(mean))*rstd)+beta[j];
			}
		});
		return y;
	}

	NDArrays Backward(const NDArray& gradient,const NDArrays& in) override
	{
		const NDArray g = gradient.Contiguous();
		const NDArray x = in[0].Contiguous();
		const FP* gamma = in[1].Contiguous().Raw();
		const int C = fused::Cols(x);
		const int M = fused::Rows(x);
		NDArray dx = NDData::New(x.Shape());
		NDArray dgamma = NDData::New(in[1].Shape(),0.0);
		NDArray dbeta = NDData::New(in[2].Shape(),0.0);
		const FP* xr = x.Raw();
		const FP* gr = g.Raw();
		FP* dxr = dx.Raw();
		FP* dgr = dgamma.Raw();
		FP* dbr = dbeta.Raw();
		std::mutex lock;
		ParallelForRange(0,M,32,[&](int r0,int r1)
		{
			std::vector<FP> dg(C,FP(0)),db(C,FP(0));
			for(int i=r0;i<r1;++i)
			{
				const FP* xi = xr+(size_t)i*C;
				const FP* gi = gr+(size_t)i*C;
				FP* dxi = dxr+(size_t)i*C;
				const FP mean = _mean[i];
				const FP rstd = _rstd[i];
				// dxhat = g*gamma; dx = rstd*(dxhat - mean(dxhat) - xhat*mean(dxhat*xhat))
				double sum1 = 0.0,sum2 = 0.0;
				for(int j=0;j<C;++j)
				{
					const FP xhat = (xi[j]-mean)*rstd;
					const FP dxhat = gi[j]*gamma[j];
					sum1 += dxhat;
					sum2 += dxhat*xhat;
					dg[j] += gi[j]*xhat;
					db[j] += gi[j];
				}
				const FP m1 = FP(sum1/C);
				const FP m2 = FP(sum2/C);
				for(int j=0;j<C;++j)
				{
					const FP xhat = (xi[j]-mean)*rstd;
					dxi[j] = rstd*(gi[j]*gamma[j]-m1-xhat*m2);
				}
			}
			std::unique_lock<std::mutex> l(lock);
			for(int j=0;j<C;++j)
			{
				dgr[j] += dg[j];
				dbr[j] += db[j];
			}
		});
		return {dx,dgamma,dbeta};
	}
};


// Elementwise activation: GELU (tanh approximation) or ReLU.
class KoActivation : public Kernel
{
public:
	enum class Type { Gelu, Relu };

private:
	const Type _type;

	static FP Gelu(const FP x)
	{
		const FP k = FP(0.7978845608028654);	// sqrt(2/pi)
		return FP(0.5)*x*(FP(1)+std::tanh(k*(x+FP(0.044715)*x*x*x)));
	}

	static FP GeluGrad(const FP x)
	{
		const FP k = FP(0.7978845608028654);
		const FP u = k*(x+FP(0.044715)*x*x*x);
		const FP t = std::tanh(u);
		const FP du = k*(FP(1)+FP(3*0.044715)*x*x);
		return FP(0.5)*(FP(1)+t)+FP(0.5)*x*(FP(1)-t*t)*du;
	}

public:
	KoActivation(const Type type) :
		_type(type)
	{
	}

	NDArray Forward(const NDArrays& in) override
	{
		const NDArray x = in[0].Contiguous();
		NDArray y = NDData::New(x.Shape());
		const FP* xr = x.Raw();
		FP* yr = y.Raw();
		const Type type = _type;
		ParallelForRange(0,x.Size(),16384,[=](int b,int e)
		{
			if(type==Type::Gelu)
				for(int i=b;i<e;++i) yr[i] = Gelu(xr[i]);
			else
				for(int i=b;i<e;++i) yr[i] = xr[i]>FP(0)?xr[i]:FP(0);
		});
		return y;
	}

	NDArrays Backward(const NDArray& gradient,const NDArrays& in) override
	{
		const NDArray x = in[0].Contiguous();
		const NDArray g = gradient.Contiguous();
		NDArray dx = NDData::New(x.Shape());
		const FP* xr = x.Raw();
		const FP* gr = g.Raw();
		FP* dr = dx.Raw();
		const Type type = _type;
		ParallelForRange(0,x.Size(),16384,[=](int b,int e)
		{
			if(type==Type::Gelu)
				for(int i=b;i<e;++i) dr[i] = gr[i]*GeluGrad(xr[i]);
			else
				for(int i=b;i<e;++i) dr[i] = xr[i]>FP(0)?gr[i]:FP(0);
		});
		return {dx};
	}
};


// Elementwise a + b for same-shaped tensors (residual connections), multithreaded.
class KoAddSame : public Kernel
{
public:
	NDArray Forward(const NDArrays& in) override
	{
		const NDArray a = in[0].Contiguous();
		const NDArray b = in[1].Contiguous();
		if(a.Shape()!=b.Shape())
			throw IncompatibleShape();
		NDArray y = NDData::New(a.Shape());
		const FP* ar = a.Raw();
		const FP* br = b.Raw();
		FP* yr = y.Raw();
		ParallelForRange(0,a.Size(),32768,[=](int s,int e)
		{
			for(int i=s;i<e;++i)
				yr[i] = ar[i]+br[i];
		});
		return y;
	}

	NDArrays Backward(const NDArray& gradient,const NDArrays&) override
	{
		// The same buffer goes to both creators; Tensor::Backward never modifies a shared gradient in place.
		return {gradient,gradient};
	}
};


// Causal multi-head self-attention core.
//
// Input qkv (B,T,3C) holds the query, key and value projections side by side; each is split into H heads of size
// C/H. Output (B,T,C) is the heads' outputs concatenated. For each (batch,head):
//		S = Q K^T / sqrt(hs), masked so position i only sees j<=i;  P = softmax(S);  O = P V
class KoCausalAttention : public Kernel
{
	const int			_heads;
	std::vector<FP>		_P;			// Attention probabilities (B,H,T,T), kept for the backward pass.

public:
	KoCausalAttention(const int heads) :
		_heads(heads)
	{
	}

	NDArray Forward(const NDArrays& in) override
	{
		const NDArray qkv = in[0].Contiguous();
		const int B = qkv.Shape()[0];
		const int T = qkv.Shape()[1];
		const int C3 = qkv.Shape()[2];
		const int C = C3/3;
		const int H = _heads;
		const int hs = C/H;
		const FP scale = FP(1.0/std::sqrt((double)hs));
		_P.resize((size_t)B*H*T*T);
		NDArray out = NDData::New({B,T,C});
		const FP* src = qkv.Raw();
		FP* dst = out.Raw();
		ParallelFor(0,B*H,[&](const int bh)
		{
			const int b = bh/H;
			const int h = bh%H;
			const FP* q = src+(size_t)b*T*C3+h*hs;
			const FP* k = q+C;
			const FP* v = q+2*C;
			FP* P = _P.data()+(size_t)bh*T*T;
			gemm::Gemm<FP>(false,true,T,T,hs,scale,q,C3,k,C3,FP(0),P,T,false);	// S = scale * Q K^T
			for(int i=0;i<T;++i)
			{
				FP* row = P+(size_t)i*T;
				FP mx = row[0];
				for(int j=1;j<=i;++j)
					mx = (std::max)(mx,row[j]);
				FP sum = 0;
				for(int j=0;j<=i;++j)
				{
					row[j] = std::exp(row[j]-mx);
					sum += row[j];
				}
				const FP inv = FP(1)/sum;
				for(int j=0;j<=i;++j)
					row[j] *= inv;
				for(int j=i+1;j<T;++j)
					row[j] = FP(0);										// Masked (future) positions.
			}
			gemm::Gemm<FP>(false,false,T,hs,T,FP(1),P,T,v,C3,FP(0),dst+(size_t)b*T*C+h*hs,C,false);	// O = P V
		});
		return out;
	}

	NDArrays Backward(const NDArray& gradient,const NDArrays& in) override
	{
		const NDArray qkv = in[0].Contiguous();
		const NDArray g = gradient.Contiguous();
		const int B = qkv.Shape()[0];
		const int T = qkv.Shape()[1];
		const int C3 = qkv.Shape()[2];
		const int C = C3/3;
		const int H = _heads;
		const int hs = C/H;
		const FP scale = FP(1.0/std::sqrt((double)hs));
		NDArray dqkv = NDData::New(qkv.Shape());
		const FP* src = qkv.Raw();
		const FP* gr = g.Raw();
		FP* dst = dqkv.Raw();
		ParallelFor(0,B*H,[&](const int bh)
		{
			thread_local std::vector<FP> dP;
			dP.resize((size_t)T*T);
			const int b = bh/H;
			const int h = bh%H;
			const size_t off = (size_t)b*T*C3+h*hs;
			const FP* q = src+off;
			const FP* k = q+C;
			const FP* v = q+2*C;
			FP* dq = dst+off;
			FP* dk = dq+C;
			FP* dv = dq+2*C;
			const FP* dO = gr+(size_t)b*T*C+h*hs;
			const FP* P = _P.data()+(size_t)bh*T*T;
			gemm::Gemm<FP>(true,false,T,hs,T,FP(1),P,T,dO,C,FP(0),dv,C3,false);		// dV = P^T dO
			gemm::Gemm<FP>(false,true,T,T,hs,FP(1),dO,C,v,C3,FP(0),dP.data(),T,false);	// dP = dO V^T
			for(int i=0;i<T;++i)															// dS = P*(dP - rowsum(P*dP)) * scale
			{
				const FP* p = P+(size_t)i*T;
				FP* d = dP.data()+(size_t)i*T;
				FP dot = 0;
				for(int j=0;j<=i;++j)
					dot += p[j]*d[j];
				for(int j=0;j<=i;++j)
					d[j] = p[j]*(d[j]-dot)*scale;
				for(int j=i+1;j<T;++j)
					d[j] = FP(0);
			}
			gemm::Gemm<FP>(false,false,T,hs,T,FP(1),dP.data(),T,k,C3,FP(0),dq,C3,false);	// dQ = dS K
			gemm::Gemm<FP>(true,false,T,hs,T,FP(1),dP.data(),T,q,C3,FP(0),dk,C3,false);	// dK = dS^T Q
		});
		return {dqkv};
	}
};


// Mean cross entropy of logits (N,V) against integer targets (N), with softmax fused in.
class KoSoftmaxCrossEntropy : public Kernel
{
	NDArray			_targets;
	std::vector<FP>	_probs;

public:
	KoSoftmaxCrossEntropy(const NDArray& targets) :
		_targets(targets.Contiguous())
	{
	}

	NDArray Forward(const NDArrays& in) override
	{
		const NDArray logits = in[0].Contiguous();
		const int N = fused::Rows(logits);
		const int V = fused::Cols(logits);
		_probs.resize((size_t)N*V);
		const FP* lr = logits.Raw();
		const FP* tr = _targets.Raw();
		std::vector<double> losses(N);
		ParallelForRange(0,N,64,[&](int r0,int r1)
		{
			for(int i=r0;i<r1;++i)
			{
				const FP* row = lr+(size_t)i*V;
				FP* p = _probs.data()+(size_t)i*V;
				FP mx = row[0];
				for(int j=1;j<V;++j)
					mx = (std::max)(mx,row[j]);
				double sum = 0.0;
				for(int j=0;j<V;++j)
				{
					p[j] = std::exp(row[j]-mx);
					sum += p[j];
				}
				const FP inv = FP(1.0/sum);
				for(int j=0;j<V;++j)
					p[j] *= inv;
				const int t = (int)tr[i];
				losses[i] = -(row[t]-mx-std::log(sum));
			}
		});
		double total = 0.0;
		for(auto l:losses)
			total += l;
		return NDData::New({},FP(total/N));
	}

	NDArrays Backward(const NDArray& gradient,const NDArrays& in) override
	{
		const int N = fused::Rows(in[0]);
		const int V = fused::Cols(in[0]);
		const FP scale = gradient[{}]/FP(N);
		NDArray d = NDData::New(in[0].Shape());
		FP* dr = d.Raw();
		const FP* tr = _targets.Raw();
		ParallelForRange(0,N,64,[&](int r0,int r1)
		{
			for(int i=r0;i<r1;++i)
			{
				const FP* p = _probs.data()+(size_t)i*V;
				FP* di = dr+(size_t)i*V;
				for(int j=0;j<V;++j)
					di[j] = p[j]*scale;
				di[(int)tr[i]] -= scale;
			}
		});
		return {d};
	}
};
