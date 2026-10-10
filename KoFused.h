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


// GPU dispatch: in a CUDA build with NDConfig::useGpu set, kernels run their *Gpu variant on device-resident arrays.
#ifdef AUTOGRAD_CUDA
#define GPU_FORWARD		bool DeviceAware() const override { return true; }
#define DISPATCH_FORWARD	if(NDConfig::useGpu) return ForwardGpu(in);
#define DISPATCH_BACKWARD	if(NDConfig::useGpu) return BackwardGpu(gradient,in);
#else
#define GPU_FORWARD
#define DISPATCH_FORWARD
#define DISPATCH_BACKWARD
#endif


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
		DISPATCH_FORWARD
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
		DISPATCH_BACKWARD
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

#ifdef AUTOGRAD_CUDA
	GPU_FORWARD

	NDArray ForwardGpu(const NDArrays& in)
	{
		const NDArray x = in[0].Contiguous();
		const NDArray W = in[1].Contiguous();
		const int K = fused::Cols(x);
		const int M = fused::Rows(x);
		const int N = W.Shape()[1];
		NDShape shape = x.Shape();
		shape[shape.size()-1] = N;
		NDArray y = NDData::New(shape);
		float* yd = y.DevWrite();
		if(_bias)
			dev::FillRows(yd,in[2].Contiguous().Dev(),M,N);
		dev::Gemm(false,false,M,N,K,1.0f,x.Dev(),K,W.Dev(),N,_bias?1.0f:0.0f,yd,N);
		return y;
	}

	NDArrays BackwardGpu(const NDArray& gradient,const NDArrays& in)
	{
		const NDArray g = gradient.Contiguous();
		const NDArray x = in[0].Contiguous();
		const NDArray W = in[1].Contiguous();
		const int K = fused::Cols(x);
		const int M = fused::Rows(x);
		const int N = W.Shape()[1];
		NDArray dx = NDData::New(x.Shape());
		NDArray dW = NDData::New(W.Shape());
		dev::Gemm(false,true,M,K,N,1.0f,g.Dev(),N,W.Dev(),N,0.0f,dx.DevWrite(),K);
		dev::Gemm(true,false,K,N,M,1.0f,x.Dev(),K,g.Dev(),N,0.0f,dW.DevWrite(),N);
		if(!_bias)
			return {dx,dW};
		NDArray db = NDData::New(in[2].Shape());
		dev::ColumnSum(g.Dev(),M,N,db.DevWrite());
		return {dx,dW,db};
	}
#endif
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
		DISPATCH_FORWARD
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
		DISPATCH_BACKWARD
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

#ifdef AUTOGRAD_CUDA
	GPU_FORWARD

	NDArray _dmean,_drstd;

	NDArray ForwardGpu(const NDArrays& in)
	{
		const NDArray x = in[0].Contiguous();
		const int C = fused::Cols(x);
		const int M = fused::Rows(x);
		// NDArray::operator= copies elements into an existing array, so members are (re)bound with _Attach.
		_dmean._Attach(NDData::New(NDShape{M}));
		_drstd._Attach(NDData::New(NDShape{M}));
		NDArray y = NDData::New(x.Shape());
		dev::LayerNormForward(x.Dev(),in[1].Contiguous().Dev(),in[2].Contiguous().Dev(),y.DevWrite(),_dmean.DevWrite(),_drstd.DevWrite(),M,C,_eps);
		return y;
	}

	NDArrays BackwardGpu(const NDArray& gradient,const NDArrays& in)
	{
		const NDArray g = gradient.Contiguous();
		const NDArray x = in[0].Contiguous();
		const int C = fused::Cols(x);
		const int M = fused::Rows(x);
		NDArray dx = NDData::New(x.Shape());
		NDArray dg = NDData::New(in[1].Shape());
		NDArray db = NDData::New(in[2].Shape());
		dev::LayerNormBackward(x.Dev(),in[1].Contiguous().Dev(),g.Dev(),_dmean.Dev(),_drstd.Dev(),dx.DevWrite(),dg.DevWrite(),db.DevWrite(),M,C);
		return {dx,dg,db};
	}
#endif
};


// Elementwise activation: GELU (tanh approximation) or ReLU.
class KoActivation : public Kernel
{
public:
	enum class Type { Gelu, Relu, Relu2 };

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
		DISPATCH_FORWARD
		const NDArray x = in[0].Contiguous();
		NDArray y = NDData::New(x.Shape());
		const FP* xr = x.Raw();
		FP* yr = y.Raw();
		const Type type = _type;
		ParallelForRange(0,x.Size(),16384,[=](int b,int e)
		{
			if(type==Type::Gelu)
				for(int i=b;i<e;++i) yr[i] = Gelu(xr[i]);
			else if(type==Type::Relu2)
				for(int i=b;i<e;++i) yr[i] = xr[i]>FP(0)?xr[i]*xr[i]:FP(0);
			else
				for(int i=b;i<e;++i) yr[i] = xr[i]>FP(0)?xr[i]:FP(0);
		});
		return y;
	}

	NDArrays Backward(const NDArray& gradient,const NDArrays& in) override
	{
		DISPATCH_BACKWARD
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
			else if(type==Type::Relu2)
				for(int i=b;i<e;++i) dr[i] = xr[i]>FP(0)?FP(2)*xr[i]*gr[i]:FP(0);
			else
				for(int i=b;i<e;++i) dr[i] = xr[i]>FP(0)?gr[i]:FP(0);
		});
		return {dx};
	}

#ifdef AUTOGRAD_CUDA
	GPU_FORWARD

	NDArray ForwardGpu(const NDArrays& in)
	{
		const NDArray x = in[0].Contiguous();
		NDArray y = NDData::New(x.Shape());
		dev::ActivationForward((int)_type,x.Dev(),y.DevWrite(),x.Size());
		return y;
	}

	NDArrays BackwardGpu(const NDArray& gradient,const NDArrays& in)
	{
		const NDArray x = in[0].Contiguous();
		const NDArray g = gradient.Contiguous();
		NDArray dx = NDData::New(x.Shape());
		dev::ActivationBackward((int)_type,x.Dev(),g.Dev(),dx.DevWrite(),x.Size());
		return {dx};
	}
#endif
};


// Elementwise a + b for same-shaped tensors (residual connections), multithreaded.
class KoAddSame : public Kernel
{
public:
	NDArray Forward(const NDArrays& in) override
	{
		DISPATCH_FORWARD
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

#ifdef AUTOGRAD_CUDA
	GPU_FORWARD

	NDArray ForwardGpu(const NDArrays& in)
	{
		const NDArray a = in[0].Contiguous();
		const NDArray b = in[1].Contiguous();
		if(a.Shape()!=b.Shape())
			throw IncompatibleShape();
		NDArray y = NDData::New(a.Shape());
		dev::Add(a.Dev(),b.Dev(),y.DevWrite(),a.Size());
		return y;
	}
#endif
};


// Causal multi-head self-attention core.
//
// Input qkv (B,T,3C) holds the query, key and value projections side by side; each is split into H heads of size
// C/H. Output (B,T,C) is the heads' outputs concatenated. For each (batch,head):
//		S = Q K^T / sqrt(hs), masked so position i only sees j<=i;  P = softmax(S);  O = P V
//
// With rope=true, rotary position embeddings are applied to Q and K first: within each head, feature pairs
// (i, i+hs/2) are rotated by angle t*base^(-2i/hs), so Q.K depends on the relative distance between positions.
class KoCausalAttention : public Kernel
{
	const int			_heads;
	const bool			_rope;
	std::vector<FP>		_P;			// Attention probabilities (B,H,T,T), kept for the backward pass.
	std::vector<FP>		_cos,_sin;	// RoPE tables (T, hs/2).

	void Tables(const int T,const int hs)
	{
		if(!_rope||(int)_cos.size()==T*(hs/2))
			return;
		_cos.resize((size_t)T*(hs/2));
		_sin.resize((size_t)T*(hs/2));
		for(int t=0;t<T;++t)
			for(int i=0;i<hs/2;++i)
			{
				const double angle = t*std::pow(10000.0,-2.0*i/hs);
				_cos[(size_t)t*(hs/2)+i] = FP(std::cos(angle));
				_sin[(size_t)t*(hs/2)+i] = FP(std::sin(angle));
			}
	}

	// dst (T,hs) contiguous = rotate(src rows with stride ld). inverse=true applies the transpose rotation.
	void Rotate(const FP* src,const int ld,FP* dst,const int ldd,const int T,const int hs,const bool inverse) const
	{
		const int h2 = hs/2;
		for(int t=0;t<T;++t)
		{
			const FP* x = src+(size_t)t*ld;
			FP* y = dst+(size_t)t*ldd;
			if(!_rope)
			{
				std::copy(x,x+hs,y);
				continue;
			}
			const FP* c = _cos.data()+(size_t)t*h2;
			const FP* sn = _sin.data()+(size_t)t*h2;
			for(int i=0;i<h2;++i)
			{
				const FP a = x[i],b = x[i+h2];
				const FP si = inverse?-sn[i]:sn[i];
				y[i] = a*c[i]-b*si;
				y[i+h2] = a*si+b*c[i];
			}
		}
	}

public:
	KoCausalAttention(const int heads,const bool rope=false) :
		_heads(heads),
		_rope(rope)
	{
	}

	NDArray Forward(const NDArrays& in) override
	{
		DISPATCH_FORWARD
		const NDArray qkv = in[0].Contiguous();
		const int B = qkv.Shape()[0];
		const int T = qkv.Shape()[1];
		const int C3 = qkv.Shape()[2];
		const int C = C3/3;
		const int H = _heads;
		const int hs = C/H;
		const FP scale = FP(1.0/std::sqrt((double)hs));
		Tables(T,hs);
		_P.resize((size_t)B*H*T*T);
		NDArray out = NDData::New({B,T,C});
		const FP* src = qkv.Raw();
		FP* dst = out.Raw();
		ParallelFor(0,B*H,[&](const int bh)
		{
			thread_local std::vector<FP> qr,kr;
			qr.resize((size_t)T*hs);
			kr.resize((size_t)T*hs);
			const int b = bh/H;
			const int h = bh%H;
			const FP* q = src+(size_t)b*T*C3+h*hs;
			const FP* k = q+C;
			const FP* v = q+2*C;
			Rotate(q,C3,qr.data(),hs,T,hs,false);
			Rotate(k,C3,kr.data(),hs,T,hs,false);
			FP* P = _P.data()+(size_t)bh*T*T;
			gemm::Gemm<FP>(false,true,T,T,hs,scale,qr.data(),hs,kr.data(),hs,FP(0),P,T,false);	// S = scale * Q K^T
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
		DISPATCH_BACKWARD
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
			thread_local std::vector<FP> dP,qr,kr,dqr,dkr;
			dP.resize((size_t)T*T);
			qr.resize((size_t)T*hs);
			kr.resize((size_t)T*hs);
			dqr.resize((size_t)T*hs);
			dkr.resize((size_t)T*hs);
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
			Rotate(q,C3,qr.data(),hs,T,hs,false);
			Rotate(k,C3,kr.data(),hs,T,hs,false);
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
			gemm::Gemm<FP>(false,false,T,hs,T,FP(1),dP.data(),T,kr.data(),hs,FP(0),dqr.data(),hs,false);	// dQr = dS Kr
			gemm::Gemm<FP>(true,false,T,hs,T,FP(1),dP.data(),T,qr.data(),hs,FP(0),dkr.data(),hs,false);	// dKr = dS^T Qr
			Rotate(dqr.data(),hs,dq,C3,T,hs,true);		// Undo the rotation for the gradient.
			Rotate(dkr.data(),hs,dk,C3,T,hs,true);
		});
		return {dqkv};
	}

#ifdef AUTOGRAD_CUDA
	GPU_FORWARD

	NDArray _dP,_dQr,_dKr;	// Device: attention probabilities (B*H,T,T) and rotated Q/K (B*H,T,hs).

	NDArray ForwardGpu(const NDArrays& in)
	{
		const NDArray qkv = in[0].Contiguous();
		const int B = qkv.Shape()[0];
		const int T = qkv.Shape()[1];
		const int C = qkv.Shape()[2]/3;
		const int H = _heads;
		const int hs = C/H;
		const float scale = float(1.0/std::sqrt((double)hs));
		const float* src = qkv.Dev();
		_dQr._Attach(NDData::New({B*H,T,hs}));
		_dKr._Attach(NDData::New({B*H,T,hs}));
		_dP._Attach(NDData::New({B*H,T,T}));
		float* Q = _dQr.DevWrite();
		float* K = _dKr.DevWrite();
		float* P = _dP.DevWrite();
		dev::GatherHeads(src,Q,B,T,C,H,0,_rope);
		dev::GatherHeads(src,K,B,T,C,H,1,_rope);

		dev::Batch flat;
		flat.count = B*H;
		flat.aOuter = (long long)T*hs; flat.bOuter = (long long)T*hs; flat.cOuter = (long long)T*T;
		dev::Gemm(false,true,T,T,hs,scale,Q,hs,K,hs,0.0f,P,T,flat);			// S = scale Q K^T
		dev::CausalSoftmax(P,B*H*T,T);

		NDArray out = NDData::New({B,T,C});
		dev::Batch heads;
		heads.count = B*H; heads.inner = H;
		heads.aOuter = (long long)H*T*T; heads.aInner = (long long)T*T;		// P
		heads.bOuter = (long long)T*3*C; heads.bInner = hs;					// V inside qkv
		heads.cOuter = (long long)T*C; heads.cInner = hs;					// out
		dev::Gemm(false,false,T,hs,T,1.0f,P,T,src+2*C,3*C,0.0f,out.DevWrite(),C,heads);	// O = P V
		return out;
	}

	NDArrays BackwardGpu(const NDArray& gradient,const NDArrays& in)
	{
		const NDArray qkv = in[0].Contiguous();
		const NDArray g = gradient.Contiguous();
		const int B = qkv.Shape()[0];
		const int T = qkv.Shape()[1];
		const int C = qkv.Shape()[2]/3;
		const int H = _heads;
		const int hs = C/H;
		const float scale = float(1.0/std::sqrt((double)hs));
		const float* src = qkv.Dev();
		const float* dO = g.Dev();
		const float* P = _dP.Dev();
		NDArray dqkv = NDData::New(qkv.Shape());
		float* dst = dqkv.DevWrite();

		dev::Batch heads;
		heads.count = B*H; heads.inner = H;
		heads.aOuter = (long long)H*T*T; heads.aInner = (long long)T*T;		// P
		heads.bOuter = (long long)T*C; heads.bInner = hs;					// dO
		heads.cOuter = (long long)T*3*C; heads.cInner = hs;					// dV inside dqkv
		dev::Gemm(true,false,T,hs,T,1.0f,P,T,dO,C,0.0f,dst+2*C,3*C,heads);	// dV = P^T dO

		NDArray dS = NDData::New({B*H,T,T});
		float* dSp = dS.DevWrite();
		dev::Batch h2;
		h2.count = B*H; h2.inner = H;
		h2.aOuter = (long long)T*C; h2.aInner = hs;							// dO
		h2.bOuter = (long long)T*3*C; h2.bInner = hs;						// V
		h2.cOuter = (long long)H*T*T; h2.cInner = (long long)T*T;			// dP
		dev::Gemm(false,true,T,T,hs,1.0f,dO,C,src+2*C,3*C,0.0f,dSp,T,h2);	// dP = dO V^T
		dev::CausalSoftmaxBackward(P,dSp,B*H*T,T,scale);					// dS

		NDArray dQ = NDData::New({B*H,T,hs});
		NDArray dK = NDData::New({B*H,T,hs});
		dev::Batch flat;
		flat.count = B*H;
		flat.aOuter = (long long)T*T; flat.bOuter = (long long)T*hs; flat.cOuter = (long long)T*hs;
		dev::Gemm(false,false,T,hs,T,1.0f,dSp,T,_dKr.Dev(),hs,0.0f,dQ.DevWrite(),hs,flat);	// dQr = dS Kr
		dev::Gemm(true,false,T,hs,T,1.0f,dSp,T,_dQr.Dev(),hs,0.0f,dK.DevWrite(),hs,flat);	// dKr = dS^T Qr
		dev::ScatterHeads(dQ.Dev(),dst,B,T,C,H,0,_rope);
		dev::ScatterHeads(dK.Dev(),dst,B,T,C,H,1,_rope);
		return {dqkv};
	}
#endif
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
		DISPATCH_FORWARD
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
		DISPATCH_BACKWARD
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

#ifdef AUTOGRAD_CUDA
	GPU_FORWARD

	NDArray _dprobs;

	NDArray ForwardGpu(const NDArrays& in)
	{
		const NDArray logits = in[0].Contiguous();
		const int N = fused::Rows(logits);
		const int V = fused::Cols(logits);
		_dprobs._Attach(NDData::New({N,V}));
		NDArray loss = NDData::New({},FP(0));
		dev::SoftmaxCrossEntropyForward(logits.Dev(),_targets.Dev(),_dprobs.DevWrite(),loss.DevWrite(),N,V);
		return loss;
	}

	NDArrays BackwardGpu(const NDArray& gradient,const NDArrays& in)
	{
		const int N = fused::Rows(in[0]);
		const int V = fused::Cols(in[0]);
		NDArray d = NDData::New(in[0].Shape());
		dev::SoftmaxCrossEntropyBackward(_dprobs.Dev(),_targets.Dev(),gradient.Contiguous().Dev(),d.DevWrite(),N,V);
		return {d};
	}
#endif
};


namespace fused
{
	// C(MxN) = alpha*op(A)@op(B) + beta*C, parallelised over column chunks of C. Better than the block-parallel
	// GEMM when M is small (e.g. a recurrent step where M is the batch size).
	inline void GemmColumns(const bool transA,const bool transB,const int M,const int N,const int K,const FP alpha,
		const FP* A,const int lda,const FP* B,const int ldb,const FP beta,FP* C,const int ldc)
	{
		const int chunk = 64;
		ParallelFor(0,(N+chunk-1)/chunk,[&](const int c)
		{
			const int j0 = c*chunk;
			const int n = (std::min)(chunk,N-j0);
			const FP* b = transB?B+(size_t)j0*ldb:B+j0;
			gemm::Gemm<FP>(transA,transB,M,n,K,alpha,A,lda,b,ldb,beta,C+j0,ldc,false);
		});
	}

	// A weight matrix op(W) (KxN) reused by every timestep: C(MxN) = A(MxK) @ op(W) + beta*C.
	// In float builds the matrix is packed for the GEMM micro-kernel once; otherwise it falls back to GemmColumns.
	class RecurrentWeights
	{
		const FP*	_w;
		const int	_ld;
		const bool	_trans;
		const int	_K,_N;
#ifndef AUTOGRAD_DOUBLE
		gemm::PackedB _packed;
#endif
	public:
		RecurrentWeights(const FP* w,const int ld,const bool trans,const int K,const int N) :
			_w(w),_ld(ld),_trans(trans),_K(K),_N(N)
		{
#ifndef AUTOGRAD_DOUBLE
			_packed.Pack(w,ld,trans,K,N);
#endif
		}

		void MultiplyAdd(const int M,const FP* A,const int lda,FP* C,const int ldc,const FP beta) const
		{
#ifndef AUTOGRAD_DOUBLE
			gemm::SgemmPacked(false,M,1.0f,A,lda,_packed,beta,C,ldc);
#else
			GemmColumns(false,_trans,M,_N,_K,FP(1),A,lda,_w,_ld,beta,C,ldc);
#endif
		}
	};

	inline FP Sigmoid(const FP x)
	{
		return FP(1)/(FP(1)+std::exp(-x));
	}
}


// Single LSTM layer over a whole sequence, zero initial state.
//
// Inputs x (B,T,I), Wx (I,4H), Wh (H,4H), b (4H); output h (B,T,H). Gate order in the 4H dimension: i,f,g,o.
//		z_t = x_t Wx + h_{t-1} Wh + b;  i,f,o = sigmoid;  g = tanh
//		c_t = f*c_{t-1} + i*g;  h_t = o*tanh(c_t)
class KoLSTM : public Kernel
{
	std::vector<FP>	_gates;		// Activated gates (B,T,4H).
	std::vector<FP>	_cells;		// Cell states (B,T,H).

public:
	NDArray Forward(const NDArrays& in) override
	{
		DISPATCH_FORWARD
		const NDArray x = in[0].Contiguous();
		const NDArray Wx = in[1].Contiguous();
		const NDArray Wh = in[2].Contiguous();
		const FP* bias = in[3].Contiguous().Raw();
		const int B = x.Shape()[0];
		const int T = x.Shape()[1];
		const int I = x.Shape()[2];
		const int H = Wh.Shape()[0];
		const int G = 4*H;
		_gates.resize((size_t)B*T*G);
		_cells.resize((size_t)B*T*H);
		NDArray out = NDData::New({B,T,H});
		FP* h = out.Raw();
		FP* z = _gates.data();
		FP* c = _cells.data();

		// Input contribution for every timestep at once: z = x Wx + b.
		ParallelForRange(0,B*T,64,[&](int r0,int r1)
		{
			for(int r=r0;r<r1;++r)
				std::copy(bias,bias+G,z+(size_t)r*G);
		});
		gemm::Gemm<FP>(false,false,B*T,G,I,FP(1),x.Raw(),I,Wx.Raw(),G,FP(1),z,G);

		fused::RecurrentWeights whPacked(Wh.Raw(),G,false,H,G);
		for(int t=0;t<T;++t)
		{
			// Recurrent contribution: z_t += h_{t-1} Wh. Rows of z_t and h_t are T steps apart in memory.
			if(t>0)
				whPacked.MultiplyAdd(B,h+(size_t)(t-1)*H,T*H,z+(size_t)t*G,T*G,FP(1));
			ParallelForRange(0,B,1,[&](int b0,int b1)
			{
				for(int b=b0;b<b1;++b)
				{
					FP* zt = z+((size_t)b*T+t)*G;
					FP* ct = c+((size_t)b*T+t)*H;
					FP* ht = h+((size_t)b*T+t)*H;
					const FP* cp = t>0?ct-H:nullptr;
					for(int j=0;j<H;++j)
					{
						const FP ig = fused::Sigmoid(zt[j]);
						const FP fg = fused::Sigmoid(zt[H+j]);
						const FP gg = std::tanh(zt[2*H+j]);
						const FP og = fused::Sigmoid(zt[3*H+j]);
						zt[j] = ig; zt[H+j] = fg; zt[2*H+j] = gg; zt[3*H+j] = og;
						const FP cv = (cp?fg*cp[j]:FP(0))+ig*gg;
						ct[j] = cv;
						ht[j] = og*std::tanh(cv);
					}
				}
			});
		}
		return out;
	}

	NDArrays Backward(const NDArray& gradient,const NDArrays& in) override
	{
		DISPATCH_BACKWARD
		const NDArray x = in[0].Contiguous();
		const NDArray Wx = in[1].Contiguous();
		const NDArray Wh = in[2].Contiguous();
		const NDArray g = gradient.Contiguous();
		const int B = x.Shape()[0];
		const int T = x.Shape()[1];
		const int I = x.Shape()[2];
		const int H = Wh.Shape()[0];
		const int G = 4*H;
		const FP* a = _gates.data();
		const FP* c = _cells.data();
		const FP* gr = g.Raw();

		// Wh^T (GxH), packed once for every recurrent step.
		fused::RecurrentWeights whT(Wh.Raw(),G,true,G,H);

		std::vector<FP> dz((size_t)B*T*G);		// Pre-activation gradients.
		std::vector<FP> dh((size_t)B*H,FP(0));	// Gradient flowing into h_t from step t+1.
		std::vector<FP> dc((size_t)B*H,FP(0));	// Gradient flowing into c_t from step t+1.
		for(int t=T-1;t>=0;--t)
		{
			ParallelForRange(0,B,1,[&](int b0,int b1)
			{
				for(int b=b0;b<b1;++b)
				{
					const FP* at = a+((size_t)b*T+t)*G;
					const FP* ct = c+((size_t)b*T+t)*H;
					const FP* cp = t>0?ct-H:nullptr;
					const FP* gt = gr+((size_t)b*T+t)*H;
					FP* dzt = dz.data()+((size_t)b*T+t)*G;
					FP* dhb = dh.data()+(size_t)b*H;
					FP* dcb = dc.data()+(size_t)b*H;
					for(int j=0;j<H;++j)
					{
						const FP ig = at[j],fg = at[H+j],gg = at[2*H+j],og = at[3*H+j];
						const FP tc = std::tanh(ct[j]);
						const FP dhv = gt[j]+dhb[j];
						const FP dcv = dhv*og*(FP(1)-tc*tc)+dcb[j];
						dzt[j] = dcv*gg*ig*(FP(1)-ig);
						dzt[H+j] = (cp?dcv*cp[j]:FP(0))*fg*(FP(1)-fg);
						dzt[2*H+j] = dcv*ig*(FP(1)-gg*gg);
						dzt[3*H+j] = dhv*tc*og*(FP(1)-og);
						dcb[j] = dcv*fg;
					}
				}
			});
			// dh_{t-1} = dz_t Wh^T.
			if(t>0)
				whT.MultiplyAdd(B,dz.data()+(size_t)t*G,T*G,dh.data(),H,FP(0));
		}

		// h_{t-1} for every (b,t), with zeros at t=0, so dWh is one GEMM: dWh = Hprev^T dz.
		std::vector<FP> hprev((size_t)B*T*H);
		{
			// Recompute h from the stored gates and cells: h_t = o_t * tanh(c_t).
			ParallelForRange(0,B,1,[&](int b0,int b1)
			{
				for(int b=b0;b<b1;++b)
					for(int t=0;t<T;++t)
					{
						FP* hp = hprev.data()+((size_t)b*T+t)*H;
						if(t==0)
						{
							std::fill(hp,hp+H,FP(0));
							continue;
						}
						const FP* ap = a+((size_t)b*T+t-1)*G;
						const FP* cpv = c+((size_t)b*T+t-1)*H;
						for(int j=0;j<H;++j)
							hp[j] = ap[3*H+j]*std::tanh(cpv[j]);
					}
			});
		}

		NDArray dx = NDData::New(x.Shape());
		NDArray dWx = NDData::New(Wx.Shape());
		NDArray dWh = NDData::New(Wh.Shape());
		NDArray db = NDData::New(in[3].Shape());
		gemm::Gemm<FP>(false,true,B*T,I,G,FP(1),dz.data(),G,Wx.Raw(),G,FP(0),dx.Raw(),I);		// dx = dz Wx^T
		gemm::Gemm<FP>(true,false,I,G,B*T,FP(1),x.Raw(),I,dz.data(),G,FP(0),dWx.Raw(),G);		// dWx = x^T dz
		gemm::Gemm<FP>(true,false,H,G,B*T,FP(1),hprev.data(),H,dz.data(),G,FP(0),dWh.Raw(),G);	// dWh = Hprev^T dz
		fused::ColumnSum(dz.data(),B*T,G,db.Raw());
		return {dx,dWx,dWh,db};
	}

#ifdef AUTOGRAD_CUDA
	GPU_FORWARD

	NDArray _dz,_dc;	// Device: activated gates (B,T,4H) and cells (B,T,H).

	NDArray ForwardGpu(const NDArrays& in)
	{
		const NDArray x = in[0].Contiguous();
		const NDArray Wx = in[1].Contiguous();
		const NDArray Wh = in[2].Contiguous();
		const int B = x.Shape()[0];
		const int T = x.Shape()[1];
		const int I = x.Shape()[2];
		const int H = Wh.Shape()[0];
		const int G = 4*H;
		_dz._Attach(NDData::New({B,T,G}));
		_dc._Attach(NDData::New({B,T,H}));
		NDArray out = NDData::New({B,T,H});
		float* z = _dz.DevWrite();
		float* c = _dc.DevWrite();
		float* h = out.DevWrite();
		const float* wh = Wh.Dev();
		dev::FillRows(z,in[3].Contiguous().Dev(),B*T,G);
		dev::Gemm(false,false,B*T,G,I,1.0f,x.Dev(),I,Wx.Dev(),G,1.0f,z,G);
		for(int t=0;t<T;++t)
		{
			if(t>0)
				dev::Gemm(false,false,B,G,H,1.0f,h+(size_t)(t-1)*H,T*H,wh,G,1.0f,z+(size_t)t*G,T*G);
			dev::LstmStepForward(z+(size_t)t*G,T*G,t>0?c+(size_t)(t-1)*H:nullptr,T*H,c+(size_t)t*H,h+(size_t)t*H,T*H,B,H);
		}
		return out;
	}

	NDArrays BackwardGpu(const NDArray& gradient,const NDArrays& in)
	{
		const NDArray x = in[0].Contiguous();
		const NDArray Wx = in[1].Contiguous();
		const NDArray Wh = in[2].Contiguous();
		const NDArray g = gradient.Contiguous();
		const int B = x.Shape()[0];
		const int T = x.Shape()[1];
		const int I = x.Shape()[2];
		const int H = Wh.Shape()[0];
		const int G = 4*H;
		const float* a = _dz.Dev();
		const float* c = _dc.Dev();
		const float* gr = g.Dev();
		const float* wh = Wh.Dev();
		NDArray dzA = NDData::New({B,T,G});
		NDArray dhA = NDData::New({B,H});
		NDArray dcA = NDData::New({B,H});
		float* dz = dzA.DevWrite();
		float* dh = dhA.DevWrite();
		float* dc = dcA.DevWrite();
		dev::Zero(dh,(size_t)B*H);
		dev::Zero(dc,(size_t)B*H);
		for(int t=T-1;t>=0;--t)
		{
			dev::LstmStepBackward(a+(size_t)t*G,T*G,c+(size_t)t*H,t>0?c+(size_t)(t-1)*H:nullptr,T*H,gr+(size_t)t*H,T*H,dh,dc,dz+(size_t)t*G,T*G,B,H);
			if(t>0)
				dev::Gemm(false,true,B,H,G,1.0f,dz+(size_t)t*G,T*G,wh,G,0.0f,dh,H);
		}
		NDArray hprev = NDData::New({B,T,H});
		dev::LstmPreviousH(a,c,hprev.DevWrite(),B,T,H);
		NDArray dx = NDData::New(x.Shape());
		NDArray dWx = NDData::New(Wx.Shape());
		NDArray dWh = NDData::New(Wh.Shape());
		NDArray db = NDData::New(in[3].Shape());
		dev::Gemm(false,true,B*T,I,G,1.0f,dz,G,Wx.Dev(),G,0.0f,dx.DevWrite(),I);
		dev::Gemm(true,false,I,G,B*T,1.0f,x.Dev(),I,dz,G,0.0f,dWx.DevWrite(),G);
		dev::Gemm(true,false,H,G,B*T,1.0f,hprev.Dev(),H,dz,G,0.0f,dWh.DevWrite(),G);
		dev::ColumnSum(dz,B*T,G,db.DevWrite());
		return {dx,dWx,dWh,db};
	}
#endif
};



// Token embedding lookup, optionally plus a learned position embedding: y[b,t] = tok[idx[b,t]] (+ pos[t]).
// Inputs: tok (V,C) [, pos (block,C)]; the indices (B,T) are a constant.
class KoEmbedding : public Kernel
{
	NDArray _idx;

public:
	KoEmbedding(const NDArray& idx) :
		_idx(idx.Contiguous())
	{
	}

	NDArray Forward(const NDArrays& in) override
	{
		DISPATCH_FORWARD
		const int B = _idx.Shape()[0];
		const int T = _idx.Shape()[1];
		const NDArray tok = in[0].Contiguous();
		const int C = tok.Shape()[1];
		const FP* tr = tok.Raw();
		const FP* pr = in.size()>1?in[1].Contiguous().Raw():nullptr;
		const FP* ir = _idx.Raw();
		NDArray y = NDData::New({B,T,C});
		FP* yr = y.Raw();
		ParallelForRange(0,B*T,64,[&](int r0,int r1)
		{
			for(int r=r0;r<r1;++r)
			{
				const FP* src = tr+(size_t)((int)ir[r])*C;
				const FP* p = pr?pr+(size_t)(r%T)*C:nullptr;
				FP* dst = yr+(size_t)r*C;
				for(int c=0;c<C;++c)
					dst[c] = src[c]+(p?p[c]:FP(0));
			}
		});
		return y;
	}

	NDArrays Backward(const NDArray& gradient,const NDArrays& in) override
	{
		DISPATCH_BACKWARD
		const int B = _idx.Shape()[0];
		const int T = _idx.Shape()[1];
		const int C = in[0].Shape()[1];
		const NDArray g = gradient.Contiguous();
		const FP* gr = g.Raw();
		const FP* ir = _idx.Raw();
		NDArray dtok = NDData::New(in[0].Shape(),0.0);
		FP* dt = dtok.Raw();
		// Parallel over feature columns so no two threads write the same element.
		ParallelForRange(0,C,8,[&](int c0,int c1)
		{
			for(int r=0;r<B*T;++r)
			{
				FP* dst = dt+(size_t)((int)ir[r])*C;
				const FP* src = gr+(size_t)r*C;
				for(int c=c0;c<c1;++c)
					dst[c] += src[c];
			}
		});
		if(in.size()==1)
			return {dtok};
		NDArray dpos = NDData::New(in[1].Shape(),0.0);
		FP* dp = dpos.Raw();
		for(int b=0;b<B;++b)
			for(int t=0;t<T;++t)
			{
				const FP* src = gr+((size_t)b*T+t)*C;
				FP* dst = dp+(size_t)t*C;
				for(int c=0;c<C;++c)
					dst[c] += src[c];
			}
		return {dtok,dpos};
	}

#ifdef AUTOGRAD_CUDA
	GPU_FORWARD

	NDArray ForwardGpu(const NDArrays& in)
	{
		const int B = _idx.Shape()[0];
		const int T = _idx.Shape()[1];
		const NDArray tok = in[0].Contiguous();
		const int C = tok.Shape()[1];
		NDArray y = NDData::New({B,T,C});
		dev::EmbeddingForward(tok.Dev(),in.size()>1?in[1].Contiguous().Dev():nullptr,_idx.Dev(),y.DevWrite(),B,T,C);
		return y;
	}

	NDArrays BackwardGpu(const NDArray& gradient,const NDArrays& in)
	{
		const int B = _idx.Shape()[0];
		const int T = _idx.Shape()[1];
		const int V = in[0].Shape()[0];
		const int C = in[0].Shape()[1];
		NDArray dtok = NDData::New(in[0].Shape());
		if(in.size()==1)
		{
			dev::EmbeddingBackward(gradient.Contiguous().Dev(),_idx.Dev(),dtok.DevWrite(),nullptr,B,T,C,V);
			return {dtok};
		}
		NDArray dpos = NDData::New(in[1].Shape());
		float* dp = dpos.DevWrite();
		dev::Zero(dp,dpos.Size());
		dev::EmbeddingBackward(gradient.Contiguous().Dev(),_idx.Dev(),dtok.DevWrite(),dp,B,T,C,V);
		return {dtok,dpos};
	}
#endif
};


// Inverted dropout with a stateless hashed mask: element i is kept if hash(seed,i) >= p, and kept values are scaled
// by 1/(1-p). The mask is recomputed in the backward pass instead of stored, and is identical on CPU and GPU.
class KoDropoutHashed : public Kernel
{
	const FP		_p;
	const unsigned	_seed;

public:
	static float Uniform(const unsigned seed,const unsigned long long i)
	{
		unsigned h = (unsigned)i*0x9E3779B1u^seed^(unsigned)(i>>32)*0x7FEB352Du;
		h ^= h>>16; h *= 0x85EBCA6Bu; h ^= h>>13; h *= 0xC2B2AE35u; h ^= h>>16;
		return h*(1.0f/4294967296.0f);
	}

	KoDropoutHashed(const FP p,const unsigned seed) :
		_p(p),
		_seed(seed)
	{
	}

	NDArray Forward(const NDArrays& in) override
	{
		DISPATCH_FORWARD
		return Apply(in[0]);
	}

	NDArrays Backward(const NDArray& gradient,const NDArrays& in) override
	{
		DISPATCH_BACKWARD
		return {Apply(gradient)};
	}

private:
	NDArray Apply(const NDArray& a) const
	{
		const NDArray x = a.Contiguous();
		NDArray y = NDData::New(x.Shape());
		const FP* xr = x.Raw();
		FP* yr = y.Raw();
		const FP p = _p;
		const unsigned seed = _seed;
		ParallelForRange(0,x.Size(),16384,[=](int b,int e)
		{
			for(int i=b;i<e;++i)
				yr[i] = Uniform(seed,i)>=p?xr[i]/(FP(1)-p):FP(0);
		});
		return y;
	}

public:
#ifdef AUTOGRAD_CUDA
	GPU_FORWARD

	NDArray ForwardGpu(const NDArrays& in)
	{
		return ApplyGpu(in[0]);
	}

	NDArrays BackwardGpu(const NDArray& gradient,const NDArrays&)
	{
		return {ApplyGpu(gradient)};
	}

	NDArray ApplyGpu(const NDArray& a) const
	{
		const NDArray x = a.Contiguous();
		NDArray y = NDData::New(x.Shape());
		dev::Dropout(x.Dev(),y.DevWrite(),x.Size(),float(_p),_seed);
		return y;
	}
#endif
};
