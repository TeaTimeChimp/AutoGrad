#pragma once

// Muon: MomentUm Orthogonalized by Newton-schulz (Keller Jordan, 2024).
//
// For a weight matrix, Adam scales each element independently. Muon instead takes the (Nesterov) momentum matrix M
// and replaces it by the nearest semi-orthogonal matrix, i.e. U V^T from M's SVD, so every direction in the update
// gets the same step size regardless of how large its gradient happened to be. Rare-but-important directions are
// no longer drowned out by a few dominant ones. U V^T is approximated cheaply with five iterations of a quintic
// Newton-Schulz polynomial that only needs matrix multiplies:
//
//		X = M/|M|;  repeat 5x:  A = X X^T;  X = a X + (b A + c A A) X
//
// Muon is only for 2D hidden-layer weights. Embeddings, the output head, biases and norm gains should use Adam.

#include <vector>
#include <cmath>
#include <map>
#include "Tensor.h"
#include "Optimiser.h"
#include "Gemm.h"
#include "ParallelFor.h"


class Muon : public Optimiser
{
	FP					_lr;
	const FP			_momentum;
	std::vector<std::vector<FP>>	_buffers;

	// In-place approximate orthogonalisation of the r x c row-major matrix G (requires r <= c).
	static void NewtonSchulz(FP* G,const int r,const int c,const int steps=5)
	{
		const FP a = FP(3.4445),b = FP(-4.7750),cc = FP(2.0315);
		double sq = 0.0;
		for(size_t i=0,n=(size_t)r*c;i<n;++i)
			sq += double(G[i])*G[i];
		const FP inv = FP(1.0/(std::sqrt(sq)+1e-7));
		for(size_t i=0,n=(size_t)r*c;i<n;++i)
			G[i] *= inv;

		std::vector<FP> A((size_t)r*r),B((size_t)r*r),X((size_t)r*c);
		for(int s=0;s<steps;++s)
		{
			gemm::Gemm<FP>(false,true,r,r,c,FP(1),G,c,G,c,FP(0),A.data(),r);				// A = X X^T
			gemm::Gemm<FP>(false,false,r,r,r,cc,A.data(),r,A.data(),r,FP(0),B.data(),r);	// B = c A A
			for(size_t i=0,n=(size_t)r*r;i<n;++i)
				B[i] += b*A[i];																// B += b A
			std::copy(G,G+(size_t)r*c,X.data());
			gemm::Gemm<FP>(false,false,r,c,r,FP(1),B.data(),r,X.data(),c,a,G,c);			// X = a X + B X
		}
	}

	static void Transpose(const FP* src,const int r,const int c,FP* dst)
	{
		for(int i=0;i<r;++i)
			for(int j=0;j<c;++j)
				dst[(size_t)j*r+i] = src[(size_t)i*c+j];
	}

#ifdef AUTOGRAD_CUDA
	std::vector<NDArray>	_dbuf;	// Device momentum buffers (default-constructed NDArrays are empty handles,
	std::vector<bool>		_dinit;	// so track which ones have been allocated).

	// GPU step: parameters with the same shape are orthogonalised together with batched GEMMs, so a model with
	// many equally-shaped layers costs a handful of large launches instead of hundreds of tiny ones.
	void StepAllGpu()
	{
		if(_dbuf.size()<_parameters.size())
		{
			_dbuf.resize(_parameters.size());
			_dinit.resize(_parameters.size(),false);
		}
		std::map<std::pair<int,int>,std::vector<size_t>> groups;
		for(size_t k=0;k<_parameters.size();++k)
			if(_parameters[k]->Gradient())
				groups[{_parameters[k]->Data().Shape()[0],_parameters[k]->Data().Shape()[1]}].push_back(k);

		const float a = 3.4445f,b = -4.7750f,cc = 2.0315f;
		for(auto& [shape,ks]:groups)
		{
			const int rows = shape.first,cols = shape.second;
			const int G = (int)ks.size();
			const size_t n = (size_t)rows*cols;
			const bool trans = rows>cols;
			const int r = trans?cols:rows,c = trans?rows:cols;	// Orthogonalise as r x c with r <= c.

			NDArray U = NDData::New({G,r,c});
			NDArray X = NDData::New({G,r,c});
			NDArray A = NDData::New({G,r,r});
			NDArray Bm = NDData::New({G,r,r});
			float* u = U.DevWrite();
			float* x = X.DevWrite();
			float* A_ = A.DevWrite();
			float* B_ = Bm.DevWrite();

			// Nesterov momentum per matrix, stacked (transposed if needed) into U.
			for(int i=0;i<G;++i)
			{
				const size_t k = ks[i];
				if(!_dinit[k])
				{
					_dbuf[k]._Attach(NDData::New({rows,cols},0.0));
					_dinit[k] = true;
				}
				float* g = _parameters[k]->Gradient()->Data().DevModify();
				if(trans)
				{
					dev::NesterovMomentum(g,_dbuf[k].DevModify(),x+i*n,n,_momentum);
					dev::Transpose(x+i*n,u+i*n,rows,cols);
				}
				else
					dev::NesterovMomentum(g,_dbuf[k].DevModify(),u+i*n,n,_momentum);
			}

			dev::NormalizeBatched(u,G,n,1e-7f);
			dev::Batch xx; xx.count = G; xx.aOuter = (long long)n; xx.bOuter = (long long)n; xx.cOuter = (long long)r*r;
			dev::Batch aa; aa.count = G; aa.aOuter = (long long)r*r; aa.bOuter = (long long)r*r; aa.cOuter = (long long)r*r;
			dev::Batch bx; bx.count = G; bx.aOuter = (long long)r*r; bx.bOuter = (long long)n; bx.cOuter = (long long)n;
			for(int s=0;s<5;++s)
			{
				dev::Gemm(false,true,r,r,c,1.0f,u,c,u,c,0.0f,A_,r,xx);			// A = X X^T
				dev::Gemm(false,false,r,r,r,cc,A_,r,A_,r,0.0f,B_,r,aa);		// B = c A A
				dev::Axpy(B_,A_,b,(size_t)G*r*r);								// B += b A
				dev::Copy(x,u,(size_t)G*n);
				dev::Gemm(false,false,r,c,r,1.0f,B_,r,x,c,a,u,c,bx);			// X = a X + B X
			}

			const float scale = _lr*float(std::sqrt((std::max)(1.0,double(cols)/double(rows))));
			for(int i=0;i<G;++i)
			{
				auto& p = _parameters[ks[i]];
				const float* upd = u+i*n;
				if(trans)
				{
					dev::Transpose(u+i*n,x+i*n,cols,rows);
					upd = x+i*n;
				}
				dev::Axpy(p->Data().DevModify(),upd,-scale,n);
				dev::Zero(p->Gradient()->Data().DevModify(),n);
				p->ClearChildren();
			}
		}
	}
#endif

public:
	Muon(const std::vector<TensorPtr>& parameters,const FP lr=FP(0.02),const FP momentum=FP(0.95)) :
		Optimiser(parameters),
		_lr(lr),
		_momentum(momentum)
	{
		for(auto& p:_parameters)
		{
			if(p->Data().Shape().size()!=2)
				throw InvalidArgument();
			_buffers.emplace_back(p->Data().Size(),FP(0));
		}
	}

	void SetLearningRate(const FP lr)
	{
		_lr = lr;
	}

	void Step(bool zero=true) override
	{
#ifdef AUTOGRAD_CUDA
		if(NDConfig::useGpu)
		{
			StepAllGpu();
			return;
		}
#endif
		for(size_t k=0;k<_parameters.size();++k)
		{
			auto& p = _parameters[k];
			if(!p->Gradient())
				continue;
			const int rows = p->Data().Shape()[0];
			const int cols = p->Data().Shape()[1];
			const int n = rows*cols;

			FP* w = p->Data().Raw();
			FP* g = p->Gradient()->Data().Raw();
			FP* m = _buffers[k].data();

			// Nesterov momentum: m = mu*m + g;  update = g + mu*m.
			std::vector<FP> u(n);
			for(int i=0;i<n;++i)
			{
				m[i] = _momentum*m[i]+g[i];
				u[i] = g[i]+_momentum*m[i];
			}

			// Orthogonalise along the smaller dimension.
			if(rows<=cols)
				NewtonSchulz(u.data(),rows,cols);
			else
			{
				std::vector<FP> t(n);
				Transpose(u.data(),rows,cols,t.data());
				NewtonSchulz(t.data(),cols,rows);
				Transpose(t.data(),cols,rows,u.data());
			}

			// Scale so the update's RMS is roughly independent of the matrix's aspect ratio.
			const FP scale = _lr*FP(std::sqrt((std::max)(1.0,double(cols)/double(rows))));
			for(int i=0;i<n;++i)
			{
				w[i] -= scale*u[i];
				if(zero)
					g[i] = FP(0);
			}
			p->ClearChildren();
		}
	}
};
