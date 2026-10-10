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
