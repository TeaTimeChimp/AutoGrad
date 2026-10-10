#pragma once

// General matrix multiply for row-major matrices:
//
//		C = alpha * op(A) @ op(B) + beta * C
//
// where op(X) is X or X^T. A is MxK after op, B is KxN after op, C is MxN.
// lda/ldb/ldc are the row strides of the matrices as stored in memory.
//
// The float path packs panels of A and B into contiguous buffers and runs a 6x16 AVX2/FMA micro-kernel,
// parallelised over blocks of C. Any other FP type (e.g. double for gradient checking) uses a simple loop.

#include <immintrin.h>
#include <vector>
#include <algorithm>
#include <type_traits>
#include "DType.h"
#include "ParallelFor.h"


namespace gemm
{
	constexpr int MR = 6;		// Micro-kernel rows.
	constexpr int NR = 16;		// Micro-kernel columns (2 AVX registers).
	constexpr int KC = 256;		// Depth of a packed block.
	constexpr int MC = 72;		// Rows of A per task (multiple of MR).
	constexpr int NC = 256;		// Columns of B per task (multiple of NR).


	// Element (i,j) of op(X).
	template<typename T>
	inline T At(const T* X,const int ld,const bool trans,const int i,const int j)
	{
		return trans?X[(size_t)j*ld+i]:X[(size_t)i*ld+j];
	}


	// Reference implementation (any type).
	template<typename T>
	void Reference(const bool transA,const bool transB,const int M,const int N,const int K,const T alpha,
		const T* A,const int lda,const T* B,const int ldb,const T beta,T* C,const int ldc)
	{
		for(int i=0;i<M;++i)
		{
			T* c = C+(size_t)i*ldc;
			if(beta==T(0))
				std::fill(c,c+N,T(0));
			else if(beta!=T(1))
				for(int j=0;j<N;++j)
					c[j] *= beta;
			for(int k=0;k<K;++k)
			{
				const T a = alpha*At(A,lda,transA,i,k);
				if(a==T(0))
					continue;
				if(!transB)
				{
					const T* b = B+(size_t)k*ldb;
					for(int j=0;j<N;++j)
						c[j] += a*b[j];
				}
				else
				{
					for(int j=0;j<N;++j)
						c[j] += a*B[(size_t)j*ldb+k];
				}
			}
		}
	}


	// Pack an (mc x kc) block of op(A) starting at (i0,k0) into MR-row panels: [panel][k][MR], zero padded.
	inline void PackA(const float* A,const int lda,const bool transA,const int i0,const int k0,const int mc,const int kc,float* dst)
	{
		for(int p=0;p<mc;p+=MR)
		{
			const int rows = (std::min)(MR,mc-p);
			for(int k=0;k<kc;++k)
			{
				int r = 0;
				if(!transA)
				{
					for(;r<rows;++r)
						dst[r] = A[(size_t)(i0+p+r)*lda+(k0+k)];
				}
				else
				{
					const float* src = A+(size_t)(k0+k)*lda+(i0+p);
					for(;r<rows;++r)
						dst[r] = src[r];
				}
				for(;r<MR;++r)
					dst[r] = 0.0f;
				dst += MR;
			}
		}
	}


	// Pack a (kc x nc) block of op(B) starting at (k0,j0) into NR-column panels: [panel][k][NR], zero padded.
	inline void PackB(const float* B,const int ldb,const bool transB,const int k0,const int j0,const int kc,const int nc,float* dst)
	{
		for(int p=0;p<nc;p+=NR)
		{
			const int cols = (std::min)(NR,nc-p);
			for(int k=0;k<kc;++k)
			{
				if(!transB)
				{
					const float* src = B+(size_t)(k0+k)*ldb+(j0+p);
					if(cols==NR)
					{
						_mm256_storeu_ps(dst,_mm256_loadu_ps(src));
						_mm256_storeu_ps(dst+8,_mm256_loadu_ps(src+8));
					}
					else
					{
						int c = 0;
						for(;c<cols;++c)
							dst[c] = src[c];
						for(;c<NR;++c)
							dst[c] = 0.0f;
					}
				}
				else
				{
					int c = 0;
					for(;c<cols;++c)
						dst[c] = B[(size_t)(j0+p+c)*ldb+(k0+k)];
					for(;c<NR;++c)
						dst[c] = 0.0f;
				}
				dst += NR;
			}
		}
	}


	// 6x16 micro-kernel: acc = Apanel(6xkc) @ Bpanel(kcx16), then C(rows x cols) = alpha*acc + (first ? beta*C : C).
	inline void Kernel(const int kc,const float* a,const float* b,float* C,const int ldc,const int rows,const int cols,const float alpha,const float beta,const bool first)
	{
		__m256 c00 = _mm256_setzero_ps(),c01 = _mm256_setzero_ps();
		__m256 c10 = _mm256_setzero_ps(),c11 = _mm256_setzero_ps();
		__m256 c20 = _mm256_setzero_ps(),c21 = _mm256_setzero_ps();
		__m256 c30 = _mm256_setzero_ps(),c31 = _mm256_setzero_ps();
		__m256 c40 = _mm256_setzero_ps(),c41 = _mm256_setzero_ps();
		__m256 c50 = _mm256_setzero_ps(),c51 = _mm256_setzero_ps();
		for(int k=0;k<kc;++k)
		{
			const __m256 b0 = _mm256_loadu_ps(b);
			const __m256 b1 = _mm256_loadu_ps(b+8);
			__m256 av;
			av = _mm256_broadcast_ss(a+0); c00 = _mm256_fmadd_ps(av,b0,c00); c01 = _mm256_fmadd_ps(av,b1,c01);
			av = _mm256_broadcast_ss(a+1); c10 = _mm256_fmadd_ps(av,b0,c10); c11 = _mm256_fmadd_ps(av,b1,c11);
			av = _mm256_broadcast_ss(a+2); c20 = _mm256_fmadd_ps(av,b0,c20); c21 = _mm256_fmadd_ps(av,b1,c21);
			av = _mm256_broadcast_ss(a+3); c30 = _mm256_fmadd_ps(av,b0,c30); c31 = _mm256_fmadd_ps(av,b1,c31);
			av = _mm256_broadcast_ss(a+4); c40 = _mm256_fmadd_ps(av,b0,c40); c41 = _mm256_fmadd_ps(av,b1,c41);
			av = _mm256_broadcast_ss(a+5); c50 = _mm256_fmadd_ps(av,b0,c50); c51 = _mm256_fmadd_ps(av,b1,c51);
			a += MR;
			b += NR;
		}

		alignas(32) float acc[MR][NR];
		_mm256_store_ps(acc[0],c00); _mm256_store_ps(acc[0]+8,c01);
		_mm256_store_ps(acc[1],c10); _mm256_store_ps(acc[1]+8,c11);
		_mm256_store_ps(acc[2],c20); _mm256_store_ps(acc[2]+8,c21);
		_mm256_store_ps(acc[3],c30); _mm256_store_ps(acc[3]+8,c31);
		_mm256_store_ps(acc[4],c40); _mm256_store_ps(acc[4]+8,c41);
		_mm256_store_ps(acc[5],c50); _mm256_store_ps(acc[5]+8,c51);

		const __m256 va = _mm256_set1_ps(alpha);
		const __m256 vb = _mm256_set1_ps(beta);
		for(int r=0;r<rows;++r)
		{
			float* c = C+(size_t)r*ldc;
			if(cols==NR)
			{
				__m256 v0 = _mm256_mul_ps(va,_mm256_load_ps(acc[r]));
				__m256 v1 = _mm256_mul_ps(va,_mm256_load_ps(acc[r]+8));
				if(first)
				{
					if(beta!=0.0f)
					{
						v0 = _mm256_fmadd_ps(vb,_mm256_loadu_ps(c),v0);
						v1 = _mm256_fmadd_ps(vb,_mm256_loadu_ps(c+8),v1);
					}
				}
				else
				{
					v0 = _mm256_add_ps(_mm256_loadu_ps(c),v0);
					v1 = _mm256_add_ps(_mm256_loadu_ps(c+8),v1);
				}
				_mm256_storeu_ps(c,v0);
				_mm256_storeu_ps(c+8,v1);
			}
			else
			{
				for(int j=0;j<cols;++j)
				{
					const float v = alpha*acc[r][j];
					if(first)
						c[j] = (beta!=0.0f?beta*c[j]:0.0f)+v;
					else
						c[j] += v;
				}
			}
		}
	}


	// Optimised float GEMM. If 'parallel' is false the calling thread does all the work (use when the caller is already parallel).
	inline void Sgemm(const bool transA,const bool transB,const int M,const int N,const int K,const float alpha,
		const float* A,const int lda,const float* B,const int ldb,const float beta,float* C,const int ldc,const bool parallel=true)
	{
		if(M<=0||N<=0)
			return;
		if(K<=0)
		{
			for(int i=0;i<M;++i)
				for(int j=0;j<N;++j)
					C[(size_t)i*ldc+j] = beta==0.0f?0.0f:beta*C[(size_t)i*ldc+j];
			return;
		}

		const int mBlocks = (M+MC-1)/MC;
		const int nBlocks = (N+NC-1)/NC;
		const int tasks = mBlocks*nBlocks;

		// Each task computes one MCxNC block of C over the full K range (no write conflicts between tasks).
		auto task = [&](const int t)
		{
			thread_local std::vector<float> aBuf,bBuf;
			aBuf.resize((size_t)MC*KC);
			bBuf.resize((size_t)KC*NC);
			const int mb = t/nBlocks;
			const int nb = t%nBlocks;
			const int i0 = mb*MC;
			const int j0 = nb*NC;
			const int mc = (std::min)(MC,M-i0);
			const int nc = (std::min)(NC,N-j0);
			for(int k0=0;k0<K;k0+=KC)
			{
				const int kc = (std::min)(KC,K-k0);
				const bool first = k0==0;
				PackA(A,lda,transA,i0,k0,mc,kc,aBuf.data());
				PackB(B,ldb,transB,k0,j0,kc,nc,bBuf.data());
				for(int jp=0;jp<nc;jp+=NR)
				{
					const float* bp = bBuf.data()+(size_t)(jp/NR)*kc*NR;
					const int cols = (std::min)(NR,nc-jp);
					for(int ip=0;ip<mc;ip+=MR)
					{
						const float* ap = aBuf.data()+(size_t)(ip/MR)*kc*MR;
						const int rows = (std::min)(MR,mc-ip);
						Kernel(kc,ap,bp,C+(size_t)(i0+ip)*ldc+(j0+jp),ldc,rows,cols,alpha,beta,first);
					}
				}
			}
		};

		// Small problems are not worth the threading overhead.
		const double flops = 2.0*M*N*K;

		// Few output blocks but a long inner dimension (e.g. weight gradients X^T@dY where K=B*T): split K across
		// threads, compute partial products into private buffers, then reduce.
		const int threads = ParallelPool::Get().Threads();
		if(parallel&&flops>=4.0e5&&tasks*2<threads&&K>=4*KC)
		{
			const int splits = (std::min)(threads/tasks,K/KC);
			const int kPer = ((K+splits-1)/splits+KC-1)/KC*KC;
			const int used = (K+kPer-1)/kPer;
			std::vector<float> partial((size_t)used*M*N);
			ParallelFor(0,used,[&](const int s)
			{
				const int k0 = s*kPer;
				const int kc = (std::min)(kPer,K-k0);
				const float* a = transA?A+(size_t)k0*lda:A+k0;
				const float* b = transB?B+k0:B+(size_t)k0*ldb;
				Sgemm(transA,transB,M,N,kc,alpha,a,lda,b,ldb,0.0f,partial.data()+(size_t)s*M*N,N,false);
			});
			ParallelForRange(0,M,8,[&](int r0,int r1)
			{
				for(int i=r0;i<r1;++i)
				{
					float* c = C+(size_t)i*ldc;
					for(int j=0;j<N;++j)
					{
						float sum = beta==0.0f?0.0f:beta*c[j];
						for(int s=0;s<used;++s)
							sum += partial[((size_t)s*M+i)*N+j];
						c[j] = sum;
					}
				}
			});
			return;
		}

		if(!parallel||tasks==1||flops<4.0e5)
		{
			for(int t=0;t<tasks;++t)
				task(t);
		}
		else
			ParallelFor(0,tasks,task);
	}


	// op(B) (KxN) packed once into NR-column panels per KC-deep block, for reuse across many multiplies
	// (e.g. a recurrent weight matrix used at every timestep).
	struct PackedB
	{
		int K = 0,N = 0;
		std::vector<float> data;	// [kblock][panel][kc][NR]

		void Pack(const float* B,const int ldb,const bool transB,const int K_,const int N_)
		{
			K = K_;
			N = N_;
			const int panels = (N+NR-1)/NR;
			data.resize((size_t)K*panels*NR);
			ParallelFor(0,(K+KC-1)/KC,[&](const int kb)
			{
				const int k0 = kb*KC;
				const int kc = (std::min)(KC,K-k0);
				PackB(B,ldb,transB,k0,0,kc,N,data.data()+(size_t)k0*panels*NR);
			});
		}

		const float* Block(const int k0) const
		{
			return data.data()+(size_t)k0*((N+NR-1)/NR)*NR;
		}
	};


	// C(MxN) = alpha*op(A)@B + beta*C with B pre-packed, parallel over column panels. Suits small M.
	inline void SgemmPacked(const bool transA,const int M,const float alpha,const float* A,const int lda,const PackedB& B,
		const float beta,float* C,const int ldc)
	{
		const int K = B.K;
		const int N = B.N;
		const int panels = (N+NR-1)/NR;
		const int panelsPerTask = (std::max)(1,(std::min)(4,panels/ParallelPool::Get().Threads()));
		const int tasks = (panels+panelsPerTask-1)/panelsPerTask;
		ParallelFor(0,tasks,[&](const int task)
		{
			thread_local std::vector<float> aBuf;
			aBuf.resize((size_t)MC*KC);
			const int p0 = task*panelsPerTask;
			const int p1 = (std::min)(panels,p0+panelsPerTask);
			for(int i0=0;i0<M;i0+=MC)
			{
				const int mc = (std::min)(MC,M-i0);
				for(int k0=0;k0<K;k0+=KC)
				{
					const int kc = (std::min)(KC,K-k0);
					PackA(A,lda,transA,i0,k0,mc,kc,aBuf.data());
					const float* bBlock = B.Block(k0);
					for(int p=p0;p<p1;++p)
					{
						const int jp = p*NR;
						const int cols = (std::min)(NR,N-jp);
						const float* bp = bBlock+(size_t)p*kc*NR;
						for(int ip=0;ip<mc;ip+=MR)
						{
							const int rows = (std::min)(MR,mc-ip);
							Kernel(kc,aBuf.data()+(size_t)(ip/MR)*kc*MR,bp,C+(size_t)(i0+ip)*ldc+jp,ldc,rows,cols,alpha,beta,k0==0);
						}
					}
				}
			}
		});
	}


	template<typename T>
	inline void Gemm(const bool transA,const bool transB,const int M,const int N,const int K,const T alpha,
		const T* A,const int lda,const T* B,const int ldb,const T beta,T* C,const int ldc,const bool parallel=true)
	{
		if constexpr(std::is_same_v<T,float>)
			Sgemm(transA,transB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,parallel);
		else
			Reference(transA,transB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
	}
}
