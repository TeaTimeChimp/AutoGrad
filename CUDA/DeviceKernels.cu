// CUDA kernels for the fused AutoGrad kernels (float only). See DeviceKernels.h for the interface.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include "DeviceKernels.h"


#define CK(x) do{ cudaError_t e_ = (x); if(e_!=cudaSuccess){ fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e_),__FILE__,__LINE__); exit(3);} }while(0)

namespace
{
	inline int Blocks(const size_t n,const int threads)
	{
		return (int)((n+threads-1)/threads);
	}

	// ---- Caching allocator ----
	// cudaMalloc/cudaFree are slow and synchronising, so freed blocks are kept in per-size free lists.
	struct Cache
	{
		std::mutex lock;
		std::vector<void*> free[48];
		std::unordered_map<void*,int> bucketOf;
	};
	Cache& TheCache()
	{
		static Cache c;
		return c;
	}
	int Bucket(const size_t bytes)
	{
		int b = 8;	// Minimum 256 bytes.
		while(((size_t)1<<b)<bytes)
			++b;
		return b;
	}
}


namespace dev
{
	bool Available()
	{
		int n = 0;
		return cudaGetDeviceCount(&n)==cudaSuccess&&n>0;
	}

	void* Alloc(size_t bytes)
	{
		const int b = Bucket(bytes);
		Cache& c = TheCache();
		std::unique_lock<std::mutex> l(c.lock);
		if(!c.free[b].empty())
		{
			void* p = c.free[b].back();
			c.free[b].pop_back();
			return p;
		}
		void* p = nullptr;
		CK(cudaMalloc(&p,(size_t)1<<b));
		c.bucketOf[p] = b;
		return p;
	}

	void Free(void* p)
	{
		if(!p)
			return;
		Cache& c = TheCache();
		std::unique_lock<std::mutex> l(c.lock);
		c.free[c.bucketOf.at(p)].push_back(p);
	}

	void Upload(float* dst,const float* host,size_t n)
	{
		CK(cudaMemcpy(dst,host,n*sizeof(float),cudaMemcpyHostToDevice));
	}

	void Download(float* host,const float* src,size_t n)
	{
		CK(cudaMemcpy(host,src,n*sizeof(float),cudaMemcpyDeviceToHost));
	}

	void Copy(float* dst,const float* src,size_t n)
	{
		CK(cudaMemcpyAsync(dst,src,n*sizeof(float),cudaMemcpyDeviceToDevice));
	}

	void Zero(float* dst,size_t n)
	{
		CK(cudaMemsetAsync(dst,0,n*sizeof(float)));
	}

	void Synchronize()
	{
		CK(cudaDeviceSynchronize());
	}
}


// ---- GEMM ----
//
// Tiled SGEMM: each block computes a BM x BN tile of C with 256 threads, each thread a TM x TN micro-tile held in
// registers. A and B tiles (BK deep) are staged in shared memory, A stored k-major so both are read as rows.

template<int BM,int BN,int BK,int TM,int TN>
__global__ void GemmKernel(const bool transA,const bool transB,const int M,const int N,const int K,const float alpha,
	const float* __restrict__ A,const int lda,const float* __restrict__ B,const int ldb,const float beta,float* __restrict__ C,const int ldc,
	const int inner,const long long aO,const long long aI,const long long bO,const long long bI,const long long cO,const long long cI)
{
	const int z = blockIdx.z;
	A += (z/inner)*aO+(z%inner)*aI;
	B += (z/inner)*bO+(z%inner)*bI;
	C += (z/inner)*cO+(z%inner)*cI;

	__shared__ float As[BK][BM];
	__shared__ float Bs[BK][BN];

	const int tid = threadIdx.x;
	const int tx = tid%(BN/TN);
	const int ty = tid/(BN/TN);
	const int row0 = blockIdx.y*BM;
	const int col0 = blockIdx.x*BN;

	float acc[TM][TN];
#pragma unroll
	for(int i=0;i<TM;++i)
#pragma unroll
		for(int j=0;j<TN;++j)
			acc[i][j] = 0.0f;

	for(int k0=0;k0<K;k0+=BK)
	{
		// Load A tile (BM x BK) as As[k][i]; index order chosen so consecutive threads read consecutive memory.
		for(int e=tid;e<BM*BK;e+=256)
		{
			int i,k;
			if(!transA) { i = e/BK; k = e%BK; }
			else { k = e/BM; i = e%BM; }
			const int gi = row0+i,gk = k0+k;
			float v = 0.0f;
			if(gi<M&&gk<K)
				v = transA?A[(size_t)gk*lda+gi]:A[(size_t)gi*lda+gk];
			As[k][i] = v;
		}
		// Load B tile (BK x BN) as Bs[k][j].
		for(int e=tid;e<BK*BN;e+=256)
		{
			int k,j;
			if(!transB) { k = e/BN; j = e%BN; }
			else { j = e/BK; k = e%BK; }
			const int gk = k0+k,gj = col0+j;
			float v = 0.0f;
			if(gk<K&&gj<N)
				v = transB?B[(size_t)gj*ldb+gk]:B[(size_t)gk*ldb+gj];
			Bs[k][j] = v;
		}
		__syncthreads();

#pragma unroll
		for(int k=0;k<BK;++k)
		{
			float a[TM],b[TN];
#pragma unroll
			for(int i=0;i<TM;++i)
				a[i] = As[k][ty*TM+i];
#pragma unroll
			for(int j=0;j<TN;++j)
				b[j] = Bs[k][tx*TN+j];
#pragma unroll
			for(int i=0;i<TM;++i)
#pragma unroll
				for(int j=0;j<TN;++j)
					acc[i][j] += a[i]*b[j];
		}
		__syncthreads();
	}

#pragma unroll
	for(int i=0;i<TM;++i)
	{
		const int r = row0+ty*TM+i;
		if(r>=M)
			continue;
#pragma unroll
		for(int j=0;j<TN;++j)
		{
			const int c = col0+tx*TN+j;
			if(c<N)
			{
				float* p = C+(size_t)r*ldc+c;
				*p = beta==0.0f?alpha*acc[i][j]:alpha*acc[i][j]+beta*(*p);
			}
		}
	}
}



// Faster 128x128x8 SGEMM: float4 global loads, double-buffered shared memory, and an 8x8 micro-tile per thread
// split into four 4x4 quadrants (rows ty*4 and 64+ty*4, cols tx*4 and 64+tx*4) so shared-memory reads are
// conflict-free float4s. Requires 16-byte aligned operands with leading dimensions (and batch strides) divisible
// by 4; edges are handled with guarded scalar loads.
template<bool TA,bool TB>
__global__ void __launch_bounds__(256) GemmFastKernel(const int M,const int N,const int K,const float alpha,
	const float* __restrict__ A,const int lda,const float* __restrict__ B,const int ldb,const float beta,float* __restrict__ C,const int ldc,
	const int inner,const long long aO,const long long aI,const long long bO,const long long bI,const long long cO,const long long cI)
{
	const int z = blockIdx.z;
	A += (z/inner)*aO+(z%inner)*aI;
	B += (z/inner)*bO+(z%inner)*bI;
	C += (z/inner)*cO+(z%inner)*cI;

	__shared__ __align__(16) float As[2][8][128];
	__shared__ __align__(16) float Bs[2][8][128];

	const int tid = threadIdx.x;
	const int tx = tid%16;
	const int ty = tid/16;
	const int row0 = blockIdx.y*128;
	const int col0 = blockIdx.x*128;

	// Each thread loads one float4 of A and one of B per k-tile.
	// A not transposed: row = tid/2, k = (tid%2)*4 (vector along K).  A transposed: k = tid/32, row = (tid%32)*4.
	// B not transposed: k = tid/32, col = (tid%32)*4 (vector along N). B transposed: col = tid/2, k = (tid%2)*4.
	const int aR = TA?(tid%32)*4:tid/2;
	const int aK = TA?tid/32:(tid%2)*4;
	const int bC = TB?tid/2:(tid%32)*4;
	const int bK = TB?(tid%2)*4:tid/32;

	auto loadA = [&](const int k0,float4& v)
	{
		const int gr = row0+aR,gk = k0+aK;
		if(!TA)
		{
			const float* p = A+(size_t)gr*lda+gk;
			if(gr<M&&gk+3<K)
				v = *reinterpret_cast<const float4*>(p);
			else
			{
				v.x = (gr<M&&gk<K)?p[0]:0.0f; v.y = (gr<M&&gk+1<K)?p[1]:0.0f;
				v.z = (gr<M&&gk+2<K)?p[2]:0.0f; v.w = (gr<M&&gk+3<K)?p[3]:0.0f;
			}
		}
		else
		{
			const float* p = A+(size_t)gk*lda+gr;
			if(gk<K&&gr+3<M)
				v = *reinterpret_cast<const float4*>(p);
			else
			{
				v.x = (gk<K&&gr<M)?p[0]:0.0f; v.y = (gk<K&&gr+1<M)?p[1]:0.0f;
				v.z = (gk<K&&gr+2<M)?p[2]:0.0f; v.w = (gk<K&&gr+3<M)?p[3]:0.0f;
			}
		}
	};
	auto loadB = [&](const int k0,float4& v)
	{
		const int gc = col0+bC,gk = k0+bK;
		if(!TB)
		{
			const float* p = B+(size_t)gk*ldb+gc;
			if(gk<K&&gc+3<N)
				v = *reinterpret_cast<const float4*>(p);
			else
			{
				v.x = (gk<K&&gc<N)?p[0]:0.0f; v.y = (gk<K&&gc+1<N)?p[1]:0.0f;
				v.z = (gk<K&&gc+2<N)?p[2]:0.0f; v.w = (gk<K&&gc+3<N)?p[3]:0.0f;
			}
		}
		else
		{
			const float* p = B+(size_t)gc*ldb+gk;
			if(gc<N&&gk+3<K)
				v = *reinterpret_cast<const float4*>(p);
			else
			{
				v.x = (gc<N&&gk<K)?p[0]:0.0f; v.y = (gc<N&&gk+1<K)?p[1]:0.0f;
				v.z = (gc<N&&gk+2<K)?p[2]:0.0f; v.w = (gc<N&&gk+3<K)?p[3]:0.0f;
			}
		}
	};
	auto storeA = [&](const int buf,const float4& v)
	{
		if(!TA)
		{
			As[buf][aK+0][aR] = v.x; As[buf][aK+1][aR] = v.y; As[buf][aK+2][aR] = v.z; As[buf][aK+3][aR] = v.w;
		}
		else
			*reinterpret_cast<float4*>(&As[buf][aK][aR]) = v;
	};
	auto storeB = [&](const int buf,const float4& v)
	{
		if(!TB)
			*reinterpret_cast<float4*>(&Bs[buf][bK][bC]) = v;
		else
		{
			Bs[buf][bK+0][bC] = v.x; Bs[buf][bK+1][bC] = v.y; Bs[buf][bK+2][bC] = v.z; Bs[buf][bK+3][bC] = v.w;
		}
	};

	// Out-of-range loads are skipped by the guards, but the pointers must still be valid to form: clamp rows/cols.
	float acc[8][8];
#pragma unroll
	for(int i=0;i<8;++i)
#pragma unroll
		for(int j=0;j<8;++j)
			acc[i][j] = 0.0f;

	float4 va,vb;
	loadA(0,va);
	loadB(0,vb);
	storeA(0,va);
	storeB(0,vb);
	__syncthreads();

	int buf = 0;
	for(int k0=0;k0<K;k0+=8)
	{
		const bool more = k0+8<K;
		if(more)
		{
			loadA(k0+8,va);
			loadB(k0+8,vb);
		}
#pragma unroll
		for(int k=0;k<8;++k)
		{
			const float4 a0 = *reinterpret_cast<const float4*>(&As[buf][k][ty*4]);
			const float4 a1 = *reinterpret_cast<const float4*>(&As[buf][k][64+ty*4]);
			const float4 b0 = *reinterpret_cast<const float4*>(&Bs[buf][k][tx*4]);
			const float4 b1 = *reinterpret_cast<const float4*>(&Bs[buf][k][64+tx*4]);
			const float a[8] = {a0.x,a0.y,a0.z,a0.w,a1.x,a1.y,a1.z,a1.w};
			const float b[8] = {b0.x,b0.y,b0.z,b0.w,b1.x,b1.y,b1.z,b1.w};
#pragma unroll
			for(int i=0;i<8;++i)
#pragma unroll
				for(int j=0;j<8;++j)
					acc[i][j] += a[i]*b[j];
		}
		if(more)
		{
			storeA(buf^1,va);
			storeB(buf^1,vb);
		}
		__syncthreads();
		buf ^= 1;
	}

#pragma unroll
	for(int i=0;i<8;++i)
	{
		const int r = row0+(i<4?ty*4+i:64+ty*4+i-4);
		if(r>=M)
			continue;
#pragma unroll
		for(int j=0;j<8;++j)
		{
			const int c = col0+(j<4?tx*4+j:64+tx*4+j-4);
			if(c<N)
			{
				float* p = C+(size_t)r*ldc+c;
				*p = beta==0.0f?alpha*acc[i][j]:alpha*acc[i][j]+beta*(*p);
			}
		}
	}
}

__global__ void SplitKReduceKernel(const float* partial,const int splits,const int M,const int N,const float beta,float* C,const int ldc)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x;
	if(i>=(size_t)M*N)
		return;
	float s = 0.0f;
	for(int k=0;k<splits;++k)
		s += partial[(size_t)k*M*N+i];
	float* p = C+(i/N)*ldc+(i%N);
	*p = beta==0.0f?s:s+beta*(*p);
}

namespace dev
{
	void Gemm(bool transA,bool transB,int M,int N,int K,float alpha,const float* A,int lda,const float* B,int ldb,
		float beta,float* C,int ldc,const Batch& batch)
	{
		if(M<=0||N<=0||batch.count<=0)
			return;

		// Split-K: few output tiles but a long inner dimension (weight gradients, K = batch*time). Each split computes
		// a partial product over a slice of K into a workspace, then a reduction adds them up.
		if(batch.count==1&&K>=2048)
		{
			const int tiles = ((M+127)/128)*((N+127)/128);
			int splits = 1;
			while(tiles*splits<2*82&&splits<32&&K%(splits*2*8)==0&&K/(splits*2)>=256)
				splits *= 2;
			if(splits>1&&((size_t)A%16==0)&&((size_t)B%16==0)&&lda%4==0&&ldb%4==0)
			{
				const int kc = K/splits;
				float* work = (float*)Alloc((size_t)splits*M*N*sizeof(float));
				Batch b;
				b.count = splits;
				b.aOuter = transA?(long long)kc*lda:kc;
				b.bOuter = transB?kc:(long long)kc*ldb;
				b.cOuter = (long long)M*N;
				Gemm(transA,transB,M,N,kc,alpha,A,lda,B,ldb,0.0f,work,N,b);
				SplitKReduceKernel<<<Blocks((size_t)M*N,256),256>>>(work,splits,M,N,beta,C,ldc);
				Free(work);	// Returned to the cache; stream ordering makes reuse safe.
				return;
			}
		}
		// Big tiles when there are enough of them to fill the GPU, otherwise smaller tiles for more parallelism.
		const long long bigTiles = (long long)((M+127)/128)*((N+127)/128)*batch.count;
		const bool aligned = ((size_t)A%16==0)&&((size_t)B%16==0)&&lda%4==0&&ldb%4==0&&
			batch.aOuter%4==0&&batch.aInner%4==0&&batch.bOuter%4==0&&batch.bInner%4==0;
		if(bigTiles>=100&&aligned&&K>=8)
		{
			dim3 grid((N+127)/128,(M+127)/128,batch.count);
#define LAUNCH(ta,tb) GemmFastKernel<ta,tb><<<grid,256>>>(M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,batch.inner,batch.aOuter,batch.aInner,batch.bOuter,batch.bInner,batch.cOuter,batch.cInner)
			if(!transA&&!transB) LAUNCH(false,false);
			else if(!transA&&transB) LAUNCH(false,true);
			else if(transA&&!transB) LAUNCH(true,false);
			else LAUNCH(true,true);
#undef LAUNCH
		}
		else if(bigTiles>=160)
		{
			dim3 grid((N+127)/128,(M+127)/128,batch.count);
			GemmKernel<128,128,8,8,8><<<grid,256>>>(transA,transB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,
				batch.inner,batch.aOuter,batch.aInner,batch.bOuter,batch.bInner,batch.cOuter,batch.cInner);
		}
		else
		{
			dim3 grid((N+63)/64,(M+63)/64,batch.count);
			GemmKernel<64,64,16,4,4><<<grid,256>>>(transA,transB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,
				batch.inner,batch.aOuter,batch.aInner,batch.bOuter,batch.bInner,batch.cOuter,batch.cInner);
		}
		CK(cudaGetLastError());
	}
}


// ---- Elementwise and reductions ----

__global__ void FillRowsKernel(float* y,const float* row,const int M,const int N)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x;
	if(i<(size_t)M*N)
		y[i] = row[i%N];
}

// Column sums: block (32 columns x 8 row-groups); each thread sums a strip of rows then atomically adds.
__global__ void ColumnSumKernel(const float* a,const int M,const int N,float* out,const int rowsPerBlock)
{
	const int j = blockIdx.x*32+threadIdx.x;
	const int r0 = blockIdx.y*rowsPerBlock;
	const int r1 = min(M,r0+rowsPerBlock);
	if(j>=N)
		return;
	float s = 0.0f;
	for(int r=r0+threadIdx.y;r<r1;r+=blockDim.y)
		s += a[(size_t)r*N+j];
	atomicAdd(out+j,s);
}

__global__ void AddKernel(const float* a,const float* b,float* y,const size_t n)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n)
		y[i] = a[i]+b[i];
}

__global__ void AddInPlaceKernel(float* y,const float* a,const size_t n)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n)
		y[i] += a[i];
}

__global__ void ScaleKernel(float* y,const float s,const size_t n)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n)
		y[i] *= s;
}

__global__ void AxpyKernel(float* y,const float* x,const float a,const size_t n)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n)
		y[i] += a*x[i];
}

__global__ void SumSquaresKernel(const float* a,const size_t n,double* out)
{
	__shared__ double s[256];
	double v = 0.0;
	for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x)
		v += (double)a[i]*a[i];
	s[threadIdx.x] = v;
	__syncthreads();
	for(int w=128;w>0;w>>=1)
	{
		if(threadIdx.x<w)
			s[threadIdx.x] += s[threadIdx.x+w];
		__syncthreads();
	}
	if(threadIdx.x==0)
		atomicAdd(out,s[0]);
}

__device__ __forceinline__ float WarpSum(float v)
{
	for(int o=16;o>0;o>>=1)
		v += __shfl_xor_sync(0xffffffff,v,o);
	return v;
}

__device__ __forceinline__ float WarpMax(float v)
{
	for(int o=16;o>0;o>>=1)
		v = fmaxf(v,__shfl_xor_sync(0xffffffff,v,o));
	return v;
}

// One warp per row.
__global__ void LayerNormForwardKernel(const float* x,const float* g,const float* b,float* y,float* mean,float* rstd,const int M,const int C,const float eps)
{
	const int row = blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
	const int lane = threadIdx.x%32;
	if(row>=M)
		return;
	const float* xi = x+(size_t)row*C;
	float s = 0.0f;
	for(int j=lane;j<C;j+=32)
		s += xi[j];
	const float m = WarpSum(s)/C;
	float v = 0.0f;
	for(int j=lane;j<C;j+=32)
	{
		const float d = xi[j]-m;
		v += d*d;
	}
	const float r = rsqrtf(WarpSum(v)/C+eps);
	float* yi = y+(size_t)row*C;
	for(int j=lane;j<C;j+=32)
		yi[j] = g[j]*((xi[j]-m)*r)+b[j];
	if(lane==0)
	{
		mean[row] = m;
		rstd[row] = r;
	}
}

__global__ void LayerNormBackwardKernel(const float* x,const float* g,const float* dy,const float* mean,const float* rstd,float* dx,const int M,const int C)
{
	const int row = blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
	const int lane = threadIdx.x%32;
	if(row>=M)
		return;
	const float* xi = x+(size_t)row*C;
	const float* gi = dy+(size_t)row*C;
	const float m = mean[row],r = rstd[row];
	float s1 = 0.0f,s2 = 0.0f;
	for(int j=lane;j<C;j+=32)
	{
		const float xhat = (xi[j]-m)*r;
		const float dxhat = gi[j]*g[j];
		s1 += dxhat;
		s2 += dxhat*xhat;
	}
	const float m1 = WarpSum(s1)/C;
	const float m2 = WarpSum(s2)/C;
	float* dxi = dx+(size_t)row*C;
	for(int j=lane;j<C;j+=32)
	{
		const float xhat = (xi[j]-m)*r;
		dxi[j] = r*(gi[j]*g[j]-m1-xhat*m2);
	}
}

// dgamma/dbeta: like ColumnSumKernel but sums dy*xhat and dy.
__global__ void LayerNormParamGradKernel(const float* x,const float* dy,const float* mean,const float* rstd,float* dg,float* db,const int M,const int C,const int rowsPerBlock)
{
	const int j = blockIdx.x*32+threadIdx.x;
	const int r0 = blockIdx.y*rowsPerBlock;
	const int r1 = min(M,r0+rowsPerBlock);
	if(j>=C)
		return;
	float sg = 0.0f,sb = 0.0f;
	for(int r=r0+threadIdx.y;r<r1;r+=blockDim.y)
	{
		const float d = dy[(size_t)r*C+j];
		sg += d*(x[(size_t)r*C+j]-mean[r])*rstd[r];
		sb += d;
	}
	atomicAdd(dg+j,sg);
	atomicAdd(db+j,sb);
}

__device__ __forceinline__ float GeluF(const float x)
{
	const float k = 0.7978845608028654f;
	return 0.5f*x*(1.0f+tanhf(k*(x+0.044715f*x*x*x)));
}

__device__ __forceinline__ float GeluGradF(const float x)
{
	const float k = 0.7978845608028654f;
	const float t = tanhf(k*(x+0.044715f*x*x*x));
	return 0.5f*(1.0f+t)+0.5f*x*(1.0f-t*t)*k*(1.0f+3.0f*0.044715f*x*x);
}

__global__ void ActivationForwardKernel(const int type,const float* x,float* y,const size_t n)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x;
	if(i>=n)
		return;
	const float v = x[i];
	y[i] = type==0?GeluF(v):(v>0.0f?(type==2?v*v:v):0.0f);
}

__global__ void ActivationBackwardKernel(const int type,const float* x,const float* dy,float* dx,const size_t n)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x;
	if(i>=n)
		return;
	const float v = x[i];
	dx[i] = type==0?dy[i]*GeluGradF(v):(v>0.0f?(type==2?2.0f*v*dy[i]:dy[i]):0.0f);
}

// ---- Attention helpers ----

// One thread per (bh,t,i<hs/2) pair, or per (bh,t,i<hs) element when not rotating.
__global__ void GatherHeadsKernel(const float* qkv,float* dst,const int B,const int T,const int C,const int H,const int which,const bool rope,const bool scatter,float* dqkv)
{
	const int hs = C/H;
	const int h2 = hs/2;
	const size_t n = (size_t)B*H*T*h2;
	const size_t idx = (size_t)blockIdx.x*blockDim.x+threadIdx.x;
	if(idx>=n)
		return;
	const int i = (int)(idx%h2);
	const int t = (int)((idx/h2)%T);
	const int bh = (int)(idx/((size_t)h2*T));
	const int b = bh/H;
	const int h = bh%H;
	const size_t src = ((size_t)b*T+t)*3*C+(size_t)which*C+h*hs;
	const size_t d = ((size_t)bh*T+t)*hs;
	float c = 1.0f,s = 0.0f;
	if(rope)
	{
		const float angle = t*powf(10000.0f,-2.0f*i/hs);
		sincosf(angle,&s,&c);
	}
	if(!scatter)
	{
		const float a = qkv[src+i],bb = qkv[src+i+h2];
		dst[d+i] = a*c-bb*s;
		dst[d+i+h2] = a*s+bb*c;
	}
	else
	{
		// Inverse rotation of the gradient.
		const float a = dst[d+i],bb = dst[d+i+h2];
		dqkv[src+i] = a*c+bb*s;
		dqkv[src+i+h2] = -a*s+bb*c;
	}
}

// One warp per row of a (rows,T) score matrix; row r is query position r%T; keys j>r%T are masked.
__global__ void CausalSoftmaxKernel(float* P,const int rows,const int T)
{
	const int row = blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
	const int lane = threadIdx.x%32;
	if(row>=rows)
		return;
	const int q = row%T;
	float* p = P+(size_t)row*T;
	float mx = -INFINITY;
	for(int j=lane;j<=q;j+=32)
		mx = fmaxf(mx,p[j]);
	mx = WarpMax(mx);
	float s = 0.0f;
	for(int j=lane;j<=q;j+=32)
	{
		const float e = __expf(p[j]-mx);
		p[j] = e;
		s += e;
	}
	const float inv = 1.0f/WarpSum(s);
	for(int j=lane;j<T;j+=32)
		p[j] = j<=q?p[j]*inv:0.0f;
}

__global__ void CausalSoftmaxBackwardKernel(const float* P,float* dP,const int rows,const int T,const float scale)
{
	const int row = blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
	const int lane = threadIdx.x%32;
	if(row>=rows)
		return;
	const int q = row%T;
	const float* p = P+(size_t)row*T;
	float* d = dP+(size_t)row*T;
	float dot = 0.0f;
	for(int j=lane;j<=q;j+=32)
		dot += p[j]*d[j];
	dot = WarpSum(dot);
	for(int j=lane;j<T;j+=32)
		d[j] = j<=q?p[j]*(d[j]-dot)*scale:0.0f;
}

// One warp per row.
__global__ void SoftmaxCrossEntropyForwardKernel(const float* logits,const float* targets,float* probs,float* loss,const int N,const int V)
{
	const int row = blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
	const int lane = threadIdx.x%32;
	if(row>=N)
		return;
	const float* l = logits+(size_t)row*V;
	float* p = probs+(size_t)row*V;
	float mx = -INFINITY;
	for(int j=lane;j<V;j+=32)
		mx = fmaxf(mx,l[j]);
	mx = WarpMax(mx);
	float s = 0.0f;
	for(int j=lane;j<V;j+=32)
	{
		const float e = __expf(l[j]-mx);
		p[j] = e;
		s += e;
	}
	s = WarpSum(s);
	const float inv = 1.0f/s;
	for(int j=lane;j<V;j+=32)
		p[j] *= inv;
	if(lane==0)
	{
		const int t = (int)targets[row];
		atomicAdd(loss,(-(l[t]-mx-logf(s)))/N);
	}
}

__global__ void SoftmaxCrossEntropyBackwardKernel(const float* probs,const float* targets,const float* dloss,float* d,const int N,const int V)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x;
	if(i>=(size_t)N*V)
		return;
	const int row = (int)(i/V);
	const int j = (int)(i%V);
	const float scale = dloss[0]/N;
	d[i] = (probs[i]-(j==(int)targets[row]?1.0f:0.0f))*scale;
}

__global__ void EmbeddingForwardKernel(const float* tok,const float* pos,const float* idx,float* y,const int B,const int T,const int C)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x;
	if(i>=(size_t)B*T*C)
		return;
	const int c = (int)(i%C);
	const size_t bt = i/C;
	const int t = (int)(bt%T);
	const int id = (int)idx[bt];
	y[i] = tok[(size_t)id*C+c]+(pos?pos[(size_t)t*C+c]:0.0f);
}

__global__ void EmbeddingBackwardTokKernel(const float* dy,const float* idx,float* dtok,const int B,const int T,const int C)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x;
	if(i>=(size_t)B*T*C)
		return;
	const int c = (int)(i%C);
	const size_t bt = i/C;
	atomicAdd(dtok+(size_t)((int)idx[bt])*C+c,dy[i]);
}

__global__ void EmbeddingBackwardPosKernel(const float* dy,float* dpos,const int B,const int T,const int C)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x;
	if(i>=(size_t)T*C)
		return;
	float s = 0.0f;
	for(int b=0;b<B;++b)
		s += dy[(size_t)b*T*C+i];
	dpos[i] = s;
}

__device__ __forceinline__ float SigmoidF(const float x)
{
	return 1.0f/(1.0f+__expf(-x));
}

__global__ void LstmStepForwardKernel(float* z,const int ldz,const float* cprev,const int ldc,float* c,float* h,const int ldh,const int B,const int H)
{
	const int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i>=B*H)
		return;
	const int b = i/H;
	const int j = i%H;
	float* zt = z+(size_t)b*ldz;
	const float ig = SigmoidF(zt[j]);
	const float fg = SigmoidF(zt[H+j]);
	const float gg = tanhf(zt[2*H+j]);
	const float og = SigmoidF(zt[3*H+j]);
	zt[j] = ig; zt[H+j] = fg; zt[2*H+j] = gg; zt[3*H+j] = og;
	const float cv = (cprev?fg*cprev[(size_t)b*ldc+j]:0.0f)+ig*gg;
	c[(size_t)b*ldc+j] = cv;
	h[(size_t)b*ldh+j] = og*tanhf(cv);
}

__global__ void LstmStepBackwardKernel(const float* a,const int lda,const float* c,const float* cprev,const int ldc,const float* g,const int ldg,
	float* dh,float* dc,float* dz,const int ldz,const int B,const int H)
{
	const int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i>=B*H)
		return;
	const int b = i/H;
	const int j = i%H;
	const float* at = a+(size_t)b*lda;
	const float ig = at[j],fg = at[H+j],gg = at[2*H+j],og = at[3*H+j];
	const float tc = tanhf(c[(size_t)b*ldc+j]);
	const float dhv = g[(size_t)b*ldg+j]+dh[i];
	const float dcv = dhv*og*(1.0f-tc*tc)+dc[i];
	float* dzt = dz+(size_t)b*ldz;
	dzt[j] = dcv*gg*ig*(1.0f-ig);
	dzt[H+j] = (cprev?dcv*cprev[(size_t)b*ldc+j]:0.0f)*fg*(1.0f-fg);
	dzt[2*H+j] = dcv*ig*(1.0f-gg*gg);
	dzt[3*H+j] = dhv*tc*og*(1.0f-og);
	dc[i] = dcv*fg;
}

__global__ void LstmPreviousHKernel(const float* a,const float* c,float* hprev,const int B,const int T,const int H)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x;
	if(i>=(size_t)B*T*H)
		return;
	const int j = (int)(i%H);
	const size_t bt = i/H;
	const int t = (int)(bt%T);
	if(t==0)
	{
		hprev[i] = 0.0f;
		return;
	}
	hprev[i] = a[(bt-1)*4*H+3*H+j]*tanhf(c[(bt-1)*H+j]);
}

__global__ void AdamKernel(float* w,float* g,float* m,float* v,const size_t n,const float lr,const float b1,const float b2,const float eps,const float c1,const float c2,const float decay)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x;
	if(i>=n)
		return;
	const float gi = g[i];
	const float mi = b1*m[i]+(1.0f-b1)*gi;
	const float vi = b2*v[i]+(1.0f-b2)*gi*gi;
	m[i] = mi;
	v[i] = vi;
	w[i] = w[i]*decay-lr*(mi*c1)/(sqrtf(vi*c2)+eps);
	g[i] = 0.0f;
}

__global__ void NesterovKernel(const float* g,float* m,float* u,const size_t n,const float mu)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x;
	if(i>=n)
		return;
	const float mi = mu*m[i]+g[i];
	m[i] = mi;
	u[i] = g[i]+mu*mi;
}

__global__ void TransposeKernel(const float* src,float* dst,const int rows,const int cols)
{
	__shared__ float tile[32][33];
	const int x = blockIdx.x*32+threadIdx.x;
	const int y = blockIdx.y*32+threadIdx.y;
	if(x<cols&&y<rows)
		tile[threadIdx.y][threadIdx.x] = src[(size_t)y*cols+x];
	__syncthreads();
	const int tx = blockIdx.y*32+threadIdx.x;
	const int ty = blockIdx.x*32+threadIdx.y;
	if(tx<rows&&ty<cols)
		dst[(size_t)ty*rows+tx] = tile[threadIdx.x][threadIdx.y];
}


namespace dev
{
	void FillRows(float* y,const float* row,int M,int N)
	{
		const size_t n = (size_t)M*N;
		FillRowsKernel<<<Blocks(n,256),256>>>(y,row,M,N);
	}

	void ColumnSum(const float* a,int M,int N,float* out)
	{
		Zero(out,N);
		const int rowsPerBlock = 256;
		dim3 grid((N+31)/32,(M+rowsPerBlock-1)/rowsPerBlock);
		ColumnSumKernel<<<grid,dim3(32,8)>>>(a,M,N,out,rowsPerBlock);
	}

	void Add(const float* a,const float* b,float* y,size_t n)
	{
		AddKernel<<<Blocks(n,256),256>>>(a,b,y,n);
	}

	void AddInPlace(float* y,const float* a,size_t n)
	{
		AddInPlaceKernel<<<Blocks(n,256),256>>>(y,a,n);
	}

	void Scale(float* y,float s,size_t n)
	{
		ScaleKernel<<<Blocks(n,256),256>>>(y,s,n);
	}

	void Axpy(float* y,const float* x,float a,size_t n)
	{
		AxpyKernel<<<Blocks(n,256),256>>>(y,x,a,n);
	}

	double SumSquares(const float* a,size_t n)
	{
		static double* d = nullptr;
		if(!d)
			CK(cudaMalloc(&d,sizeof(double)));
		CK(cudaMemsetAsync(d,0,sizeof(double)));
		SumSquaresKernel<<<(int)std::min<size_t>(1024,Blocks(n,256)),256>>>(a,n,d);
		double h = 0.0;
		CK(cudaMemcpy(&h,d,sizeof(double),cudaMemcpyDeviceToHost));
		return h;
	}

	void LayerNormForward(const float* x,const float* g,const float* b,float* y,float* mean,float* rstd,int M,int C,float eps)
	{
		LayerNormForwardKernel<<<Blocks(M,8),256>>>(x,g,b,y,mean,rstd,M,C,eps);
	}

	void LayerNormBackward(const float* x,const float* g,const float* dy,const float* mean,const float* rstd,float* dx,float* dg,float* db,int M,int C)
	{
		LayerNormBackwardKernel<<<Blocks(M,8),256>>>(x,g,dy,mean,rstd,dx,M,C);
		Zero(dg,C);
		Zero(db,C);
		const int rowsPerBlock = 256;
		dim3 grid((C+31)/32,(M+rowsPerBlock-1)/rowsPerBlock);
		LayerNormParamGradKernel<<<grid,dim3(32,8)>>>(x,dy,mean,rstd,dg,db,M,C,rowsPerBlock);
	}

	void ActivationForward(int type,const float* x,float* y,size_t n)
	{
		ActivationForwardKernel<<<Blocks(n,256),256>>>(type,x,y,n);
	}

	void ActivationBackward(int type,const float* x,const float* dy,float* dx,size_t n)
	{
		ActivationBackwardKernel<<<Blocks(n,256),256>>>(type,x,dy,dx,n);
	}

	void GatherHeads(const float* qkv,float* dst,int B,int T,int C,int H,int which,bool rope)
	{
		const size_t n = (size_t)B*T*C/2;
		GatherHeadsKernel<<<Blocks(n,256),256>>>(qkv,dst,B,T,C,H,which,rope,false,nullptr);
	}

	void ScatterHeads(const float* src,float* dqkv,int B,int T,int C,int H,int which,bool rope)
	{
		const size_t n = (size_t)B*T*C/2;
		GatherHeadsKernel<<<Blocks(n,256),256>>>(nullptr,const_cast<float*>(src),B,T,C,H,which,rope,true,dqkv);
	}

	void CausalSoftmax(float* P,int rows,int T)
	{
		CausalSoftmaxKernel<<<Blocks(rows,8),256>>>(P,rows,T);
	}

	void CausalSoftmaxBackward(const float* P,float* dP,int rows,int T,float scale)
	{
		CausalSoftmaxBackwardKernel<<<Blocks(rows,8),256>>>(P,dP,rows,T,scale);
	}

	void SoftmaxCrossEntropyForward(const float* logits,const float* targets,float* probs,float* loss,int N,int V)
	{
		Zero(loss,1);
		SoftmaxCrossEntropyForwardKernel<<<Blocks(N,8),256>>>(logits,targets,probs,loss,N,V);
	}

	void SoftmaxCrossEntropyBackward(const float* probs,const float* targets,const float* dloss,float* dlogits,int N,int V)
	{
		const size_t n = (size_t)N*V;
		SoftmaxCrossEntropyBackwardKernel<<<Blocks(n,256),256>>>(probs,targets,dloss,dlogits,N,V);
	}

	void EmbeddingForward(const float* tok,const float* pos,const float* idx,float* y,int B,int T,int C)
	{
		const size_t n = (size_t)B*T*C;
		EmbeddingForwardKernel<<<Blocks(n,256),256>>>(tok,pos,idx,y,B,T,C);
	}

	void EmbeddingBackward(const float* dy,const float* idx,float* dtok,float* dpos,int B,int T,int C,int V)
	{
		Zero(dtok,(size_t)V*C);
		const size_t n = (size_t)B*T*C;
		EmbeddingBackwardTokKernel<<<Blocks(n,256),256>>>(dy,idx,dtok,B,T,C);
		if(dpos)
			EmbeddingBackwardPosKernel<<<Blocks((size_t)T*C,256),256>>>(dy,dpos,B,T,C);
	}

	void LstmStepForward(float* z,int ldz,const float* cprev,int ldc,float* c,float* h,int ldh,int B,int H)
	{
		LstmStepForwardKernel<<<Blocks((size_t)B*H,256),256>>>(z,ldz,cprev,ldc,c,h,ldh,B,H);
	}

	void LstmStepBackward(const float* a,int lda,const float* c,const float* cprev,int ldc,const float* g,int ldg,float* dh,float* dc,float* dz,int ldz,int B,int H)
	{
		LstmStepBackwardKernel<<<Blocks((size_t)B*H,256),256>>>(a,lda,c,cprev,ldc,g,ldg,dh,dc,dz,ldz,B,H);
	}

	void LstmPreviousH(const float* a,const float* c,float* hprev,int B,int T,int H)
	{
		const size_t n = (size_t)B*T*H;
		LstmPreviousHKernel<<<Blocks(n,256),256>>>(a,c,hprev,B,T,H);
	}

	void Adam(float* w,float* g,float* m,float* v,size_t n,float lr,float b1,float b2,float eps,float c1,float c2,float decay)
	{
		AdamKernel<<<Blocks(n,256),256>>>(w,g,m,v,n,lr,b1,b2,eps,c1,c2,decay);
	}

	void NesterovMomentum(const float* g,float* m,float* u,size_t n,float mu)
	{
		NesterovKernel<<<Blocks(n,256),256>>>(g,m,u,n,mu);
	}

	void Transpose(const float* src,float* dst,int rows,int cols)
	{
		dim3 grid((cols+31)/32,(rows+31)/32);
		TransposeKernel<<<grid,dim3(32,32)>>>(src,dst,rows,cols);
	}
}


__host__ __device__ __forceinline__ float DropoutUniform(const unsigned seed,const unsigned long long i)
{
	unsigned h = (unsigned)i*0x9E3779B1u^seed^(unsigned)(i>>32)*0x7FEB352Du;
	h ^= h>>16; h *= 0x85EBCA6Bu; h ^= h>>13; h *= 0xC2B2AE35u; h ^= h>>16;
	return h*(1.0f/4294967296.0f);
}

__global__ void DropoutKernel(const float* x,float* y,const size_t n,const float p,const unsigned seed)
{
	const size_t i = (size_t)blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n)
		y[i] = DropoutUniform(seed,i)>=p?x[i]/(1.0f-p):0.0f;
}

namespace dev
{
	void Dropout(const float* x,float* y,size_t n,float p,unsigned seed)
	{
		DropoutKernel<<<Blocks(n,256),256>>>(x,y,n,p,seed);
	}
}
