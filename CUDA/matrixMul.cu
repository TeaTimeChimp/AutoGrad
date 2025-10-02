// System includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_cuda.h"


// mm.cu
#include <cuda_runtime.h>


// Basic C++/CUDA explanation
//  https://www.nvidia.com/docs/io/116711/sc11-cuda-c-basics.pdf


// CUDA Kernel, matrix multiply.
//
//      a(M,K) @ b(K,N) = c(M,N)
//
template<int TILE>
__global__
void DeviceMatMul(const float* __restrict__ A,const float* __restrict__ B,float* __restrict__ C,int M,int K,int N)
{
    // Row of C = blockIdx.y*TILE + threadIdx.y
    // Col of C = blockIdx.x*TILE + threadIdx.x
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    // +1 to avoid shared memory bank conflicts on square tiles
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    float acc = 0.0f;

    // Loop over tiles of K dimension
    for(int tk=0;tk<K;tk+=TILE)
    {
        // Predicated loads (zero-pad out-of-range)
        const int a_col = tk + threadIdx.x;
        const int b_row = tk + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row<M && a_col<K) ? A[row*K+a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row<K && col<N) ? B[b_row*N+col] : 0.0f;

        __syncthreads();

        // Compute this tile
        #pragma unroll
        for(int k=0;k<TILE;++k)
        {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Guarded store
    if(row<M&&col<N)
    {
        C[row*N+col] = acc;
    }
}


// Matrix multiplication - Host code.
//
//      a(M,K) @ b(K,N) = c(M,N)
//
void CudaMatMul(const float* const h_A,const float* const h_B,float* const h_C,const unsigned int M,const unsigned int K,const unsigned int N)
{
    constexpr unsigned int block_size = 32;

    // Allocate host memory for matrices A, B and C.
    float* d_A;
    float* d_B;
    float* d_C;
    unsigned int mem_size_A = M*K*sizeof(float);
    unsigned int mem_size_B = K*N*sizeof(float);
    unsigned int mem_size_C = M*N*sizeof(float);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_A),mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_B),mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_C),mem_size_C));

	// Create a stream - operations in the stream will synchronize with each other and not block concurrent streams.
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Copy host memory to device
    checkCudaErrors(cudaMemcpyAsync(d_A,h_A,mem_size_A,cudaMemcpyHostToDevice,stream));
    checkCudaErrors(cudaMemcpyAsync(d_B,h_B,mem_size_B,cudaMemcpyHostToDevice,stream));

    // Setup execution parameters
    dim3 threads(block_size,block_size);
    dim3 grid(((N-1)/block_size)+1,((M-1)/block_size)+1);

    // Call matrix multiply kernal.
    DeviceMatMul<block_size><<<grid,threads,0,stream>>>(d_A,d_B,d_C,M,K,N);
    checkCudaErrors(cudaStreamSynchronize(stream));

    // Copy result from device to host
    checkCudaErrors(cudaMemcpyAsync(h_C,d_C,mem_size_C,cudaMemcpyDeviceToHost,stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    // Free sevice memory
    checkCudaErrors(cudaStreamDestroy(stream));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
}
