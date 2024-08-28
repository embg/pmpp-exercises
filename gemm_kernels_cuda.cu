#include <cuda.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>

#include "gemm_kernels.h"

#define cdiv(a, b) (((a) + ((b) - 1)) / (b))

///////////// HELPERS /////////////

// Does not own memory!
class Vector {
  public:
    __device__ __forceinline__
    Vector(const float* basePtr, size_t size, size_t stride) 
        : basePtr_(basePtr), size_(size), stride_(stride) {}
    
    __device__ __forceinline__
    float operator[](size_t idx) const {
        assert(idx < size());
        return basePtr_[idx * stride_];
    }
    
    __device__ __forceinline__
    size_t size() const {
        return size_;
    }
    
  private:
    const float* basePtr_;
    size_t size_;
    size_t stride_;
};

__device__ __forceinline__
float dotProd(Vector A, Vector B) {
    assert(A.size() == B.size());
    float result = 0;
    for (size_t i = 0; i < A.size(); i++) {
        result += A[i] * B[i];
    }
    return result;
}

///////////// KERNELS /////////////

__global__ void ex1A(
    float* C, const float* A, const float* B, size_t size)
{
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < size) {
        // All dot products will use the same row from A
        Vector aVec(&A[row * size], size /* size */, 1 /* stride */);

        for (size_t col = 0; col < size; col++) {
            // Each dot product needs a different col from B
            Vector bVec(&B[col], size /* size */, size /* stride */);
            C[row * size + col] = dotProd(aVec, bVec);
        }
    }
}

void launchEx1A(float* C, const float* A, const float* B, size_t size)
{
    dim3 blockShape(1, BLOCK_SIZE);
    dim3 gridShape(1, cdiv(size, BLOCK_SIZE));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ex1A<<<gridShape, blockShape, 0, stream>>>(C, A, B, size);
}

__global__ void ex1B(
    float* C, const float* A, const float* B, size_t size)
{
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < size) {
        // All dot products will use the same col from B
        Vector bVec(&B[col], size /* size */, size /* stride */);

        for (size_t row = 0; row < size; row++) {
            // Each dot product needs a different row from A
            Vector aVec(&A[row * size], size /* size */, 1 /* stride */);
            C[row * size + col] = dotProd(aVec, bVec);
        }
    }
}

void launchEx1B(float* C, const float* A, const float* B, size_t size)
{
    dim3 blockShape(BLOCK_SIZE);
    dim3 gridShape(cdiv(size, BLOCK_SIZE));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ex1B<<<gridShape, blockShape, 0, stream>>>(C, A, B, size);
}

__global__ void ex2(
    float* c, const float* A, const float* b, size_t size)
{
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < size) {
        Vector aVec(&A[row * size], size /* size */, 1 /* stride */);
        Vector bVec(&b[0], size /* size */, 1 /* stride */);
        c[row] = dotProd(aVec, bVec);
    }
}

void launchEx2(float* c, const float* A, const float* b, size_t size)
{
    dim3 blockShape(1, BLOCK_SIZE);
    dim3 gridShape(1, cdiv(size, BLOCK_SIZE));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ex2<<<gridShape, blockShape, 0, stream>>>(c, A, b, size);
}

__global__ void ex1B(
    float* C, const float* A, const float* B, size_t size)
{
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < size) {
        // All dot products will use the same col from B
        Vector bVec(&B[col], size /* size */, size /* stride */);

        for (size_t row = 0; row < size; row++) {
            // Each dot product needs a different row from A
            Vector aVec(&A[row * size], size /* size */, 1 /* stride */);
            C[row * size + col] = dotProd(aVec, bVec);
        }
    }
}

// Note: for this kernel, B is col-major
void launchChap6(float* C, const float* A, const float* B, size_t size)
{
    dim3 blockShape(BLOCK_SIZE_CHAP6);
    dim3 gridShape(cdiv(size, BLOCK_SIZE_CHAP6), cdiv(size, BLOCK_SIZE_CHAP6));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    chap6<<<gridShape, blockShape, 48 * 1024, stream>>>(C, A, B, size);
}

// Note: for this kernel, B is col-major
// Note: I DID NOT RUN THIS CODE
void chap6_NotDebugged(
    float* C, const float* A, const float* B, size_t size)
{
    // B is col-major, but tileB is row-major
    __shared__ float tileA[BLOCK_SIZE_CHAP6 * BLOCK_SIZE_CHAP6];
    __shared__ float tileB[BLOCK_SIZE_CHAP6 * BLOCK_SIZE_CHAP6];
    
    // We hope this ends up in registers
    float acc[BLOCK_SIZE_CHAP6];
    for (int row = 0; row < BLOCK_SIZE_CHAP6; row++) {
        acc[row] = 0.0f;
    }

    // Top-left row and col of the output tile
    const size_t outRowStart = blockIdx.y * BLOCK_SIZE_CHAP6;
    const size_t outRowEnd = std::min(outRowStart + BLOCK_SIZE_CHAP6, size);
    const size_t outColStart = blockIdx.x * BLOCK_SIZE_CHAP6;
    const size_t outColEnd = std::min(outColStart + BLOCK_SIZE_CHAP6, size);
    
    for (size_t k = threadIdx.x; k < size; k += BLOCK_SIZE_CHAP6) {
        for (int row = outRowStart; row < outRowEnd; row++) {
            tileA[row * BLOCK_SIZE_CHAP6 + k] = A[row * size + k];
        }

        for (int col = outColStart; col < outColEnd; col++) {
            tileB[k * BLOCK_SIZE_CHAP6 + col] = B[col * size + k];
        }
        
        __syncthreads();
        
        float tileColB[BLOCK_SIZE_CHAP6];
        for (int row = 0; row < BLOCK_SIZE_CHAP6; row++) {
            tileColB[row] = tileB[row * BLOCK_SIZE_CHAP6 + threadIdx.x];
        }
        for (int row = 0; row < BLOCK_SIZE_CHAP6; row++) {
            for (int col = 0; col < BLOCK_SIZE_CHAP6; col++) {
                acc[row] += tileColB[row] * tileA[row * BLOCK_SIZE_CHAP6 + col];
            }
        }
        
        __syncthreads();
    }
    
    for (int i = 0; i < BLOCK_SIZE_CHAP6; i++) {
        const size_t outRow = outRowStart + i;
        const size_t outCol = outColStart + threadIdx.x;
        if (outRow < outRowEnd && outCol < outColEnd) {
            C[outRow * size + outCol] = acc[i];
        }
    }
}
