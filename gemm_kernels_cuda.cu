#include <cuda.h>
#include <ATen/cuda/CUDAContext.h>

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
