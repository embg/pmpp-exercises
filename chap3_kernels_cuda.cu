#include <cuda.h>
#include <ATen/cuda/CUDAContext.h>

#include "chap3_kernels.h"

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
    float* C, const float* A, const float* B, size_t Width)
{
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < Width) {
        // All dot products will use the same row from A
        Vector aVec(&A[row * Width], Width /* size */, 1 /* stride */);

        for (size_t col = 0; col < Width; col++) {
            // Each dot product needs a different row from B
            Vector bVec(&B[col], Width /* size */, Width /* stride */);
            C[row * Width + col] = dotProd(aVec, bVec);
        }
    }
}

void launchEx1A(float* C, const float* A, const float* B, size_t Width)
{
    dim3 blockShape(1, BLOCK_SIZE);
    dim3 gridShape(1, cdiv(Width, BLOCK_SIZE));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ex1A<<<gridShape, blockShape>>>(C, A, B, Width);
}

__global__ void ex1B(
    float* C, const float* A, const float* B, size_t Width)
{
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < Width) {
        // All dot products will use the same col from B
        Vector bVec(&B[col], Width /* size */, Width /* stride */);

        for (size_t row = 0; row < Width; row++) {
            // Each dot product needs a different row from B
            Vector aVec(&A[row * Width], Width /* size */, 1 /* stride */);
            C[row * Width + col] = dotProd(aVec, bVec);
        }
    }
}

void launchEx1B(float* C, const float* A, const float* B, size_t Width)
{
    dim3 blockShape(BLOCK_SIZE);
    dim3 gridShape(cdiv(Width, BLOCK_SIZE));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ex1B<<<gridShape, blockShape>>>(C, A, B, Width);
}


