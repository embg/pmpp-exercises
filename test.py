import torch
import numpy as np
torch.ops.load_library("gemm_kernels.so")
torch.manual_seed(0)

ITERS = 5

def run_ex1A(C, A, B):
    return torch.ops.gemm_kernels.pyEx1A(C, A, B)
    
def run_ex1B(C, A, B):
    return torch.ops.gemm_kernels.pyEx1B(C, A, B)

def run_ex2(c, A, b):
    return torch.ops.gemm_kernels.pyEx2(c, A, b)
    
def run_chap6(C, A, B):
    return torch.ops.gemm_kernels.pyChap6(C, A, B.T.contiguous())
    
def test_mm(func, shapeA, shapeB):
    for _ in range(ITERS):
        A = torch.randn(shapeA, device="cuda")
        B = torch.randn(shapeB, device="cuda")
        C = torch.randn((shapeA[0], shapeB[1]), device="cuda")
        func(C, A, B)
        delta = (C - torch.matmul(A, B)).cpu().detach().numpy()
        print(delta)
        assert torch.allclose(C, torch.matmul(A, B), atol=1e-2, rtol=1e-2)

if __name__ == "__main__":
    # Only square matrices are supported
    sizes = [
        1, 2, 3, 254, 255, 256, 1023, 1024, 1025
    ]
    for size in sizes:
        test_mm(run_ex1A, (size, size), (size, size))
        test_mm(run_ex1B, (size, size), (size, size))
        test_mm(run_ex2, (size, size), (size, 1))
        test_mm(run_chap6, (size, size), (size, size))
        print(f"Passed all tests for size = {size}")    
