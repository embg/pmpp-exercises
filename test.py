import torch
import numpy as np
torch.ops.load_library("chap3_kernels.so")

def run_ex1A(C, A, B):
    return torch.ops.chap3_kernels.pyEx1A(C, A, B)
    
def run_ex1B(C, A, B):
    return torch.ops.chap3_kernels.pyEx1B(C, A, B)
    
def test_mm(func, size):
    A = torch.randn((size, size), device="cuda")
    B = torch.randn((size, size), device="cuda")
    C = torch.randn((size, size), device="cuda")
    func(C, A, B)
    assert torch.allclose(C, torch.matmul(A, B), atol=1e-4, rtol=1e-4)

if __name__ == "__main__":
    # Only square matrices are supported
    sizes = [
        1, 2, 3, 254, 255, 256, 1023, 1024, 1025
    ]
    for size in sizes:
        test_mm(run_ex1A, size)
        test_mm(run_ex1B, size)
        print(f"Passed all tests for size = {size}")    
