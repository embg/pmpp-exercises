#include "gemm_kernels.h"
#include "ATen/ATen.h"
#include "torch/extension.h"

// Only supports square matrices
void pyEx1A(at::Tensor C, at::Tensor A, at::Tensor B) {
  launchEx1A(
      C.data_ptr<float>(),
      A.data_ptr<float>(),
      B.data_ptr<float>(),
      A.size(0)
  );
}

// Only supports square matrices
void pyEx1B(at::Tensor C, at::Tensor A, at::Tensor B) {
  launchEx1B(
      C.data_ptr<float>(),
      A.data_ptr<float>(),
      B.data_ptr<float>(),
      A.size(0)
  );
}

// A is a square matrix. c and b are be vectors with A.size(0) elements
void pyEx2(at::Tensor c, at::Tensor A, at::Tensor b) {
  launchEx2(
      c.data_ptr<float>(),
      A.data_ptr<float>(),
      b.data_ptr<float>(),
      A.size(0)
  );
}


TORCH_LIBRARY(gemm_kernels, m) {
  m.def("pyEx1A", pyEx1A);
  m.def("pyEx1B", pyEx1B);
  m.def("pyEx2", pyEx2);
}
