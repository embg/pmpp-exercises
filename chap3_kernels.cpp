#include "chap3_kernels.h"
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

void pyEx1B(at::Tensor C, at::Tensor A, at::Tensor B) {
  launchEx1B(
      C.data_ptr<float>(),
      A.data_ptr<float>(),
      B.data_ptr<float>(),
      A.size(0)
  );
}


TORCH_LIBRARY(chap3_kernels, m) {
  m.def("pyEx1A", pyEx1A);
  m.def("pyEx1B", pyEx1B);
}
