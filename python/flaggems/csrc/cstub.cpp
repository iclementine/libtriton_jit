#include <pybind11/pybind11.h>
#include "operators/operators.h"

PYBIND11_MODULE(c_operators, m) {
}

namespace flaggems {
TORCH_LIBRARY(flaggems, m) {
  m.def("sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
  m.def("add_tensor(Tensor self, Tensor other) -> Tensor");
}

TORCH_LIBRARY_IMPL(flaggems, CUDA, m) {
  m.impl("sum.dim_IntList", TORCH_FN(sum_dim));
  m.impl("add_tensor", TORCH_FN(add_tensor));
}
}  // namespace flaggems
