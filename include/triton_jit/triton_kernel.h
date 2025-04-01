#pragma once

#include <stdexcept>
#include <string>
#include "cuda.h"

namespace triton_jit {
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// Error handling function using exceptions instead of exit()
inline void __checkCudaErrors(CUresult code, const char *file, const int line) {
  if (code != CUDA_SUCCESS) {
    const char *error_string;
    cuGetErrorString(code, &error_string);
    fprintf(stderr,
            "CUDA Driver API error = %04d from file <%s>, line %i. Detail: <%s>\n",
            code,
            file,
            line,
            error_string);
    throw std::runtime_error(error_string);
  }
}

class TritonKernel {
 private:
  std::string dir_;
  std::string kernel_name_;

  mutable bool loaded_ = false;
  mutable unsigned int share_ = 0;
  mutable CUmodule module_ = nullptr;
  mutable CUfunction function_ = nullptr;

  void lazy_init_handle() const;

 public:
  TritonKernel(std::string_view dir, std::string_view kernel_name) : dir_(dir), kernel_name_(kernel_name) {
  }

  // consider using a variadic template
  void launch(unsigned int grid_x,
              unsigned int grid_y,
              unsigned int grid_z,
              int num_warps,
              CUstream stream,
              void **args) const;
};
}  // namespace triton_jit
