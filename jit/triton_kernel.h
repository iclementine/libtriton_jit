#pragma once

#include <string>
#include "cuda.h"

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// Is this good enough for error handling, exit directly?
inline void __checkCudaErrors(CUresult code, const char *file, const int line) {
  if (code != CUDA_SUCCESS) {
    const char *error_string;
    cuGetErrorString(code, &error_string);
    fprintf(stderr,
            "CUDA Driver API error = %04d from file <%s>, line %i. detail: <%s>\n",
            code,
            file,
            line,
            error_string);
    exit(-1);
  }
}

class TritonKernel {
 private:
  std::string dir_;
  std::string kernel_name_;

  mutable bool loaded_ = false;
  mutable unsigned int share_;
  mutable CUmodule module_;
  mutable CUfunction function_;

  void lazy_init_handle() const;

 public:
  TritonKernel(std::string_view dir, std::string_view kernel_name)
      : dir_(dir),
        kernel_name_(kernel_name),
        loaded_(false),
        share_(0),
        module_(nullptr),
        function_(nullptr) {
  }

  // consider using a variadic template
  void launch(unsigned int grid_x,
              unsigned int grid_y,
              unsigned int grid_z,
              int num_warps,
              CUstream stream,
              void **args) const;
};
