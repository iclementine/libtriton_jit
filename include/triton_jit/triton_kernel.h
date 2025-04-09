#pragma once

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
  std::string
      dir_; /* The directory that contain the IRs(ttir, ttgir, llir, ptx, cubin) & metadata(json file))*/
  std::string kernel_name_; /* name of the kernel in cubin */
  unsigned int shared_;     /* amount of static shared memory per block (in bytes) required for the cubin*/
  unsigned int arch_;       /* cuda arch */
  mutable std::unordered_map<int /*device id*/, CUmodule /*module*/>
      modules_; /*loaded modules, possibly one per device if the arch matches*/

 public:
  TritonKernel(std::string_view dir, std::string_view kernel_name);

  void launch(unsigned int grid_x,
              unsigned int grid_y,
              unsigned int grid_z,
              int num_warps,
              CUstream stream,
              void **args) const;

 private:
  /* load cubin into a cumodule for a device */
  void lazy_init_handle(CUdevice device_index) const;
};
}  // namespace triton_jit
