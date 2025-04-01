#include "triton_jit/triton_kernel.h"

#include <fstream>
#include <string>

#include "fmt/core.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace triton_jit {
void TritonKernel::lazy_init_handle() const {
  if (this->loaded_) return;

  // Maybe geting shared with driver API is better
  std::string metadata_path = fmt::format("{}/{}.json", this->dir_, this->kernel_name_);
  std::ifstream f(metadata_path.c_str());
  json meta_data = json::parse(f);
  this->share_ = meta_data["shared"];

  std::string cubin_path = fmt::format("{}/{}.cubin", this->dir_, this->kernel_name_);
  // is it possible to have more than one function in a cubin file for triton
  // compiler? no
  checkCudaErrors(cuModuleLoad(&(this->module_), cubin_path.c_str()));
  checkCudaErrors(cuModuleGetFunction(&this->function_, this->module_, this->kernel_name_.c_str()));
  this->loaded_ = true;
}

// consider using a variadic template
void TritonKernel::launch(unsigned int grid_x,
                          unsigned int grid_y,
                          unsigned int grid_z,
                          int num_warps,
                          CUstream stream,
                          void **args) const {
  this->lazy_init_handle();
  // TODO: some kernels need to be launched via cuLaunchKernelEx, add that later?
  checkCudaErrors(cuLaunchKernel(this->function_,
                                 grid_x,
                                 grid_y,
                                 grid_z,
                                 32 * num_warps,
                                 1,
                                 1,
                                 this->share_,
                                 stream,
                                 args,
                                 nullptr));
}
}  // namespace triton_jit
