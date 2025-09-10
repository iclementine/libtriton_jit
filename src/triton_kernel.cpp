#include "triton_jit/triton_kernel.h"

#include <fstream>
#include <iostream>
#include <string>

#include "c10/util/Logging.h"  // use torch's logging
#include "fmt/core.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace triton_jit {
TritonKernel::TritonKernel(std::string_view dir, std::string_view kernel_name)
    : dir_(std::string(dir)), kernel_name_(std::string(kernel_name)) {
  std::string metadata_path = fmt::format("{}/{}.json", this->dir_, this->kernel_name_);
  std::ifstream f(metadata_path.c_str());
  json meta_data = json::parse(f);
  // shared and arch are bound to a kernel dir
  this->shared_ = meta_data["shared"];

  this->arch_ = meta_data["target"]["arch"];
  LOG(INFO) << fmt::format("TritonKernel Metadata loaded arch: {} shared: {}", this->arch_, this->shared_);
}

void TritonKernel::lazy_init_handle() const {
  if (this->loaded_) {
    return;
  }

  // check cuda arch
  CUdevice device_index;
  checkCudaErrors(cuCtxGetDevice(&device_index));
  int major = 0, minor = 0;
  checkCudaErrors(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_index));
  checkCudaErrors(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_index));
  unsigned int arch = major * 10 + minor;
  if (arch != this->arch_) {
    throw std::runtime_error("compute architecture mismatch!");
  }

  // load module
  std::string cubin_path = fmt::format("{}/{}.cubin", this->dir_, this->kernel_name_);
  LOG(INFO) << fmt::format("Loading cubin {} into device {}", cubin_path, device_index);
  checkCudaErrors(cuModuleLoad(&this->mod_, cubin_path.c_str()));

  // get function
  checkCudaErrors(cuModuleGetFunction(&this->fn_, this->mod_, this->kernel_name_.c_str()));

  // check required shared memory does not exceeds max shared memory per block
  int shared_optin;
  cuDeviceGetAttribute(&shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device_index);
  if (this->shared_ > shared_optin) {
    throw std::runtime_error(
        fmt::format("Out0fResources: Requested shared memory ({}) bytes exceeds GPU's maximum ({}) bytes.",
                    this->shared_,
                    shared_optin));
  }

  // increase shared memory if required
  if (this->shared_ > 49152 && shared_optin > 49152) {
    LOG(INFO) << fmt::format(
        "Condition met: this->shared_ ={} && shared_optin = {}. Setting CU_FUNC_CACHE_PREFER_SHARED.",
        this->shared_,
        shared_optin);
    checkCudaErrors(cuFuncSetCacheConfig(this->fn_, CU_FUNC_CACHE_PREFER_SHARED));
    int shared_total, shared_static;
    checkCudaErrors(cuDeviceGetAttribute(&shared_total,
                                         CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                                         device_index));
    checkCudaErrors(cuFuncGetAttribute(&shared_static, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, this->fn_));
    LOG(INFO) << fmt::format("current shared memory total {}", shared_total);
    LOG(INFO) << fmt::format("current shared memory static {}", shared_static);
    checkCudaErrors(cuFuncSetAttribute(this->fn_,
                                       CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                       shared_optin - shared_static));
    LOG(INFO) << fmt::format("shared memory to add {}", shared_optin - shared_static);
  }
  this->loaded_ = true;
}

// consider using a variadic template
void TritonKernel::launch(unsigned int grid_x,
                          unsigned int grid_y,
                          unsigned int grid_z,
                          int num_warps,
                          CUstream stream,
                          void** args) const {
  // get the context associated with the stream
  this->lazy_init_handle();

  LOG(INFO) << "cuLaunchKernel";
  checkCudaErrors(cuLaunchKernel(this->fn_,
                                 /*grid*/ grid_x,
                                 grid_y,
                                 grid_z,
                                 /*block*/ 32 * num_warps,
                                 1,
                                 1,
                                 /*shared & stream*/ this->shared_,
                                 /*stream*/ stream,
                                 /*args*/ args,
                                 nullptr));
}
}  // namespace triton_jit
