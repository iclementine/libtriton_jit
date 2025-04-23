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

void TritonKernel::lazy_init_handle(CUdevice device_index) const {
  if (modules_.count(device_index)) {
    LOG(INFO) << fmt::format("cubin is loaded on device {}", device_index);
    return;
  }

  // check cuda arch
  int major = 0, minor = 0;
  checkCudaErrors(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_index));
  checkCudaErrors(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_index));
  unsigned int arch = major * 10 + minor;
  if (arch != this->arch_) {
    throw std::runtime_error("compute architecture mismatch!");
  }

  CUmodule module;
  std::string cubin_path = fmt::format("{}/{}.cubin", this->dir_, this->kernel_name_);
  LOG(INFO) << fmt::format("Loading cubin {} into {}", cubin_path, device_index);
  checkCudaErrors(cuModuleLoad(&module, cubin_path.c_str()));
  this->modules_.emplace(device_index, module);
}

// consider using a variadic template
void TritonKernel::launch(unsigned int grid_x,
                          unsigned int grid_y,
                          unsigned int grid_z,
                          int num_warps,
                          CUstream stream,
                          void** args) const {
  // get the context associated with the stream
  CUcontext ctx;
  checkCudaErrors(cuStreamGetCtx(stream, &ctx));
  checkCudaErrors(cuCtxSetCurrent(ctx));
  CUdevice d;
  checkCudaErrors(
      cuCtxGetDevice(&d));  // device management is done with torch, assume one CUcontext per device

  this->lazy_init_handle(d);
  // TODO: some kernels need to be launched via cuLaunchKernelEx, add that later?
  CUfunction f;
  checkCudaErrors(
      cuModuleGetFunction(&f,
                          this->modules_.at(d),
                          this->kernel_name_.c_str()));  // maybe check CUfunction instead of CUmodule?
  checkCudaErrors(cuLaunchKernel(f,
                                 /*grid*/
                                 grid_x,
                                 grid_y,
                                 grid_z,
                                 /*block*/
                                 32 * num_warps,
                                 1,
                                 1,
                                 /*shared & stream*/
                                 this->shared_,
                                 stream,
                                 /*args*/
                                 args,
                                 nullptr));
}
}  // namespace triton_jit
