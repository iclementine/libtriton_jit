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
  LOG(INFO) << fmt::format("Loading cubin {} into device {}", cubin_path, device_index);
  checkCudaErrors(cuModuleLoad(&module, cubin_path.c_str()));
  this->modules_.emplace(device_index, module);
  int shared_optin;
  CUdevice d;
  checkCudaErrors(cuCtxGetDevice(&d)); 
  cuDeviceGetAttribute(&shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,d);
  if(this->shared_> shared_optin){
    throw std::runtime_error(fmt::format("Out0fResources: Requested shared memory ({}) bytes exceeds GPU's maximum ({}) bytes.",
                                      this->shared_, shared_optin));
  }
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
  int shared_optin;
  cuDeviceGetAttribute(&shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,d);
  int shared_memory = this->shared_;
  if (this->shared_ > 49152 && shared_optin > 49152) {
    LOG(INFO) << fmt::format("Condition met: this->shared_ ={} && shared_optin = {}. Setting CU_FUNC_CACHE_PREFER_SHARED.",this->shared_,shared_optin);
    checkCudaErrors(cuFuncSetCacheConfig(f, CU_FUNC_CACHE_PREFER_SHARED));
    int shared_total, shared_static;
    checkCudaErrors(cuDeviceGetAttribute(&shared_total, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,d));
    checkCudaErrors(cuFuncGetAttribute(&shared_static, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f));
    LOG(INFO) << fmt::format("current shared memory total {}",shared_total);
    LOG(INFO) << fmt::format("current shared memory static {}",shared_static);
    checkCudaErrors(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,shared_optin - shared_static));
    shared_memory = shared_optin - shared_static;
    LOG(INFO) << fmt::format("shared memory to add {}",shared_memory);
  }
  LOG(INFO) << "cuLaunchKernel" ;
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
                                 shared_memory,
                                 stream,
                                 /*args*/
                                 args,
                                 nullptr));
}
}  // namespace triton_jit
