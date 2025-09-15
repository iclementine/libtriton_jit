#include "triton_jit/triton_jit_function.h"

#include <algorithm>
#include <string>
#include <vector>

#include <type_traits>
#include <utility>
#include "c10/util/Logging.h"  // use torch's logging
#include "fmt/core.h"
#include "nlohmann/json.hpp"

#include "pybind11/embed.h"

namespace triton_jit {
std::unordered_map<std::string, std::shared_ptr<TritonJITFunction>> TritonJITFunction::functions_;

void ensure_initialized() {
  // When using libtriton_jit with a python C-extension, it is already initialized
  c10::initLogging();
  if (!Py_IsInitialized()) {
    Py_InitializeEx(false);
  }
}

TritonJITFunction::TritonJITFunction(std::string_view path, std::string_view name)
    : file_path_(std::string(path)), function_name_(std::string(name)) {
  // embed python
  namespace py = pybind11;
  ensure_initialized();
  py::gil_scoped_acquire gil;

  std::filesystem::path script_dir = get_script_dir();
  py::module_ sys = py::module_::import("sys");
  sys.attr("path").attr("insert")(0, script_dir.c_str());
  py::module_ mod = py::module_::import("gen_ssig");
  py::object fn = mod.attr("extract_static_signature");
  py::object ans = fn(this->file_path_, this->function_name_);
  py::list arg_types_raw = ans.cast<py::list>();

  int num_args = arg_types_raw.size();
  std::vector<ArgType> arg_types;
  arg_types.reserve(num_args);
  for (auto item : arg_types_raw) {
    try {
      arg_types.push_back(ArgType(item.cast<int>()));
    } catch (const py::cast_error& e) {
      std::cerr << "Type error: " << e.what() << std::endl;
    }
  }
  this->static_sig_ = StaticSignature {num_args, arg_types};
}

std::shared_ptr<TritonKernel> TritonJITFunction::get_kernel(std::string_view _signature,
                                                  int num_warps,
                                                  int num_stages,
                                                  CUdevice device_index) const {
  std::string signature(_signature);
  std::string key = fmt::format("{};{}", signature, device_index);
  // std::cout << "Standalone script to debug: \n"
  //           << fmt::format(
  //                  "python standalone_compile.py {} -n {} --device-id {} --num-warps {} --num-stages {} "
  //                  "--signature {}",
  //                  this->file_path_,
  //                  this->function_name_,
  //                  device_index,
  //                  num_warps,
  //                  num_stages,
  //                  signature)
  //           << std::endl;
  auto pos = this->overloads_.find(key);

  if (pos == this->overloads_.end()) {
    // embed python
    namespace py = pybind11;
    ensure_initialized();
    py::gil_scoped_acquire gil;

    std::filesystem::path script_dir = get_script_dir();
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, script_dir.c_str());
    py::module_ mod = py::module_::import("standalone_compile");
    py::object fn = mod.attr("compile_a_kernel");
    py::object ans;
    try {
      ans = fn(this->file_path_, this->function_name_, signature, num_warps, num_stages, device_index);
    } catch (const py::error_already_set& e) {
      std::cerr << "Python exception: " << e.what() << std::endl;
    }
    std::string cache_dir = ans.cast<std::string>();
    LOG(INFO) << "Output: " << cache_dir;

    TritonKernel kernel(cache_dir, this->function_name_);
    LOG(INFO) << fmt::format("kernel_dir: {}", cache_dir);
    LOG(INFO) << fmt::format("kernel_name: {}", this->function_name_);
    auto kernel_ptr = std::make_shared<TritonKernel>(cache_dir, this->function_name_);
    auto result = this->overloads_.insert({key, kernel_ptr});
    if (result.second) {
      pos = result.first;
    } else {
      throw std::runtime_error("Unable to emplace the kernel into TritonJITFunction's cache");
    }
  }
  return pos->second;
}

std::shared_ptr<TritonJITFunction> TritonJITFunction::getInstance(
  const std::string& path,
  const std::string& name) {
  //std::lock_guard<std::mutex> lock(mutex_);
  auto key = path + ":" + name;
  auto it = functions_.find(key);
  if (it != functions_.end()) {
    return it->second;
  }

  auto instance = std::shared_ptr<TritonJITFunction>(new TritonJITFunction(path, name));
  functions_[key] = instance;
  return instance;
}

void TritonJITFunction::launch_with_raw_args(CUstream stream,
                                             unsigned int grid_x,
                                             unsigned int grid_y,
                                             unsigned int grid_z,
                                             unsigned int num_warps,
                                             unsigned int num_stages,
                                             std::string full_signature,
                                             void** args) const {
  CUcontext ctx;
  checkCudaErrors(cuStreamGetCtx(stream, &ctx));
  checkCudaErrors(cuCtxSetCurrent(ctx));
  CUdevice d;
  checkCudaErrors(cuCtxGetDevice(&d));
  LOG(INFO) << fmt::format("launching kernel");
  auto kernel = this->get_kernel(full_signature, num_warps, num_stages, d);
  kernel->launch(grid_x, grid_y, grid_z, num_warps, stream, args);
}
}  // namespace triton_jit
