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
std::unordered_map<std::string, TritonJITFunction> TritonJITFunction::functions_;

TritonJITFunction::TritonJITFunction(std::string_view path, std::string_view name)
    : file_path_(std::string(path)), function_name_(std::string(name)) {
  // sys command
  // std::string cmd =
  //     fmt::format("{} {} -n {} {}", get_python_executable(), get_gen_static_sig_script(), name, path);
  // LOG(INFO) << "(Extracting Static Signature) Command: " << cmd;
  // using json = nlohmann::json;
  // std::string signature = execute_command(cmd);
  // LOG(INFO) << "Output: " << signature;

  // json j = json::parse(std::stringstream(signature));
  // std::vector<int> arg_types_raw = j.get<std::vector<int>>();
  // std::vector<ArgType> arg_types(arg_types_raw.size());
  // std::transform(arg_types_raw.begin(), arg_types_raw.end(), arg_types.begin(), [](int tag) {
  //   return ArgType(tag);
  // });
  // int num_args = arg_types.size();
  // this->static_sig_ = StaticSignature {num_args, arg_types};
  // LOG(INFO) << arg_types_raw;

  // embed python
  namespace py = pybind11;
  // py::scoped_interpreter guard{};

  auto tstate = PyGILState_Ensure();
  pybind11::get_shared_data("");  // setup the internals pointer
  PyGILState_Release(tstate);
  pybind11::gil_scoped_acquire guard {};

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

const TritonKernel& TritonJITFunction::get_kernel(std::string_view _signature,
                                                  int num_warps,
                                                  int num_stages,
                                                  CUdevice device_index) const {
  std::string signature(_signature);
  auto pos = this->overloads_.find(signature);
  if (pos == this->overloads_.end()) {
    // embed python
    namespace py = pybind11;
    // py::scoped_interpreter guard{};
    auto tstate = PyGILState_Ensure();
    pybind11::get_shared_data("");  // setup the internals pointer
    PyGILState_Release(tstate);
    pybind11::gil_scoped_acquire guard {};

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
    std::string hash = ans.cast<std::string>();
    LOG(INFO) << "Output: " << hash;

    // sys call
    // std::string cmd = fmt::format(
    //     "{} {} "
    //     "--kernel-name {} "
    //     "--signature {} "
    //     "--num-warps {} --num-stages {} "
    //     "--device-id {} "
    //     "{}",
    //     get_python_executable(),
    //     get_standalone_compile_script(),
    //     this->function_name_,
    //     signature,
    //     num_warps,
    //     num_stages,
    //     device_index,
    //     this->file_path_);
    // LOG(INFO) << "(JIT compiling) Command: " << cmd;
    // std::string hash = execute_command(cmd);
    // LOG(INFO) << "Output: " << hash;

    std::string kernel_dir = std::string(get_cache_path() / hash);
    TritonKernel kernel(kernel_dir, this->function_name_);
    LOG(INFO) << fmt::format("kernel_dir: {}", kernel_dir);
    LOG(INFO) << fmt::format("kernel_name: {}", this->function_name_);
    auto result = this->overloads_.insert({signature, kernel});
    if (result.second) {
      pos = result.first;
    } else {
      throw std::runtime_error("Unable to emplace the kernel into TritonJITFunction's cache");
    }
  }
  return pos->second;
}

TritonJITFunction& TritonJITFunction::getInstance(std::string_view path, std::string_view name) {
  std::string function_id = fmt::format("{}:{}", path, name);
  auto pos = TritonJITFunction::functions_.find(function_id);

  if (pos == TritonJITFunction::functions_.end()) {
    TritonJITFunction f(path, name);
    auto result = TritonJITFunction::functions_.insert({function_id, f});
    if (result.second) {
      pos = result.first;
    } else {
      throw std::runtime_error("Unable to emplace the TritonJITFunction into Multiton cache.");
    }
  }
  return pos->second;
}
}  // namespace triton_jit
