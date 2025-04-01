#include "triton_jit/triton_jit_function.h"

#include <algorithm>
#include <string>
#include <vector>

#include "fmt/core.h"
#include "nlohmann/json.hpp"

namespace triton_jit {
std::unordered_map<std::string, TritonJITFunction> TritonJITFunction::functions_;

TritonJITFunction::TritonJITFunction(std::string_view path, std::string_view name)
    : file_path_(path), function_name_(name) {
  std::string cmd =
      fmt::format("{} {} -n {} {}", get_python_executable(), get_gen_static_sig_script(), name, path);
  std::cout << "Command: " << cmd << std::endl;
  using json = nlohmann::json;
  std::string signature = execute_command(cmd);
  std::cout << "Output: " << signature << std::endl;

  json j = json::parse(std::stringstream(signature));
  std::vector<int> arg_types_raw = j.get<std::vector<int>>();
  std::vector<ArgType> arg_types(arg_types_raw.size());
  std::transform(arg_types_raw.begin(), arg_types_raw.end(), arg_types.begin(), [](int tag) {
    return ArgType(tag);
  });
  int num_args = arg_types.size();
  this->static_sig_ = StaticSignature {num_args, arg_types};
  std::cout << j.dump() << std::endl;
}

const TritonKernel &TritonJITFunction::get_kernel(const std::string &signature,
                                                  int num_warps,
                                                  int num_stages) const {
  auto pos = this->overloads_.find(signature);
  if (pos == this->overloads_.end()) {
    std::string cmd = fmt::format(
        "{} {} "
        "--kernel-name {} "
        "--signature {} "
        "--num-warps {} --num-stages {} "
        "{}",
        get_python_executable(),
        get_standalone_compile_script(),
        this->function_name_,
        signature,
        num_warps,
        num_stages,
        this->file_path_);
    std::cout << "Command: " << cmd << std::endl;
    std::string hash = execute_command(cmd);
    std::cout << "Output: " << hash << std::endl;

    std::string kernel_dir = std::string(get_cache_path() / hash);
    TritonKernel kernel(kernel_dir, this->function_name_);
    pos = this->overloads_.emplace(signature, kernel).first;
  }
  return pos->second;
}

TritonJITFunction &TritonJITFunction::getInstance(std::string_view path, std::string_view name) {
  const std::string function_id = fmt::format("{}:{}", path, name);
  auto pos = TritonJITFunction::functions_.find(function_id);

  if (pos == TritonJITFunction::functions_.end()) {
    TritonJITFunction f(path, name);
    pos = TritonJITFunction::functions_.emplace(function_id, f).first;
  }
  return pos->second;
}
}  // namespace triton_jit
