#include "triton_jit_function.h"
#include "fmt/core.h"

std::unordered_map<std::string, TritonJITFunction>
    TritonJITFunction::functions_;

const TritonKernel &TritonJITFunction::get_kernel(const std::string &signature,
                                                  int num_warps,
                                                  int num_stages) const {
  auto pos = this->overloads_.find(signature);
  if (pos == this->overloads_.end()) {
    std::string cmd = fmt::format(
        "/home/clement/.virtualenvs/dev/bin/python "
        "/home/clement/projects/libtorch_example/tools/standalone_compile.py "
        "--kernel-name {} "
        "--signature {} "
        "--num-warps {} --num-stages {} "
        "{}",
        this->function_name_, signature, num_warps, num_stages,
        this->file_path_);
    std::cout << "Command: " << cmd << std::endl;
    std::string hash = executePythonScript(cmd);
    std::cout << "Output: " << hash << std::endl;

    c10::string kernel_dir =
        fmt::format("/home/clement/.flaggems/triton_cache/{}", hash);
    TritonKernel kernel(kernel_dir, this->function_name_);
    pos = this->overloads_.emplace(signature, kernel).first;
  }
  return pos->second;
}

TritonJITFunction &TritonJITFunction::getInstance(std::string_view path,
                                                  std::string_view name) {
  const std::string function_id = fmt::format("{}:{}", path, name);
  auto pos = TritonJITFunction::functions_.find(function_id);

  if (pos == TritonJITFunction::functions_.end()) {
    TritonJITFunction f(path, name);
    pos = TritonJITFunction::functions_.emplace(function_id, f).first;
  }
  return pos->second;
}