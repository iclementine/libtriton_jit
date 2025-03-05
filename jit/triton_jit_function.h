#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>
#include <sstream>
#include <type_traits>
#include "cuda.h"
#include "fmt/core.h"
#include "jit_utils.h"
#include "triton_kernel.h"

struct StaticSignature {
  int num_args;
  std::vector<int> arg_type;
};

class TritonJITFunction {
 public:
  static TritonJITFunction &getInstance(std::string_view path, std::string_view name);

  template <typename... Args>
  void operator()(CUstream stream,
                  unsigned int grid_x,
                  unsigned int grid_y,
                  unsigned int grid_z,
                  unsigned int num_warps,
                  unsigned int num_stages,
                  Args... args) const {
    // requires:
    // 1. signature from kernel
    // build signature (requires args processing) -> build a source for the
    // function & call it. get kernel (requires this*(for path & name)) pick
    // argument to the kernel & launch (requires signature from kernel)
    const int num_args = this->static_sig_.num_args;

    std::vector<void *> data_pointers;
    data_pointers.reserve(num_args);

    std::vector<void *> kernel_args;
    kernel_args.reserve(num_args);

    std::vector<std::string> signature;
    signature.reserve(num_args);

    int idx = 0;

    auto arg_handle = [&](const auto &item) {
      if constexpr (has_data_ptr<decltype(item)>::value) {
        void *p_item = item.data_ptr();
        data_pointers.push_back(p_item);
        kernel_args.push_back(&(data_pointers[idx]));

        const char *dtype = to_triton_typename(item.scalar_type());
        const char *specialization = "";
        if (this->static_sig_.arg_type[idx] == 1) {
          specialization = spec(reinterpret_cast<std::uintptr_t>(data_pointers[idx]));
        }
        std::string sig_for_idx = fmt::format("*{}{}", dtype, specialization);
        signature.push_back(sig_for_idx);
      } else if constexpr (std::is_same_v<decltype(item), std::nullopt_t>) {
        signature.push_back("*i8");
      } else if (this->static_sig_.arg_type[idx] == 2) {  // constexpr
        signature.push_back(fmt::format("{}", item));
      } else if (this->static_sig_.arg_type[idx] == 1) {  // specialzied
        const char *dtype = triton_type<decltype(item)>::name;
        const char *specialization = spec(item);
        if constexpr (std::is_integral_v<decltype(item)>) {
          if (specialization != ":1") {
            const void *p_item = &item;
            kernel_args.push_back(const_cast<void *>(p_item));
          }
        } else {
          const void *p_item = &item;
          kernel_args.push_back(const_cast<void *>(p_item));
        }
        std::string sig_for_idx = fmt::format("{}{}", dtype, specialization);
        signature.push_back(sig_for_idx);
      } else {
        const void *p_item = &item;
        kernel_args.push_back(const_cast<void *>(p_item));
        const char *dtype = triton_type<decltype(item)>::name;
        signature.push_back(dtype);
      }
      idx++;
    };
    (arg_handle(args), ...);

    std::string full_signature;
    for (int i = 0; i < signature.size(); i++) {
      if (i == 0) {
        full_signature += signature[i];
      } else {
        full_signature += ",";
        full_signature += signature[i];
      }
    }

    const TritonKernel &kernel = this->get_kernel(full_signature, num_warps, num_stages);
    kernel.launch(grid_x, grid_y, grid_z, num_warps, stream, kernel_args.data());
    return;
  }

 private:
  std::string file_path_;
  std::string function_name_;
  StaticSignature static_sig_;
  mutable std::unordered_map<std::string, TritonKernel> overloads_;

  static std::unordered_map<std::string, TritonJITFunction> functions_;

 private:
  TritonJITFunction(std::string_view path, std::string_view name) : file_path_(path), function_name_(name) {
    // Can we load the function with signature now?
    // We need to know whether an argument is a constexpr
    std::string cmd =
        fmt::format("{} {} -n {} {}", get_python_executable(), get_gen_static_sig_script(), name, path);
    std::cout << "Command: " << cmd << std::endl;
    using json = nlohmann::json;
    std::string signature = execute_command(cmd);
    std::cout << "Output: " << signature << std::endl;

    json j = json::parse(std::stringstream(signature));
    std::vector<int> arg_types = j.get<std::vector<int>>();
    int num_args = arg_types.size();
    this->static_sig_ = StaticSignature {num_args, arg_types};
    std::cout << j.dump() << std::endl;
  }

  const TritonKernel &get_kernel(const std::string &signature, int num_warps, int num_stages) const;
};
