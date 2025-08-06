#include "triton_jit/jit_utils.h"

#include <dlfcn.h>  // dladdr
#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace triton_jit {
std::filesystem::path get_path_of_this_library() {
  // This function gives the library path of this library as runtime, similar to the $ORIGIN
  // that is used for run path (RPATH), but unfortunately, for custom dependencies (instead of linking)
  // there is no build system generator to take care of this.
  static const std::filesystem::path cached_path = []() {
    Dl_info dl_info;
    if (dladdr(reinterpret_cast<void *>(&get_path_of_this_library), &dl_info) && dl_info.dli_fname) {
      return std::filesystem::canonical(dl_info.dli_fname);  // Ensure absolute, resolved path
    } else {
      throw std::runtime_error("cannot get the path of libjit_utils.so");
    }
  }();
  return cached_path;
}

std::filesystem::path get_script_dir() {
  const static std::filesystem::path script_dir = []() {
    std::filesystem::path installed_script_dir = get_path_of_this_library().parent_path().parent_path() / "share" / "triton_jit" / "scripts";

    if (std::filesystem::exists(installed_script_dir)) {
      return installed_script_dir;
    } else {
      std::filesystem::path source_script_dir =
          std::filesystem::path(__FILE__).parent_path().parent_path() / "scripts";
      return source_script_dir;
    }
  }();
  return script_dir;
}

const char *get_gen_static_sig_script() {
  std::filesystem::path script_dir = get_script_dir();
  return (script_dir / "gen_ssig.py").c_str();
}

const char *get_standalone_compile_script() {
  std::filesystem::path script_dir = get_script_dir();
  return (script_dir / "standalone_compile.py").c_str();
}

std::filesystem::path get_home_directory() {
  const static std::filesystem::path home_dir = []() {
#ifdef _WIN32
    const char *home_dir_path = std::getenv("USERPROFILE");
#else
    const char *home_dir_path = std::getenv("HOME");
#endif
    return std::filesystem::path(home_dir_path);
  }();
  return home_dir;
}

std::filesystem::path get_cache_path() {
  const static std::filesystem::path cache_dir = []() {
    const char *cache_env = std::getenv("FLAGGEMS_TRITON_CACHE_DIR");
    std::filesystem::path cache_dir =
        cache_env ? std::filesystem::path(cache_env) : (get_home_directory() / ".flaggems" / "triton_cache");
    return cache_dir;
  }();
  return cache_dir;
}
// TODO: support reinterpret_and_print_args through full_signature
// now only support binary add
void reinterpret_and_print_args(void** raw_args_list, std::string full_signature) {
    LOG(INFO) << "Reinterpreting raw arguments:" << std::endl;
    LOG(INFO) << "-------------------------------" << std::endl;
    
    // 重新解释并打印前三个 Tensor 的数据指针
    void* tensor_ptr_a = raw_args_list[0];
    void* tensor_ptr_b = raw_args_list[1];
    void* tensor_ptr_out = raw_args_list[2];
    
    LOG(INFO) << "Tensor A Data Pointer: " << tensor_ptr_a << std::endl;
    LOG(INFO) << "Tensor B Data Pointer: " << tensor_ptr_b << std::endl;
    LOG(INFO) << "Output Tensor Data Pointer: " << tensor_ptr_out << std::endl;
    
    LOG(INFO) << "-------------------------------" << std::endl;
    
    int64_t* n_ptr = static_cast<int64_t*>(raw_args_list[3]);
    int64_t n_value = *n_ptr;
    
    LOG(INFO) << "Scalar n (reinterpreted as int64): " << n_value << std::endl;
}
}  // namespace triton_jit
