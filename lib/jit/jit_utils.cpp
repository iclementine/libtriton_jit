#include "jit/jit_utils.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace triton_jit {
std::string execute_command(std::string_view command) {
  std::array<char, 128> buffer;
  std::string result;

  // Open the process and read its output
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.data(), "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }

  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  if (!result.empty() && result.back() == '\n') {
    result.pop_back();
  }
  return result;
}

const char *get_python_executable() {
  const static std::string python_executable_path = []() {
    std::string python_exe;
    const char *python_env = std::getenv("PYTHON_EXECUTABLE");
    python_exe = python_env ? std::string(python_env) : execute_command("which python");
    std::cout << "python executable: " << python_exe << std::endl;
    if (python_exe.empty()) {
      throw std::runtime_error("cannot find python executable!");
    }
    return python_exe;
  }();
  return python_executable_path.c_str();
}

const char *get_gen_static_sig_script() {
  const static std::filesystem::path gen_spec_script =
      std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "tools" / "gen_spec.py";
  return gen_spec_script.c_str();
}

const char *get_standalone_compile_script() {
  const static std::filesystem::path compile_script =
      std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "tools" /
      "standalone_compile.py";
  return compile_script.c_str();
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

std::filesystem::path get_triton_src_path() {
  const static std::filesystem::path triton_src_dir =
      std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / "triton_src";
  return triton_src_dir;
}
}  // namespace triton_jit
