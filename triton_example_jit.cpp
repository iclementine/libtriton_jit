#include <cstdint>
#include "torch/torch.h"
#include "c10/cuda/CUDAStream.h"
#include "cuda.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include "c10/util/flat_hash_map.h"
#include "fmt/core.h"
#include <string>
#include <iostream>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>
#include <unordered_map>

using json = nlohmann::json;

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors( CUresult err, const char *file, const int line )
{
    if( CUDA_SUCCESS != err) {
        const char* errorName;
        cuGetErrorName(err, &errorName);
        fprintf(stderr,
                "CUDA Driver API error = %04d from file <%s>, line %i. detail: <%s>\n",
                err, file, line, errorName);
        exit(-1);
    }
}

std::string strip(const std::string& str) {
    // Find the first non-whitespace character
    size_t start = str.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
        return ""; // String is all whitespace
    }

    // Find the last non-whitespace character
    size_t end = str.find_last_not_of(" \t\n\r");

    // Return the substring without leading/trailing whitespace
    return str.substr(start, end - start + 1);
}

std::string executePythonScript(const std::string& command) {
    std::array<char, 128> buffer;
    std::string result;

    // Open the process and read its output
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return strip(result);
}



class TritonKernel {
 private:
  std::string dir_;
  std::string kernel_name_;

  bool loaded_ = false;
  unsigned int share_;
  CUmodule module_;
  CUfunction function_;

  void lazy_init_handle(){

    std::cout << (this -> dir_) << std::endl;
    std::cout << (this -> kernel_name_) << std::endl;
    std::cout << "INIT HANDLE" << std::endl;

    if (this -> loaded_) return;

    std::string metadata_path = fmt::format("{}/{}.json", this -> dir_, this -> kernel_name_);
    std::cout <<"META PATH: " <<  metadata_path << std::endl;
    std::ifstream f(metadata_path.c_str());
    json meta_data = json::parse(f);
    this->share_ = meta_data["shared"];

    std::string cubin_path = fmt::format("{}/{}.cubin", this -> dir_, this -> kernel_name_);
    checkCudaErrors(cuModuleLoad(&(this->module_), cubin_path.c_str()));
    checkCudaErrors(cuModuleGetFunction(&this->function_, this->module_, this->kernel_name_.c_str()));
    this->loaded_ = true;
    std::cout << "INIT DONE" << std::endl;
  }

 public:
  TritonKernel(const std::string& dir, const std::string& kernel_name)
    :dir_(dir), kernel_name_(kernel_name), loaded_(false), share_(0), module_(nullptr), function_(nullptr){
    std::cout << "Construct" << std::endl;
    std::cout << (this -> dir_) << std::endl;
    std::cout << (this -> kernel_name_) << std::endl;
  }

  // consider using a variadic template
  void launch(unsigned int grid_x, unsigned int grid_y, unsigned int grid_z, int num_warps, CUstream stream, void** args) {
    this->lazy_init_handle();
    checkCudaErrors(cuLaunchKernel(this->function_, grid_x, grid_y, grid_z, 32 * num_warps, 1, 1, this->share_, stream, args, nullptr));
  }
};

inline const char* to_triton_typename(c10::ScalarType t) {
  switch(t) {
    case c10::ScalarType::Float:
      return "fp32";
    case c10::ScalarType::Double:
      return "fp64";
    case c10::ScalarType::Half:
      return "fp16";
    case c10::ScalarType::BFloat16:
      return "bf16";
    case c10::ScalarType::Int:
      return "i32";
    case c10::ScalarType::Long:
      return "i64";
    case c10::ScalarType::Bool:
      return "i1";
    default:
      return "<unsupported_type>";
  }
}

template <typename T>
struct triton_type {
};

template <> struct triton_type<bool> {static constexpr const char* name = "i1";};
template <> struct triton_type<int> {static constexpr const char* name = "i32";};
template <> struct triton_type<int64_t> {static constexpr const char* name = "i64";};
template <> struct triton_type<float> {static constexpr const char* name = "f32";};
template <> struct triton_type<double> {static constexpr const char* name = "f64";};
template <> struct triton_type<std::nullptr_t> {static constexpr const char* name = "*i8";};


template <typename T>
inline const char* spec(T v) {
  return v % 16 == 0
          ? ":16"
          : v == 1
            ? ":1"
            : "";
}

class TritonJITFunction {
 private:
  std::string path_;
  std::string function_name_;

  mutable std::unordered_map<std::string, TritonKernel> overloads_;

 public:
  TritonJITFunction(std::string_view path, std::string_view function_name)
    :path_(path), function_name_(function_name){}

  void invoke(
    unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
    unsigned int num_warps, unsigned int num_stages,
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& out, int64_t numel, int64_t tile_size,
    CUstream stream) const
  {
    std::string signature = fmt::format("\"*{}{}, *{}{}, *{}{}, {}, {}\"",
      to_triton_typename(a.scalar_type()), spec(reinterpret_cast<std::uintptr_t>(a.data_ptr())),
      to_triton_typename(b.scalar_type()), spec(reinterpret_cast<std::uintptr_t>(b.data_ptr())),
      to_triton_typename(out.scalar_type()), spec(reinterpret_cast<std::uintptr_t>(out.data_ptr())),
      triton_type<decltype(numel)>::name,
      tile_size);

    auto pos = this->overloads_.find(signature);
    if (pos == this->overloads_.end()) {
      std::string cmd = fmt::format(
        "/home/clement/.virtualenvs/dev/bin/python "
        "/home/clement/projects/libtorch_example/standalone_compile.py "
        "--kernel-name {} "
        "--signature {} "
        "--num-warps {} --num-stages {} "
        "{}",
        this->function_name_,
        signature,
        num_warps, num_stages,
        this -> path_
      );
      std::cout << "Command: " << cmd  << std::endl;
      std::string hash = executePythonScript(cmd);

      c10::string kernel_dir = fmt::format(
        "/home/clement/.flaggems/triton_cache/{}",
        hash
      );
      TritonKernel kernel(kernel_dir, this->function_name_);
      pos = this->overloads_.insert({signature, kernel}).first;
    }
    void* pa = a.data_ptr();
    void* pb = b.data_ptr();
    void* pout = out.data_ptr();
    void* args[] = {&pa, &pb, &pout, &numel};
    pos->second.launch(grid_x, grid_y, grid_z, num_warps, stream, args);
  }
};




at::Tensor add_tensor(const at::Tensor& a_, const at::Tensor& b_) {
    auto res = torch::broadcast_tensors({a_, b_});
    const at::Tensor& a = res[0];
    const at::Tensor& b = res[1];
    at::ScalarType out_dtype = at::promote_types(a.scalar_type(), b.scalar_type());
    at::Tensor out = at::empty(
      a.sizes(),
      at::TensorOptions().dtype(out_dtype).device(a.device()));
    int64_t rank = out.ndimension();

    static TritonJITFunction f(
      "/home/clement/projects/libtorch_example/binary_add.py",
      "binary_pointwise_kernel");

    // add utility to build this automatically
    const int64_t tile_size = 1024;
    const int num_warps = 8;
    const int num_stages = 1;
    int64_t n = out.numel();
    const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    CUstream raw_stream = static_cast<CUstream>(stream.stream());
    f.invoke(num_blocks, 1, 1, num_warps, num_stages, a, b, out, n, tile_size, raw_stream);
    // std::cout << out << std::endl;
    return out;
}

#include <iostream>

int main() {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({10, 10}, device);
  torch::Tensor b = torch::randn({10, 10}, device);
  torch::Tensor tmp1 = a + b;
  torch::Tensor tmp2 = add_tensor(a, b);

  std::cout << "EAGER" << std::endl ;
  torch::Tensor out1 = a + b;
  std::cout << out1 << std::endl ;
  std::cout << std::endl;

  std::cout << "TRITON" << std::endl ;
  torch::Tensor out2 = add_tensor(a, b);
  std::cout << out2 << std::endl;
  return 0;

}


