#include "torch/torch.h"
#include "c10/cuda/CUDAStream.h"
#include "cuda.h"

// an example of using cuda driver c++ api to launch cubin compile from triton kernel

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


at::Tensor add_tensor(const at::Tensor& a_, const at::Tensor& b_) {
    auto res = torch::broadcast_tensors({a_, b_});
    const at::Tensor& a = res[0];
    const at::Tensor& b = res[1];
    at::ScalarType out_dtype = at::promote_types(a.scalar_type(), b.scalar_type());
    at::Tensor out = at::empty(
      a.sizes(),
      at::TensorOptions().dtype(out_dtype).device(a.device()));
    int64_t rank = out.ndimension();

    // add utility to build this automatically
    const int64_t tile_size = 1024;
    c10::string signature = "fp32:16, fp32:16, fp32:16, i64, 1024";
    const int num_warps = 4;
    const int num_stages = 1;



    checkCudaErrors(cuInit(0));
    CUdevice device;
    checkCudaErrors(cuDeviceGet(&device, 0));
    char name[100];
    cuDeviceGetName(name, 100, device);
    std::cout << name << std::endl;
    CUcontext context;
    cuCtxCreate(&context, 0, device);

    // move it out
    CUmodule module;
    checkCudaErrors(cuModuleLoad(&module, "/home/clement/projects/libtorch_example/this_cache/419ac661d9e136e585ea07663f6b91a2e482e187d2958a47d033a8d985679e19/binary_pointwise_kernel.cubin"));
    CUfunction cuf;
    checkCudaErrors(cuModuleGetFunction(&cuf, module, "binary_pointwise_kernel"));

    int64_t n = out.numel();
    const unsigned int num_blocks = (n + tile_size - 1) / tile_size;
    void* da = a_.data_ptr();
    void* db = b_.data_ptr();
    void* dout = out.data_ptr();

    void* args[] = {&da, &db, &dout, static_cast<void*>(&n)};

    CUstream stream;
    checkCudaErrors(cuStreamCreate(&stream, 0));
    checkCudaErrors(cuLaunchKernel(
      cuf,
      num_blocks, 1, 1,
      32 * num_warps, 1, 1,
      /*shared*/0, stream,
      args, 0));
    checkCudaErrors(cuStreamSynchronize(stream));
    std::cout << out << std::endl;
    return out;
}

#include <iostream>

int main() {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({4096}, device);
  torch::Tensor b = torch::randn({4096}, device);
  torch::Tensor out = a + b;

  torch::Tensor out2 = add_tensor(a, b);
  // std::cout << out << std::endl;
  return 0;

}


