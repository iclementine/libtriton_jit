#include "c10/cuda/CUDAStream.h"
#include "torch/torch.h"
#include <iostream>
#include "triton_jit_function.h"

at::Tensor add_tensor(const at::Tensor &a_, const at::Tensor &b_) {
  auto res = torch::broadcast_tensors({a_, b_});
  const at::Tensor &a = res[0];
  const at::Tensor &b = res[1];
  at::ScalarType out_dtype =
      at::promote_types(a.scalar_type(), b.scalar_type());
  at::Tensor out = at::empty(
      a.sizes(), at::TensorOptions().dtype(out_dtype).device(a.device()));
  int64_t rank = out.ndimension();

  const TritonJITFunction& f = TritonJITFunction::getInstance(
      "/home/clement/projects/libtorch_example/triton_src/binary_add.py",
      "binary_pointwise_kernel");

  // add utility to build this automatically
  int64_t tile_size = 1024;
  const int num_warps = 8;
  const int num_stages = 1;
  int64_t n = out.numel();
  const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  f(stream, num_blocks, 1, 1, num_warps, num_stages, a, b, out, n, tile_size);
  return out;
}

int main() {
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({10, 10}, device);
  torch::Tensor b = torch::randn({10, 10}, device);
  torch::Tensor tmp1 = a + b;
  torch::Tensor tmp2 = add_tensor(a, b);
  std::cout << "ATEN:\n" << tmp1 << std::endl;
  std::cout << "TRITON:\n" << tmp2 << std::endl;

  for (int i = 0; i < 10; i++) {
    torch::Tensor out1 = a + b;
  }

  for (int i = 0; i < 10; i++) {
    torch::Tensor out2 = add_tensor(a, b);
  }
  return 0;
}
