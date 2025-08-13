
#include "add_op.h"
#include "c10/cuda/CUDAFunctions.h"
#include "torch/torch.h"

int main() {
  at::Tensor a = at::rand({128 * 1024}, at::kCUDA);
  at::Tensor b = at::rand({128 * 1024}, at::kCUDA);
  // warm up
  at::Tensor result1 = my_ops::add_tensor(a, b);
  at::Tensor result2 = at::add(a, b);
  assert(torch::allclose(result1, result2, /*rtol=*/1e-5, /*atol=*/1e-8) && "Results are not equal!");
  c10::cuda::device_synchronize();
  for (int i = 0; i < 10; ++i) {
    auto tmp = at::add(a, b);
  }
  c10::cuda::device_synchronize();
  for (int i = 0; i < 10; ++i) {
    auto tmp = my_ops::add_tensor(a, b);
  }
  c10::cuda::device_synchronize();
  return 0;
}
