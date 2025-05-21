
#include "c10/cuda/CUDAFunctions.h"
#include "sum_op.h"
#include "torch/torch.h"

int main() {
  at::Tensor tensor = at::rand({16, 4 * 1024}, at::kCUDA);
  // warm up
  at::Tensor result1 = my_ops::sum_dim(tensor, {1}, false, c10::nullopt);
  at::Tensor result2 = at::sum(tensor, {1}, false, c10::nullopt);

  c10::cuda::device_synchronize();
  for (int i = 0; i < 10; ++i) {
    auto tmp = at::sum(tensor, {1}, false, c10::nullopt);
  }
  c10::cuda::device_synchronize();
  for (int i = 0; i < 10; ++i) {
    auto tmp = my_ops::sum_dim(tensor, {1}, false, c10::nullopt);
  }
  c10::cuda::device_synchronize();
  return 0;
}
