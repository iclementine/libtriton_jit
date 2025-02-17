#include "cuda.h"
#include "torch/torch.h"
#include <iostream>

template <typename T, int TILE_SIZE, int BLOCK_SIZE>
__global__ void tensor_add(const T *a, const T *b, T *out, int64_t *strides_a,
                           int64_t *strides_b, int64_t *strides_out,
                           int64_t *task_shape, int num_tasks, int rank) {
  int elements_per_thread = TILE_SIZE / BLOCK_SIZE;
  int block_offset = TILE_SIZE * blockIdx.x;

#pragma unroll
  for (int i = 0; i < elements_per_thread; i++) {
    int task_id = block_offset + threadIdx.x + i * BLOCK_SIZE;
    if (task_id < num_tasks) {
      int offsets_a = 0;
      int offsets_b = 0;
      int offsets_out = 0;
      for (int j = rank - 1; j >= 0; j--) {
        int idx = task_id % task_shape[j];
        offsets_a += strides_a[j] * idx;
        offsets_b += strides_b[j] * idx;
        offsets_out += strides_out[j] * idx;
        task_id /= task_shape[j];
      }
      out[offsets_out] = a[offsets_a] + b[offsets_b];
    }
  }
}

template <typename T>
at::Tensor add_tensor(const at::Tensor &a_, const at::Tensor &b_) {
  auto res = torch::broadcast_tensors({a_, b_});
  const at::Tensor &a = res[0];
  const at::Tensor &b = res[1];
  at::ScalarType out_dtype =
      at::promote_types(a.scalar_type(), b.scalar_type());
  at::Tensor out = at::empty(
      a.sizes(), at::TensorOptions().dtype(out_dtype).device(a.device()));
  int64_t rank = out.ndimension();

  at::Tensor strides_a = at::empty(
      {rank}, at::TensorOptions().dtype(torch::kInt64).device(a.device()));
  cudaMemcpy(strides_a.mutable_data_ptr<int64_t>(), a.strides().data(),
             rank * sizeof(int64_t), cudaMemcpyHostToDevice);

  at::Tensor strides_b = at::empty(
      {rank}, at::TensorOptions().dtype(torch::kInt64).device(a.device()));
  cudaMemcpy(strides_b.mutable_data_ptr<int64_t>(), b.strides().data(),
             rank * sizeof(int64_t), cudaMemcpyHostToDevice);

  at::Tensor strides_out = at::empty(
      {rank}, at::TensorOptions().dtype(torch::kInt64).device(a.device()));
  cudaMemcpy(strides_out.mutable_data_ptr<int64_t>(), out.strides().data(),
             rank * sizeof(int64_t), cudaMemcpyHostToDevice);

  at::Tensor task_shape = at::empty(
      {rank}, at::TensorOptions().dtype(torch::kInt64).device(a.device()));
  cudaMemcpy(task_shape.mutable_data_ptr<int64_t>(), out.sizes().data(),
             rank * sizeof(int64_t), cudaMemcpyHostToDevice);

  constexpr int tile_size = 128;
  int num_blocks = (a.numel() + tile_size - 1) / tile_size;
  constexpr int block_size = 32;
  tensor_add<T, tile_size, block_size><<<num_blocks, block_size>>>(
      a.const_data_ptr<T>(), b.const_data_ptr<T>(), out.data_ptr<T>(),
      strides_a.data_ptr<int64_t>(), strides_b.data_ptr<int64_t>(),
      strides_out.data_ptr<int64_t>(), task_shape.data_ptr<int64_t>(),
      out.numel(), rank);
  return out;
}

int main() {
  torch::Tensor MM = torch::randn({100, 100});
  cuInit(static_cast<unsigned int>(0));
  const torch::Device device(torch::kCUDA, 0);
  torch::Tensor a = torch::randn({10, 10}, device);
  torch::Tensor b = torch::randn({10}, device);
  torch::Tensor out = a + b;

  std::cout << out << std::endl;
  std::cout << (add_tensor<float>(a, b) - out).abs().sum() << std::endl;
  return 0;
}
