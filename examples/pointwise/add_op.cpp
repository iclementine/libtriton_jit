

#include "add_op.h"
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace my_ops {
using namespace triton_jit;

at::Tensor add_tensor(const at::Tensor &a_, const at::Tensor &b_) {
  auto res = torch::broadcast_tensors({a_, b_});
  res[0] = res[0].contiguous();
  res[1] = res[1].contiguous();
  const at::Tensor &a = res[0];
  const at::Tensor &b = res[1];

  at::ScalarType out_dtype = at::promote_types(a.scalar_type(), b.scalar_type());
  at::Tensor out = at::empty(a.sizes(), at::TensorOptions().dtype(out_dtype).device(a.device()));

  const TritonJITFunction &f =
      TritonJITFunction::getInstance(std::string("add.py"), "binary_pointwise_kernel");

  // add utility to build this automatically
  int64_t tile_size = 1024;
  const int num_warps = 8;
  const int num_stages = 1;
  int64_t n = out.numel();
  const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

  // getCurrentCUDAStream ensures that the stream is initialized, a default stream for each device
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  c10::DeviceGuard guard(out.device());
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  f(stream, num_blocks, 1, 1, num_warps, num_stages, a, b, out, n, tile_size);
  return out;
}
// demo: add tensor using `launch_with_raw_args`
at::Tensor add_tensor(const at::Tensor& a_, const at::Tensor& b_) {
    auto res = torch::broadcast_tensors({a_, b_});
    const at::Tensor& a = res[0].contiguous();
    const at::Tensor& b = res[1].contiguous();
    void* a_ptr = a.data_ptr();
    void* b_ptr = b.data_ptr();
    at::ScalarType out_dtype = at::promote_types(a.scalar_type(), b.scalar_type());
    at::Tensor out = at::empty(a.sizes(), at::TensorOptions().dtype(out_dtype).device(a.device()));
    void* out_ptr = out.data_ptr();
    const TritonJITFunction& f =
        TritonJITFunction::getInstance(std::string(utils::get_triton_src_path() / "binary_add.py"),
                                       "binary_pointwise_kernel");
    int64_t tile_size = 1024;
    const int64_t n = out.numel();
    std::vector<void*> raw_args_list;
    raw_args_list.push_back(&a_ptr);
    raw_args_list.push_back(&b_ptr);
    raw_args_list.push_back(&out_ptr);
    raw_args_list.push_back(const_cast<int64_t*>(&n));
    void *global_scratch = nullptr;
    raw_args_list.push_back(&global_scratch);
    std::string signature = "*fp32:16,*fp32:16,*fp32:16,i64,1024";

    reinterpret_and_print_args(raw_args_list.data(),signature);
    const int num_warps = 8;
    const int num_stages = 1;
    const unsigned int num_blocks = (n + tile_size - 1) / tile_size;
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    c10::DeviceGuard guard(out.device());
    CUstream raw_stream = static_cast<CUstream>(stream.stream());

    f.launch_with_raw_args(raw_stream, num_blocks, 1, 1, num_warps, num_stages, signature, raw_args_list.data());

    return out;
}

TORCH_LIBRARY(my_ops, m) {
  m.def("add_tensor(Tensor self, Tensor other) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_ops, CUDA, m) {
  m.impl("add_tensor", TORCH_FN(add_tensor));
}
}  // namespace my_ops
