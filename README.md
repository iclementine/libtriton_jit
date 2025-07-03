# libtriton_jit: Triton JIT C++ runtime.

## Backgrounds

This project offer shims to make using Triton lang inside a c++ based project easier. It provides user experience similar to Triton's python interface. You can define Jit functions in python and run them in c++ code. It aims to reduce the inevitable python overhead when use triton in python code. For many kernels, the execution time of the kernel is much shorter that the cpu overhead, which composes of mainly three parts:

- wrapper overhead(tensor metadata computation and argument preprocessing);
- jit overhead(extracting information from arguments to get the kernel, including type and specialization of arguments and value of constexpr arguments);
- kernel launch overhead(in cuda, cuLaunchKernel introduces about 6us per kernel launch);

This project the wrapper overhead and jit overhead can be moderated by using c++.

## Triton JIT C++ runtime

The most user-facing part of this project is class `TritonJitFunction`, which stands for a JitFunction in python. It jit compiles kernels and caches them in a per `TritonJitFunction` fashion. The compilation is done via some glue code to call `triton.compile`. The cache of compiled kernels is managed by triton's `CacheManager`.

`TritonJitFunction` has a variadic function template `operator()` to capture the types of the arguments at call-site. The call-site signature, along with the static signature provided by the JitFunction (mainly via `tl.constexpr` type hint and `do_not_specialize` argument to the `triton.jit` decorator, which describe how to route the parameters, to pass to the compiler, or the compiled kernel, to specialize or not) make up the logic to handle arguments. It builds a full signature to compile a kernel for, and pick all the arguments for the kernel launch.

Once the full signature is acquired, a standalond script is excuted to compile a kernel and returns the path of the compiled kernel (see class `TritonKernel` for more details), which is then loaded into a per `TritonJitFunction` cache.

Then the arguemnts are used to launch the kernel via a low level driver API. Now it supports cuda driver API. The cuda driver API `cuLaunchKernel` erases type of all arguments by taking addresses of all arguments to the kernel via a pointer to void(`void*`). Backends with similar API can adapt the code to launch kernels. But other backends are also considered. For backends without such indirect call API via type erasure, the captured types from call-site can be used to redirect the call to the kernel. Hopefully we may see them soon.

This part is the main facilities for calling jit functions from c++, which can be used to write operators.

## Usage

The basic usage of this library is via `TritonJITFunction`. First get a `TritonJITFunction` via `TritonJITFunction::getInstance(source_path, function_name)`. Then call it.

The operator() of TritonJITFunction is a variadic template. The arguemnts consists of 2 parts.
- The fixed part is basically launch config and compile options for triton jit function.
- The variadic part is the arguments of the triton jit function.

A simple example to add two tensors.

```cpp
at::Tensor add_tensor(const at::Tensor &a_, const at::Tensor &b_) {
  auto res = torch::broadcast_tensors({a_, b_});
  res[0] = res[0].contiguous();
  res[1] = res[1].contiguous();
  const at::Tensor &a = res[0];
  const at::Tensor &b = res[1];

  at::ScalarType out_dtype = at::promote_types(a.scalar_type(), b.scalar_type());
  at::Tensor out = at::empty(a.sizes(), at::TensorOptions().dtype(out_dtype).device(a.device()));

  const triton_jit::TritonJITFunction &f =
      triton_jit::TritonJITFunction::getInstance("add.py", "binary_pointwise_kernel");

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
```


Since we are mainly focus on Torch now, operators means some functions that

- handles torch tensors and
- dynamically dispatch to different backend-specific implementations based on arguments.

They can use a lot of APIs provided by libtorch, including utility functions for meta-data computation and also all other aten operators. But when the focus is to implement operators with triton lang, we mainly use those utility functions for meta-data computation and operators for output allocation and reviewing(viewing a tensor into another with different meta data and leave the underlying storage untouched).

The the operators can be register into a torch library via `TORCH_LIBRARY` APIs. Then the operators can be used both from c++ and python. You don't even need to explicitly write python bindings for them, since torch already provides a unified(boxed) way to call operators via the dispatcher.

We have examples on pointwise add and reduce sum.


## How to build

1. Install dependencies.

   Though this project is a c++project, it embeds python interpreter to execute some python code, so it has some python dependencies. Also, those python packages is not pure-python, this project also uses their cmake packages, headers and libraries. You can install them in a python virtual environment.

   command: `pip install torch triton cmake ninja packaging pybind11`

2. Configure & Generate build system. Remember to specify which python root to use, since the python root is used to find libtorch and pybind11.

   command: `cmake -S . -B build -DPython_ROOT="$(which python)/../.."`

   You can also specify build type via `-DCMAKE_BUILD_TYPE` and install prefix by `-DCMAKE_INSTALL_PREFIX`.
3. Build:

   command: `cmake --build build --parallel`.
4. Install(optional):

   command: `cmake --install build`.

## How to use it in a c++ project

TritonJIT provides cmake packages, so it can be used with cmake. It can be used in 2 ways.

1. use the installed package, via `find_package`.
2. add the project as a sub-project, via `FetchContent`, `ExternProjectAdd` or `add_subdirectory`.


## Debug
Enable debug LOGGING
