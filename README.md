# libtorch_example
use libtorch to do interesting things

This project offer shims to make using Triton lang inside a c++ based project easier. It provides user experience similar to Triton's python interface. You can define Jit functions in python and run them in c++ code. It aims to reduce the inevitable python overhead when use triton in python code. For many kernels, the execution time of the kernel is much shorter that the cpu overhead, which composes of mainly three parts:

- wrapper overhead(tensor metadata computation and argument preprocessing);
- jit overhead(extracting information from arguments to get the kernel, including type and specialization of arguments and value of constexpr arguments);
- kernel launch overhead(in cuda, cuLaunchKernel introduces about 6us per kernel launch);

This project the wrapper overhead and jit overhead can be moderated by using c++.


## structure

### jit

The jit part mainly provides class `TritonJitFunction`, which stands for a JitFunction in python. It jit compiles kernels and caches them in a per `TritonJitFunction` fashion. The compilation is done via some glue code to call `triton.compile`. The cache of compiled kernels is managed by triton's `CacheManager`.

`TritonJitFunction` has a variadic function template `operator()` to capture the types of the arguments at call-site. The call-site signature, along with the static signature provided by the JitFunction (mainly via `tl.constexpr` type hint and `do_not_specialize` argument to the `triton.jit` decorator, which describe how to route the parameters, to pass to the compiler, or the compiled kernel, to specialize or not) make up the logic the handle arguments, mainly build a full signature to compile a kernel for, and pick all the arguments for the kernel launch.

Once the full signature is acquired, a standalond script is excuted to compile a kernel and returns the path of the compiled kernel, which is then loaded into a per `TritonJitFunction` cache.

Then the arguemnts are used to launch the kernel via a low level driver API. Now it supports cuda driver API. The cuda driver API `cuLaunchKernel` erases type of all arguments by taking addresses of all arguments to the kernel via a pointer to void(`void*`). Backends with similar API can adapt the code to launch kernels. But other backends are also considered. For backends without such indirect call API via type erasure, the captured types from call-site can be used to redirect the call to the kernel. Hopefully we may see them soon.

This part is the main facilities for calling jit functions from c++, which can be used to write operators.

### operators

Since we are mainly focus on Torch now, operators means some function that 1) handles torch tensors and 2)dynamically dispatch to different backend-specific implementations based on arguments.

An implementation that handles torch tensors can use a lot of APIs provided by libtorch, including utility functions for meta-data computation and also all other aten operators. But when the focus is to implement operators via torch, we mainly use those utility functions for meta-data computation and operators for output allocation and review(viewing a tensor into another with different meta data and leave the underlying storage untouched).

The the operators can be register into a torch library via `TORCH_LIBRARY` APIs. Then the operators can be used both from c++ and python. You don't even need to explicitly write python bindings for them, since torch already provides a unified(boxed) way to call operators via the dispatcher.

### tests

This directory includes some code to test the operators. The operators should be tested both from c++ and python to ensure that they work as expected. But now the test is not done.

## how to build

1. Activate a python virtual environment where torch is installed;
2. Configure & Generate build system: specify which python root to use, we would use the torch installed within it.
  command: `cmake -S . -B build -DPython_ROOT="$(which python)/../.."`
3. Build: `cmake --build build --parallel`

## how to use

1. via find_package
2. via FetchContent
