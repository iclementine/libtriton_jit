An example for show how to implement an custom operator in torch with libtriton_jit. There are basically 4 steps to go:

1. define the triton function to be use with the wrapper;
2. define the wrapper in c++, which typically uses aten APIs to do meta data computation and use TritonJITFunction to wrap the triton function and call it;
3. define a torch library, define an operator and register the wrapper as an implementation of it;
4. (optional) add a python C-extension which includes the content in step-3;
5. load the torch library after loading libtorch(either by torch.ops.load_library or importing the C-extension) ;
6. add other function to support torch subsystems(register_fake to support torch.compile and register_autograd to support torch.autograd, Note that any part of the code in the backward function that accesses the data of tensors should be wrapped into a custom op, too)

We compare the performances in 6 cases:

1. (c++) at::op_name in c++ (for torch-builtin ops only);
2. (python) torch.op_name in python (for torch-builtin ops only);
3. (python) python wrapper that calls triton function in triton python runtime;
4. (python) custom op that registers 3 as implementation (via torch.ops.lib_name.op_name);
5. (c++) c++ wrapper that calls TritonJITFunction(libtriton_jit);
6. (python) custom op that registers 5 as implementation (via torch.ops.lib_name.op_name);

we can also add
7. (python) torch.ops.lib_name.op_name for torch-builtin operators to show that boxed-call is slower.

Typically, we warm up a function first; then synchronize the device, run the function repeatedly for 10 times and synchronize again. We use nsys to profile to program.

The metric we are interested now is
1. the average execution time of the kernel (excluding the warmup steo);
2. the average interval for kernel launches (this is an approximation of the wrapper's overhead).
