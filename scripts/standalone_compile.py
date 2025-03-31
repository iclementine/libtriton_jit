import importlib.util
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, Union

import triton

# use a separate cache for flaggems triton kernels
os.environ["TRITON_CACHE_DIR"] = str(Path.home() / ".flaggems" / "triton_cache")
# pylint: disable-next=wrong-import-position

DESC = """
Script to compile Triton Jit functions into Compiled Kernel and cache it into a cache dir.
We return the kernel name and subdir path in which the kernel files site.

This program compiles the kernel with name `kernel-name` in the file at the
provided `path` into self-contained C source-code that embeds the `cubin`
data along with utilities to load, unload and launch the kernel.

signature is provided as a list of (optionally divisibility-hinted) types
or constexpr values, e.g.

`compile.py --kernel-name kernel --signature "*fp32:16, i32:16, 1024, i32" --out-name kernel /path/to/kernel.py`

will compile triton.JITFunction of name `kernel` inside the file `/path/to/kernel.py`.
Said kernel will be specialized such that argument 0, 1 are assumed to be multiple of 16,
and argument 2 is assumed to be a compile-time constant of value 1024, i.e. it won't be part of the generated prototype.
"""


# backends/nvidia/driver.py
def ty_to_cpp(ty):
    if ty[0] == "*":
        return "CUdeviceptr"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


# compiler/code_generator.py
def kernel_suffix(signature, specialization):
    # suffix format:
    # <argid><'c' if equal to 1><'d' if divisible by 16><'e' if divisible by 8>
    suffix = ""
    for i, _ in enumerate(signature):
        suffix += str(i)
        if i in specialization.equal_to_1:
            suffix += "c"
        if i in specialization.divisible_by_16:
            suffix += "d"
    return suffix


def compile_a_kernel(
    fn: triton.runtime.JITFunction,
    signature: str,
    num_warps: int = 4,
    num_stages: int = 3,
) -> Tuple[str, str]:
    """compile a kernel."""
    # validate and parse signature
    # example "*fp32, *fp32:16, i32, 1024"
    # for bool use i1, for boolean values, use 0 or 1.
    # split it
    signature: List[str] = list(map(lambda s: s.strip(" "), signature.split(",")))

    def constexpr(s: str) -> Union[int, float]:
        """Extract constexpr from signature"""
        try:
            ret = int(s)
            return ret
        except ValueError:
            pass
        try:
            ret = float(s)
            return ret
        except ValueError:
            pass
        return None

    # constants
    constants = {i: constexpr(s) for i, s in enumerate(signature)}
    constants = {k: v for k, v in constants.items() if v is not None}

    # signature, no specializations here
    signature_without_spec = {
        i: s.split(":")[0] for i, s in enumerate(signature) if i not in constants
    }

    # specialization: divisibility by 16 or equal to 1
    hints = {i: constexpr(s.split(":")[1]) for i, s in enumerate(signature) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}
    for h in hints.values():
        assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"
    divisible_by_16 = [i for i, h in hints.items() if h == 16]
    equal_to_1 = [i for i, h in hints.items() if h == 1]
    attrs = triton.compiler.AttrsDescriptor(
        divisible_by_16=divisible_by_16, equal_to_1=equal_to_1
    )

    # 1 are added into constants
    # we shall also specialize None
    # also the type f None should be *i8
    for i in equal_to_1:
        constants.update({i: 1})

    # STEP1: JITFunction, constants, signature, specialization
    src = triton.compiler.ASTSource(
        fn=fn,
        constants=constants,
        signature=signature_without_spec,
        attrs=attrs,
    )
    # STEP2: compile options for the backend
    opts = {"num_warps": num_warps, "num_stages": num_stages}

    # STEP3: ast source, target, compile options
    target: triton.backends.compiler.GPUTarget = (
        triton.runtime.driver.active.get_current_target()
    )
    ccinfo: triton.compiler.CompiledKernel = triton.compile(src, target, options=opts)
    return ccinfo.name, ccinfo.hash


if __name__ == "__main__":
    # command-line arguments
    parser = ArgumentParser(description=DESC)
    parser.add_argument(
        "path",
        type=Path,
        help="Path to Python source containing desired kernel in its scope. File will be executed.",
    )
    parser.add_argument(
        "--kernel-name",
        "-n",
        type=str,
        default="",
        help="Name of the kernel to compile",
        required=True,
    )
    parser.add_argument(
        "--num-warps",
        "-w",
        type=int,
        default=4,
        help="Number of warps to launch the kernel",
    )
    parser.add_argument(
        "--num-stages",
        "-ns",
        type=int,
        default=3,
        help="Number of stages (meta-parameter of the kernel)",
    )
    parser.add_argument(
        "--signature", "-s", type=str, help="Signature of the kernel", required=True
    )
    args = parser.parse_args()

    # execute python sources and extract functions wrapped in JITFunction
    arg_path = Path(args.path).expanduser()

    spec = importlib.util.spec_from_file_location(arg_path.stem, arg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    jitfn: triton.JITFunction = getattr(mod, args.kernel_name)

    kernel = compile_a_kernel(jitfn, args.signature, args.num_warps, args.num_stages)
    print(kernel[1])
