import importlib.util
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import List

import triton


@dataclass
class Signature:
    num_args: int
    constexpr_indices: List[int]
    non_constexpr_indices: List[int]
    specialised_indices: List[int]


def static_signature(f: triton.runtime.JITFunction):
    args_names = f.arg_names
    arg_num = len(args_names)
    constexpr_indices = [i for (i, p) in enumerate(f.params) if p.is_constexpr]
    non_constexpr_indices = [i for (i, p) in enumerate(f.params) if not p.is_constexpr]
    specialised_indices = [
        i
        for (i, p) in enumerate(f.params)
        if (not p.do_not_specialize) and (not p.is_constexpr)
    ]
    return Signature(
        arg_num, constexpr_indices, non_constexpr_indices, specialised_indices
    )


def extract_static_signature(source_path, fn_name):
    source_path = Path(source_path)
    spec = importlib.util.spec_from_file_location(source_path.stem, source_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, fn_name)

    # unwrap JITFunction from Autotuner or Heuristics, contarct: decorated fn is stored in the fn attribute
    # while not (type(fn) is triton.runtime.JITFunction):
    #     fn = fn.fn

    sig = static_signature(fn)

    # convert to list of int for c++ processing
    arg_types = []
    for i in range(sig.num_args):
        if i in sig.constexpr_indices:
            arg_types.append(2)
        elif i in sig.specialised_indices:
            arg_types.append(1)
        else:  # non-specialzed
            arg_types.append(0)
    return arg_types


if __name__ == "__main__":
    # command-line arguments
    parser = ArgumentParser(
        description="generate server-side signature, that is, static part of the full signature"
    )
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

    args = parser.parse_args()

    # execute python sources and extract functions wrapped in JITFunction
    arg_path = Path(args.path).expanduser()
    arg_types = extract_static_signature(arg_path, args.kernel_name)

    print(arg_types)
