from flag_gems.utils.pointwise_dynamic import KernelGenerator,FunctionSchema
from flag_gems.utils.codegen_config_utils import CodeGenConfig,get_codegen_config
from flag_gems.utils.code_utils import IndentedBuffer,write_atomic
from flag_gems.utils.code_cache import code_cache_dir
import triton
from pathlib import Path

def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import math")
    code.writeline("from typing import Union")
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("from triton import language as tl")
    code.newline()
    code.writeline("from flag_gems.utils.shape_utils import (")
    code.writeline("    heuristics_for_tile_size,")
    code.writeline("    heuristics_for_num_warps,")
    code.writeline("    stride_order,")
    code.writeline(")")
    code.writeline("from flag_gems.utils.tensor_wrapper import StridedBuffer")
    code.writeline("from flag_gems.utils.libentry import libentry, libtuner")
    code.writeline("from flag_gems.utils import triton_lang_extension as tle")
    code.writeline("from flag_gems.runtime import torch_device_fn")
    code.newline()
    code.newline()
    return code

def validate_file_path(file_name):
    file_path = code_cache_dir() / file_name
    if Path(file_path).exists():
        print(f"File exists: {file_path}")
    else:
        print(f"File does not exist: {file_path}")

def gen_add(ndim,is_tensor=None,dtypes=None):
    @triton.jit
    def add_func(x, y, alpha):
        return x + y * alpha
    op_desc = FunctionSchema(
                num_inputs=2,
                is_tensor=is_tensor,
                dtypes=dtypes,
                num_outputs=1,
                promotion_methods=[(0, 1, "DEFAULT")],
            )
    config = get_codegen_config()
    config.prefer_1d_tile = True
    kernel_gen = KernelGenerator(
        op_desc, add_func, ndim, "add_func", config
    )

    code = IndentedBuffer()
    code = generate_imports(code)
    code = kernel_gen.codegen_1d_tile(code)
    
    kernel_name = f"add_func_kernel_rank_{ndim}"

    file_name = (
        f"pointwise_dynamic_{add_func.cache_key}_{kernel_name}_"
        f"{'1d_tile_' if config.prefer_1d_tile else ''}"
        f"{'bptr' if (not config.prefer_1d_tile and config.prefer_block_pointer) else ''}"
        ".py"
    )
    # TODO save code to file_name
    file_path = code_cache_dir() / file_name
    write_atomic(file_path, code.getvalue())

    return kernel_name, str(file_path)

'''
kernel_name, file_name = gen_add(1)  
validate_file_path(file_name) 
'''
