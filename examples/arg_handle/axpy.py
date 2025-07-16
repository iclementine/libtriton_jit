import triton
from triton import language as tl


@triton.jit
def axpy_kernel(X, Y, Out, a, n, BLOCK_N: tl.constexpr):
    # ax + y
    pid = tl.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n

    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    o = a * x + y
    tl.store(Out + offsets, o, mask=mask)


@triton.jit
def axpy2_kernel(X, Y, Out, a, n, BLOCK_N: tl.constexpr):
    # ax + y
    pid = tl.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n

    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    if a is None:
        o = x + y
    else:
        o = a * x + y
    tl.store(Out + offsets, o, mask=mask)


@triton.jit
def axpy3_kernel(X, Y, Out, a, n, BLOCK_N: tl.constexpr):
    # ax + y
    pid = tl.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n

    x = tl.load(X + offsets, mask=mask)
    o = x
    if a is not None:
        o *= a

    if Y is not None:
        y = tl.load(Y + offsets, mask=mask)
        o += y
    tl.store(Out + offsets, o, mask=mask)
