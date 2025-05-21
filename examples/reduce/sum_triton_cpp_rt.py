import torch

# in this way, we can skip building a python extension
torch.ops.load_library("libsum_op.so")

if __name__ == "__main__":
    x = torch.randn(16, 4 * 1024, device="cuda")
    result1 = torch.ops.my_ops.sum.dim_IntList(x, [1])
    result2 = torch.sum(x, [1])
    # print(result1)
    # print(result2)

    torch.cuda.synchronize()
    for _ in range(10):
        torch.sum(x, [1])
    torch.cuda.synchronize()
    for _ in range(10):
        torch.ops.my_ops.sum.dim_IntList(x, [1])
    torch.cuda.synchronize()
