import torch

torch.ops.load_library("libadd_op.so")

if __name__ == "__main__":
    x = torch.randn(128 * 1024, device="cuda")
    y = torch.randn(128 * 1024, device="cuda")
    result1 = torch.ops.my_ops.add_tensor(x, y)
    result2 = torch.add(x, y)
    # print(result1)
    # print(result2)

    torch.cuda.synchronize()
    for _ in range(10):
        torch.add(x, y)
    torch.cuda.synchronize()
    for _ in range(10):
        torch.ops.my_ops.add_tensor(x, y)
    torch.cuda.synchronize()
