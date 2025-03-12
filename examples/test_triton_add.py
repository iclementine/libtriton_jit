import torch
from flaggems import c_operators  # noqa: F401

x = torch.randn(3, 10, device="cuda")
y = torch.randn(10, device="cuda")

out_ref = x + y
out_hyp = torch.ops.flaggems.add_tensor(x, y)
print("ATEN:\n", out_ref)
print("FLAGGEMS:\n", out_hyp)
