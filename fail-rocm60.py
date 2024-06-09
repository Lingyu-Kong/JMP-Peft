# %%
import torch
from torch_scatter import segment_coo

src = torch.randn(10, 6, 64).cuda().contiguous()
index = torch.tensor([0, 0, 1, 1, 1, 2]).cuda().contiguous()
index = index.view(1, -1)  # Broadcasting in the first and last dim.
index = index.contiguous()
print(index.layout, src.layout)

# %%
out = segment_coo(src.cpu(), index.cpu(), reduce="sum")
print(out.size())

out = segment_coo(src, index, reduce="sum")

print(out.size())

# %%
