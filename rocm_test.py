# %%
import torch

x = torch.tensor([7056, 7056], dtype=torch.long)
print(x)
x = x.cuda()
print(x)
x = x.cumsum(0)
print(x)


# %%
def monkey_patch_torch_scatter():
    import torch
    import torch_scatter

    def segment_coo(
        src: torch.Tensor,
        index: torch.Tensor,
        out: torch.Tensor | None = None,
        dim_size: int | None = None,
        reduce: str = "sum",
    ):
        # Dim should be the first (and only) non-broadcastable dimension in index.
        dims_to_squeeze: list[int] = []
        dim: int = -1
        for dim_idx in range(index.dim()):
            if index.size(dim_idx) == 1:
                dims_to_squeeze.append(dim)
                continue

            if dim != -1:
                raise ValueError(
                    "Found multiple non-broadcastable dimensions in index."
                )
            dim = dim_idx

        index = index.squeeze(dims_to_squeeze)
        return torch_scatter.scatter(src, index, dim, out, dim_size, reduce)

    torch_scatter.segment_coo = segment_coo

    print("Monkey-patched torch_scatter.segment_coo")


# monkey_patch_torch_scatter()

# %%
from torch_scatter import segment_csr

src = torch.randn(10, 6, 64).cuda()
indptr = torch.tensor([0, 2, 5, 6]).cuda()
indptr = indptr.view(1, -1)  # Broadcasting in the first and last dim.

out = segment_csr(src, indptr, reduce="sum")

print(out.size())

# %%
from torch_scatter import segment_coo

src = torch.randn(10, 6, 64).cuda().contiguous()
index = torch.tensor([0, 0, 1, 1, 1, 2]).cuda().contiguous()
index = index.view(1, -1)  # Broadcasting in the first and last dim.

out = segment_coo(src, index, reduce="sum")

print(out.size())

# %%
# Torch Scatter Tests
from torch_scatter import scatter_add, segment_coo, segment_csr

src = torch.randn(10, 6, 64).cuda()

index = torch.tensor([0, 0, 1, 1, 1, 2]).cuda()
out_scatter = scatter_add(src, index, dim=1)
print(out_scatter.shape)

index = torch.tensor([0, 0, 1, 1, 1, 2]).cuda()
index = index.view(1, -1)  # Broadcasting in the first and last dim.

out_coo = segment_coo(src, index, reduce="sum")
print(out_coo.shape)

indptr = torch.tensor([0, 2, 5, 6]).cuda()
indptr = indptr.view(1, -1)  # Broadcasting in the first and last dim.

out_csr = segment_csr(src, indptr, reduce="sum")

torch.testing.assert_allclose(out_scatter, out_coo)
torch.testing.assert_allclose(out_scatter, out_csr)

# %%
import torch
from torch_sparse import coalesce

index = torch.tensor([[1, 0, 1, 0, 2, 1], [0, 1, 1, 1, 0, 0]]).cuda()
value = torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]).cuda()

index, value = coalesce(index, value, m=3, n=2)

# %%
import torch
from torch_sparse import transpose

index = torch.tensor([[1, 0, 1, 0, 2, 1], [0, 1, 1, 1, 0, 0]]).cuda()
value = torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]).cuda()

index, value = transpose(index, value, 3, 2)

# %%
import torch
from torch_sparse import spmm

index = torch.tensor([[0, 0, 1, 2, 2], [0, 2, 1, 0, 1]]).cuda()
value = torch.tensor([1, 2, 4, 1, 3]).cuda()
matrix = torch.tensor([[1, 4], [2, 5], [3, 6]]).cuda()

out = spmm(index, value, 3, 3, matrix)

# %%
# import torch
# from torch_sparse import spspmm

# indexA = torch.tensor([[0, 0, 1, 2, 2], [1, 2, 0, 0, 1]]).cudA()
# valueA = torch.tensor([1, 2, 3, 4, 5]).cudA()

# indexB = torch.tensor([[0, 2], [1, 0]]).cudA()
# valueB = torch.tensor([2, 4]).cudA()

# indexC, valueC = spspmm(indexA, valueA, indexB, valueB, 3, 3, 2)

# %%
from torch_sparse import SparseTensor

row = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 4]).cuda()
col = torch.tensor([1, 2, 0, 2, 0, 1, 3, 2, 4, 3]).cuda()
num_edges = row.size(0)
num_nodes = int(max(row.max().item(), col.max().item())) + 1
adj = SparseTensor(
    row=row,
    col=col,
    value=torch.arange(num_edges).cuda(),
    sparse_sizes=(num_nodes, num_nodes),
)

in_ = adj.storage.value()
out_ = adj.storage.row()

print(in_.shape, out_.shape)
