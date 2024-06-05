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
