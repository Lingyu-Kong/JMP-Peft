import torch

ENABLE_POLYFILL = False
if not ENABLE_POLYFILL:
    from torch_scatter import scatter as scatter
    from torch_scatter import scatter_add as scatter_add
    from torch_scatter import scatter_max as scatter_max
    from torch_scatter import scatter_mean as scatter_mean
    from torch_scatter import scatter_min as scatter_min
    from torch_scatter import scatter_mul as scatter_mul
    from torch_scatter import scatter_sum as scatter_sum
    from torch_scatter import segment_coo as segment_coo
    from torch_scatter import segment_csr as segment_csr
else:

    def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
        if dim < 0:
            dim = other.dim() + dim
        if src.dim() == 1:
            for _ in range(0, dim):
                src = src.unsqueeze(0)
        for _ in range(src.dim(), other.dim()):
            src = src.unsqueeze(-1)
        src = src.expand(other.size())
        return src

    def scatter_sum(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = -1,
        out: torch.Tensor | None = None,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        index = broadcast(index, src, dim)
        if out is None:
            size = list(src.size())
            if dim_size is not None:
                size[dim] = dim_size
            elif index.numel() == 0:
                size[dim] = 0
            else:
                size[dim] = int(index.max()) + 1
            out = torch.zeros(size, dtype=src.dtype, device=src.device)
            return out.scatter_add_(dim, index, src)
        else:
            return out.scatter_add_(dim, index, src)

    def scatter_add(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = -1,
        out: torch.Tensor | None = None,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        return scatter_sum(src, index, dim, out, dim_size)

    def scatter_mul(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = -1,
        out: torch.Tensor | None = None,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def scatter_mean(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = -1,
        out: torch.Tensor | None = None,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        out = scatter_sum(src, index, dim, out, dim_size)
        dim_size = out.size(dim)

        index_dim = dim
        if index_dim < 0:
            index_dim = index_dim + src.dim()
        if index.dim() <= index_dim:
            index_dim = index.dim() - 1

        ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
        count = scatter_sum(ones, index, index_dim, None, dim_size)
        count[count < 1] = 1
        count = broadcast(count, out, dim)
        if out.is_floating_point():
            out.true_divide_(count)
        else:
            out.div_(count, rounding_mode="floor")
        return out

    def scatter_min(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = -1,
        out: torch.Tensor | None = None,
        dim_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def scatter_max(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = -1,
        out: torch.Tensor | None = None,
        dim_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def scatter(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = -1,
        out: torch.Tensor | None = None,
        dim_size: int | None = None,
        reduce: str = "sum",
    ) -> torch.Tensor:
        r"""
        |

        .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
                master/docs/source/_figures/add.svg?sanitize=true
            :align: center
            :width: 400px

        |

        Reduces all values from the :attr:`src` tensor into :attr:`out` at the
        indices specified in the :attr:`index` tensor along a given axis
        :attr:`dim`.
        For each value in :attr:`src`, its output index is specified by its index
        in :attr:`src` for dimensions outside of :attr:`dim` and by the
        corresponding value in :attr:`index` for dimension :attr:`dim`.
        The applied reduction is defined via the :attr:`reduce` argument.

        Formally, if :attr:`src` and :attr:`index` are :math:`n`-dimensional
        tensors with size :math:`(x_0, ..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})`
        and :attr:`dim` = `i`, then :attr:`out` must be an :math:`n`-dimensional
        tensor with size :math:`(x_0, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})`.
        Moreover, the values of :attr:`index` must be between :math:`0` and
        :math:`y - 1`, although no specific ordering of indices is required.
        The :attr:`index` tensor supports broadcasting in case its dimensions do
        not match with :attr:`src`.

        For one-dimensional tensors with :obj:`reduce="sum"`, the operation
        computes

        .. math::
            \mathrm{out}_i = \mathrm{out}_i + \sum_j~\mathrm{src}_j

        where :math:`\sum_j` is over :math:`j` such that
        :math:`\mathrm{index}_j = i`.

        .. note::

            This operation is implemented via atomic operations on the GPU and is
            therefore **non-deterministic** since the order of parallel operations
            to the same value is undetermined.
            For floating-point variables, this results in a source of variance in
            the result.

        :param src: The source tensor.
        :param index: The indices of elements to scatter.
        :param dim: The axis along which to index. (default: :obj:`-1`)
        :param out: The destination tensor.
        :param dim_size: If :attr:`out` is not given, automatically create output
            with size :attr:`dim_size` at dimension :attr:`dim`.
            If :attr:`dim_size` is not given, a minimal sized output tensor
            according to :obj:`index.max() + 1` is returned.
        :param reduce: The reduce operation (:obj:`"sum"`, :obj:`"mul"`,
            :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`). (default: :obj:`"sum"`)

        :rtype: :class:`Tensor`

        .. code-block:: python

            from jmppeft.modules.torch_scatter_polyfill import scatter

            src = torch.randn(10, 6, 64)
            index = torch.tensor([0, 1, 0, 1, 2, 1])

            # Broadcasting in the first and last dim.
            out = scatter(src, index, dim=1, reduce="sum")

            print(out.size())

        .. code-block::

            torch.Size([10, 3, 64])
        """
        if reduce == "sum" or reduce == "add":
            return scatter_sum(src, index, dim, out, dim_size)
        if reduce == "mul":
            return scatter_mul(src, index, dim, out, dim_size)
        elif reduce == "mean":
            return scatter_mean(src, index, dim, out, dim_size)
        elif reduce == "min":
            return scatter_min(src, index, dim, out, dim_size)[0]
        elif reduce == "max":
            return scatter_max(src, index, dim, out, dim_size)[0]
        else:
            raise ValueError

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
        return scatter(src, index, dim, out, dim_size, reduce)

    def segment_csr(
        src: torch.Tensor,
        indptr: torch.Tensor,
        out: torch.Tensor | None = None,
        reduce: str = "sum",
    ):
        # Dim should be the first (and only) non-broadcastable dimension in indptr.
        dims_to_squeeze: list[int] = []
        dim: int = -1
        for dim_idx in range(indptr.dim()):
            if indptr.size(dim_idx) == 1:
                dims_to_squeeze.append(dim)
                continue

            if dim != -1:
                raise ValueError(
                    "Found multiple non-broadcastable dimensions in index."
                )
            dim = dim_idx

        indptr = indptr.squeeze(dims_to_squeeze)

        # Convert CSR indptr to COO index.
        index_counts = indptr[1:] - indptr[:-1]
        index = torch.repeat_interleave(
            torch.arange(index_counts.size(0), device=indptr.device),
            index_counts,
            dim=0,
        )

        return scatter(src, index, dim, out, indptr.size(0) - 1, reduce)
