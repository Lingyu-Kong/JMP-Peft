import itertools
import logging
import math
from collections.abc import Callable
from contextlib import contextmanager
from typing import TypedDict

import torch
import torch.nn as nn


class _Stats(TypedDict):
    variance_in: float
    variance_out: float
    n_samples: int


IndexFn = Callable[[], None]


def _check_consistency(old: torch.Tensor, new: torch.Tensor, key: str):
    if not torch.allclose(old, new):
        raise ValueError(
            f"Scale factor parameter {key} is inconsistent with the loaded state dict.\n"
            f"Old: {old}\n"
            f"Actual: {new}"
        )


class ScaleFactor(nn.Module):
    scale_factor: torch.Tensor

    dim_size: int | None = None
    name: str | None = None
    index_fn: IndexFn | None = None
    stats: _Stats | None = None

    def __init__(
        self,
        name: str | None = None,
        enforce_consistency: bool = True,
        dim_size: int | None = None,
    ):
        super().__init__()

        self.name = name
        self.dim_size = dim_size

        self.index_fn = None
        self.stats = None
        self.register_buffer("scale_factor", torch.tensor(0.0))
        if enforce_consistency:
            _ = self._register_load_state_dict_pre_hook(self._enforce_consistency)

    def _enforce_consistency(
        self,
        state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ):
        if not self.fitted:
            return

        persistent_buffers = {
            k: v
            for k, v in self._buffers.items()
            if k not in self._non_persistent_buffers_set
        }
        local_name_params = itertools.chain(
            self._parameters.items(), persistent_buffers.items()
        )
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key not in state_dict:
                continue

            input_param = state_dict[key]
            _check_consistency(old=param, new=input_param, key=key)

    @property
    def fitted(self):
        return bool((self.scale_factor != 0.0).item())

    @torch.jit.unused
    def reset_(self):
        self.scale_factor.zero_()

    @torch.jit.unused
    def set_(self, scale: float | torch.Tensor):
        if self.fitted:
            _check_consistency(
                old=self.scale_factor,
                new=torch.tensor(scale) if isinstance(scale, (float, int)) else scale,
                key="scale_factor",
            )
        self.scale_factor.fill_(scale)

    @torch.jit.unused
    def initialize_(self, *, index_fn: IndexFn | None = None):
        self.index_fn = index_fn

    @contextmanager
    @torch.jit.unused
    def fit_context_(self):
        self.stats = _Stats(variance_in=0.0, variance_out=0.0, n_samples=0)
        yield
        del self.stats
        self.stats = None

    @torch.jit.unused
    def fit_(self):
        assert self.stats, "Stats not set"
        for k, v in self.stats.items():
            assert v > 0, f"{k} is {v}"

        self.stats["variance_in"] = self.stats["variance_in"] / self.stats["n_samples"]
        self.stats["variance_out"] = (
            self.stats["variance_out"] / self.stats["n_samples"]
        )

        ratio = self.stats["variance_out"] / self.stats["variance_in"]
        value = math.sqrt(1 / ratio)

        self.set_(value)

        stats = dict(**self.stats)
        return stats, ratio, value

    @torch.no_grad()
    @torch.jit.unused
    def _observe(self, x: torch.Tensor, ref: torch.Tensor | None = None):
        if self.stats is None:
            logging.debug("Observer not initialized but self.observe() called")
            return

        n_samples = x.shape[0]
        self.stats["variance_out"] += torch.mean(torch.var(x, dim=0)).item() * n_samples

        if ref is None:
            self.stats["variance_in"] += n_samples
        else:
            self.stats["variance_in"] += (
                torch.mean(torch.var(ref, dim=0)).item() * n_samples
            )
        self.stats["n_samples"] += n_samples

    def forward(
        self,
        x: torch.Tensor,
        *,
        ref: torch.Tensor | None = None,
    ):
        if self.index_fn is not None:
            self.index_fn()

        if self.fitted:
            x = x * self.scale_factor

        if not torch.jit.is_scripting():
            self._observe(x, ref=ref)

        if self.dim_size is not None:
            assert (
                x.shape[-1] == self.dim_size
            ), f"Expected {self.dim_size} but got {x.shape[-1]=}"

        return x
