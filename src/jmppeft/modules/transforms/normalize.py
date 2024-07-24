from collections.abc import Mapping
from logging import getLogger
from typing import cast

import numpy as np
import torch
from nshtrainer.ll import TypedConfig
from torch_geometric.data.data import BaseData
from typing_extensions import TypeVar

log = getLogger(__name__)

T = TypeVar("T", float, torch.Tensor, np.ndarray, infer_variance=True)


def _process_value(value: T) -> torch.Tensor:
    return cast(
        torch.Tensor,
        torch.tensor(value) if not torch.is_tensor(value) else value,
    )


class NormalizationConfig(TypedConfig):
    mean: float = 1.0
    std: float = 1.0

    def normalize(self, value: T) -> T:
        return (value - self.mean) / self.std

    def denormalize(self, value: T) -> T:
        return (value * self.std) + self.mean


def normalize(properties: Mapping[str, NormalizationConfig]):
    def _normalize(data: BaseData):
        nonlocal properties

        for key, d in properties.items():
            if (value := getattr(data, key, None)) is None:
                raise ValueError(f"Property {key} not found in data")

            value = _process_value(value)
            new_value = d.normalize(value)
            # log.critical(f"Normalized {key}: {value} -> {new_value}.")
            value = new_value
            setattr(data, key, value)
            setattr(data, f"{key}_norm_mean", torch.full_like(value, d.mean))
            setattr(data, f"{key}_norm_std", torch.full_like(value, d.std))

        return data

    return _normalize


def denormalize_batch(
    batch: BaseData,
    additional_tensors: dict[str, torch.Tensor] | None = None,
):
    if additional_tensors is None:
        additional_tensors = {}

    keys: set[str] = set(batch.keys())

    # find all keys that have a denorm_mean and denorm_std
    norm_keys: set[str] = {
        key.replace("_norm_mean", "") for key in keys if key.endswith("_norm_mean")
    } & {key.replace("_norm_std", "") for key in keys if key.endswith("_norm_std")}

    for key in norm_keys:
        mean = getattr(batch, f"{key}_norm_mean")
        std = getattr(batch, f"{key}_norm_std")
        value = getattr(batch, key)

        new_value = (value * std) + mean
        # log.critical(
        #     f"Denormalized batch {key}: {value} -> {new_value}. {additional_tensors.keys()=}"
        # )
        setattr(batch, key, new_value)

        if (additional_value := additional_tensors.pop(key, None)) is not None:
            new_value = (additional_value * std) + mean
            # log.critical(
            #     f"Denormalized additional tensor {key}: {additional_value} -> {new_value}"
            # )
            additional_tensors[key] = new_value

    return batch, additional_tensors
