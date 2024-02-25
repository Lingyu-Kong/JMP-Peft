import torch
from torch_geometric.data.data import BaseData


def atomref_transform(
    data: BaseData,
    refs: dict[str, torch.Tensor],
    keep_raw: bool = False,
):
    z: torch.Tensor = data.atomic_numbers
    for target, coeffs in refs.items():
        value = getattr(data, target)
        if keep_raw:
            setattr(data, f"{target}_raw", value.clone())
        value = value - coeffs[z].sum()
        setattr(data, target, value)

    return data
