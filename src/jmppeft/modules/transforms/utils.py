from collections.abc import Callable

from torch_geometric.data.data import BaseData

Transform = Callable[[BaseData], BaseData]


def compose(transforms: list[Transform]):
    def composed(data: BaseData):
        for transform in transforms:
            data = transform(data)
        return data

    return composed
