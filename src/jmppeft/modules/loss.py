from abc import ABC, abstractmethod
from typing import Annotated, Literal, TypeAlias

import ll
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data.data import BaseData
from typing_extensions import assert_never, override

Reduction: TypeAlias = Literal["mean", "sum", "none"]


class LossConfigBase(ll.TypedConfig, ABC):
    @abstractmethod
    def compute(
        self,
        data: BaseData,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        reduction: Reduction = "mean",
    ) -> torch.Tensor: ...


class MAELossConfig(LossConfigBase):
    name: Literal["mae"] = "mae"

    @override
    def compute(
        self,
        data: BaseData,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        return F.l1_loss(y_pred, y_true, reduction=reduction)


class MSELossConfig(LossConfigBase):
    name: Literal["mse"] = "mse"

    @override
    def compute(
        self,
        data: BaseData,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        return F.mse_loss(y_pred, y_true, reduction=reduction)


class HuberLossConfig(LossConfigBase):
    name: Literal["huber"] = "huber"

    delta: float

    @override
    def compute(
        self,
        data: BaseData,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        return F.huber_loss(y_pred, y_true, delta=self.delta, reduction=reduction)


class MACEHuberLossConfig(LossConfigBase):
    name: Literal["mace_huber"] = "mace_huber"

    # steps: list[tuple[tuple[float, float], float]]

    # @classmethod
    # def mace_force_loss(cls, delta: float = 0.01):
    #     return cls(
    #         steps=[
    #             ((0.0, 100.0), delta),
    #             ((100.0, 200.0), 0.7 * delta),
    #             ((200.0, 300.0), 0.4 * delta),
    #             ((300.0, float("inf")), 0.1 * delta),
    #         ]
    #     )

    delta: float

    @override
    def compute(
        self,
        data: BaseData,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        # Define the multiplication factors for each condition
        factors = self.delta * np.array([1.0, 0.7, 0.4, 0.1])

        # Apply multiplication factors based on conditions
        c1 = torch.norm(y_true, dim=-1) < 100
        c2 = (torch.norm(y_true, dim=-1) >= 100) & (torch.norm(y_true, dim=-1) < 200)
        c3 = (torch.norm(y_true, dim=-1) >= 200) & (torch.norm(y_true, dim=-1) < 300)
        c4 = ~(c1 | c2 | c3)

        se = torch.zeros_like(y_pred)

        se[c1] = F.huber_loss(
            y_true[c1], y_pred[c1], reduction="none", delta=factors[0]
        )
        se[c2] = F.huber_loss(
            y_true[c2], y_pred[c2], reduction="none", delta=factors[1]
        )
        se[c3] = F.huber_loss(
            y_true[c3], y_pred[c3], reduction="none", delta=factors[2]
        )
        se[c4] = F.huber_loss(
            y_true[c4], y_pred[c4], reduction="none", delta=factors[3]
        )

        match reduction:
            case "mean":
                return se.mean()
            case "sum":
                return se.sum()
            case "none":
                return se
            case _:
                assert_never(reduction)


class MACEHuberEnergyLossConfig(LossConfigBase):
    name: Literal["mace_huber_energy"] = "mace_huber_energy"

    delta: float

    @override
    def compute(
        self,
        data: BaseData,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        assert (
            natoms := getattr(data, "natoms", None)
        ) is not None, "natoms is required"

        # First, divide the energy by the number of atoms
        y_pred = y_pred / natoms
        y_true = y_true / natoms

        # Compute the loss
        loss = F.huber_loss(y_pred, y_true, reduction=reduction, delta=self.delta)

        return loss


class L2MAELossConfig(LossConfigBase):
    name: Literal["l2mae"] = "l2mae"

    p: int | float = 2

    @override
    def compute(
        self,
        data: BaseData,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        loss = F.pairwise_distance(y_pred, y_true, p=self.p)

        match reduction:
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()
            case "none":
                return loss
            case _:
                assert_never(reduction)


LossConfig: TypeAlias = Annotated[
    MAELossConfig
    | MSELossConfig
    | HuberLossConfig
    | MACEHuberLossConfig
    | MACEHuberEnergyLossConfig
    | L2MAELossConfig,
    ll.Field(discriminator="name"),
]
