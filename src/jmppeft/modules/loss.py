from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import ll
import ll.typecheck as tc
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data.data import BaseData
from torch_scatter import scatter
from typing_extensions import assert_never, override

Reduction: TypeAlias = Literal["mean", "sum", "none"]


class LossConfigBase(ll.TypedConfig, ABC):
    def compute(
        self,
        data: BaseData,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        loss = self.compute_impl(data, y_pred, y_true, reduction)
        return loss

    @abstractmethod
    def compute_impl(
        self,
        data: BaseData,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        reduction: Reduction = "mean",
    ) -> torch.Tensor: ...


class MAELossConfig(LossConfigBase):
    name: Literal["mae"] = "mae"

    @override
    def compute_impl(
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
    def compute_impl(
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
    def compute_impl(
        self,
        data: BaseData,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        return F.huber_loss(y_pred, y_true, delta=self.delta, reduction=reduction)


class MACEHuberLossConfig(LossConfigBase):
    name: Literal["mace_huber"] = "mace_huber"

    delta: float

    @override
    def compute_impl(
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


def _apply_focal_loss_per_atom(
    *,
    loss: tc.Float[torch.Tensor, "natoms"],
    freq_ratios: tc.Float[torch.Tensor, "atom_type"],
    atomic_numbers: tc.Int[torch.Tensor, "natoms"],
    gamma: float,
):
    # Apply the frequency factor to each sample & compute the mean frequency factor for each graph
    freq_factor = freq_ratios[atomic_numbers]
    tc.tassert(tc.Float[torch.Tensor, "natoms"], freq_factor)

    # Compute difficulty factor
    # We'll use the Huber loss value instead of MAE for the difficulty
    normalized_loss = loss / loss.max()
    difficulty = (1 - torch.exp(-normalized_loss)) ** gamma
    tc.tassert(tc.Float[torch.Tensor, "natoms"], difficulty)

    # Combine factors
    loss = loss * freq_factor * difficulty

    return loss


class MACEHuberForceFocalLossConfig(LossConfigBase):
    name: Literal["mace_huber_force_focal"] = "mace_huber_force_focal"

    delta: float  # Huber loss delta

    freq_ratios_path: Path
    gamma: float = 2.0  # Focal loss gamma

    @cached_property
    def freq_ratios(self) -> tc.Float[torch.Tensor, "atom_type"]:
        freq_ratios = np.load(self.freq_ratios_path)
        return torch.from_numpy(freq_ratios).float()

    @override
    def compute_impl(
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

        # Reduce over the 3 force components
        loss = se.mean(dim=-1)
        tc.tassert(tc.Float[torch.Tensor, "natoms"], loss)

        # Compute frequency factor for each sample based on its atom type
        loss = _apply_focal_loss_per_atom(
            loss=loss,
            freq_ratios=self.freq_ratios.to(loss.device),
            atomic_numbers=data.atomic_numbers,
            gamma=self.gamma,
        )
        tc.tassert(tc.Float[torch.Tensor, "natoms"], loss)

        match reduction:
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()
            case "none":
                return loss
            case _:
                assert_never(reduction)


class MACEHuberEnergyLossConfig(LossConfigBase):
    name: Literal["mace_huber_energy"] = "mace_huber_energy"

    delta: float

    @override
    def compute_impl(
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


def _apply_focal_loss_per_graph(
    *,
    loss: tc.Float[torch.Tensor, "bsz"],
    freq_ratios: tc.Float[torch.Tensor, "atom_type"],
    atomic_numbers: tc.Int[torch.Tensor, "natoms"],
    batch: tc.Int[torch.Tensor, "natoms"],
    gamma: float,
):
    # Apply the frequency factor to each sample & compute the mean frequency factor for each graph
    freq_factor = scatter(
        freq_ratios[atomic_numbers],
        batch,
        dim=0,
        dim_size=loss.shape[0],
        reduce="mean",
    )
    tc.tassert(tc.Float[torch.Tensor, "bsz"], freq_factor)

    # Compute difficulty factor
    # We'll use the Huber loss value instead of MAE for the difficulty
    normalized_loss = loss / loss.max()
    difficulty = (1 - torch.exp(-normalized_loss)) ** gamma
    tc.tassert(tc.Float[torch.Tensor, "bsz"], difficulty)

    # Combine factors
    loss = loss * freq_factor * difficulty

    return loss


class MACEHuberEnergyFocalLossConfig(LossConfigBase):
    name: Literal["mace_huber_energy_focal"] = "mace_huber_energy_focal"

    delta: float  # Huber loss delta

    freq_ratios_path: Path
    gamma: float = 2.0  # Focal loss gamma

    @cached_property
    def freq_ratios(self) -> tc.Float[torch.Tensor, "atom_type"]:
        freq_ratios = np.load(self.freq_ratios_path)
        return torch.from_numpy(freq_ratios).float()

    @override
    def compute_impl(
        self,
        data: BaseData,
        y_pred: tc.Float[torch.Tensor, "bsz"],
        y_true: tc.Float[torch.Tensor, "bsz"],
        reduction: Reduction = "mean",
    ) -> tc.Float[torch.Tensor, ""] | tc.Float[torch.Tensor, "bsz"]:
        assert (
            natoms := getattr(data, "natoms", None)
        ) is not None, "natoms is required"

        # First, divide the energy by the number of atoms
        y_pred = y_pred / natoms
        y_true = y_true / natoms

        # Compute the loss
        loss = F.huber_loss(y_pred, y_true, reduction="none", delta=self.delta)
        tc.tassert(tc.Float[torch.Tensor, "bsz"], loss)

        # Compute frequency factor for each sample based on its atom type
        loss = _apply_focal_loss_per_graph(
            loss=loss,
            freq_ratios=self.freq_ratios.to(loss.device),
            atomic_numbers=data.atomic_numbers,
            batch=data.batch,
            gamma=self.gamma,
        )
        tc.tassert(tc.Float[torch.Tensor, "bsz"], loss)

        match reduction:
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()
            case "none":
                return loss
            case _:
                assert_never(reduction)


class HuberStressFocalLossConfig(LossConfigBase):
    name: Literal["huber_stress_focal"] = "huber_stress_focal"

    delta: float  # Huber loss delta

    freq_ratios_path: Path
    gamma: float = 2.0  # Focal loss gamma

    @cached_property
    def freq_ratios(self) -> tc.Float[torch.Tensor, "atom_type"]:
        freq_ratios = np.load(self.freq_ratios_path)
        return torch.from_numpy(freq_ratios).float()

    @override
    def compute_impl(
        self,
        data: BaseData,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        # Compute the loss
        loss = F.huber_loss(y_pred, y_true, reduction="none", delta=self.delta)
        tc.tassert(tc.Float[torch.Tensor, "bsz 3 3"], loss)

        # Reduce over the 3x3 stress tensor
        loss = loss.mean(dim=(-2, -1))
        tc.tassert(tc.Float[torch.Tensor, "bsz"], loss)

        # Compute frequency factor for each sample based on its atom type
        loss = _apply_focal_loss_per_graph(
            loss=loss,
            freq_ratios=self.freq_ratios.to(loss.device),
            atomic_numbers=data.atomic_numbers,
            batch=data.batch,
            gamma=self.gamma,
        )
        tc.tassert(tc.Float[torch.Tensor, "bsz"], loss)

        match reduction:
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()
            case "none":
                return loss
            case _:
                assert_never(reduction)


class L2MAELossConfig(LossConfigBase):
    name: Literal["l2mae"] = "l2mae"

    p: int | float = 2

    @override
    def compute_impl(
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
    | MACEHuberForceFocalLossConfig
    | MACEHuberEnergyFocalLossConfig
    | HuberStressFocalLossConfig
    | L2MAELossConfig,
    ll.Field(discriminator="name"),
]
