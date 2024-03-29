import math

import torch
import torch.nn as nn
from ll.typecheck import Float
from torch_scatter import scatter
from typing_extensions import override

from ...models.gemnet.layers.base_layers import Dense, ResidualLayer
from ..lora import LoraConfig
from ..scaling import ScaleFactor
from ._config import DLoraConfig


class AtomUpdateBlock(nn.Module):
    """
    Aggregate the message embeddings of the atoms

    Arguments
    ---------
    emb_size_atom: int
        Embedding size of the atoms.
    emb_size_edge: int
        Embedding size of the edges.
    emb_size_rbf: int
        Embedding size of the radial basis.
    nHidden: int
        Number of residual blocks.
    activation: callable/str
        Name of the activation function to use in the dense layers.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        activation: str | None = None,
        *,
        dropout: float | None,
    ):
        super().__init__()

        self.dense_rbf = Dense(
            emb_size_rbf,
            emb_size_edge,
            activation=None,
            bias=False,
            dropout=dropout,
            lora=LoraConfig.disabled(),
        )
        self.scale_sum = ScaleFactor()

        self.layers = self.get_mlp(
            emb_size_edge,
            emb_size_atom,
            nHidden,
            activation,
            dropout,
        )

    def get_mlp(
        self,
        units_in: int,
        units: int,
        nHidden: int,
        activation: str | None,
        dropout: float | None,
    ):
        mlp: list[nn.Module]
        if units_in != units:
            dense1 = Dense(
                units_in,
                units,
                activation=activation,
                bias=False,
                dropout=dropout,
                lora=LoraConfig.disabled(),
            )
            mlp = [dense1]
        else:
            mlp = []
        res = [
            ResidualLayer(
                units,
                nLayers=2,
                activation=activation,
                dropout=dropout,
                lora=LoraConfig.disabled(),
            )
            for i in range(nHidden)
        ]
        mlp.extend(res)
        return nn.ModuleList(mlp)

    @override
    def forward(self, h, m, basis_rad, idx_atom):
        raise NotImplementedError


class OutputBlock(AtomUpdateBlock):
    """
    Combines the atom update block and subsequent final dense layer.

    Arguments
    ---------
    emb_size_atom: int
        Embedding size of the atoms.
    emb_size_edge: int
        Embedding size of the edges.
    emb_size_rbf: int
        Embedding size of the radial basis.
    nHidden: int
        Number of residual blocks before adding the atom embedding.
    nHidden_afteratom: int
        Number of residual blocks after adding the atom embedding.
    activation: str
        Name of the activation function to use in the dense layers.
    direct_forces: bool
        If true directly predict forces, i.e. without taking the gradient
        of the energy potential.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        nHidden_afteratom: int,
        activation: str | None = None,
        direct_forces: bool = True,
        *,
        edge_dropout: float | None,
        dropout: float | None,
    ):
        super().__init__(
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=nHidden,
            activation=activation,
            dropout=dropout,
        )

        self.direct_forces = direct_forces
        self.edge_dropout = edge_dropout

        self.seq_energy_pre = self.layers  # inherited from parent class
        if nHidden_afteratom >= 1:
            self.seq_energy2 = self.get_mlp(
                emb_size_atom,
                emb_size_atom,
                nHidden_afteratom,
                activation,
                dropout,
            )
            self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        else:
            self.seq_energy2 = None

        if self.direct_forces:
            self.scale_rbf_F = ScaleFactor()
            self.seq_forces = self.get_mlp(
                emb_size_edge,
                emb_size_edge,
                nHidden,
                activation,
                dropout,
            )
            self.dense_rbf_F = Dense(
                emb_size_rbf,
                emb_size_edge,
                activation=None,
                bias=False,
                dropout=dropout,
                lora=LoraConfig.disabled(),
            )

    def _drop_edge_boost_activations(self, x: torch.Tensor):
        if not self.training or not self.edge_dropout:
            return x

        x = x / (1 - self.edge_dropout)
        return x

    def forward(
        self,
        h,
        m,
        basis_rad,
        idx_atom,
    ):
        """
        Returns
        -------
        torch.Tensor, shape=(nAtoms, emb_size_atom)
            Output atom embeddings.
        torch.Tensor, shape=(nEdges, emb_size_edge)
            Output edge embeddings.
        """
        nAtoms = h.shape[0]

        # ------------------------ Atom embeddings ------------------------ #
        basis_emb_E = self.dense_rbf(basis_rad)  # (nEdges, emb_size_edge)
        x = m * basis_emb_E

        x_E = scatter(
            x, idx_atom, dim=0, dim_size=nAtoms, reduce="sum"
        )  # (nAtoms, emb_size_edge)

        x_E = self._drop_edge_boost_activations(x_E)

        x_E = self.scale_sum(x_E, ref=m)

        for layer in self.seq_energy_pre:
            x_E = layer(x_E)  # (nAtoms, emb_size_atom)

        if self.seq_energy2 is not None:
            x_E = x_E + h
            x_E = x_E * self.inv_sqrt_2
            for layer in self.seq_energy2:
                x_E = layer(x_E)  # (nAtoms, emb_size_atom)

        # ------------------------- Edge embeddings ------------------------ #
        if self.direct_forces:
            x_F = m
            for i, layer in enumerate(self.seq_forces):
                x_F = layer(x_F)  # (nEdges, emb_size_edge)

            basis_emb_F = self.dense_rbf_F(basis_rad)
            # (nEdges, emb_size_edge)
            x_F_basis = x_F * basis_emb_F
            x_F = self.scale_rbf_F(x_F_basis, ref=x_F)
        else:
            x_F = 0
        # ------------------------------------------------------------------ #

        return x_E, x_F
