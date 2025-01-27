import nshtrainer.ll as ll
import torch
from jmppeft.modules.torch_scatter_polyfill import scatter
from nshtrainer.ll.typecheck import Float, Int, tassert
from typing_extensions import override

from ...models.gemnet.layers.atom_update_block import OutputBlock
from ..lora import LoraConfig
from ._adapter import (
    AdapterLayer,
    AdapterOutput,
    run_adapters,
    run_adapters_existing_output,
)
from ._config import DLoraConfig


class DLoraOutputBlock(OutputBlock):
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
        dlora: DLoraConfig,
        edge_dropout: float | None,
        dropout: float | None,
        lora: LoraConfig,
    ):
        assert not lora.enabled, "DLoraOutputBlock does not support LoraConfig"

        super().__init__(
            emb_size_atom,
            emb_size_edge,
            emb_size_rbf,
            nHidden,
            nHidden_afteratom,
            activation,
            direct_forces,
            edge_dropout=edge_dropout,
            dropout=dropout,
            lora=lora,
        )

        # Initialize DLora Adapter layers
        self.dlora = dlora

        self.seq_energy2_adapters = ll.nn.TypedModuleList(
            [
                AdapterLayer(self.dlora.seq_energy2_output_block)
                for _ in range(self.dlora.num_heads)
            ]
        )

        if self.dlora.seq_energy_pre_output_block is not None:
            self.seq_energy_pre_adapters = ll.nn.TypedModuleList(
                [
                    AdapterLayer(self.dlora.seq_energy_pre_output_block)
                    for _ in range(self.dlora.num_heads)
                ]
            )

        if self.direct_forces:
            assert (
                self.dlora.seq_forces_output_block is not None
            ), "force_output_block must be provided when direct_forces=True"

            self.seq_forces_adapters = ll.nn.TypedModuleList(
                [
                    AdapterLayer(self.dlora.seq_forces_output_block)
                    for _ in range(self.dlora.num_heads)
                ]
            )

    @override
    def forward(
        self,
        h: Float[torch.Tensor, "num_atoms emb_size_atom"],
        m: Float[torch.Tensor, "num_edges emb_size_edge"],
        basis_rad: Float[torch.Tensor, "num_edges emb_size_rbf"],
        idx_atom: Int[torch.Tensor, "num_edges"],
    ) -> tuple[AdapterOutput, AdapterOutput | None]:
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
        tassert(Float[torch.Tensor, "nAtoms emb_size_edge"], x_E)

        if self.dlora.seq_energy_pre_output_block is None:
            for layer in self.seq_energy_pre:
                x_E = layer(x_E)  # (nAtoms, emb_size_atom)
        else:
            x_E = run_adapters(
                x_E,
                config=self.dlora,
                adapters=self.seq_energy_pre_adapters,
                original_mlp=self.seq_energy_pre,
            )

        if self.seq_energy2 is not None:
            if not isinstance(x_E, AdapterOutput):
                x_E = x_E + h
                x_E = x_E * self.inv_sqrt_2
            else:
                x_E = x_E.vmap_output(lambda x_E: (x_E + h) * self.inv_sqrt_2)

            if self.dlora.seq_energy2_output_block is None:
                # If seq_energy_pre adapters were used, we need to vectorize this.
                if not isinstance(x_E, AdapterOutput):
                    for layer in self.seq_energy2:
                        x_E = layer(x_E)  # (nAtoms, emb_size_atom)
                else:
                    raise NotImplementedError(
                        "You need to either vectorize both adapters, the last adapter, or none."
                    )
            else:
                if isinstance(x_E, AdapterOutput):
                    x_E = run_adapters_existing_output(
                        x_E,
                        config=self.dlora,
                        adapters=self.seq_energy2_adapters,
                        original_mlp=self.seq_energy2,
                    )
                else:
                    x_E = run_adapters(
                        x_E,
                        config=self.dlora,
                        adapters=self.seq_energy2_adapters,
                        original_mlp=self.seq_energy2,
                    )

            # Make sure we return an AdapterOutput
            if not isinstance(x_E, AdapterOutput):
                x_E = AdapterOutput.from_single_output(x_E)
        else:
            raise ValueError("seq_energy2 is None. Needed for DLoraOutputBlock")

        # ------------------------- Edge embeddings ------------------------ #
        if self.direct_forces:
            x_F = m
            tassert(Float[torch.Tensor, "num_edges emb_size_edge"], x_F)

            if self.dlora.seq_forces_output_block is None:
                for i, layer in enumerate(self.seq_forces):
                    x_F = layer(x_F)  # (nEdges, emb_size_edge)

                basis_emb_F = self.dense_rbf_F(basis_rad)
                # (nEdges, emb_size_edge)
                x_F_basis = x_F * basis_emb_F
                x_F = self.scale_rbf_F(x_F_basis, ref=x_F)

                x_F = AdapterOutput.from_single_output(x_F)
            else:
                x_F = run_adapters(
                    x_F,
                    config=self.dlora,
                    adapters=self.seq_forces_adapters,
                    original_mlp=self.seq_forces,
                )

                basis_emb_F = self.dense_rbf_F(basis_rad)
                tassert(Float[torch.Tensor, "num_edges emb_size_edge"], basis_emb_F)
                tassert(
                    Float[torch.Tensor, "num_adapters num_edges emb_size_edge"],
                    x_F.output,
                )

                # x_F_basis = x_F * basis_emb_F
                x_F_basis = x_F.vmap_output(lambda x_F: x_F * basis_emb_F)
                x_F = x_F.map_output(
                    lambda x_F: self.scale_rbf_F(x_F_basis.output, ref=x_F)
                )
        else:
            x_F = None
        # ------------------------------------------------------------------ #

        return x_E, x_F
