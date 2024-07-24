import numpy as np
import torch
import torch.nn as nn
from nshtrainer.ll.typecheck import Float, tassert
from typing_extensions import override

from ....modules.lora import LoraConfig, LoRALayer
from .efficient import BasisEmbedding


class LoRABasisEmbedding(BasisEmbedding, LoRALayer):
    @override
    def __init__(
        self,
        num_radial: int,
        emb_size_interm: int,
        num_spherical: int | None = None,
        *,
        lora: LoraConfig,
    ):
        assert lora is not None, "LoRABasisEmbedding requires a LoraConfig"
        assert lora.r > 0, "LoRABasisEmbedding requires a positive r"

        BasisEmbedding.__init__(
            self,
            num_radial,
            emb_size_interm,
            num_spherical,
            lora=LoraConfig.disabled(),
        )
        LoRALayer.__init__(self, **lora.as_kwargs())

        r"""
        Weight initialization code for the original BasisEmbedding class:
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        if num_spherical is None:
            self.weight = torch.nn.Parameter(
                torch.empty(emb_size_interm, num_radial),
                requires_grad=True,
            )
        else:
            self.weight = torch.nn.Parameter(
                torch.empty(num_radial, num_spherical, emb_size_interm),
                requires_grad=True,
            )

        Now, we want to initialize a low-rank weight matrix, \delta W.
        """

        self.lora = lora
        # Initialize LoRA parameters
        if num_spherical is not None:
            # LoRA A: (r, num_radial * num_spherical)
            # LoRA B: (emb_size_interm, r)
            self.lora_A = nn.Parameter(torch.empty(num_radial, num_spherical * lora.r))
            self.lora_B = nn.Parameter(torch.empty(lora.r, emb_size_interm))
        else:
            # LoRA A: (r, num_radial)
            # LoRA B: (emb_size_interm, r)
            self.lora_A = nn.Parameter(torch.empty(num_radial, lora.r))
            self.lora_B = nn.Parameter(torch.empty(lora.r, emb_size_interm))

        self.weight.requires_grad = False

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        # Reset LoRA parameters
        if self.lora:
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
            nn.init.zeros_(self.lora_B)

    @override
    def forward(
        self,
        rad_basis,
        sph_basis=None,
        idx_rad_outer=None,
        idx_rad_inner=None,
        idx_sph_outer=None,
        idx_sph_inner=None,
        num_atoms=None,
    ):
        """

        Arguments
        ---------
        rad_basis: torch.Tensor, shape=(num_edges, num_radial or num_orders * num_radial)
            Raw radial basis.
        sph_basis: torch.Tensor, shape=(num_triplets or num_quadruplets, num_spherical)
            Raw spherical or circular basis.
        idx_rad_outer: torch.Tensor, shape=(num_edges)
            Atom associated with each radial basis value.
            Optional, used for efficient edge aggregation.
        idx_rad_inner: torch.Tensor, shape=(num_edges)
            Enumerates radial basis values per atom.
            Optional, used for efficient edge aggregation.
        idx_sph_outer: torch.Tensor, shape=(num_triplets or num_quadruplets)
            Edge associated with each circular/spherical basis value.
            Optional, used for efficient triplet/quadruplet aggregation.
        idx_sph_inner: torch.Tensor, shape=(num_triplets or num_quadruplets)
            Enumerates circular/spherical basis values per edge.
            Optional, used for efficient triplet/quadruplet aggregation.
        num_atoms: int
            Total number of atoms.
            Optional, used for efficient edge aggregation.

        Returns
        -------
        rad_W1: torch.Tensor, shape=(num_edges, emb_size_interm, num_spherical)
        sph: torch.Tensor, shape=(num_edges, Kmax, num_spherical)
            Kmax = maximum number of neighbors of the edges
        """
        num_edges = rad_basis.shape[0]

        if self.num_spherical is not None:
            tassert(Float[torch.Tensor, "E R"], rad_basis)
            # MatMul: mul + sum over num_radial
            rad_W1 = torch.bmm(
                rad_basis[..., None, :],
                self.weight[None]
                .expand(num_edges, -1, -1, -1)
                .view(num_edges, self.weight.shape[0], -1),
            )
            tassert(Float[torch.Tensor, "E 1 interm_times_sph"], rad_W1)
            num_spherical = sph_basis.shape[-1]
            rad_W1 = rad_W1.view(num_edges, -1, num_spherical)
            tassert(Float[torch.Tensor, "E emb_size_interm num_spherical"], rad_W1)

            # Compute LoRA component
            if self.lora:
                x = self.lora_dropout(rad_basis[..., None, :])
                tassert(Float[torch.Tensor, "E 1 R"], x)
                rad_W1 += torch.bmm(
                    torch.bmm(x, self.lora_A[None].expand(num_edges, -1, -1)).view(
                        num_edges, num_spherical, -1
                    ),
                    self.lora_B[None].expand(num_edges, -1, -1),
                ).transpose(-2, -1)
        else:
            # MatMul: mul + sum over num_radial
            rad_W1 = rad_basis @ self.weight.T
            tassert(Float[torch.Tensor, "E interm"], rad_W1)
            # (num_edges, emb_size_interm)

            if self.lora:
                x = self.lora_dropout(rad_basis[..., None, :])
                tassert(Float[torch.Tensor, "E 1 R"], x)

                rad_W1 += torch.bmm(
                    torch.bmm(x, self.lora_A[None].expand(num_edges, -1, -1)).view(
                        num_edges, 1, -1
                    ),
                    self.lora_B[None].expand(num_edges, -1, -1),
                ).squeeze(-2)

        if idx_rad_inner is not None:
            # Zero padded dense matrix
            # maximum number of neighbors
            if idx_rad_outer.shape[0] == 0:
                # catch empty idx_rad_outer
                Kmax = 0
            else:
                Kmax = torch.max(idx_rad_inner) + 1

            rad_W1_padded = rad_W1.new_zeros([num_atoms, Kmax] + list(rad_W1.shape[1:]))
            rad_W1_padded[idx_rad_outer, idx_rad_inner] = rad_W1
            # (num_atoms, Kmax, emb_size_interm, ...)
            rad_W1_padded = torch.transpose(rad_W1_padded, 1, 2)
            # (num_atoms, emb_size_interm, Kmax, ...)
            rad_W1_padded = rad_W1_padded.reshape(num_atoms, rad_W1.shape[1], -1)
            # (num_atoms, emb_size_interm, Kmax2 * ...)
            rad_W1 = rad_W1_padded

        if idx_sph_inner is not None:
            # Zero padded dense matrix
            # maximum number of neighbors
            if idx_sph_outer.shape[0] == 0:
                # catch empty idx_sph_outer
                Kmax = 0
            else:
                Kmax = torch.max(idx_sph_inner) + 1

            sph2 = sph_basis.new_zeros(num_edges, Kmax, sph_basis.shape[-1])
            sph2[idx_sph_outer, idx_sph_inner] = sph_basis
            # (num_edges, Kmax, num_spherical)
            sph2 = torch.transpose(sph2, 1, 2)
            # (num_edges, num_spherical, Kmax)

        if sph_basis is None:
            return rad_W1
        else:
            if idx_sph_inner is None:
                rad_W1 = rad_W1[idx_sph_outer]
                # (num_triplets, emb_size_interm, num_spherical)

                sph_W1 = rad_W1 @ sph_basis[:, :, None]
                # (num_triplets, emb_size_interm, num_spherical)
                return sph_W1.squeeze(-1)
            else:
                return rad_W1, sph2
