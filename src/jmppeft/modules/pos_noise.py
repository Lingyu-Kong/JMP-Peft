from typing import Literal

import ll
import numpy as np
import torch
from torch_geometric.data import Data


class PositionNoiseAugmentationConfig(ll.TypedConfig):
    """
    Configuration class for adding noise to atomic coordinates.
    """

    name: Literal["pos_noise"] = "pos_noise"

    noise_std: float
    r"""The standard deviation of the noise.

    The noise standard deviation $\sigma_{\text{denoise}}$ denotes the standard deviation of Gaussian noise added to each xyz component of atomic coordinates."""

    system_corrupt_prob: float
    r"""The system corruption probability.
    The system corruption probability $p_{\text{denoise}}$ denotes the probability of adding noise to atomic coordinates and optimizing for both the auxiliary task and the original task.
    Using $p^{\text{system}}_{\text{denoise}} < 1$ enables taking original atomistic structures without any noise as inputs and optimizing for only the original task for some training iterations."""

    atom_corrupt_prob: float
    """The atom corruption probability.
    The atom corruption probability $r_{\text{denoise}}$ denotes the probability of adding noise to each atom in the structure.
    Using $p^{\text{atom}}_{\text{denoise}} < 1$ allows only adding noise to and denoising a subset of atoms within a structure.
    """

    def apply_transform_(self, data: Data):
        assert data.pos is not None, "Data object does not have `pos`"
        assert not hasattr(
            data, "pos_noise"
        ), "Data object already has a pos_noise attribute"

        device = data.pos.device

        # Compute the noise to add
        # With probability 1 - denoising_prob, don't add noise
        if np.random.rand() > self.system_corrupt_prob:
            noise = torch.zeros_like(data.pos, device=device)
        else:
            noise = torch.randn_like(data.pos, device=device) * self.noise_std

        # Zero out the noise for the atoms that are not corrupted
        num_atoms = data.atomic_numbers.numel()
        corrupt_mask = torch.rand((num_atoms,), device=device) < self.atom_corrupt_prob
        noise[corrupt_mask] = 0

        # Add the noise to the positions
        data.pos = data.pos + noise

        # Store the noise in the data object
        data.pos_noise = noise

    def apply_transform(self, data: Data):
        data = data.clone()
        self.apply_transform_(data)
        return data
