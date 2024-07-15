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

    std: float
    r"""The standard deviation of the noise.

    The noise standard deviation $\sigma_{\text{denoise}}$ denotes the standard deviation of Gaussian noise added to each xyz component of atomic coordinates."""

    denoising_prob: float
    r"""The denoising probability.
    The denoising probability $p_{\text{denoise}}$ denotes the probability of adding noise to atomic coordinates and optimizing for both the auxiliary task and the original task.
    Using $p_{\text{denoise}} < 1$ enables taking original atomistic structures without any noise as inputs and optimizing for only the original task for some training iterations."""

    corrupt_ratio: float
    """The corruption ratio.
    The corrupt ratio $r_{\text{denoise}}$ denotes the ratio of the number of atoms, which we add noise to and denoise, to the total number of atoms.
    Using $r_{\text{denoise}} < 1$ allows only adding noise to and denoising a subset of atoms within a structure.
    """

    def apply_transform_(self, data: Data):
        assert data.pos is not None, "Data object does not have `pos`"

        # Get a random subset of the atoms to corrupt
        num_atoms = data.atomic_numbers.numel()
        num_corrupt_atoms = int(num_atoms * self.corrupt_ratio)
        corrupt_indices = torch.randperm(num_atoms)[:num_corrupt_atoms]

        # Compute the noise to add
        # With probability 1 - denoising_prob, don't add noise
        if np.random.rand() > self.denoising_prob:
            noise = torch.zeros_like(data.pos)
        else:
            noise = torch.randn_like(data.pos) * self.std
            # Zero out the noise for the atoms that are not corrupted
            noise[corrupt_indices] = 0

        # Add the noise to the positions
        data.pos = data.pos + noise

        # Store the noise in the data object
        assert not hasattr(
            data, "pos_noise"
        ), "Data object already has a pos_noise attribute"
        data.pos_noise = noise

    def apply_transform(self, data: Data):
        data = data.clone()
        self.apply_transform_(data)
        return data
