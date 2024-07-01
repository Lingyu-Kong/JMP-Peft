"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging

import ll.typecheck as tc
import torch
import torch.nn as nn
from einops import rearrange
from lightning.fabric.utilities.distributed import _all_gather_ddp_if_available


class ForceScaler:
    """
    Scales up the energy and then scales down the forces
    to prevent NaNs and infs in calculations using AMP.
    Inspired by torch.cuda.amp.GradScaler.
    """

    def __init__(
        self,
        init_scale=2.0**8,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        max_force_iters=50,
        enabled=True,
    ):
        self.scale_factor = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.max_force_iters = max_force_iters
        self.enabled = enabled
        self.finite_force_results = 0

    def scale(self, energy):
        return energy * self.scale_factor if self.enabled else energy

    def unscale(self, forces):
        return forces / self.scale_factor if self.enabled else forces

    def calc_forces(self, energy, pos):
        energy_scaled = self.scale(energy)
        forces_scaled = -torch.autograd.grad(
            energy_scaled,
            pos,
            grad_outputs=torch.ones_like(energy_scaled),
            create_graph=True,
        )[0]
        # (nAtoms, 3)
        forces = self.unscale(forces_scaled)
        return forces

    def calc_forces_and_update(self, energy, pos):
        if self.enabled:
            found_nans_or_infs = True
            force_iters = 0

            # Re-calculate forces until everything is nice and finite.
            while found_nans_or_infs:
                forces = self.calc_forces(energy, pos)

                found_nans_or_infs = not torch.all(forces.isfinite())
                if found_nans_or_infs:
                    self.finite_force_results = 0

                    # Prevent infinite loop
                    force_iters += 1
                    if force_iters == self.max_force_iters:
                        logging.warning(
                            "Too many non-finite force results in a batch. "
                            "Breaking scaling loop."
                        )
                        break
                    else:
                        # Delete graph to save memory
                        del forces
                else:
                    self.finite_force_results += 1
                self.update()
        else:
            forces = self.calc_forces(energy, pos)
        return forces

    def update(self):
        if self.finite_force_results == 0:
            self.scale_factor *= self.backoff_factor

        if self.finite_force_results == self.growth_interval:
            self.scale_factor *= self.growth_factor
            self.finite_force_results = 0

        logging.info(f"finite force step count: {self.finite_force_results}")
        logging.info(f"scaling factor: {self.scale_factor}")


class ForceStressScaler(nn.Module):
    """
    Scales up the energy and then scales down the forces
    to prevent NaNs and infs in calculations using AMP.
    Inspired by torch.cuda.amp.GradScaler.
    """

    def __init__(
        self,
        init_scale=2.0**8,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        max_force_iters=50,
        enabled=True,
    ):
        super().__init__()

        self.force_stress_scale_factor = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.max_force_iters = max_force_iters
        self.enabled = enabled
        self.finite_force_stress_results = 0

    def scale(self, energy):
        return energy * self.force_stress_scale_factor if self.enabled else energy

    def unscale(self, forces, virial_scaled):
        forces = forces / self.force_stress_scale_factor if self.enabled else forces
        virial_scaled = (
            virial_scaled / self.force_stress_scale_factor
            if self.enabled
            else virial_scaled
        )
        return forces, virial_scaled

    def calc_forces(
        self,
        energy: torch.Tensor,
        pos: torch.Tensor,
        displacement: torch.Tensor,
        cell: torch.Tensor,
    ):
        energy_scaled = self.scale(energy)
        grad = torch.autograd.grad(
            energy_scaled,
            [pos, displacement],
            grad_outputs=torch.ones_like(energy),
            create_graph=True,
        )
        forces_scaled = -1 * grad[0]
        tc.tassert(tc.Float[torch.Tensor, "n_atoms 3"], forces_scaled)
        virial_scaled = grad[1]
        forces, virial = self.unscale(forces_scaled, virial_scaled)

        volume = torch.linalg.det(cell).abs()
        tc.tassert(tc.Float[torch.Tensor, "bsz"], volume)
        stress = virial / rearrange(volume, "b -> b 1 1")

        return forces, stress

    def calc_forces_and_update(
        self,
        energy: torch.Tensor,
        pos: torch.Tensor,
        displacement: torch.Tensor,
        cell: torch.Tensor,
    ):
        if self.enabled:
            found_nans_or_infs = True
            force_iters = 0

            # Re-calculate forces until everything is nice and finite.
            while found_nans_or_infs:
                forces, stress = self.calc_forces(energy, pos, displacement, cell)

                # Check both forces and stress for NaNs or Infs
                with torch.no_grad():
                    found_nans_or_infs = (~torch.all(forces.isfinite())) | (
                        ~torch.all(stress.isfinite())
                    )
                    found_nans_or_infs = found_nans_or_infs.view(-1)
                    found_nans_or_infs = _all_gather_ddp_if_available(
                        found_nans_or_infs
                    )
                    # ^ (world_size, 1)
                    found_nans_or_infs = bool(found_nans_or_infs.any().item())

                if found_nans_or_infs:
                    self.finite_force_stress_results = 0

                    # Prevent infinite loop
                    force_iters += 1
                    if force_iters == self.max_force_iters:
                        logging.warning(
                            "Too many non-finite force/stress results in a batch. "
                            "Breaking scaling loop."
                        )
                        break
                    else:
                        # Delete graph to save memory
                        del forces, stress

                        self.update()
                        continue
                else:
                    self.finite_force_stress_results += 1

                self.update()
        else:
            forces, stress = self.calc_forces(energy, pos, displacement, cell)

        return forces, stress

    def update(self):
        if not self.training:
            return

        if self.finite_force_stress_results == 0:
            self.force_stress_scale_factor *= self.backoff_factor

        if self.finite_force_stress_results == self.growth_interval:
            self.force_stress_scale_factor *= self.growth_factor
            self.finite_force_stress_results = 0

        logging.info(f"finite force step count: {self.finite_force_stress_results}")
        logging.info(f"scaling factor: {self.force_stress_scale_factor}")
