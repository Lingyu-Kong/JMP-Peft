import contextlib
from typing import Literal

import nshutils.typecheck as tc
import torch
import torch.nn as nn
from einops import rearrange
from torch_geometric.data.data import BaseData
from typing_extensions import NotRequired, TypedDict, override

from ....models.gemnet.backbone import GOCBackboneOutput
from ....models.gemnet.layers.force_scaler import ForceScaler, ForceStressScaler
from ....modules.loss import L2MAELossConfig, LossConfig, MAELossConfig
from ._base import BaseTargetConfig
from ._util import enable_grad


class GradientStressTargetConfig(BaseTargetConfig):
    r"""
    Description of this layer:

    **Before the backbone forward pass:**

    1. **Create the displacement tensor**: We begin by creating a small displacement tensor, which represents an infinitesimal deformation of the system. This tensor is denoted as $\mathbf{displacement}$.

    2. **Compute the symmetric part of the displacement tensor**: We then compute the symmetric part of the displacement tensor, denoted as $\mathbf{symmetric\_displacement}$, using the formula:

    $\mathbf{symmetric\_displacement} = \frac{1}{2}\left(\mathbf{displacement} + \mathbf{displacement}^\top\right)$

    This ensures that the displacement tensor is symmetric, as the stress tensor should be.

    3. **Apply the deformation to the atom positions**: We apply the symmetric displacement tensor to the atom positions using the following formula:

    $\mathbf{r}' = \mathbf{r} + \mathbf{r} \cdot \mathbf{symmetric\_displacement}$

    Here, $\mathbf{r}$ and $\mathbf{r}'$ are the original and deformed atom positions, respectively.

    4. **Apply the deformation to the cell (if present)**: If the cell information is available, we apply the symmetric displacement tensor to the cell vectors using the following formula:

    $\mathbf{h}' = \mathbf{h} + \mathbf{h} \cdot \mathbf{symmetric\_displacement}$

    Here, $\mathbf{h}$ and $\mathbf{h}'$ are the original and deformed cell vectors, respectively.

    **After the backbone forward pass:**

    1. **Compute the virial**: We compute the virial $\mathbf{W}$ as the negative of the gradient of the total energy $E$ with respect to the displacement tensor $\mathbf{displacement}$:

    $\mathbf{W} = -\frac{\partial E}{\partial \mathbf{displacement}}$

    2. **Compute the volume**: If the cell information is available, we compute the volume $V$ as the absolute value of the determinant of the cell vectors $\mathbf{h}$:

    $V = |\det(\mathbf{h})|$

    3. **Compute the stress tensor**: The stress tensor $\mathbf{\sigma}$ is defined as the negative of the virial $\mathbf{W}$ divided by the volume $V$:

    $\mathbf{\sigma} = -\frac{1}{V}\mathbf{W}$

    The key equations used in this process are:

    $\mathbf{symmetric\_displacement} = \frac{1}{2}\left(\mathbf{displacement} + \mathbf{displacement}^\top\right)$
    $\mathbf{r}' = \mathbf{r} + \mathbf{r} \cdot \mathbf{symmetric\_displacement}$
    $\mathbf{h}' = \mathbf{h} + \mathbf{h} \cdot \mathbf{symmetric\_displacement}$
    $\mathbf{W} = -\frac{\partial E}{\partial \mathbf{displacement}}$
    $V = |\det(\mathbf{h})|$
    $\mathbf{\sigma} = -\frac{1}{V}\mathbf{W}$
    """

    kind: Literal["gradient_stress"] = "gradient_stress"

    loss: LossConfig = MAELossConfig()
    """The loss function to use for the target"""

    energy_name: str
    """
    The name of the energy target. This target must
    be registered as a graph scalar target.
    """

    forces: bool = False
    """Whether to compute the forces as well"""

    @override
    def is_classification(self) -> bool:
        return False

    @override
    def construct_output_head(
        self,
        output_config,
        d_model_node,
        d_model_edge,
        activation_cls,
    ):
        return GradientStressOutputHead(self)

    @override
    @contextlib.contextmanager
    def model_forward_context(self, data: BaseData):
        with contextlib.ExitStack() as stack:
            enable_grad(stack)

            if not data.pos.requires_grad:
                data.pos.requires_grad_(True)

            # Not necessary in our case as we don't actually
            # use data.cell in the computation graph.
            # if not data.cell.requires_grad:
            #     data.cell.requires_grad_(True)

            data.displacement = torch.zeros(
                (len(data), 3, 3), dtype=data.pos.dtype, device=data.pos.device
            )
            data.displacement.requires_grad_(True)
            symmetric_displacement = 0.5 * (
                data.displacement + data.displacement.transpose(-1, -2)
            )

            # # Disable autograd here so that we don't downcast the `pos` and `cell`
            # # tensors to [b]float16.
            # with torch.autocast(device_type=data.pos.device.type, enabled=False):
            data.pos = data.pos + torch.bmm(
                data.pos.unsqueeze(-2), symmetric_displacement[data.batch]
            ).squeeze(-2)
            data.cell = data.cell + torch.bmm(data.cell, symmetric_displacement)

            yield

    @override
    def supports_inference_mode(self):
        return False


class GradientStressOutputHeadInput(TypedDict):
    data: BaseData
    backbone_output: GOCBackboneOutput
    graph_preds: dict[str, torch.Tensor]
    _stress_precomputed_forces: NotRequired[torch.Tensor]


class GradientStressOutputHead(nn.Module):
    @override
    def __init__(self, target_config: GradientStressTargetConfig):
        super().__init__()

        self.target_config = target_config
        if target_config.forces:
            self.force_stress_scaler = ForceStressScaler()

    @override
    def forward(self, input: GradientStressOutputHeadInput) -> torch.Tensor:
        # Get the computed energy
        assert (graph_preds := input.get("graph_preds")), "Graph predictions not found"
        energy = graph_preds[self.target_config.energy_name]
        tc.tassert(tc.Float[torch.Tensor, "bsz"], energy)

        data = input["data"]
        # Displacement must be in data
        if "displacement" not in data:
            raise ValueError("Displacement tensor not found in data")

        if self.target_config.forces:
            # grad = torch.autograd.grad(
            #     energy,
            #     [data.pos, data.displacement],
            #     grad_outputs=torch.ones_like(energy),
            #     create_graph=self.training,
            # )
            # forces = -1 * grad[0]
            # virial = grad[1]

            # volume = torch.linalg.det(data.cell).abs()
            # tc.tassert(tc.Float[torch.Tensor, "bsz"], volume)
            # stress = virial / rearrange(volume, "b -> b 1 1")

            forces, stress = self.force_stress_scaler.calc_forces_and_update(
                energy, data.pos, data.displacement, data.cell
            )

            # Store the forces in the input dict so that they can be used
            # by the force head.
            input["_stress_precomputed_forces"] = forces
        else:
            grad = torch.autograd.grad(
                energy,
                [data.displacement],
                grad_outputs=torch.ones_like(energy),
                create_graph=self.training,
            )
            # forces = -1 * grad[0]
            virial = grad[0]

            volume = torch.linalg.det(data.cell).abs()
            tc.tassert(tc.Float[torch.Tensor, "bsz"], volume)
            stress = virial / rearrange(volume, "b -> b 1 1")

        return stress


class GradientForcesTargetConfig(BaseTargetConfig):
    kind: Literal["gradient_forces"] = "gradient_forces"

    loss: LossConfig = L2MAELossConfig()
    """The loss function to use for the target"""

    energy_name: str
    """
    The name of the energy target. This target must
    be registered as a graph scalar target.
    """

    use_stress_forces: bool = False
    """If True, assumes that the stress head has already computed the forces."""

    @override
    def is_classification(self) -> bool:
        return False

    @override
    def construct_output_head(
        self,
        output_config,
        d_model_node,
        d_model_edge,
        activation_cls,
    ):
        return GradientForcesOutputHead(self)

    @override
    @contextlib.contextmanager
    def model_forward_context(self, data: BaseData):
        with contextlib.ExitStack() as stack:
            enable_grad(stack)

            if not data.pos.requires_grad:
                data.pos.requires_grad_(True)
            yield

    @override
    def supports_inference_mode(self):
        return False


class GradientOutputHeadInput(TypedDict):
    data: BaseData
    backbone_output: GOCBackboneOutput
    graph_preds: dict[str, torch.Tensor]
    _stress_precomputed_forces: NotRequired[torch.Tensor]


class GradientForcesOutputHead(nn.Module):
    @override
    def __init__(self, target_config: GradientForcesTargetConfig):
        super().__init__()

        self.target_config = target_config
        self.force_scaler = ForceScaler()

    @override
    def forward(self, input: GradientOutputHeadInput) -> torch.Tensor:
        if self.target_config.use_stress_forces:
            assert (
                forces := input.pop("_stress_precomputed_forces")
            ) is not None, "Forces not found"
            return forces

        data = input["data"]
        assert (graph_preds := input.get("graph_preds")), "Graph predictions not found"
        energy = graph_preds[self.target_config.energy_name]
        return self.force_scaler.calc_forces_and_update(energy, data.pos)
