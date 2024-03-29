import functools
from collections.abc import Sequence
from typing import TypeAlias, cast

import ll.nn
import torch
import torch.func
import torch.nn as nn
from ll.typecheck import Float

MLP: TypeAlias = nn.Sequential | ll.nn.ResidualSequential


# "Functional" forward method for a single head
def _forward_mlp(
    x: Float[torch.Tensor, "... d_model"],
    param_buffer_dicts: dict[str, torch.Tensor]
    | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    *,
    reference_mlp: MLP,
):
    x = torch.func.functional_call(
        reference_mlp,
        param_buffer_dicts,
        args=(x,),
        kwargs={},
        strict=True,
    )
    return x


def run_mlps_in_parallel(
    mlps: Sequence[MLP],
    x: Float[torch.Tensor, "n d_model"],
) -> Float[torch.Tensor, "num_mlps n d_model"]:
    assert mlps, "At least one MLP is required."

    # Now, we stack the head weights.
    params, buffers = torch.func.stack_module_state(cast(list[nn.Module], mlps))

    # Vmap the forward method over the stacked weight dimension.
    fn = torch.func.vmap(
        functools.partial(_forward_mlp, reference_mlp=mlps[0]),
        in_dims=(
            None,  # `x` does not have a stacked dimension.
            0,  # `params` and `buffers` are batched over the stacked dimension.
        ),
    )

    return fn(x, (params, buffers))
