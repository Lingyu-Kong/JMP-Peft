import functools
from collections.abc import Iterable
from typing import TypeAlias, cast

import torch
import torch.func
import torch.nn as nn
from nshtrainer.ll.typecheck import Float

MLP: TypeAlias = nn.Module


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
    mlps: Iterable[MLP],
    x: Float[torch.Tensor, "n d_in"] | Float[torch.Tensor, "num_mlps n d_in"],
    is_x_stacked: bool = False,
) -> Float[torch.Tensor, "num_mlps n d_out"]:
    assert mlps, "At least one MLP is required."
    reference_mlp = next(iter(mlps))

    # Now, we stack the head weights.
    params, buffers = torch.func.stack_module_state(cast(list[nn.Module], mlps))

    # Vmap the forward method over the stacked weight dimension.
    fn = torch.func.vmap(
        functools.partial(_forward_mlp, reference_mlp=reference_mlp),
        in_dims=(
            None  # `x` does not have a stacked dimension.
            if not is_x_stacked
            else 0,  # `x` is batched over the stacked dimension.
            0,  # `params` and `buffers` are batched over the stacked dimension.
        ),
    )

    return fn(x, (params, buffers))
