import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

import torch.utils.checkpoint
from ll import TypedConfig


class GradientCheckpointingConfig(TypedConfig):
    preserve_rng_state: bool = False
    """
    Whether to preserve the RNG state when checkpointing.
    Incurs a small overhead if set to `True`.
    """

    use_reentrant: bool = False
    """
    Whether to use reentrant checkpointing.
    This is recommended to be `False`, see https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
    """

    checkpoint_early_stop: bool | None = None
    """
    Non-reentrant checkpoint stops recomputation as soon as all needed intermediate activations have been recomputed.
    Set this to `True` to enable this optimization. Set to `None` to use the default value from PyTorch.

    See https://pytorch.org/docs/stable/checkpoint.html
    """

    @contextlib.contextmanager
    def context(self):
        """
        Context manager to temporarily set the config.
        """
        with contextlib.ExitStack() as stack:
            if (early_stop := self.checkpoint_early_stop) is not None:
                stack.enter_context(
                    torch.utils.checkpoint.set_checkpoint_early_stop(early_stop)
                )
            yield


TParams = ParamSpec("TParams")
TReturn = TypeVar("TReturn")


if TYPE_CHECKING:

    def checkpoint(
        fn: Callable[TParams, TReturn],
        config: GradientCheckpointingConfig | None,
    ) -> Callable[TParams, TReturn]: ...
else:

    def checkpoint(
        fn: Callable,
        config: GradientCheckpointingConfig | None,
    ) -> Callable:
        # If checkpointing is disabled, just return the function
        if config is None:
            return fn

        def inner(*args: Any, **kwargs: Any) -> TReturn:
            nonlocal config
            if kwargs:
                raise ValueError("Keyword arguments are not supported.")

            out = torch.utils.checkpoint.checkpoint(
                fn,
                *args,
                use_reentrant=config.use_reentrant,
                preserve_rng_state=config.preserve_rng_state,
            )
            return cast(TReturn, out)

        return inner
