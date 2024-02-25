from typing import Any

import torch
from jaxtyping._storage import get_shape_memo, shape_str
from lovely_tensors import lovely
from typing_extensions import TypeVar


def _make_error_str(input: Any, t: Any) -> str:
    error_components: list[str] = []
    error_components.append(f"JaxTyping error:")
    if hasattr(t, "__instancecheck_str__"):
        error_components.append(t.__instancecheck_str__(input))
    if torch.is_tensor(input):
        error_components.append(repr(lovely(input)))
    error_components.append(shape_str(get_shape_memo()))

    return "\n".join(error_components)


T = TypeVar("T", infer_variance=True)


def tassert(t: Any, input: T) -> T:
    assert isinstance(input, t), _make_error_str(input, t)
    return input
