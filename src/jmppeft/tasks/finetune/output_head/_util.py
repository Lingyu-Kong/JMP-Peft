import contextlib

import torch


def enable_grad(stack: contextlib.ExitStack):
    if torch.is_inference_mode_enabled():
        stack.enter_context(torch.inference_mode(mode=False))
    if not torch.is_grad_enabled():
        stack.enter_context(torch.enable_grad())
