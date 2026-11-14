###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Run a module's forward pre-hooks without calling its ``forward`` (Muse Spark's fused
post-expert norm).

Fused kernels read a submodule's weights without entering ``submodule.forward``.
Hooks registered on that submodule (e.g. DDP's ``overlap_param_gather`` wait for
the weight's all-gather) must still run. :func:`run_forward_pre_hooks` calls them
as ``torch.nn.Module._call_impl`` does: global hooks first, then the module's own,
``with_kwargs=True`` hooks with the kwargs dict, and a hook's return value
replaces the call's args (and kwargs).
"""

from typing import Any, Dict, Tuple

import torch
from torch.nn.modules import module as _torch_module


def run_forward_pre_hooks(
    module: torch.nn.Module, args: Any, kwargs: Dict[str, Any] = None
) -> Tuple[tuple, Dict[str, Any]]:
    """Run ``module``'s forward pre-hooks and return ``(args, kwargs)`` after them."""
    if not isinstance(args, tuple):
        args = (args,)
    kwargs = {} if kwargs is None else kwargs
    with_kwargs = getattr(module, "_forward_pre_hooks_with_kwargs", {})
    hooks = (
        *getattr(_torch_module, "_global_forward_pre_hooks", {}).items(),
        *module._forward_pre_hooks.items(),
    )
    for hook_id, hook in hooks:
        if hook_id in with_kwargs:
            result = hook(module, args, kwargs)
            if result is not None:
                if isinstance(result, tuple) and len(result) == 2:
                    args, kwargs = result
                else:
                    raise RuntimeError(
                        "forward pre-hook must return None or a tuple of (new_args, new_kwargs), "
                        f"but got {result}."
                    )
        else:
            result = hook(module, args)
            if result is not None:
                args = result if isinstance(result, tuple) else (result,)
    return args, kwargs
