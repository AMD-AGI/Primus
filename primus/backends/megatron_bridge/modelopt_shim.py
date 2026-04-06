###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Work around broken or missing ``modelopt`` when using Megatron-Bridge.

Some venvs ship a conflicting PyPI ``modelopt`` package (or a partial install)
that breaks NVIDIA ModelOpt imports. Megatron-Bridge imports several
``modelopt.torch.*`` modules at import time (``train``, ``checkpointing``,
``gpt_provider``, …).

When the real stack is not usable, we register minimal in-memory packages with
no-op callables for training without ModelOpt quantization / distillation /
checkpoint extras.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

_MEGATRON_DISTILL = "modelopt.torch.distill.plugins.megatron"
_OPT_PLUGINS = "modelopt.torch.opt.plugins"
_INSTALLED = False


def _real_modelopt_usable() -> bool:
    """True only if distillation Megatron hooks and opt checkpoint plugins all import."""
    try:
        m1 = importlib.import_module(_MEGATRON_DISTILL)
        if not callable(getattr(m1, "get_tensor_shapes_adjust_fn_for_distillation", None)):
            return False
        m2 = importlib.import_module(_OPT_PLUGINS)
        for name in (
            "save_modelopt_state",
            "save_sharded_modelopt_state",
            "restore_modelopt_state",
            "restore_sharded_modelopt_state",
        ):
            if not callable(getattr(m2, name, None)):
                return False
        return True
    except Exception:
        return False


def install_modelopt_stub_if_needed() -> None:
    """If ModelOpt cannot be loaded, install no-op stubs once."""
    global _INSTALLED
    if _INSTALLED:
        return

    if _real_modelopt_usable():
        _INSTALLED = True
        return

    for name in list(sys.modules):
        if name == "modelopt" or name.startswith("modelopt."):
            del sys.modules[name]

    def _make_pkg(fullname: str) -> types.ModuleType:
        mod = types.ModuleType(fullname)
        mod.__file__ = f"<primus modelopt shim {fullname}>"
        mod.__package__ = fullname.rpartition(".")[0] if "." in fullname else ""
        mod.__path__ = []
        mod.__loader__ = None
        sys.modules[fullname] = mod
        return mod

    _make_pkg("modelopt")
    _make_pkg("modelopt.torch")
    _make_pkg("modelopt.torch.distill")
    _make_pkg("modelopt.torch.distill.plugins")
    meg = _make_pkg(_MEGATRON_DISTILL)

    def get_tensor_shapes_adjust_fn_for_distillation(*_a: Any, **_k: Any) -> None:
        return None

    meg.get_tensor_shapes_adjust_fn_for_distillation = get_tensor_shapes_adjust_fn_for_distillation

    _make_pkg("modelopt.torch.opt")
    opt_plugins = _make_pkg(_OPT_PLUGINS)

    def _noop(*_a: Any, **_k: Any) -> None:
        return None

    opt_plugins.save_modelopt_state = _noop
    opt_plugins.save_sharded_modelopt_state = _noop
    opt_plugins.restore_modelopt_state = _noop
    opt_plugins.restore_sharded_modelopt_state = _noop

    _INSTALLED = True

    try:
        from primus.modules.module_utils import log_rank_0

        log_rank_0(
            "[Primus:Megatron-Bridge] modelopt NVIDIA stack not usable; "
            "using no-op stubs (OK for pretrain/SFT without ModelOpt). "
            "For ModelOpt features, install `nvidia-modelopt[torch]` and remove conflicting `modelopt` wheels."
        )
    except Exception:
        pass
