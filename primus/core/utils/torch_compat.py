###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Compatibility shims for torch APIs that newer torch builds drop."""


def ensure_torch_jit_script() -> None:
    """Restore ``torch.jit.script`` on torch builds that no longer export it.

    torch 2.16 nightlies drop the public ``torch.jit.script`` alias while keeping
    ``torch.jit._script.script``. Megatron-LM (``megatron/core/jit.py``), the
    Megatron-LM copy inside Megatron-Bridge, and modelopt still call
    ``torch.jit.script`` at import time, so the alias must exist before they are
    imported. No-op on torch builds that still export it.
    """
    import torch.jit

    if hasattr(torch.jit, "script"):
        return
    from torch.jit._script import script

    torch.jit.script = script
