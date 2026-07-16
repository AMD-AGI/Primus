# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Pytest fixtures for Flux diffusion model testing.

Re-exports shared fixtures from the parent megatron conftest.
"""

import pytest

from tests.utils import install_aiter_deepbind_hook

# Install Primus' production aiter RTLD_DEEPBIND import hook before any diffusion
# test imports the aiter mha kernels. On gfx942/gfx950, transformer_engine's
# stale vendored libmha interposes the global ``aiter::mha_bwd`` over Turbo's
# pinned aiter, crashing attention backward (ROCm/aiter#1332). Real training
# installs this via the ``megatron.turbo.aiter_deepbind`` before_train patch;
# unit tests call flash_attn_func directly and never hit that phase, so we wire
# up the same hook here. No-op without a GPU.
install_aiter_deepbind_hook()

from tests.unit_tests.backends.megatron.conftest import (  # noqa: F401,E402
    init_parallel_state,
)


@pytest.fixture(autouse=True)
def _unset_nvte_attention_env(monkeypatch):
    """Clear the TE attention-backend env vars for diffusion tests.

    Some container images bake ``NVTE_FLASH_ATTN=0`` (they target the fused/CK
    attention path). Flux's ``DiffusionModule._set_attention_backend()`` defaults
    to the ``auto`` backend, which validates that ``NVTE_FLASH_ATTN``/
    ``NVTE_FUSED_ATTN``/``NVTE_UNFUSED_ATTN`` are unset-or-1, so the baked ``0``
    makes every Flux model construction fail. Mirror Megatron's own harness
    (``Utils.initialize_distributed`` in ``tests/unit_tests/test_utilities.py``),
    which pops these three vars so a baked/leaked value cannot poison the auto
    backend check. Use ``monkeypatch.delenv`` (not ``os.environ.pop``):
    ``_set_attention_backend`` writes ``os.environ[var] = 1`` after its check, so
    monkeypatch's teardown is what restores each var to its pre-test state and
    contains that write-back leak. Scoped to the diffusion suite only --
    non-diffusion Primus megatron tests never construct a model that runs this
    assertion.
    """
    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(var, raising=False)
