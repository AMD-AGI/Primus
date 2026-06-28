# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Shared fixtures for distributed diffusion integration tests.

These tests construct and run real Flux models (forward + backward through the
Primus Turbo / aiter attention path), so they need the same container-quirk
mitigations the diffusion *unit* tests already have, which the integration tree
was previously missing:

  * ``install_aiter_deepbind_hook()`` -- isolates the pinned aiter::mha_bwd from
    TransformerEngine's stale vendored libmha so the hd128 attention backward
    does not abort the process (ROCm/aiter#1332). The root conftest installs
    this early (before its warmup); calling it again here is idempotent and
    keeps the integration tree self-sufficient.
  * ``_unset_nvte_attention_env`` -- clears baked ``NVTE_FLASH_ATTN=0`` etc. so
    Flux's ``auto`` attention-backend check does not reject model construction.

Re-exports ``init_parallel_state`` and ``_unset_nvte_attention_env`` from the
unit conftests so both trees share one definition.
"""

from tests.utils import install_aiter_deepbind_hook

# Install Primus' production aiter RTLD_DEEPBIND import hook before any test
# imports the aiter mha kernels. No-op without a GPU / when already installed.
install_aiter_deepbind_hook()

from tests.unit_tests.backends.megatron.conftest import (  # noqa: F401,E402
    init_parallel_state,
)
from tests.unit_tests.backends.megatron.diffusion.conftest import (  # noqa: F401,E402
    _unset_nvte_attention_env,
)
