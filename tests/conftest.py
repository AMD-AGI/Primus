###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import sys
from pathlib import Path


def _warmup_aiter_nondeterministic_mha_bwd() -> None:
    """Load aiter's nondeterministic flash-attention backward kernel before TE.

    aiter and TransformerEngine both statically bundle composable_kernel
    (``ck_tile``). If ``transformer_engine`` is imported before aiter's
    nondeterministic ``mha_bwd`` JIT module is first loaded, that kernel's
    ``.so`` resolves its ck_tile host launch path against TE's (different) CK
    copy and launches with an invalid grid/block config -> at runtime the
    backward dies with::

        HIP Function Failed (.../ck_tile/host/kernel_launch.hpp,110)
        invalid configuration argument

    Triggering one tiny nondeterministic backward here (in the root conftest's
    ``pytest_configure``, before any test module imports TE) loads that kernel
    against aiter's own CK first, after which it stays correct for the rest of
    the session. The deterministic kernel is unaffected, so we only need to warm
    the nondeterministic variant. Best-effort: never let warmup break the suite.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return

        import math

        import primus_turbo.pytorch as pt

        q, k, v = (
            torch.randn(1, 8, 1, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
            for _ in range(3)
        )
        out = pt.ops.flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=1.0 / math.sqrt(64),
            causal=False,
            window_size=(-1, -1),
            deterministic=False,
            return_lse=False,
        )
        out.float().sum().backward()
        torch.cuda.synchronize()
    except Exception:
        # Warmup is a best-effort mitigation; if aiter/CUDA is unavailable or
        # the API shifts, fall through silently and let the affected tests
        # surface their own errors.
        pass


def _selection_includes_megatron(config) -> bool:
    """True if the pytest selection could include a backends/megatron suite.

    Covers both the unit (``tests/unit_tests/backends/megatron``) and integration
    (``tests/integration_tests/backends/megatron``) trees: the diffusion
    integration tests run the same aiter attention backward and need the same
    crash mitigations (deepbind hook + warmup).

    Path-level heuristic only (does not inspect -k / -m); the mitigations are
    best-effort, so a conservative path check is sufficient. Fails OPEN: any
    detection error returns True, because a missed mitigation degrades to a hard
    mha_bwd crash, not just a slowdown.

    Uses ``config.args`` (populated post-parse, includes the default rootdir on
    a bare run) rather than ``config.getoption("file_or_dir")`` (empty on a bare
    run) -- do not "simplify" to the latter, it flips the empty-case semantics.
    """
    try:
        base = Path(__file__).resolve().parent
        megatron_dirs = (
            base / "unit_tests" / "backends" / "megatron",
            base / "integration_tests" / "backends" / "megatron",
        )
        args = list(getattr(config, "args", []) or [])
        if not args:
            return True  # whole-suite run (defensive; config.args is never empty)
        for arg in args:
            path = Path(str(arg).split("::", 1)[0]).resolve()
            for megatron_dir in megatron_dirs:
                # arg is the megatron dir, an ancestor of it, or a path inside it
                if (
                    path == megatron_dir
                    or megatron_dir.is_relative_to(path)
                    or path.is_relative_to(megatron_dir)
                ):
                    return True
        return False
    except Exception:
        # Fail open: never let a detection error suppress the crash-prevention mitigations.
        return True


def pytest_configure(config):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    megatron_path = os.environ.get("MEGATRON_PATH")
    if megatron_path is None or not os.path.exists(megatron_path):
        megatron_path = project_root / "third_party" / "Megatron-LM"
    if str(megatron_path) not in sys.path:
        sys.path.append(str(megatron_path))

    # Only needed for the megatron suite, so skip for unrelated selections
    # (fail-open: detection errors still run it).
    if _selection_includes_megatron(config):
        # ORDER MATTERS: install the aiter RTLD_DEEPBIND import hook BEFORE the
        # warmup below (or any test) first imports aiter's mha backward
        # extension. The hook wraps importlib.import_module so the pinned
        # aiter::mha_bwd binds its own ck_tile instead of TE's stale vendored
        # libmha (ROCm/aiter#1332); once ``module_fmha_v3_bwd`` is already in
        # sys.modules the hook is a no-op, so the warmup MUST NOT run first or
        # the hd128 backward launches with an invalid grid config and aborts the
        # whole process. Mirrors the production megatron.turbo.aiter_deepbind
        # before_train patch (which has no such ordering hazard).
        from tests.utils import install_aiter_deepbind_hook

        install_aiter_deepbind_hook()
        # Must run before any test module imports transformer_engine. See the
        # helper docstring for the aiter<->TE composable_kernel load-order issue.
        _warmup_aiter_nondeterministic_mha_bwd()
    else:
        print("[conftest] skipping aiter mha_bwd mitigations (no megatron tests selected)")
