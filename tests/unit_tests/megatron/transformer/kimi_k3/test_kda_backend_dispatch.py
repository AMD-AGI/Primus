###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The KDA backend-dispatch surface.

``kda_kernels`` must import with **no** optional dependency present (the
eager reference is the always-available fallback), and selecting a
backend whose dependency is missing must raise an actionable
:class:`ImportError` rather than crash the import of the whole package.
This mirrors ``test_v4_backend_import_gating.py`` for the DeepSeek-V4
kernels.
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def test_importing_the_package_does_not_import_fla():
    """``import kda_kernels`` must not pull in ``fla``.

    Forcing a genuine re-import means evicting the package from
    ``sys.modules``, which is a global side effect: any module that has
    already done ``from ...kda_kernels import eager_chunk_kda`` keeps a
    reference to the *old* function object, so a later test comparing
    identity against a fresh import would fail for reasons of its own.
    The original module objects are therefore put back on the way out.
    """
    module_name = "primus.backends.megatron.core.transformer.kimi_k3.kda_kernels"
    real_import = builtins.__import__
    touched = []

    def spy(name, *args, **kwargs):
        if name.split(".")[0] == "fla":
            touched.append(name)
        return real_import(name, *args, **kwargs)

    saved = {name: mod for name, mod in sys.modules.items() if name.startswith(module_name)}
    for name in saved:
        del sys.modules[name]

    builtins.__import__ = spy
    try:
        importlib.import_module(module_name)
    finally:
        builtins.__import__ = real_import
        for name in [n for n in sys.modules if n.startswith(module_name)]:
            del sys.modules[name]
        sys.modules.update(saved)
    assert not touched, f"importing kda_kernels eagerly imported fla: {touched}"


def test_eager_backends_resolve_without_any_optional_dependency():
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
        KDA_BACKENDS,
        eager_chunk_kda,
        eager_recurrent_kda,
        resolve_kda_backend,
    )

    assert set(KDA_BACKENDS) == {"eager", "eager_recurrent", "fla", "flydsl"}
    assert resolve_kda_backend("eager") is eager_chunk_kda
    assert resolve_kda_backend("eager_recurrent") is eager_recurrent_kda


def test_unknown_backend_raises_value_error():
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
        resolve_kda_backend,
    )

    with pytest.raises(ValueError, match="Unknown KDA backend"):
        resolve_kda_backend("triton_v99")


def test_flydsl_backend_resolves_or_explains_itself():
    """With flydsl on gfx950 the entry resolves; otherwise the error names the fallbacks.

    This replaces the WP9-era "not implemented yet" assertion. What still
    matters is that selecting an unavailable backend raises an
    :class:`ImportError` naming what to use instead — never an
    ``AttributeError`` or a bare ``ModuleNotFoundError`` from inside the
    kernel package.
    """
    import torch

    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
        resolve_kda_backend,
    )

    have_flydsl = True
    try:
        import flydsl  # noqa: F401
    except ImportError:
        have_flydsl = False
    on_gfx950 = torch.cuda.is_available() and str(
        getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    ).startswith("gfx950")

    if have_flydsl and on_gfx950:
        assert callable(resolve_kda_backend("flydsl"))
    else:
        with pytest.raises(ImportError, match="eager | eager_recurrent | fla"):
            resolve_kda_backend("flydsl")


def test_importing_kda_kernels_does_not_import_flydsl():
    """Selecting ``eager`` must not pay for — or crash on — the flydsl import."""
    import importlib
    import sys

    prefixes = ("flydsl", "primus.backends.megatron.core.transformer.kimi_k3.kda_kernels")
    saved = {n: m for n, m in sys.modules.items() if n.startswith(prefixes)}
    for name in saved:
        del sys.modules[name]
    try:
        importlib.import_module(
            "primus.backends.megatron.core.transformer.kimi_k3.kda_kernels"
        )
        touched = sorted(n for n in sys.modules if n.startswith("flydsl"))
    finally:
        sys.modules.update(saved)
    assert not touched, f"importing kda_kernels eagerly imported flydsl: {touched}"


def test_fla_backend_resolves_or_explains_itself():
    """With ``fla`` installed the entry resolves; without it the error is actionable."""
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
        resolve_kda_backend,
    )

    try:
        import fla.ops.kda  # noqa: F401

        have_fla = True
    except ImportError:
        have_fla = False

    if have_fla:
        assert callable(resolve_kda_backend("fla"))
    else:
        with pytest.raises(ImportError, match="fla-core"):
            resolve_kda_backend("fla")


def test_all_backends_share_one_signature():
    """The eager entries must be call-compatible so they can be swapped."""
    import inspect

    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
        eager_chunk_kda,
        eager_recurrent_kda,
    )

    common = {
        "q",
        "k",
        "v",
        "g",
        "beta",
        "scale",
        "initial_state",
        "output_final_state",
        "use_qk_l2norm_in_kernel",
        "chunk_size",
    }
    for fn in (eager_chunk_kda, eager_recurrent_kda):
        params = set(inspect.signature(fn).parameters)
        assert common <= params, f"{fn.__name__} is missing {sorted(common - params)}"
