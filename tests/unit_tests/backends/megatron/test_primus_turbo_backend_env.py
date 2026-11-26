###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``turbo_gemm_backend`` / ``turbo_grouped_gemm_backend`` reach Primus-Turbo's dispatcher.

A Turbo layer publishes them as ``PRIMUS_TURBO_GEMM_BACKEND`` /
``PRIMUS_TURBO_GROUPED_GEMM_BACKEND`` when it is built. Primus-Turbo since
AMD-AGI/Primus-Turbo#605 caches each variable at its first read, so Primus writes it with
``primus_turbo.common.envs.set_env()``; older Primus-Turbo reads ``os.environ`` at every
dispatch. The tests read the backend once before publishing it, so on either version a write
the dispatcher does not see fails them.
"""

import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformer_engine.pytorch")
pytest.importorskip("primus_turbo")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="primus_turbo needs a GPU")

HELPERS = ("_apply_turbo_gemm_backend_env", "_apply_turbo_grouped_gemm_backend_env")
VARIABLES = ("PRIMUS_TURBO_GEMM_BACKEND", "PRIMUS_TURBO_GROUPED_GEMM_BACKEND")

# helper, Primus argument, Primus-Turbo variable, GlobalBackendManager getter, value, BF16 backend
CASES = [
    (
        "_apply_turbo_gemm_backend_env",
        "turbo_gemm_backend",
        "PRIMUS_TURBO_GEMM_BACKEND",
        "get_gemm_backend",
        "hipblaslt",
        "HIPBLASLT",
    ),
    (
        "_apply_turbo_grouped_gemm_backend_env",
        "turbo_grouped_gemm_backend",
        "PRIMUS_TURBO_GROUPED_GEMM_BACKEND",
        "get_grouped_gemm_backend",
        "bf16:flydsl",
        "FLYDSL",
    ),
]
IDS = ["gemm", "grouped_gemm"]


@pytest.fixture
def turbo(monkeypatch):
    """The extension module, with both variables unset and Primus-Turbo's backend state reset."""
    from primus_turbo.pytorch.core.backend import GlobalBackendManager

    import primus.backends.megatron.core.extensions.primus_turbo as turbo

    saved = {name: os.environ.pop(name, None) for name in VARIABLES}
    GlobalBackendManager.reset()  # since Primus-Turbo#605 also drops the cached backend variables
    monkeypatch.setattr(turbo, "warning_rank_0", lambda *args, **kwargs: None)
    yield turbo
    for helper in HELPERS:
        getattr(turbo, helper).cache_clear()
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    GlobalBackendManager.reset()


@pytest.mark.parametrize("helper,arg,variable,getter,value,backend", CASES, ids=IDS)
def test_backend_published_after_first_read(
    turbo, monkeypatch, helper, arg, variable, getter, value, backend
):
    from primus_turbo.pytorch.core.backend import (
        BackendType,
        GlobalBackendManager,
        PrecisionType,
    )

    get_backend = getattr(GlobalBackendManager, getter)
    # read while unset: since Primus-Turbo#605 this caches the unset value
    assert get_backend(PrecisionType.BF16_FP16_FP32) is None

    monkeypatch.setattr(turbo, "get_args", lambda: SimpleNamespace(**{arg: value}))
    getattr(turbo, helper).cache_clear()
    getattr(turbo, helper)()

    assert os.environ.get(variable) == value
    choice = get_backend(PrecisionType.BF16_FP16_FP32)
    assert choice is not None and choice.backend == BackendType[backend]


@pytest.mark.parametrize("helper,arg,variable,getter,value,backend", CASES, ids=IDS)
def test_default_publishes_nothing(turbo, monkeypatch, helper, arg, variable, getter, value, backend):
    monkeypatch.setattr(turbo, "get_args", lambda: SimpleNamespace(**{arg: "default"}))
    getattr(turbo, helper).cache_clear()
    getattr(turbo, helper)()

    assert variable not in os.environ


def test_uses_primus_turbo_set_env_when_available(turbo):
    try:
        from primus_turbo.common import envs
    except ImportError:
        pytest.skip("Primus-Turbo before AMD-AGI/Primus-Turbo#605 has no primus_turbo.common.envs")

    assert turbo._set_turbo_env is envs.set_env
