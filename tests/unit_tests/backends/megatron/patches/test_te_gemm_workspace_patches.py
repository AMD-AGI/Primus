###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The shared TE GEMM workspace helper (te_patches/gemm_workspace_size_patches.py) and its two
users: GLM-5.3's and Mistral Large 4's te_gemm_workspace patches (CPU, stand-in TE module)."""

import itertools
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from primus.core.patches.context import PatchContext


@pytest.fixture
def te_gemm(monkeypatch):
    """A stand-in ``transformer_engine.pytorch.cpp_extensions.gemm`` with the layout of TE 2.17
    (the project image): ``get_cublas_workspace_size_bytes()`` and an ``lru_cache``'d
    ``get_cublas_workspace(device, ub, grouped_gemm)`` that allocates that many bytes, at TE's
    64 MiB gfx950 default. ``test_helper_on_real_te`` checks the real module on a GPU."""
    import functools

    gemm = types.ModuleType("transformer_engine.pytorch.cpp_extensions.gemm")

    def get_cublas_workspace_size_bytes():
        return 64 << 20

    @functools.lru_cache(maxsize=None)
    def get_cublas_workspace(device, ub, grouped_gemm):
        # as TE: the size function is looked up in the module at call time (a meta tensor: no memory)
        return torch.empty(gemm.get_cublas_workspace_size_bytes(), dtype=torch.uint8, device="meta")

    gemm.get_cublas_workspace_size_bytes = get_cublas_workspace_size_bytes
    gemm.get_cublas_workspace = get_cublas_workspace
    names = ("transformer_engine", "transformer_engine.pytorch", "transformer_engine.pytorch.cpp_extensions")
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.cpp_extensions.gemm", gemm)
    monkeypatch.setattr(sys.modules[names[-1]], "gemm", gemm, raising=False)
    # no TE 2.8 module.base layout: an installed real TE must not be reached (importing it raises
    # ImportError, so the helper sees only this stand-in)
    for name in ("transformer_engine.pytorch.module", "transformer_engine.pytorch.module.base"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delenv("PRIMUS_TE_GEMM_WORKSPACE_MB", raising=False)
    return gemm


def _ctx(model_type):
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={
            "backend_args": SimpleNamespace(model_type=model_type),
            "module_config": SimpleNamespace(params=SimpleNamespace(model_type=model_type)),
        },
    )


def test_helper_rocm_only_and_never_lowers(te_gemm, monkeypatch):
    from primus.backends.megatron.patches.te_patches import (
        gemm_workspace_size_patches as ws,
    )

    def allocated():  # the buffer TE hands to its GEMMs (cached per device)
        return te_gemm.get_cublas_workspace(0, False, False).numel()

    assert allocated() == 64 << 20  # cached at TE's default before the patch
    monkeypatch.setattr(torch.version, "hip", None)
    assert ws.enlarge_te_gemm_workspace() == 0  # CUDA: unchanged
    assert te_gemm.get_cublas_workspace_size_bytes() == 64 << 20  # the size function untouched
    assert allocated() == 64 << 20
    monkeypatch.setattr(torch.version, "hip", "6.4")
    assert ws.enlarge_te_gemm_workspace() == ws.DEFAULT_WORKSPACE_MB << 20
    assert allocated() == 512 << 20  # the cached 64 MiB buffer was dropped and reallocated
    monkeypatch.setenv("PRIMUS_TE_GEMM_WORKSPACE_MB", "128")
    assert ws.enlarge_te_gemm_workspace() == 512 << 20  # never lowered: the current (effective) size
    assert allocated() == 512 << 20
    monkeypatch.setenv("PRIMUS_TE_GEMM_WORKSPACE_MB", "1024")
    assert ws.enlarge_te_gemm_workspace() == 1024 << 20
    assert allocated() == 1024 << 20

    # a value that is not an integer names the variable
    monkeypatch.setenv("PRIMUS_TE_GEMM_WORKSPACE_MB", "512MB")
    with pytest.raises(ValueError, match="PRIMUS_TE_GEMM_WORKSPACE_MB"):
        ws.enlarge_te_gemm_workspace()
    # 0 / negative keeps TE's workspace as it is
    for off in ("0", "-64"):
        monkeypatch.setenv("PRIMUS_TE_GEMM_WORKSPACE_MB", off)
        assert ws.enlarge_te_gemm_workspace() == 0
        assert te_gemm.get_cublas_workspace_size_bytes() == 1024 << 20  # the size function untouched
        assert allocated() == 1024 << 20


@pytest.mark.parametrize(
    "module,patch_fn,model_type",
    [
        (
            "primus.backends.megatron.patches.glm5_next_patches",
            "patch_glm5_next_te_gemm_workspace",
            "glm5_next",
        ),
        (
            "primus.backends.megatron.patches.mistral_large4_args_patches",
            "patch_mistral_large4_te_gemm_workspace",
            "mistral_large4",
        ),
    ],
)
def test_model_patches_use_the_helper(te_gemm, monkeypatch, module, patch_fn, model_type):
    """GLM-5.3 (512 MiB on ROCm, PRIMUS_TE_GEMM_WORKSPACE_MB, nothing on CUDA) and Mistral Large 4
    both enlarge the workspace through the shared helper."""
    import importlib

    mod = importlib.import_module(module)

    def allocated():
        return te_gemm.get_cublas_workspace(0, False, False).numel()

    assert allocated() == 64 << 20
    monkeypatch.setattr(torch.version, "hip", None)
    getattr(mod, patch_fn)(_ctx(model_type))
    assert allocated() == 64 << 20  # CUDA: nothing changed
    monkeypatch.setattr(torch.version, "hip", "6.4")
    getattr(mod, patch_fn)(_ctx(model_type))  # no env var: the 512 MiB default
    assert allocated() == 512 << 20  # the GEMM buffer TE allocates is the larger one

    # the patch goes through the shared helper (one owner of the env var and default)
    calls = []
    # both patches import the helper in their body: replace it where they look it up
    from primus.backends.megatron.patches.te_patches import (
        gemm_workspace_size_patches as ws,
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ws, "enlarge_te_gemm_workspace", lambda tag="": calls.append(tag))
        getattr(mod, patch_fn)(_ctx(model_type))
    assert calls == [
        f"megatron.{'glm5_next' if model_type == 'glm5_next' else 'mistral_large4'}.te_gemm_workspace"
    ]
    # the helper's checks reach each patch: a non-integer value names the variable
    monkeypatch.setenv("PRIMUS_TE_GEMM_WORKSPACE_MB", "512MB")
    with pytest.raises(ValueError, match="PRIMUS_TE_GEMM_WORKSPACE_MB"):
        getattr(mod, patch_fn)(_ctx(model_type))


@pytest.mark.parametrize(
    "patch_id,model_type",
    [
        ("megatron.glm5_next.te_gemm_workspace", "glm5_next"),
        ("megatron.mistral_large4.te_gemm_workspace", "mistral_large4"),
    ],
)
def test_patches_registered_for_their_model_only(patch_id, model_type):
    import primus.backends.megatron.patches.glm5_next_patches  # noqa: F401  (registers)
    import primus.backends.megatron.patches.mistral_large4_args_patches  # noqa: F401
    from primus.core.patches.patch_registry import PatchRegistry

    patch = PatchRegistry.get(patch_id)
    assert patch is not None and patch.phase == "before_train"
    assert patch.condition(_ctx(model_type)) is True
    assert patch.condition(_ctx("deepseek_v3")) is False
    if model_type == "mistral_large4":
        # only when the experts run on TE GEMMs: Primus-Turbo's grouped GEMM (PrimusGroupedMLP) needs
        # Primus-Turbo enabled and installed, use_turbo_grouped_gemm and moe_grouped_gemm (the layer
        # specs' rule); with any of them off the experts run on TE GEMMs (SequentialMLP or TE's
        # grouped GEMM) and the workspace is enlarged
        from primus.backends.megatron.core.models.mistral_large4 import (
            mistral_large4_layout as layout,
        )

        for installed in (True, False):
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(
                    layout.importlib.util, "find_spec", lambda name, _i=installed: object() if _i else None
                )
                for enabled, turbo, grouped in itertools.product((True, False), repeat=3):
                    fires = not (installed and enabled and turbo and grouped)
                    c = _ctx(model_type)
                    c.extra["module_config"].params.enable_primus_turbo = enabled
                    c.extra["module_config"].params.use_turbo_grouped_gemm = turbo
                    c.extra["module_config"].params.moe_grouped_gemm = grouped
                    assert patch.condition(c) is fires, (installed, enabled, turbo, grouped)


def test_helper_on_real_te(monkeypatch):
    """With the real Transformer Engine on a ROCm GPU: the workspace TE allocates for its GEMMs
    (``get_cublas_workspace``, the tensor ``general_gemm`` passes to hipBLASLt) has the enlarged
    size after the helper ran. Skips without TE or a GPU. A TE whose ``cpp_extensions.gemm``
    has no ``get_cublas_workspace_size_bytes`` / cached ``get_cublas_workspace`` (another layout)
    is refused by the helper (RuntimeError) unless PRIMUS_TE_GEMM_WORKSPACE_MB=0."""
    gemm = pytest.importorskip("transformer_engine.pytorch.cpp_extensions.gemm")
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("needs a ROCm GPU")
    from primus.backends.megatron.patches.te_patches import (
        gemm_workspace_size_patches as ws,
    )

    if not ws.te_cpp_extensions_gemm_layout_supported(gemm):
        base = pytest.importorskip("transformer_engine.pytorch.module.base")
        if not ws.te_module_base_layout_supported(base):
            with pytest.raises(RuntimeError):
                ws.enlarge_te_gemm_workspace()
            return
        # TE 2.8 layout: the buffer get_workspace allocates has the enlarged size
        monkeypatch.setattr(base, "get_cublas_workspace_size_bytes", base.get_cublas_workspace_size_bytes)
        monkeypatch.setattr(base, "_cublas_workspace", None)
        if hasattr(base, "_multi_stream_cublas_workspace"):  # the helper resets it: restored after the test
            monkeypatch.setattr(base, "_multi_stream_cublas_workspace", base._multi_stream_cublas_workspace)
        monkeypatch.setenv("PRIMUS_TE_GEMM_WORKSPACE_MB", "512")
        assert base.get_workspace().numel() < 512 << 20
        assert ws.enlarge_te_gemm_workspace() == 512 << 20
        assert base.get_workspace().numel() == 512 << 20
        return
    monkeypatch.setattr(gemm, "get_cublas_workspace_size_bytes", gemm.get_cublas_workspace_size_bytes)
    monkeypatch.setenv("PRIMUS_TE_GEMM_WORKSPACE_MB", "512")
    try:
        before = gemm.get_cublas_workspace(torch.cuda.current_device(), False, False)
        assert before.numel() < 512 << 20  # TE's default (64 MiB on gfx950, 32 MiB elsewhere)
        assert ws.enlarge_te_gemm_workspace() == 512 << 20
        after = gemm.get_cublas_workspace(torch.cuda.current_device(), False, False)
        assert after.numel() == 512 << 20  # the cached buffer was dropped and reallocated
    finally:
        gemm.get_cublas_workspace.cache_clear()  # the next user allocates with the restored size


def test_helper_without_te_returns_0(monkeypatch):
    """On ROCm without Transformer Engine installed the helper logs and returns 0 (a TE without
    either GEMM layout raises instead: test_helper_raises_on_broken_te)."""
    from primus.backends.megatron.patches.te_patches import (
        gemm_workspace_size_patches as ws,
    )

    monkeypatch.setattr(torch.version, "hip", "6.4")
    monkeypatch.delenv("PRIMUS_TE_GEMM_WORKSPACE_MB", raising=False)
    for name in (
        "transformer_engine",
        "transformer_engine.pytorch",
        "transformer_engine.pytorch.cpp_extensions",
        "transformer_engine.pytorch.cpp_extensions.gemm",
        "transformer_engine.pytorch.module",
        "transformer_engine.pytorch.module.base",
    ):
        # importing it raises ImportError, also when an earlier test imported the real TE
        monkeypatch.setitem(sys.modules, name, None)
    assert ws.enlarge_te_gemm_workspace() == 0


def test_mistral_warns_for_te_grouped_gemm(te_gemm, monkeypatch):
    """Both TE expert paths get the larger workspace: TE's own grouped GEMM (moe_grouped_gemm
    without Primus-Turbo) with a warning that this path is untested for the model, SequentialMLP
    experts without one. Each case starts from TE's 64 MiB default."""
    from primus.backends.megatron.patches import mistral_large4_args_patches as m

    default = te_gemm.get_cublas_workspace_size_bytes
    monkeypatch.setattr(torch.version, "hip", "6.4")
    logged = []
    monkeypatch.setattr(m, "log_rank_0", lambda msg, *a, **k: logged.append(msg))
    for grouped, warned in ((True, True), (False, False)):
        te_gemm.get_cublas_workspace_size_bytes = default  # fresh: TE's default, nothing cached
        te_gemm.get_cublas_workspace.cache_clear()
        assert te_gemm.get_cublas_workspace(0, False, False).numel() == 64 << 20
        logged.clear()
        ctx = _ctx("mistral_large4")
        ctx.extra["module_config"].params.moe_grouped_gemm = grouped
        m.patch_mistral_large4_te_gemm_workspace(ctx)
        assert any("not tested" in x for x in logged) is warned
        assert te_gemm.get_cublas_workspace_size_bytes() == 512 << 20
        assert te_gemm.get_cublas_workspace(0, False, False).numel() == 512 << 20


def test_helper_on_te28_module_base_layout(monkeypatch):
    """TE 2.8 (ROCm) keeps the workspace in ``transformer_engine.pytorch.module.base``
    (``get_cublas_workspace_size_bytes`` and the cached ``_cublas_workspace`` /
    ``_multi_stream_cublas_workspace``), and its ``cpp_extensions.gemm`` has neither function.
    The helper enlarges it there, drops the cached buffers, and never lowers it."""
    from primus.backends.megatron.patches.te_patches import (
        gemm_workspace_size_patches as ws,
    )

    gemm = types.ModuleType("transformer_engine.pytorch.cpp_extensions.gemm")
    base = types.ModuleType("transformer_engine.pytorch.module.base")
    base.get_cublas_workspace_size_bytes = lambda: 64 << 20
    base._cublas_workspace = None
    base._multi_stream_cublas_workspace = []

    def get_workspace():
        if base._cublas_workspace is None:
            base._cublas_workspace = torch.empty(
                base.get_cublas_workspace_size_bytes(), dtype=torch.uint8, device="meta"
            )
        return base._cublas_workspace

    base.get_workspace = get_workspace
    for name in (
        "transformer_engine",
        "transformer_engine.pytorch",
        "transformer_engine.pytorch.cpp_extensions",
        "transformer_engine.pytorch.module",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, gemm.__name__, gemm)
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.setattr(torch.version, "hip", "6.4")
    monkeypatch.delenv("PRIMUS_TE_GEMM_WORKSPACE_MB", raising=False)

    assert not ws.te_cpp_extensions_gemm_layout_supported(gemm) and ws.te_module_base_layout_supported(base)
    assert get_workspace().numel() == 64 << 20
    base._multi_stream_cublas_workspace = ["old"]
    assert ws.enlarge_te_gemm_workspace() == 512 << 20
    assert base._multi_stream_cublas_workspace == []
    assert get_workspace().numel() == 512 << 20  # cached buffer dropped, reallocated larger
    monkeypatch.setenv("PRIMUS_TE_GEMM_WORKSPACE_MB", "128")
    assert ws.enlarge_te_gemm_workspace() == 512 << 20  # never lowered


def test_helper_enlarges_both_layouts_when_te_has_both(te_gemm, monkeypatch):
    """A TE with both layouts: each one is enlarged, so buffers from module.base.get_workspace do
    not keep the old size while cpp_extensions.gemm's grow."""
    from primus.backends.megatron.patches.te_patches import (
        gemm_workspace_size_patches as ws,
    )

    base = types.ModuleType("transformer_engine.pytorch.module.base")
    base.get_cublas_workspace_size_bytes = lambda: 64 << 20
    base._cublas_workspace = None
    base.get_workspace = lambda: torch.empty(
        base.get_cublas_workspace_size_bytes(), dtype=torch.uint8, device="meta"
    )
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.module", types.ModuleType("m"))
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.setattr(torch.version, "hip", "6.4")
    assert ws.enlarge_te_gemm_workspace() == 512 << 20
    assert te_gemm.get_cublas_workspace(0, False, False).numel() == 512 << 20
    assert base.get_workspace().numel() == 512 << 20


def test_helper_refuses_unknown_te_layout(monkeypatch):
    """A TE with neither layout raises unless PRIMUS_TE_GEMM_WORKSPACE_MB=0."""
    from primus.backends.megatron.patches.te_patches import (
        gemm_workspace_size_patches as ws,
    )

    for name in (
        "transformer_engine",
        "transformer_engine.pytorch",
        "transformer_engine.pytorch.cpp_extensions",
        "transformer_engine.pytorch.cpp_extensions.gemm",
        "transformer_engine.pytorch.module",
        "transformer_engine.pytorch.module.base",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setattr(torch.version, "hip", "6.4")
    monkeypatch.delenv("PRIMUS_TE_GEMM_WORKSPACE_MB", raising=False)
    with pytest.raises(RuntimeError, match="PRIMUS_TE_GEMM_WORKSPACE_MB=0"):
        ws.enlarge_te_gemm_workspace()
    monkeypatch.setenv("PRIMUS_TE_GEMM_WORKSPACE_MB", "0")
    assert ws.enlarge_te_gemm_workspace() == 0


def test_helper_raises_on_broken_te(monkeypatch):
    """TE installed but its GEMM modules fail to import (here: missing): the helper does not fall
    back to TE's default silently; it raises (the model needs the larger workspace). Only a TE that
    is not installed returns 0."""
    from primus.backends.megatron.patches.te_patches import (
        gemm_workspace_size_patches as ws,
    )

    monkeypatch.setitem(sys.modules, "transformer_engine", types.ModuleType("transformer_engine"))
    for name in (
        "transformer_engine.pytorch",
        "transformer_engine.pytorch.cpp_extensions",
        "transformer_engine.pytorch.cpp_extensions.gemm",
        "transformer_engine.pytorch.module",
        "transformer_engine.pytorch.module.base",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setattr(torch.version, "hip", "6.4")
    monkeypatch.delenv("PRIMUS_TE_GEMM_WORKSPACE_MB", raising=False)
    with pytest.raises(RuntimeError, match="PRIMUS_TE_GEMM_WORKSPACE_MB=0"):
        ws.enlarge_te_gemm_workspace()


def test_helper_keeps_a_broken_te_import_error(monkeypatch):
    """A TE whose GEMM module exists but fails to import (an undefined symbol, a missing .so or a
    missing dependency, all ImportError) is not taken for a TE without that module: the original
    error is raised, not the "neither layout" RuntimeError."""
    import importlib

    from primus.backends.megatron.patches.te_patches import (
        gemm_workspace_size_patches as ws,
    )

    monkeypatch.setitem(sys.modules, "transformer_engine", types.ModuleType("transformer_engine"))
    monkeypatch.setattr(torch.version, "hip", "6.4")
    monkeypatch.delenv("PRIMUS_TE_GEMM_WORKSPACE_MB", raising=False)
    real_import = importlib.import_module
    gemm_name = "transformer_engine.pytorch.cpp_extensions.gemm"

    def broken(name, *a, **k):
        if name == gemm_name:
            raise ImportError("libtransformer_engine.so: undefined symbol: nvte_cublas_gemm")
        return real_import(name, *a, **k)

    monkeypatch.setattr(ws.importlib, "import_module", broken)
    with pytest.raises(ImportError, match="undefined symbol"):
        ws.enlarge_te_gemm_workspace()

    # a missing dependency of the module (not the module itself) is a broken install too
    def missing_dep(name, *a, **k):
        if name == gemm_name:
            raise ModuleNotFoundError("No module named 'some_dep'", name="some_dep")
        return real_import(name, *a, **k)

    monkeypatch.setattr(ws.importlib, "import_module", missing_dep)
    with pytest.raises(ModuleNotFoundError, match="some_dep"):
        ws.enlarge_te_gemm_workspace()
