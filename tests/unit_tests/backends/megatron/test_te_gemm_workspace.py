###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``enlarge_te_gemm_workspace`` on fake TE modules (no TE / GPU needed)."""

import functools
import sys
import types

import pytest

from primus.backends.megatron.core.extensions import te_gemm_workspace as ws

MB = 1 << 20


@pytest.fixture
def fake_te(monkeypatch):
    """A TE whose ``module.base`` caches into globals, plus a module that imported the size fn by name."""
    base = types.ModuleType("transformer_engine.pytorch.module.base")
    base.get_cublas_workspace_size_bytes = lambda: 64 * MB
    base._cublas_workspace = ["ws"]
    base._multi_stream_cublas_workspace = ["a", "b"]
    base._dummy_wgrads = {"k": 1}  # unrelated cache: must stay

    @functools.lru_cache
    def get_multi_stream_cublas_workspace():
        return base.get_cublas_workspace_size_bytes()

    base.get_multi_stream_cublas_workspace = get_multi_stream_cublas_workspace
    user = types.ModuleType("transformer_engine.pytorch.module.linear")
    user.get_cublas_workspace_size_bytes = base.get_cublas_workspace_size_bytes
    for name, mod in {
        "transformer_engine": types.ModuleType("transformer_engine"),
        base.__name__: base,
        user.__name__: user,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    # cpp_extensions.gemm is missing in this fake TE (as in some builds)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.cpp_extensions.gemm", None)
    return base, user, get_multi_stream_cublas_workspace


def test_enlarges_every_reference_and_drops_caches(fake_te):
    base, user, cached = fake_te
    assert cached() == 64 * MB
    enlarged, unchanged, errors, notes = ws.enlarge_te_gemm_workspace(512 * MB)
    assert enlarged == [base.__name__, user.__name__] and unchanged == []
    assert base.get_cublas_workspace_size_bytes() == user.get_cublas_workspace_size_bytes() == 512 * MB
    assert base._cublas_workspace == [] and base._multi_stream_cublas_workspace == []
    assert base._dummy_wgrads == {"k": 1}
    assert cached() == 512 * MB  # the lru_cache was dropped
    # the missing optional module is a note, not an error: another module was patched
    assert errors == [] and len(notes) == 1 and "cpp_extensions.gemm" in notes[0]


def test_patch_is_max_of_te_and_ours_and_idempotent(fake_te):
    base, _, _ = fake_te
    ws.enlarge_te_gemm_workspace(512 * MB)
    patched = base.get_cublas_workspace_size_bytes
    assert ws.enlarge_te_gemm_workspace(256 * MB)[0] == []  # already >= 256 MiB: unchanged
    assert base.get_cublas_workspace_size_bytes is patched
    enlarged, _, _, _ = ws.enlarge_te_gemm_workspace(1024 * MB)  # wraps TE's original, not our wrapper
    assert base.__name__ in enlarged and base.get_cublas_workspace_size_bytes() == 1024 * MB
    assert base.get_cublas_workspace_size_bytes._primus_original() == 64 * MB


def test_larger_default_is_kept_and_reported_unchanged(fake_te):
    base, _, _ = fake_te
    big = lambda: 1024 * MB  # noqa: E731
    base.get_cublas_workspace_size_bytes = big
    enlarged, unchanged, _, _ = ws.enlarge_te_gemm_workspace(512 * MB)
    assert base.__name__ in unchanged and base.__name__ not in enlarged
    assert base.get_cublas_workspace_size_bytes is big
    assert base._cublas_workspace == ["ws"]  # nothing changed, nothing dropped


def test_second_apply_same_size_does_not_double_wrap(fake_te):
    """A second call with the same size (e.g. the patch applied twice) is a no-op."""
    base, user, _ = fake_te
    ws.enlarge_te_gemm_workspace(512 * MB)
    patched = base.get_cublas_workspace_size_bytes
    enlarged, unchanged, errors, _ = ws.enlarge_te_gemm_workspace(512 * MB)
    assert enlarged == [] and unchanged == [base.__name__, user.__name__] and errors == []
    assert base.get_cublas_workspace_size_bytes is patched
    assert not hasattr(patched._primus_original, "_primus_original")  # wraps TE's function only


def test_failed_probe_is_recorded_and_never_shrinks(fake_te):
    base, _, _ = fake_te
    state = {"gpu": False}

    def te_size():
        if not state["gpu"]:
            raise RuntimeError("No HIP GPUs are available")
        return 2048 * MB

    base.get_cublas_workspace_size_bytes = te_size
    enlarged, _, errors, notes = ws.enlarge_te_gemm_workspace(512 * MB)
    assert base.__name__ in enlarged
    assert errors == []  # the module was patched: a note, not an error
    assert any("size probe failed" in n and "No HIP GPUs" in n for n in notes)
    assert base.get_cublas_workspace_size_bytes() == 512 * MB  # TE still cannot tell
    state["gpu"] = True
    assert base.get_cublas_workspace_size_bytes() == 2048 * MB  # max(): TE's larger size wins


def test_no_te_returns_empty(monkeypatch):
    for name in list(sys.modules):
        if name == "transformer_engine" or name.startswith("transformer_engine."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "transformer_engine", None)
    enlarged, unchanged, errors, notes = ws.enlarge_te_gemm_workspace(512 * MB)
    assert enlarged == [] and unchanged == [] and len(errors) == 2 and notes == []


def test_apply_raises_on_rocm_when_nothing_patched(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.version, "hip", "6.0", raising=False)
    monkeypatch.setattr(ws, "enlarge_te_gemm_workspace", lambda size: ([], [], ["x: ImportError"], []))
    with pytest.raises(RuntimeError, match="HIPBLASLT"):
        ws.apply_rocm_te_gemm_workspace("[t]")


def test_apply_warns_only_for_errors(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.version, "hip", "6.0", raising=False)
    logs, warns = [], []
    monkeypatch.setattr(ws, "log_rank_0", logs.append)
    monkeypatch.setattr(ws, "warning_rank_0", warns.append)
    monkeypatch.setattr(
        ws, "enlarge_te_gemm_workspace", lambda size: (["m"], [], [], ["m: size probe failed"])
    )
    assert ws.apply_rocm_te_gemm_workspace("[t]") == ["m"]
    assert warns == [] and any("size probe failed" in m for m in logs)
    monkeypatch.setattr(ws, "enlarge_te_gemm_workspace", lambda size: (["m"], [], ["n: OSError: bad"], []))
    ws.apply_rocm_te_gemm_workspace("[t]")
    assert warns == ["[t] n: OSError: bad"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, 512),
        ("", 512),
        (256, 256),
        ("256", 256),
        (" 1024 ", 1024),
        (512.0, 512),
        ("512.0", 512),
        ("5e2", 500),
        (" 64.0 ", 64),
    ],
)
def test_workspace_mb_of_args(raw, expected):
    from types import SimpleNamespace

    assert ws.workspace_mb_of(SimpleNamespace(te_gemm_workspace_mb=raw)) == expected
    assert ws.workspace_mb_of(SimpleNamespace()) == 512  # key absent
    assert ws.workspace_mb_of(None) == 512


def test_workspace_mb_of_accepts_numpy_ints():
    from types import SimpleNamespace

    np = pytest.importorskip("numpy")
    assert ws.workspace_mb_of(SimpleNamespace(te_gemm_workspace_mb=np.int64(128))) == 128
    assert ws.workspace_mb_of(SimpleNamespace(te_gemm_workspace_mb=np.float32(256.0))) == 256


@pytest.mark.parametrize(
    "raw", ["abc", "512M", 0, "-5", "1.5", 1.5, True, False, float("nan"), "inf", -1.0, [512]]
)
def test_workspace_mb_of_rejects_bad_values(raw):
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="te_gemm_workspace_mb"):
        ws.workspace_mb_of(SimpleNamespace(te_gemm_workspace_mb=raw))


def test_apply_uses_the_args_size(monkeypatch):
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.version, "hip", "6.0", raising=False)
    monkeypatch.setattr(ws, "log_rank_0", lambda *a, **k: None)
    sizes = []
    monkeypatch.setattr(
        ws, "enlarge_te_gemm_workspace", lambda size: (sizes.append(size) or (["m"], [], [], []))
    )
    ws.apply_rocm_te_gemm_workspace("[t]", SimpleNamespace(te_gemm_workspace_mb=64))
    assert sizes == [64 * MB]


@pytest.mark.parametrize("bad", [0, -5, 1.5, "abc", True, float("nan")])
def test_apply_validates_an_explicit_mb(monkeypatch, bad):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.version, "hip", "6.0", raising=False)
    monkeypatch.setattr(ws, "log_rank_0", lambda *a, **k: None)
    sizes = []
    monkeypatch.setattr(
        ws, "enlarge_te_gemm_workspace", lambda size: (sizes.append(size) or (["m"], [], [], []))
    )
    with pytest.raises(ValueError, match="positive integer"):
        ws.apply_rocm_te_gemm_workspace("[t]", mb=bad)
    assert sizes == []  # nothing patched with a bad size
    ws.apply_rocm_te_gemm_workspace("[t]", mb="256")
    ws.apply_rocm_te_gemm_workspace("[t]", mb=128.0)
    assert sizes == [256 * MB, 128 * MB]


@pytest.mark.parametrize("bad", ["abc", 0, 1.5])
def test_apply_validates_the_value_off_rocm_too(monkeypatch, bad):
    # the same yaml must not pass on CUDA and fail only on ROCm
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    with pytest.raises(ValueError):
        ws.apply_rocm_te_gemm_workspace("[t]", SimpleNamespace(te_gemm_workspace_mb=bad))
    assert ws.apply_rocm_te_gemm_workspace("[t]", SimpleNamespace(te_gemm_workspace_mb=512)) == []


def test_muse_spark_patch_passes_the_config_args_to_the_shared_helper(monkeypatch):
    # the te_gemm_workspace_mb config key reaches the helper through get_args(ctx)
    from types import SimpleNamespace

    from primus.backends.megatron.patches import (
        muse_spark_te_gemm_workspace_patches as msw,
    )
    from primus.core.patches import PatchContext

    args = SimpleNamespace(te_gemm_workspace_mb=64)
    seen = []
    monkeypatch.setattr(
        msw,
        "apply_rocm_te_gemm_workspace",
        lambda tag, args=None, mb=None, mb_name="mb": seen.append((tag, args, mb)),
    )
    ctx = PatchContext(
        backend="megatron", phase="before_train", extra={"module_config": SimpleNamespace(params=args)}
    )
    msw.patch_muse_spark_te_gemm_workspace(ctx)
    assert seen == [("[Patch:megatron.muse_spark.te_gemm_workspace]", args, None)]


def test_glm5_next_patch_uses_the_shared_helper_with_its_env(fake_te, monkeypatch):
    torch = pytest.importorskip("torch")
    from primus.backends.megatron.patches import glm5_next_patches as glm

    base, user, _ = fake_te
    monkeypatch.setattr(torch.version, "hip", "6.0", raising=False)
    monkeypatch.setattr(ws, "log_rank_0", lambda *a, **k: None)
    for off in ("0", "-1"):  # an integer <= 0: no enlargement, no error
        monkeypatch.setenv(glm.TE_GEMM_WORKSPACE_ENV, off)
        glm.patch_glm5_next_te_gemm_workspace(None)
        assert base.get_cublas_workspace_size_bytes() == 64 * MB
    monkeypatch.setenv(glm.TE_GEMM_WORKSPACE_ENV, "2.56e2")  # the shared parser's forms
    glm.patch_glm5_next_te_gemm_workspace(None)
    assert base.get_cublas_workspace_size_bytes() == user.get_cublas_workspace_size_bytes() == 256 * MB


def test_glm5_next_patch_defaults_to_512_and_names_the_env_on_a_bad_value(monkeypatch):
    torch = pytest.importorskip("torch")
    from primus.backends.megatron.patches import glm5_next_patches as glm

    monkeypatch.setattr(torch.version, "hip", "6.0", raising=False)
    monkeypatch.setattr(ws, "log_rank_0", lambda *a, **k: None)
    sizes = []
    monkeypatch.setattr(
        ws, "enlarge_te_gemm_workspace", lambda size: (sizes.append(size) or (["m"], [], [], []))
    )
    monkeypatch.delenv(glm.TE_GEMM_WORKSPACE_ENV, raising=False)
    glm.patch_glm5_next_te_gemm_workspace(None)
    assert sizes == [512 * MB]
    for bad in ("512M", "", "  "):  # set but blank is a config error too
        monkeypatch.setenv(glm.TE_GEMM_WORKSPACE_ENV, bad)
        with pytest.raises(ValueError, match="PRIMUS_TE_GEMM_WORKSPACE_MB"):
            glm.patch_glm5_next_te_gemm_workspace(None)
    assert sizes == [512 * MB]


def test_workspace_mb_from_env_is_the_shared_env_policy():
    env = "SOME_MB"
    assert ws.workspace_mb_from_env(env, {}) == ws.DEFAULT_MB
    assert ws.workspace_mb_from_env(env, {env: "1_024"}) == 1024
    for off in ("0", "-3", " -1 ", "-0"):  # only an integer <= 0 opts out
        assert ws.workspace_mb_from_env(env, {env: off}) is None
    # the YAML key's parser for the rest: these raise from env as from te_gemm_workspace_mb
    for bad in (
        "",
        " ",
        "abc",
        "1.5",
        "5\uff11\uff12",
        "0.0",
        "-1e3",
        "-0.5",
        "-inf",
        "-1e400",
        "nan",
        "0.5",
    ):
        with pytest.raises(ValueError, match=env):
            ws.workspace_mb_from_env(env, {env: bad})


def test_glm5_next_patch_warns_and_goes_on_when_no_te_size_function_is_found(monkeypatch):
    # best effort: a TE build without the expected symbols keeps running with its own
    # workspace (Muse Spark raises instead)
    torch = pytest.importorskip("torch")
    from primus.backends.megatron.patches import glm5_next_patches as glm

    monkeypatch.setattr(torch.version, "hip", "6.0", raising=False)
    monkeypatch.setenv(glm.TE_GEMM_WORKSPACE_ENV, "512")
    monkeypatch.setattr(ws, "enlarge_te_gemm_workspace", lambda size: ([], [], ["x: ImportError"], []))
    warned = []
    monkeypatch.setattr(ws, "warning_rank_0", warned.append)
    glm.patch_glm5_next_te_gemm_workspace(None)
    assert len(warned) == 1 and "glm5_next.te_gemm_workspace" in warned[0] and "HIPBLASLT" in warned[0]
    assert "going on" in warned[0]


def test_size_fn_logs_the_first_te_error_once(monkeypatch):
    warned = []
    monkeypatch.setattr(ws, "warning_rank_0", warned.append)

    def broken():
        raise RuntimeError("No HIP GPUs are available")

    fn = ws._size_fn(broken, 512 * MB)
    assert fn() == fn() == 512 * MB
    assert len(warned) == 1 and "No HIP GPUs" in warned[0]


def test_glm5_next_patch_is_a_noop_off_rocm(monkeypatch):
    torch = pytest.importorskip("torch")
    from primus.backends.megatron.patches import glm5_next_patches as glm

    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setenv(glm.TE_GEMM_WORKSPACE_ENV, "not a number")  # not read on CUDA
    monkeypatch.setattr(ws, "enlarge_te_gemm_workspace", lambda size: pytest.fail("called off ROCm"))
    glm.patch_glm5_next_te_gemm_workspace(None)
