###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pinned Triton configs patch: gating, pinning real ``triton.autotune`` objects (stand-ins and the real MLA
RoPE and permutation kernels) and all-or-nothing aborts. No kernel is launched."""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("megatron")
triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

import primus.backends.megatron.patches.pinned_triton_configs_patches as P  # noqa: E402
from primus.core.patches.patch_runner import run_patches  # noqa: E402
from tests.unit_tests.backends.megatron.conftest import (  # noqa: E402
    patch_ctx,
    registered_patch,
)


def _args(**kw):
    base = dict(
        use_primus_pinned_triton_configs=True,
        multi_latent_attention=True,
        apply_rope_fusion=True,
        moe_permute_fusion=True,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _condition(args):
    return P._condition(patch_ctx(**vars(args)))


def _labels(args):
    return [label for label, _ in P._wanted_groups(args)]


def _pin_group(group):
    return list(P._pin(group, P._resolve_group(group)))


def test_condition_requires_the_flag():
    assert _condition(_args()) is True
    assert _condition(_args(use_primus_pinned_triton_configs=False)) is False


def test_condition_requires_an_enabled_fusion():
    assert _condition(_args(apply_rope_fusion=False, moe_permute_fusion=False)) is False
    assert _condition(_args(multi_latent_attention=False, moe_permute_fusion=False)) is False
    assert _labels(_args(apply_rope_fusion=False)) == ["moe_permute"]
    assert _labels(_args(moe_permute_fusion=False)) == ["mla_rope"]
    assert _labels(_args(multi_latent_attention=False)) == ["moe_permute"]
    assert _labels(_args()) == ["mla_rope", "moe_permute"]


def test_sync_free_moe_stage_counts_as_permute_fusion():
    # validate_args turns moe_permute_fusion on for stage >= 1 only after the before_train patches.
    no_rope = dict(apply_rope_fusion=False, moe_permute_fusion=False)
    assert _labels(_args(turbo_sync_free_moe_stage=1, **no_rope)) == ["moe_permute"]
    assert _labels(_args(turbo_sync_free_moe_stage=3, **no_rope)) == ["moe_permute"]
    assert _labels(_args(turbo_sync_free_moe_stage=0, **no_rope)) == []
    assert _labels(_args(turbo_sync_free_moe_stage=None, **no_rope)) == []
    assert _condition(_args(turbo_sync_free_moe_stage=1, **no_rope)) is True


def _fake_module(name):
    @triton.jit
    def _k(x_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(x_ptr + offs, tl.load(x_ptr + offs, mask=offs < n), mask=offs < n)

    tuned = triton.autotune(configs=[triton.Config({"BLOCK": b}) for b in (64, 128, 256, 512)], key=["n"])(_k)
    mod = types.ModuleType(name)
    mod.kern = tuned
    mod.plain = _k
    sys.modules[name] = mod
    return mod


@pytest.fixture
def fake_module():
    names = []

    def make(name):
        names.append(name)
        return _fake_module(name)

    yield make
    for name in names:
        sys.modules.pop(name, None)


def test_pin_on_real_autotuner(fake_module):
    mod = fake_module("_primus_test_pin_mod")
    alias = mod.kern  # a launcher module's `from ... import kern` binding
    mod.kern.cache[("k",)] = mod.kern.configs[2]
    assert _pin_group(P._PinGroup("_primus_test_pin_mod", ("kern", "plain"), {"BLOCK": 128}, 4, 3)) == [
        "kern"
    ]
    # "plain" is not autotuned: left alone
    assert len(alias.configs) == 1
    cfg = alias.configs[0]
    assert cfg.kwargs == {"BLOCK": 128} and cfg.num_warps == 4 and cfg.num_stages == 3
    # Autotuner.run benchmarks only when len(configs) > 1 and reads no stale tuning result.
    assert alias.cache == {}


def test_resolve_rejects_unknown_meta_and_missing_kernels(fake_module):
    fake_module("_primus_test_pin_mod2")
    with pytest.raises(RuntimeError, match="no argument"):
        P._resolve_group(P._PinGroup("_primus_test_pin_mod2", ("kern",), {"BLOCK_SIZE": 1024}, 4, 3))
    with pytest.raises(RuntimeError, match="not found"):
        P._resolve_group(P._PinGroup("_primus_test_pin_mod2", ("nope",), {"BLOCK": 64}, 4, 3))


@pytest.fixture
def restore_real_autotuners():
    """Pinning mutates the real module-global autotuners; put their configs back for later tests."""
    saved = []
    for group in (P._ROPE_GROUP, *P._PERMUTE_GROUPS):
        try:
            mod = importlib.import_module(group.module)
        except Exception:  # noqa: BLE001 - not importable here: nothing to restore
            continue
        for name in group.kernels:
            tuner = P._find_autotuner(getattr(mod, name, None))
            if tuner is not None:
                saved.append((tuner, list(tuner.configs), dict(tuner.cache)))
    yield
    for tuner, configs, cache in saved:
        tuner.configs = configs
        tuner.cache.clear()
        tuner.cache.update(cache)


def test_real_mla_rope_kernels_pinned(restore_real_autotuners):
    F = pytest.importorskip("megatron.core.fusions.fused_mla_yarn_rope_apply")
    if not getattr(F, "HAVE_TRITON", False):
        pytest.skip("Megatron MLA RoPE fusion has no Triton")
    assert sorted(_pin_group(P._ROPE_GROUP)) == sorted(P._ROPE_GROUP.kernels)
    for name in P._ROPE_GROUP.kernels:
        tuner = P._find_autotuner(getattr(F, name))
        assert len(tuner.configs) == 1
        c = tuner.configs[0]
        assert c.kwargs == {"BLOCK_H": 32} and c.num_warps == 4 and c.num_stages == 3


@pytest.mark.parametrize("group", P._PERMUTE_GROUPS, ids=lambda g: g.module)
def test_real_permutation_kernels_pinned(group, restore_real_autotuners):
    pytest.importorskip("transformer_engine")
    try:
        mod = importlib.import_module(group.module)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{group.module} not importable: {exc!r}")
    assert sorted(_pin_group(group)) == sorted(group.kernels)
    for name in group.kernels:
        tuner = P._find_autotuner(getattr(mod, name))
        assert len(tuner.configs) == 1
        c = tuner.configs[0]
        assert c.kwargs == {"BLOCK_SIZE": 1024} and c.num_warps == 4 and c.num_stages == 3
    if group.module.startswith("primus."):
        # The launch wrappers import the kernels by name: they must see the same pinned object.
        launch = importlib.import_module("primus.backends.transformer_engine.pytorch.triton.permutation")
        assert (
            P._find_autotuner(launch._permute_kernel).configs
            == P._find_autotuner(mod._permute_kernel).configs
        )


def test_apply_raises_when_unpinnable(monkeypatch):
    group = P._PinGroup("_no_such_module_xyz", ("k",), {}, 4, 3)
    monkeypatch.setattr(P, "_wanted_groups", lambda args: [("mla_rope", (group,))])
    with pytest.raises(RuntimeError, match="cannot pin the mla_rope"):
        P._apply_pinned_configs(_args())


def test_apply_skips_the_optional_te_module(monkeypatch, fake_module):
    fake_module("_primus_test_pin_mod4")
    te_group = P._PERMUTE_GROUPS[1]
    assert te_group.optional
    groups = (P._PinGroup("_primus_test_pin_mod4", ("kern",), {"BLOCK": 256}, 4, 3), te_group)
    monkeypatch.setattr(P, "_wanted_groups", lambda args: [("moe_permute", groups)])
    monkeypatch.setitem(sys.modules, te_group.module, None)  # import raises ModuleNotFoundError
    done = P._apply_pinned_configs(_args())
    assert list(done) == ["_primus_test_pin_mod4"]
    assert len(done["_primus_test_pin_mod4"]["kern"]) == 1


def test_unpinnable_group_changes_nothing(monkeypatch, fake_module):
    mod = fake_module("_primus_test_pin_mod5")
    fake_module("_primus_test_pin_mod6")
    good = P._PinGroup("_primus_test_pin_mod5", ("kern",), {"BLOCK": 128}, 4, 3)
    bad = P._PinGroup(
        "_primus_test_pin_mod6", ("kern", "nope"), {"BLOCK": 128}, 4, 3
    )  # second kernel missing
    monkeypatch.setattr(P, "_wanted_groups", lambda args: [("mla_rope", (good,)), ("moe_permute", (bad,))])
    with pytest.raises(RuntimeError, match="cannot pin the moe_permute"):
        P._apply_pinned_configs(_args())
    assert len(mod.kern.configs) == 4  # the valid group, resolved first, was not pinned either
    assert len(sys.modules["_primus_test_pin_mod6"].kern.configs) == 4


def test_registered_patch_aborts_the_run_when_unpinnable(monkeypatch, fake_module):
    registered_patch(P._PATCH_KEY, P)
    mod = fake_module("_primus_test_pin_mod7")
    good = P._PinGroup("_primus_test_pin_mod7", ("kern",), {"BLOCK": 128}, 4, 3)
    bad = P._PinGroup(
        "_primus_test_pin_mod7", ("kern",), {"BLOCK_SIZE": 1024}, 4, 3
    )  # no such meta-parameter
    monkeypatch.setattr(P, "_wanted_groups", lambda args: [("mla_rope", (good,)), ("moe_permute", (bad,))])
    with pytest.raises(SystemExit, match="cannot pin the moe_permute"):
        run_patches(
            backend="megatron",
            phase="before_train",
            enabled_ids=[P._PATCH_KEY],
            extra={"module_config": SimpleNamespace(params=_args())},
        )
    assert len(mod.kern.configs) == 4  # nothing pinned
