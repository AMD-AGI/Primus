###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the pinned-config patch (MLA YaRN RoPE fusion, MoE permutation fusion).

Covered:
  1. Gating: only with use_primus_pinned_triton_configs, only for the enabled fusions (moe_permute_fusion also
     through turbo_sync_free_moe_stage >= 1, which validate_args applies after the patches run).
  2. Surgery on real ``triton.autotune`` objects: one config left, tuning cache dropped, every
     ``from ... import`` binding sees the pin.
  3. The real kernels: Megatron's four MLA RoPE autotuners and the four permutation autotuners (Primus' copy
     and TE's) end up with exactly the pinned config.
  4. A requested kernel that cannot be pinned raises before any autotuner is changed, and the registered
     patch aborts the run (SystemExit, which run_patches does not swallow); TE's optional module is skipped
     when absent.
None of this launches a kernel, so it runs without a GPU.
"""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("megatron")
triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

import primus.backends.megatron.patches.triton_pinned_config_patches as P  # noqa: E402
from primus.core.patches.context import PatchContext  # noqa: E402
from primus.core.patches.patch_registry import PatchRegistry  # noqa: E402
from primus.core.patches.patch_runner import run_patches  # noqa: E402


def _args(**kw):
    base = dict(
        use_primus_pinned_triton_configs=True,
        multi_latent_attention=True,
        apply_rope_fusion=True,
        moe_permute_fusion=True,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _ctx(args):
    return PatchContext(
        backend="megatron", phase="before_train", extra={"module_config": SimpleNamespace(params=args)}
    )


def _labels(args):
    return [label for label, _ in P.wanted_groups(args)]


def test_condition_requires_the_flag():
    assert P._condition(_ctx(_args())) is True
    assert P._condition(_ctx(_args(use_primus_pinned_triton_configs=False))) is False


def test_condition_requires_an_enabled_fusion():
    assert P._condition(_ctx(_args(apply_rope_fusion=False, moe_permute_fusion=False))) is False
    assert P._condition(_ctx(_args(multi_latent_attention=False, moe_permute_fusion=False))) is False
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
    assert P._condition(_ctx(_args(turbo_sync_free_moe_stage=1, **no_rope))) is True


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


def test_pin_group_on_real_autotuner(fake_module):
    mod = fake_module("_primus_test_pin_mod")
    alias = mod.kern  # a launcher module's `from ... import kern` binding
    mod.kern.cache[("k",)] = mod.kern.configs[2]
    pinned = P.pin_group(("_primus_test_pin_mod", ("kern", "plain"), {"BLOCK": 128}, 4, 3))
    assert pinned == ["kern"]  # "plain" is not autotuned: left alone
    assert len(alias.configs) == 1
    cfg = alias.configs[0]
    assert cfg.kwargs == {"BLOCK": 128} and cfg.num_warps == 4 and cfg.num_stages == 3
    # Autotuner.run benchmarks only when len(configs) > 1 and reads no stale tuning result.
    assert alias.cache == {}


def test_pin_group_rejects_unknown_meta(fake_module):
    fake_module("_primus_test_pin_mod2")
    with pytest.raises(RuntimeError):
        P.pin_group(("_primus_test_pin_mod2", ("kern",), {"BLOCK_SIZE": 1024}, 4, 3))


def test_pin_group_missing_kernel_raises(fake_module):
    fake_module("_primus_test_pin_mod3")
    with pytest.raises(RuntimeError):
        P.pin_group(("_primus_test_pin_mod3", ("nope",), {"BLOCK": 64}, 4, 3))


def test_real_mla_rope_kernels_pinned():
    F = pytest.importorskip("megatron.core.fusions.fused_mla_yarn_rope_apply")
    if not getattr(F, "HAVE_TRITON", False):
        pytest.skip("Megatron MLA RoPE fusion has no Triton")
    pinned = P.pin_group(P._ROPE_GROUP)
    assert sorted(pinned) == sorted(P._ROPE_GROUP[1])
    for name in P._ROPE_GROUP[1]:
        tuner = P._find_autotuner(getattr(F, name))
        assert len(tuner.configs) == 1
        c = tuner.configs[0]
        assert c.kwargs == {"BLOCK_H": 32} and c.num_warps == 4 and c.num_stages == 3


@pytest.mark.parametrize("group", P._PERMUTE_GROUPS, ids=lambda g: g[0])
def test_real_permutation_kernels_pinned(group):
    pytest.importorskip("transformer_engine")
    try:
        mod = __import__(group[0], fromlist=["_"])
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{group[0]} not importable: {exc!r}")
    pinned = P.pin_group(group)
    assert sorted(pinned) == sorted(group[1])
    for name in group[1]:
        tuner = P._find_autotuner(getattr(mod, name))
        assert len(tuner.configs) == 1
        c = tuner.configs[0]
        assert c.kwargs == {"BLOCK_SIZE": 1024} and c.num_warps == 4 and c.num_stages == 3
    if group[0].startswith("primus."):
        # The launch wrappers import the kernels by name: they must see the same pinned object.
        launch = __import__("primus.backends.transformer_engine.pytorch.triton.permutation", fromlist=["_"])
        assert (
            P._find_autotuner(launch._permute_kernel).configs
            == P._find_autotuner(mod._permute_kernel).configs
        )


def test_apply_raises_when_unpinnable(monkeypatch):
    monkeypatch.setattr(
        P, "wanted_groups", lambda args: [("mla_rope", (("_no_such_module_xyz", ("k",), {}, 4, 3),))]
    )
    with pytest.raises(RuntimeError, match="cannot pin mla_rope"):
        P.apply_pinned_configs(_args())


def test_apply_skips_missing_optional_te_module(monkeypatch, fake_module):
    fake_module("_primus_test_pin_mod4")
    groups = (
        ("_primus_test_pin_mod4", ("kern",), {"BLOCK": 256}, 4, 3),
        (P._TE_PERMUTE_MODULE, P._PERMUTE_KERNELS, {"BLOCK_SIZE": 1024}, 4, 3),
    )
    monkeypatch.setattr(P, "wanted_groups", lambda args: [("moe_permute", groups)])
    monkeypatch.setitem(sys.modules, P._TE_PERMUTE_MODULE, None)  # import raises ModuleNotFoundError
    done = P.apply_pinned_configs(_args())
    assert list(done) == ["_primus_test_pin_mod4"]
    assert len(done["_primus_test_pin_mod4"]["kern"]) == 1


def test_unpinnable_group_changes_nothing(monkeypatch, fake_module):
    mod = fake_module("_primus_test_pin_mod5")
    fake_module("_primus_test_pin_mod6")
    good = ("_primus_test_pin_mod5", ("kern",), {"BLOCK": 128}, 4, 3)
    bad = ("_primus_test_pin_mod6", ("kern", "nope"), {"BLOCK": 128}, 4, 3)  # second kernel missing
    monkeypatch.setattr(P, "wanted_groups", lambda args: [("mla_rope", (good,)), ("moe_permute", (bad,))])
    with pytest.raises(RuntimeError, match="cannot pin the moe_permute"):
        P.apply_pinned_configs(_args())
    assert len(mod.kern.configs) == 4  # the valid group, resolved first, was not pinned either
    assert len(sys.modules["_primus_test_pin_mod6"].kern.configs) == 4


def _registered(patch_id):
    # Some suites clear the global registry; re-import the patch module to register it again.
    if PatchRegistry.get(patch_id) is None:
        importlib.reload(P)
    assert PatchRegistry.get(patch_id) is not None


def test_registered_patch_aborts_the_run_when_unpinnable(monkeypatch, fake_module):
    _registered(P._PATCH_KEY)
    mod = fake_module("_primus_test_pin_mod7")
    good = ("_primus_test_pin_mod7", ("kern",), {"BLOCK": 128}, 4, 3)
    bad = ("_primus_test_pin_mod7", ("kern",), {"BLOCK_SIZE": 1024}, 4, 3)  # no such meta-parameter
    monkeypatch.setattr(P, "wanted_groups", lambda args: [("mla_rope", (good,)), ("moe_permute", (bad,))])
    with pytest.raises(SystemExit, match="cannot pin the moe_permute"):
        run_patches(
            backend="megatron",
            phase="before_train",
            enabled_ids=[P._PATCH_KEY],
            extra={"module_config": SimpleNamespace(params=_args())},
        )
    assert len(mod.kern.configs) == 4  # nothing pinned
