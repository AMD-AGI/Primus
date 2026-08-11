###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the FLA KDA safe-autotune patch.

The patch drops ``num_stages >= 3`` from the autotune space of FLA's KDA
intra-chunk kernels, which AMD Triton 3.6 fails to compile.

Covered here:
  1. Gating -- ROCm only, and only for runs that actually reach FLA's KDA
     kernels. The second half is about not importing FLA into a run that has no
     use for it, so that is asserted directly rather than inferred.
  2. The surgery -- walking Triton's decorator chain to the object that owns
     ``.configs``, filtering it, and refusing to empty it.
  3. Safety -- FLA absent, kernel missing, ``.configs`` restructured, and
     re-application all end up as logged no-ops.
  4. The ``FLA_CACHE_MODE`` back door -- a cached best-config JSON is written
     straight into ``tuner.cache`` without consulting ``.configs``, so unsafe
     entries must be clamped there too, repeatedly and without losing the rest
     of the config.

Fake autotuner objects stand in for FLA, so none of this needs FLA installed or
a GPU present.
"""

import sys
import types
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest

pytest.importorskip("megatron")

import primus.backends.megatron.patches.fla_kda_autotune_patches as patch_mod
from primus.backends.megatron.patches._patch_guard import is_patched
from primus.core.patches.context import PatchContext


@dataclass
class FakeConfig:
    """Stand-in for ``triton.Config``.

    Carries the fields FLA populates on Triton >= 3.5.1 as well as the two the
    patch reads, so a config rebuilt instead of adjusted in place shows up as a
    lost field rather than passing silently.
    """

    num_warps: int
    num_stages: int
    kwargs: dict = field(default_factory=dict)
    num_ctas: int = 1
    maxnreg: int | None = None
    pre_hook: object = None
    ir_override: object = None


class FakeCachedAutotuner:
    """Mimics ``fla.ops.utils.cache.CachedAutotuner``'s relevant surface."""

    def __init__(self, configs, kernel_name="fake_kernel", cached_config=None):
        self.configs = configs
        self.cache = {}
        self.kernel_name = kernel_name
        self._cached_config = cached_config
        self.load_calls = 0

    def maybe_load_cached_config(self, autotune_key):
        self.load_calls += 1
        if self._cached_config is not None:
            # FLA builds a fresh triton.Config out of the JSON on every call, so
            # hand out a copy rather than the template -- otherwise a patch that
            # adjusts the installed config would appear to also fix the file.
            self.cache[autotune_key] = replace(self._cached_config)


class FakeHeuristics:
    """Mimics ``triton.runtime.autotuner.Heuristics``: a wrapper linked by .fn."""

    def __init__(self, fn):
        self.fn = fn


def _full_kda_space():
    """The real FLA space: num_warps [1,2,4,8] x num_stages [2,3,4]."""
    return [FakeConfig(num_warps=w, num_stages=s) for w in (1, 2, 4, 8) for s in (2, 3, 4)]


def _make_kernel(configs=None, **kwargs):
    """A Heuristics-wrapped autotuner, matching FLA's decorator order."""
    tuner = FakeCachedAutotuner(_full_kda_space() if configs is None else configs, **kwargs)
    return FakeHeuristics(tuner), tuner


@pytest.fixture
def fake_fla_module(monkeypatch):
    """Register a fake ``fla.ops.kda.chunk_intra`` and point the patch at it."""
    module = types.ModuleType("fake_fla_chunk_intra")
    tuners = {}
    for name in patch_mod._FLA_KDA_INTRA_KERNELS:
        kernel, tuner = _make_kernel(kernel_name=name)
        setattr(module, name, kernel)
        tuners[name] = tuner

    monkeypatch.setitem(sys.modules, "fake_fla_chunk_intra", module)
    monkeypatch.setattr(patch_mod, "_FLA_KDA_INTRA_MODULE", "fake_fla_chunk_intra")
    monkeypatch.setattr(patch_mod, "log_rank_0", lambda *a, **k: None)
    return module, tuners


@pytest.fixture(autouse=True)
def quiet_logs(monkeypatch):
    monkeypatch.setattr(patch_mod, "log_rank_0", lambda *a, **k: None)


@pytest.fixture
def on_rocm(monkeypatch):
    monkeypatch.setattr(patch_mod.torch.version, "hip", "7.2.0", raising=False)


def _ctx(args):
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=args)},
    )


def _apply_if_gated(ctx) -> bool:
    """Mimic the registry: evaluate the condition, run the patch only if it passes."""
    if not patch_mod._should_narrow_kda_autotune(ctx):
        return False
    patch_mod.patch_fla_kda_safe_autotune(ctx)
    return True


# ─── Gating: which configs select FLA KDA ────────────────────────────────────


@pytest.mark.parametrize(
    "fields, expected",
    [
        # zebra_llama hybrid stack.
        ({"use_fla_triton_kda": True}, "use_fla_triton_kda"),
        ({"use_fla_triton_kda": "true"}, "use_fla_triton_kda"),
        # Kimi K3 legacy chunk-kernel selector.
        ({"kda_backend": "fla"}, "kda_backend"),
        ({"kda_backend": "FLA"}, "kda_backend"),
        # The unified selector supersedes kda_backend when it is not None...
        (
            {"kda_backend": "eager", "use_kimi_k3_attention_backend": "fla"},
            "use_kimi_k3_attention_backend",
        ),
        # ...including when it turns KDA *off* despite kda_backend saying fla.
        ({"kda_backend": "fla", "use_kimi_k3_attention_backend": "eager"}, None),
        # Negatives: GDN / mamba / plain llama.
        ({}, None),
        ({"use_fla_triton_kda": False}, None),
        ({"kda_backend": "eager"}, None),
        ({"experimental_attention_variant": "gated_delta_net"}, None),
    ],
)
def test_uses_fla_kda(fields, expected):
    assert patch_mod._uses_fla_kda(SimpleNamespace(**fields)) == expected


# ─── Gating: the combined decision ───────────────────────────────────────────


@pytest.mark.parametrize(
    "fields, expected, why",
    [
        ({"use_fla_triton_kda": True}, True, "the zebra_llama spelling"),
        ({"kda_backend": "fla"}, True, "the Kimi K3 spelling"),
        ({"use_kimi_k3_attention_backend": "fla"}, True, "the unified selector"),
        ({"kda_backend": "eager"}, False, "eager KDA never enters the FLA kernel"),
        ({}, False, "GDN / mamba / plain llama do not reach the kernel"),
    ],
)
def test_gate_on_rocm_follows_the_kda_backend(on_rocm, fields, expected, why):
    assert patch_mod._should_narrow_kda_autotune(_ctx(SimpleNamespace(**fields))) is expected, why


@pytest.mark.parametrize("fields", [{"use_fla_triton_kda": True}, {"kda_backend": "fla"}])
def test_gate_is_off_on_a_non_rocm_platform(monkeypatch, fields):
    """The broken pass is in the AMD Triton backend; narrowing elsewhere is pure loss."""
    monkeypatch.setattr(patch_mod.torch.version, "hip", None, raising=False)
    assert patch_mod._should_narrow_kda_autotune(_ctx(SimpleNamespace(**fields))) is False


@pytest.mark.parametrize(
    "fields, expect_import",
    [
        ({"use_fla_triton_kda": True}, True),
        ({"kda_backend": "eager"}, False),
        ({}, False),
    ],
)
def test_fla_is_imported_only_for_kda_runs(monkeypatch, on_rocm, fields, expect_import):
    """The KDA half of the gate exists to avoid this import, so pin it.

    Narrowing a kernel that never launches would be harmless; pulling FLA into
    a run that has no other reason to load it is not.
    """
    imports = []

    def spy(name):
        imports.append(name)
        return types.ModuleType(name)

    monkeypatch.setattr(patch_mod.importlib, "import_module", spy)

    _apply_if_gated(_ctx(SimpleNamespace(**fields)))

    assert imports == ([patch_mod._FLA_KDA_INTRA_MODULE] if expect_import else [])


# ─── Finding the autotuner in Triton's decorator chain ───────────────────────


def test_find_autotuner_walks_the_wrapper_chain():
    """@triton.heuristics sits outside @fla_cache_autotune, so .configs is a hop down."""
    kernel, tuner = _make_kernel()
    assert not hasattr(kernel, "configs")
    assert patch_mod._find_autotuner(kernel) is tuner
    # Also tolerate extra decorator layers.
    assert patch_mod._find_autotuner(FakeHeuristics(FakeHeuristics(kernel))) is tuner


def test_find_autotuner_returns_none_when_absent():
    assert patch_mod._find_autotuner(SimpleNamespace()) is None
    assert patch_mod._find_autotuner(None) is None
    # A non-list .configs means the layout changed; do not guess.
    assert patch_mod._find_autotuner(SimpleNamespace(configs={"a": 1})) is None


def test_find_autotuner_does_not_loop_forever_on_a_cycle():
    node = SimpleNamespace()
    node.fn = node
    assert patch_mod._find_autotuner(node) is None


# ─── The filtering itself ────────────────────────────────────────────────────


def test_narrow_drops_unsafe_stages_and_keeps_the_warps_sweep():
    module = types.ModuleType("m")
    kernel, tuner = _make_kernel()
    module.k = kernel

    assert patch_mod._narrow_kernel_autotune(module, "k") is True

    assert len(tuner.configs) == 4
    assert {c.num_stages for c in tuner.configs} == {2}
    assert {c.num_warps for c in tuner.configs} == {1, 2, 4, 8}


def test_narrow_is_a_noop_when_the_space_is_already_safe():
    module = types.ModuleType("m")
    safe = [FakeConfig(num_warps=w, num_stages=2) for w in (1, 2, 4, 8)]
    kernel, tuner = _make_kernel(configs=safe)
    module.k = kernel

    patch_mod._narrow_kernel_autotune(module, "k")
    assert len(tuner.configs) == 4


def test_narrow_refuses_to_empty_the_autotune_space():
    """An all-unsafe space means FLA changed; an empty list would break Triton."""
    module = types.ModuleType("m")
    unsafe = [FakeConfig(num_warps=4, num_stages=4)]
    kernel, tuner = _make_kernel(configs=unsafe)
    module.k = kernel

    assert patch_mod._narrow_kernel_autotune(module, "k") is False
    assert tuner.configs == unsafe


def test_narrow_reports_false_for_a_missing_kernel():
    assert patch_mod._narrow_kernel_autotune(types.ModuleType("m"), "gone") is False


def test_narrow_reports_false_when_configs_are_unreachable():
    module = types.ModuleType("m")
    module.k = SimpleNamespace()
    assert patch_mod._narrow_kernel_autotune(module, "k") is False


# ─── Install: end to end, idempotency, safety ────────────────────────────────


def test_install_narrows_both_kda_intra_kernels(fake_fla_module):
    _module, tuners = fake_fla_module

    patch_mod._install_kda_safe_autotune_patch()

    assert set(tuners) == {
        "chunk_kda_bwd_kernel_intra",
        # Dead code today -- only reachable with safe_gate=True, which Primus
        # never sets -- but narrowed anyway so the protection is pre-positioned.
        "chunk_kda_fwd_kernel_intra_sub_chunk",
    }
    for name, tuner in tuners.items():
        assert {c.num_stages for c in tuner.configs} == {2}, name
        assert {c.num_warps for c in tuner.configs} == {1, 2, 4, 8}, name


def test_install_is_idempotent(fake_fla_module):
    module, tuners = fake_fla_module

    patch_mod._install_kda_safe_autotune_patch()
    first = {name: list(t.configs) for name, t in tuners.items()}
    guards = {name: t.maybe_load_cached_config for name, t in tuners.items()}

    patch_mod._install_kda_safe_autotune_patch()

    assert is_patched(module, patch_mod._PATCH_KEY)
    for name, tuner in tuners.items():
        assert tuner.configs == first[name]
        # The cache guard wraps, so a second wrap would double the indirection.
        assert tuner.maybe_load_cached_config is guards[name]


def test_install_is_a_noop_when_fla_is_absent(monkeypatch):
    monkeypatch.setattr(patch_mod, "_FLA_KDA_INTRA_MODULE", "primus_no_such_fla_module")
    patch_mod._install_kda_safe_autotune_patch()  # must not raise


def test_install_survives_a_kernel_disappearing(monkeypatch, fake_fla_module):
    module, tuners = fake_fla_module
    delattr(module, "chunk_kda_fwd_kernel_intra_sub_chunk")

    patch_mod._install_kda_safe_autotune_patch()  # must not raise

    surviving = tuners["chunk_kda_bwd_kernel_intra"]
    assert {c.num_stages for c in surviving.configs} == {2}


# ─── The FLA_CACHE_MODE back door ────────────────────────────────────────────


def test_cache_guard_clamps_an_unsafe_cached_config(fake_fla_module):
    """A best-config JSON goes straight into tuner.cache, bypassing .configs."""
    module, _ = fake_fla_module
    kernel, tuner = _make_kernel(cached_config=FakeConfig(num_warps=4, num_stages=4))
    module.chunk_kda_bwd_kernel_intra = kernel

    patch_mod._install_kda_safe_autotune_patch()
    key = ("some", "autotune", "key")
    tuner.maybe_load_cached_config(key)

    assert tuner.load_calls == 1, "the original loader must still run"
    assert tuner.cache[key].num_stages == patch_mod._MAX_SAFE_NUM_STAGES


def test_cache_guard_clamps_without_losing_the_rest_of_the_config(fake_fla_module):
    """Clamping must adjust num_stages, not rebuild the config around it.

    FLA populates num_ctas / maxnreg / pre_hook / ir_override on Triton >= 3.5.1,
    and a ``triton.Config(kwargs, num_warps=..., num_stages=...)`` rebuild would
    silently reset all of them to their defaults.
    """
    module, _ = fake_fla_module
    hook = object()
    cached = FakeConfig(
        num_warps=8,
        num_stages=4,
        kwargs={"BT": 64, "BS": 32},
        num_ctas=2,
        maxnreg=128,
        pre_hook=hook,
        ir_override="some/override.ttgir",
    )
    kernel, tuner = _make_kernel(cached_config=cached)
    module.chunk_kda_bwd_kernel_intra = kernel

    patch_mod._install_kda_safe_autotune_patch()
    tuner.maybe_load_cached_config("key")

    clamped = tuner.cache["key"]
    assert clamped.num_stages == patch_mod._MAX_SAFE_NUM_STAGES
    assert clamped == replace(cached, num_stages=patch_mod._MAX_SAFE_NUM_STAGES)
    assert clamped.pre_hook is hook


def test_cache_guard_keeps_always_mode_from_re_autotuning_every_step(fake_fla_module):
    """With FLA_CACHE_MODE=ALWAYS the guard runs on every kernel launch.

    ``should_check_fla_cache`` returns True unconditionally in that mode, so
    ``run()`` reinstalls the unsafe JSON config before every launch. Dropping the
    entry would leave the cache empty, sending Triton back to benchmarking, whose
    result the next reload overwrites again -- an autotune loop for the whole run.
    Clamping has to leave a safe entry in place on every pass instead.
    """
    module, _ = fake_fla_module
    kernel, tuner = _make_kernel(cached_config=FakeConfig(num_warps=4, num_stages=4))
    module.chunk_kda_bwd_kernel_intra = kernel

    patch_mod._install_kda_safe_autotune_patch()

    key = ("always", "mode", "key")
    for attempt in range(1, 4):
        tuner.maybe_load_cached_config(key)
        assert key in tuner.cache, f"pass {attempt} emptied the cache, so Triton re-autotunes"
        assert tuner.cache[key].num_stages == patch_mod._MAX_SAFE_NUM_STAGES, attempt
        assert tuner.cache[key].num_warps == 4, "clamping must not disturb the warp choice"


def test_cache_guard_clamps_entries_left_behind_by_earlier_loads(fake_fla_module):
    """The whole cache is swept, not just the key this call installed.

    Entries can predate the patch -- the guard is installed at before_train,
    after anything that already ran -- and clamping them is now free of the
    re-autotune loop that made a sweep costly.
    """
    module, _ = fake_fla_module
    kernel, tuner = _make_kernel(cached_config=FakeConfig(num_warps=1, num_stages=2))
    module.chunk_kda_bwd_kernel_intra = kernel
    tuner.cache["stale"] = FakeConfig(num_warps=2, num_stages=4)

    patch_mod._install_kda_safe_autotune_patch()
    tuner.maybe_load_cached_config("fresh")

    assert tuner.cache["stale"].num_stages == patch_mod._MAX_SAFE_NUM_STAGES
    assert tuner.cache["stale"].num_warps == 2
    assert tuner.cache["fresh"].num_stages == 2


def test_cache_guard_keeps_a_safe_cached_config(fake_fla_module):
    module, _ = fake_fla_module
    safe = FakeConfig(num_warps=4, num_stages=2)
    kernel, tuner = _make_kernel(cached_config=safe)
    module.chunk_kda_bwd_kernel_intra = kernel

    patch_mod._install_kda_safe_autotune_patch()
    tuner.maybe_load_cached_config("key")

    assert tuner.cache == {"key": safe}


def test_cache_guard_is_skipped_when_the_tuner_has_no_cache_hook():
    """Plain triton.Autotuner (no FLA config cache) must still be filtered."""
    module = types.ModuleType("m")
    tuner = SimpleNamespace(configs=_full_kda_space())
    module.k = FakeHeuristics(tuner)

    assert patch_mod._narrow_kernel_autotune(module, "k") is True
    assert {c.num_stages for c in tuner.configs} == {2}
    assert not hasattr(tuner, "maybe_load_cached_config")


# ─── Registration ────────────────────────────────────────────────────────────


def test_patch_is_registered_for_before_train():
    from primus.core.patches import PatchRegistry

    patch = next(
        (p for p in PatchRegistry.iter_patches("megatron", "before_train") if p.id == patch_mod._PATCH_KEY),
        None,
    )
    assert patch is not None, f"{patch_mod._PATCH_KEY} is not registered for megatron/before_train"
    assert patch.condition is patch_mod._should_narrow_kda_autotune


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
