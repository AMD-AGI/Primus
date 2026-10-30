###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the compile-friendly FP8 linear path (``quantization/_fp8_traceable``).

WHY THIS TEST EXISTS:
  Nothing on this path fails loudly. The quantized operands reach the GEMM as raw
  fp8 bytes plus a separate scale, so a wrong layout, a wrong scale or a stale
  buffer produces plausible numbers at full speed -- a run that trains, converges
  a little worse, and gets believed. So the tests pin:

    - bit-identity with ``Float8Linear``, forward and all three gradients, and of
      the Inductor cast with the extension cast it replaces;
    - the unpadded operand layout, which Primus-Turbo 0.4+ no longer produces by
      default (a padded K = 64 operand gives a gradient of the wrong shape);
    - the delayed-scaling schedule, the zero-amax floor, and that only a
      shared-input group's owner writes the shared scale;
    - that eval (validation) forwards leave the delayed scales alone, and that a
      compiled refresh writes the same scales as an eager one;
    - that a dependent setting without what it depends on raises instead of
      quietly measuring something else.

  The numerics and compile tests need Primus-Turbo on a GPU; the rest stub it.
"""

import logging
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from primus.backends.automodel.quantization import _fp8_traceable as mod  # noqa: E402
from primus.backends.automodel.quantization import fp8_linear  # noqa: E402

LOGGER = "primus.backends.automodel.quantization._fp8_traceable"


def _turbo_on_a_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    import importlib.util

    return importlib.util.find_spec("primus_turbo") is not None


requires_turbo_on_a_gpu = pytest.mark.skipif(
    not _turbo_on_a_gpu(), reason="the FP8 kernels need primus_turbo and a GPU"
)


@pytest.fixture(autouse=True)
def clean_module_state(monkeypatch):
    """``resolve`` binds module globals; put them back after every test."""
    monkeypatch.setattr(mod, "_DELAYED_SCALE_STATE", {"step": 0, "refresh": True})
    for name in ("_AMAX_WARMUP_STEPS", "_AMAX_REFRESH_INTERVAL", "_INDUCTOR_CAST", "_GEMM_BACKEND"):
        monkeypatch.setattr(mod, name, getattr(mod, name))
    monkeypatch.setattr(mod, "_QUANTIZE_EXTRA_ARGS", ())


# --------------------------------------------------------------------------- #
# Settings                                                                    #
# --------------------------------------------------------------------------- #
class TestSettings:
    def test_everything_is_off_by_default(self):
        assert not mod.requested()
        assert mod.read_settings() == mod.Settings(False, False, 2, 16, False, False, "hipblaslt")

    def test_not_requested_resolves_to_none_without_touching_primus_turbo(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "primus_turbo", None)
        assert mod.resolve(object()) is None

    def test_the_full_stack_reads_back(self, set_option):
        for key, value in (
            ("fp8_linear_traceable", True),
            ("fp8_linear_delayed_scaling", "true"),
            ("fp8_linear_amax_warmup_steps", 3),
            ("fp8_linear_amax_refresh_interval", "8"),
            ("fp8_linear_inductor_cast", True),
            ("fp8_linear_shared_input_scales", True),
            ("fp8_linear_gemm_backend", "FlyDSL"),
        ):
            set_option(f"primus_turbo.{key}", value)

        assert mod.read_settings() == mod.Settings(True, True, 3, 8, True, True, "flydsl")

    @pytest.mark.parametrize(
        "settings,needed",
        [
            ({"fp8_linear_delayed_scaling": True}, "fp8_linear_traceable"),
            ({"fp8_linear_inductor_cast": True}, "fp8_linear_traceable"),
            ({"fp8_linear_shared_input_scales": True}, "fp8_linear_traceable"),
            ({"fp8_linear_gemm_backend": "flydsl"}, "fp8_linear_traceable"),
            ({"fp8_linear_traceable": True, "fp8_linear_inductor_cast": True}, "fp8_linear_delayed_scaling"),
            (
                {"fp8_linear_traceable": True, "fp8_linear_shared_input_scales": True},
                "fp8_linear_delayed_scaling",
            ),
        ],
    )
    def test_a_dependent_setting_without_its_dependency_raises(self, set_option, settings, needed):
        for key, value in settings.items():
            set_option(f"primus_turbo.{key}", value)

        with pytest.raises(ValueError, match=f"needs primus_turbo.{needed}: true"):
            mod.read_settings()

    def test_an_unknown_gemm_backend_raises(self, set_option):
        set_option("primus_turbo.fp8_linear_traceable", True)
        set_option("primus_turbo.fp8_linear_gemm_backend", "hipblas")

        with pytest.raises(ValueError, match="fp8_linear_gemm_backend='hipblas' invalid"):
            mod.read_settings()

    def test_a_negative_schedule_raises(self, set_option):
        set_option("primus_turbo.fp8_linear_amax_refresh_interval", -1)

        with pytest.raises(ValueError, match="must be non-negative"):
            mod.read_settings()

    def test_a_schedule_that_never_measures_raises(self, set_option):
        set_option("primus_turbo.fp8_linear_traceable", True)
        set_option("primus_turbo.fp8_linear_delayed_scaling", True)
        set_option("primus_turbo.fp8_linear_amax_warmup_steps", 0)
        set_option("primus_turbo.fp8_linear_amax_refresh_interval", 0)

        with pytest.raises(ValueError, match="would never be measured"):
            mod.read_settings()

    def test_every_setting_is_declared(self):
        from primus.backends.automodel import options

        set_keys = {
            f"fp8_linear_{k}" for k in ("traceable", "delayed_scaling", "inductor_cast", "gemm_backend")
        }
        set_keys |= {"fp8_linear_amax_warmup_steps", "fp8_linear_amax_refresh_interval"}
        set_keys |= {"fp8_linear_shared_input_scales"}
        options.load({"primus_turbo": {key: True for key in set_keys}})

        assert options.undeclared() == []


# --------------------------------------------------------------------------- #
# Settings left on without the FP8 swap                                       #
# --------------------------------------------------------------------------- #
def _fp8_patch():
    from primus.core.patches.patch_registry import PatchRegistry

    return PatchRegistry.get("automodel.quantization.fp8_linear")


def test_traceable_settings_without_the_fp8_swap_log_a_warning(set_option, caplog):
    set_option("primus_turbo.fp8_linear_traceable", True)
    set_option("primus_turbo.fp8_linear_gemm_backend", "flydsl")

    with caplog.at_level(logging.WARNING, logger=fp8_linear.__name__):
        assert _fp8_patch().condition(None) is False

    assert "primus_turbo.fp8_linear_traceable, primus_turbo.fp8_linear_gemm_backend set" in caplog.text


@pytest.mark.parametrize("swap", [True, False])
def test_the_fp8_patch_condition_is_quiet_unless_settings_go_unused(set_option, caplog, swap):
    """With the swap on they are used; with it off and them off there is nothing to say."""
    set_option("primus_turbo.fp8_linear", swap)
    set_option("primus_turbo.fp8_linear_traceable", swap)

    with caplog.at_level(logging.WARNING, logger=fp8_linear.__name__):
        assert _fp8_patch().condition(None) is swap

    assert "not applied" not in caplog.text


# --------------------------------------------------------------------------- #
# Primus-Turbo's backend API across versions                                  #
# --------------------------------------------------------------------------- #
def _stub_backend_api(monkeypatch, members):
    """``primus_turbo.pytorch.core.backend`` with the given ``BackendType`` members."""
    import enum

    api = types.ModuleType("primus_turbo.pytorch.core.backend")
    api.BackendType = enum.Enum("BackendType", members)
    api.PrecisionType = enum.Enum("PrecisionType", ["FP8"])
    api.GlobalBackendManager = types.SimpleNamespace(
        auto_tune_enabled=lambda: False, get_gemm_backend=lambda precision: None
    )
    for name in ("primus_turbo", "primus_turbo.pytorch", "primus_turbo.pytorch.core"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, api.__name__, api)
    return api


def test_the_gemm_backend_is_bound_as_primus_turbos_id(monkeypatch):
    api = _stub_backend_api(monkeypatch, ["CK", "HIPBLASLT", "TRITON", "FLYDSL"])

    assert mod._gemm_backend_id("flydsl") == api.BackendType.FLYDSL.value


@pytest.mark.parametrize("installed", ["no_flydsl", "no_backend_api"])
def test_a_backend_the_installed_primus_turbo_lacks_is_a_clear_error(monkeypatch, installed):
    if installed == "no_flydsl":
        _stub_backend_api(monkeypatch, ["CK", "HIPBLASLT", "TRITON"])
    else:
        monkeypatch.setitem(sys.modules, "primus_turbo.pytorch.core.backend", None)

    with pytest.raises(ValueError, match="'flydsl' is not available in the installed primus_turbo"):
        mod._gemm_backend_id("flydsl")


@pytest.mark.parametrize(
    "returns,reported",
    [
        (lambda api: api.BackendType.HIPBLASLT, True),
        (lambda api: types.SimpleNamespace(backend=api.BackendType.HIPBLASLT, auto_tune=False), True),
        (lambda api: types.SimpleNamespace(backend=None, auto_tune=False), False),
        (lambda api: None, False),
    ],
    ids=["0.3", "0.4", "0.4-unset", "unset"],
)
def test_a_pinned_gemm_backend_is_reported_from_either_primus_turbo_api(
    monkeypatch, caplog, returns, reported
):
    """Primus-Turbo 0.3 returns the BackendType itself, 0.4 and later a BackendChoice."""
    api = _stub_backend_api(monkeypatch, ["HIPBLASLT", "FLYDSL"])
    api.GlobalBackendManager.get_gemm_backend = lambda precision: returns(api)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        mod._warn_if_backend_overridden("flydsl")

    assert ("pins the FP8 GEMMs to HIPBLASLT" in caplog.text) is reported


# --------------------------------------------------------------------------- #
# The quantize op's padding arguments                                         #
# --------------------------------------------------------------------------- #
def _fake_op_with_arguments(count):
    schema = types.SimpleNamespace(arguments=[object()] * count)
    op = types.SimpleNamespace(default=types.SimpleNamespace(_schema=schema))
    return types.SimpleNamespace(quantize_fp8_tensorwise=op)


@pytest.mark.parametrize("count,expected", [(3, ()), (5, (1, 1)), (6, (1, 1))])
def test_the_unpadded_arguments_follow_the_op_schema(monkeypatch, count, expected):
    """Primus-Turbo 0.3 takes (x, dtype, scale); 0.4 adds two padding arguments, 0.5 a third input."""
    monkeypatch.setattr(
        torch.ops, "primus_turbo_cpp_extension", _fake_op_with_arguments(count), raising=False
    )

    assert mod._unpadded_quantize_args() == expected


# --------------------------------------------------------------------------- #
# Casting with a known scale                                                  #
# --------------------------------------------------------------------------- #
class FakeExt:
    """The two ``primus_turbo_cpp_extension`` ops this path uses."""

    def __init__(self, scale_inv=0.25):
        self.quantized = []
        self.transposed = 0
        self._scale_inv = scale_inv

    def quantize_fp8_tensorwise(self, x, fp8_dtype, scale, *extra):
        self.quantized.append((x, fp8_dtype, scale, extra))
        return torch.zeros_like(x, dtype=torch.uint8), torch.tensor(self._scale_inv)

    def transpose_2d(self, q, dim0, dim1):
        self.transposed += 1
        return q.transpose(dim0, dim1)


@pytest.fixture
def fake_ext(monkeypatch):
    ext = FakeExt()
    monkeypatch.setattr(torch.ops, "primus_turbo_cpp_extension", ext, raising=False)
    return ext


class ForbiddenExt:
    def __getattr__(self, name):
        raise AttributeError(f"the Inductor cast must not call primus_turbo_cpp_extension.{name}")


def test_the_extension_cast_passes_the_unpadded_arguments(fake_ext, monkeypatch):
    monkeypatch.setattr(mod, "_QUANTIZE_EXTRA_ARGS", (1, 1))
    x = torch.zeros(64, 63, dtype=torch.bfloat16)

    q, q_t, scale_inv = mod.cast_transpose_scaled(x, torch.float8_e4m3fn, torch.ones(()))

    assert len(fake_ext.quantized) == 1 and fake_ext.quantized[0][3] == (1, 1)
    assert fake_ext.transposed == 1
    assert q.shape == (64, 63) and q_t.shape == (63, 64) and q_t.is_contiguous()
    assert float(scale_inv) == 0.25


def test_the_inductor_cast_bypasses_the_extension_and_saturates(monkeypatch):
    monkeypatch.setattr(torch.ops, "primus_turbo_cpp_extension", ForbiddenExt(), raising=False)
    monkeypatch.setattr(mod, "_INDUCTOR_CAST", True)
    x = torch.tensor([[1.0, -2.0, 1000.0], [0.5, -1000.0, 0.0]], dtype=torch.bfloat16)

    q, q_t, scale_inv = mod.cast_transpose_scaled(x, torch.float8_e4m3fn, torch.tensor(2.0))

    assert q.dtype == torch.float8_e4m3fn and q.shape == (2, 3)
    assert q_t.shape == (3, 2) and q_t.is_contiguous() and torch.equal(q_t, q.t())
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    assert q.float().tolist() == [[2.0, -4.0, fp8_max], [1.0, -fp8_max, 0.0]], "saturates, no inf"
    assert float(scale_inv) == 0.5


# --------------------------------------------------------------------------- #
# The autograd Functions, with the GEMM stubbed                               #
# --------------------------------------------------------------------------- #
TURBO_STUBS = (
    "primus_turbo",
    "primus_turbo.pytorch",
    "primus_turbo.pytorch.kernels",
    "primus_turbo.pytorch.kernels.gemm",
    "primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl",
)


class FakeCtx:
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


@pytest.fixture
def stub_gemm(monkeypatch):
    """Stub the FP8 GEMM; these tests are about what reaches it, not what it computes."""
    calls = []

    def gemm_fp8_impl(a, a_s, trans_a, b, b_s, trans_b, out_dtype, trans_c, **kwargs):
        calls.append({"a": a, "a_s": a_s, "b": b, "b_s": b_s, "trans_c": trans_c, **kwargs})
        return torch.zeros(4, 4)

    for name in TURBO_STUBS:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl"].gemm_fp8_impl = gemm_fp8_impl
    monkeypatch.setattr(mod, "_GEMM_BACKEND", 7)
    return calls


@pytest.fixture
def record_scaled_casts(monkeypatch):
    calls = []

    def cast_transpose_scaled(x, fp8_dtype, scale):
        calls.append((x, fp8_dtype, scale))
        return torch.zeros(4, 8, dtype=torch.uint8), torch.zeros(8, 4, dtype=torch.uint8), torch.ones(())

    monkeypatch.setattr(mod, "cast_transpose_scaled", cast_transpose_scaled)
    return calls


def run_delayed_forward(refresh, owner=True):
    ctx = FakeCtx()
    a = torch.zeros(4, 8, dtype=torch.bfloat16)
    b = torch.zeros(4, 8, dtype=torch.bfloat16)
    scales = tuple(torch.ones(()) for _ in range(3))
    mod._TraceableFP8GemmNTDelayed.forward(
        ctx, a, b, *scales, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2, 1, refresh, owner
    )
    return ctx, (a, b), scales


def test_the_dynamic_function_measures_both_operands_and_the_gradient(fake_ext, stub_gemm):
    ctx = FakeCtx()
    a = torch.zeros(4, 8, dtype=torch.bfloat16)
    b = torch.zeros(4, 8, dtype=torch.bfloat16)
    mod._TraceableFP8GemmNT.forward(ctx, a, b, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2, 1)

    grads = mod._TraceableFP8GemmNT.backward(ctx, torch.zeros(4, 4, dtype=torch.bfloat16))

    assert [q[2] for q in fake_ext.quantized] == [None, None, None], "every cast measures its amax"
    assert [q[1] for q in fake_ext.quantized] == [torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float8_e5m2]
    assert [c["trans_c"] for c in stub_gemm] == [False, False, True]
    assert all(c["default_backend"] == 7 for c in stub_gemm), "the bound backend reaches every GEMM"
    assert len(grads) == 6


def test_a_scaled_forward_casts_both_operands_with_their_stashed_scales(
    fake_ext, stub_gemm, record_scaled_casts
):
    _, (a, b), (s_a, s_b, _) = run_delayed_forward(refresh=False)

    assert [c[0] for c in record_scaled_casts] == [a, b]
    assert [c[2] for c in record_scaled_casts] == [s_a, s_b]
    assert fake_ext.quantized == [], "a scaled forward must not compute an amax"


def test_a_refresh_forward_measures_the_amax_and_stashes_its_reciprocal(
    fake_ext, stub_gemm, record_scaled_casts
):
    _, _, (s_a, s_b, _) = run_delayed_forward(refresh=True)

    assert record_scaled_casts == [], "a refresh has no scale to cast with yet"
    assert [q[2] for q in fake_ext.quantized] == [None, None]
    assert float(s_a) == 4.0 and float(s_b) == 4.0


def test_an_all_zero_operand_cannot_stash_an_infinite_scale(monkeypatch, stub_gemm):
    monkeypatch.setattr(torch.ops, "primus_turbo_cpp_extension", FakeExt(scale_inv=0.0), raising=False)

    _, _, (s_a, _, _) = run_delayed_forward(refresh=True)

    assert torch.isfinite(s_a).all()
    assert float(s_a) == pytest.approx(1.0 / mod._SCALE_INV_FLOOR)


@pytest.mark.parametrize("owner", [True, False])
def test_a_refresh_writes_the_input_scale_only_from_the_owner(fake_ext, stub_gemm, owner):
    _, _, (s_a, s_b, _) = run_delayed_forward(refresh=True, owner=owner)

    assert float(s_a) == (4.0 if owner else 1.0)
    assert float(s_b) == 4.0, "every linear owns its weight scale"


def test_the_backward_reuses_the_forwards_transposes_and_its_own_scale(
    fake_ext, stub_gemm, record_scaled_casts
):
    ctx, _, (_, _, s_g) = run_delayed_forward(refresh=False)
    grad_out = torch.zeros(4, 8, dtype=torch.bfloat16)

    grads = mod._TraceableFP8GemmNTDelayed.backward(ctx, grad_out)

    assert record_scaled_casts[-1][0] is grad_out
    assert record_scaled_casts[-1][1] is torch.float8_e5m2, "gradients take the backward-stage dtype"
    assert record_scaled_casts[-1][2] is s_g
    assert fake_ext.quantized == []
    assert len(stub_gemm) == 3, "one forward GEMM and the two backward GEMMs"
    assert len(grads) == 11, "one gradient slot per forward input"


# --------------------------------------------------------------------------- #
# The refresh schedule                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "warmup,interval,expected",
    [
        (2, 4, [True, True, False, False, True, False, False, False, True, False]),
        (2, 16, [True, True] + [False] * 8),
        (0, 3, [True, False, False, True, False, False, True, False, False, True]),
        (1, 0, [True] + [False] * 9),
    ],
)
def test_the_schedule_refreshes_in_warmup_and_on_the_interval(monkeypatch, warmup, interval, expected):
    monkeypatch.setattr(mod, "_AMAX_WARMUP_STEPS", warmup)
    monkeypatch.setattr(mod, "_AMAX_REFRESH_INTERVAL", interval)
    refreshes = []
    for _ in expected:
        mod._advance_delayed_scale_step()
        refreshes.append(mod._DELAYED_SCALE_STATE["refresh"])

    assert refreshes == expected


# --------------------------------------------------------------------------- #
# Discovering linears that share an input                                     #
# --------------------------------------------------------------------------- #
def scaled_linear():
    linear = torch.nn.Linear(4, 4, bias=False)
    linear.register_buffer("_fp8_scale_x", torch.ones(()), persistent=False)
    linear._fp8_scale_x_owner = True
    return linear


class ToyBlock(torch.nn.Module):
    """Self-attention Q/K/V and an MLP-in on x, cross-attention K/V on ctx, a reused linear."""

    def __init__(self):
        super().__init__()
        for name in ("to_q", "to_k", "to_v", "proj_mlp", "add_k", "add_v", "twice", "solo"):
            setattr(self, name, scaled_linear())
        self.plain = torch.nn.Linear(4, 4)

    def forward(self, x, ctx):
        out = self.to_q(x) + self.to_k(x) + self.to_v(x) + self.proj_mlp(x)
        out = out + self.add_k(ctx) + self.add_v(ctx)
        y = self.plain(x)
        out = out + self.twice(y) + self.twice(out)
        return self.solo(out) + self.twice(y) * 0


def run_discovery(block):
    with torch.no_grad(), mod.discover_shared_inputs(block):
        block(torch.randn(2, 4), torch.randn(2, 4))


def test_linears_reading_the_same_tensor_alias_their_first_members_scale(caplog):
    block = ToyBlock()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        run_discovery(block)

    for name in ("to_k", "to_v", "proj_mlp"):
        assert getattr(block, name)._fp8_scale_x is block.to_q._fp8_scale_x
        assert getattr(block, name)._fp8_scale_x_owner is False
    assert block.to_q._fp8_scale_x_owner is True
    assert block.add_v._fp8_scale_x is block.add_k._fp8_scale_x
    assert block.add_k._fp8_scale_x is not block.to_q._fp8_scale_x, "ctx and x must never share"
    assert block.add_k._fp8_scale_x_owner is True and block.add_v._fp8_scale_x_owner is False
    assert "6 linears in 2 groups" in caplog.text


def test_a_linear_called_more_than_once_keeps_its_own_scale():
    """Its scale reflects whichever input it saw last, so it cannot stand in for a group."""
    block = ToyBlock()
    before = block.twice._fp8_scale_x

    run_discovery(block)

    assert block.twice._fp8_scale_x is before and block.twice._fp8_scale_x_owner is True
    assert block.solo._fp8_scale_x_owner is True


def test_the_shared_buffer_is_still_a_registered_buffer():
    """FSDP and .to() see buffers through _buffers; an aliased plain attribute would drift."""
    block = ToyBlock()

    run_discovery(block)

    assert block.to_k._buffers["_fp8_scale_x"] is block.to_q._buffers["_fp8_scale_x"]


def test_discovery_removes_its_hooks():
    block = ToyBlock()

    run_discovery(block)

    assert all(not m._forward_pre_hooks for m in block.modules())


def test_a_freed_tensor_whose_memory_is_reused_is_not_mistaken_for_the_original():
    class Sequential(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = scaled_linear(), scaled_linear()

        def forward(self, x, ctx):
            h = self.a(x * 2)
            return self.b(h * 2)

    block = Sequential()

    run_discovery(block)

    assert block.b._fp8_scale_x is not block.a._fp8_scale_x


# --------------------------------------------------------------------------- #
# The step counter runs discovery once, eagerly and without grad              #
# --------------------------------------------------------------------------- #
def test_the_step_counter_discovers_on_the_first_forward_only(monkeypatch):
    stances = []
    entered = []
    original_set_stance = torch.compiler.set_stance

    def set_stance(stance, *args, **kwargs):
        stances.append(stance)
        return original_set_stance(stance, *args, **kwargs)

    original_discover = mod.discover_shared_inputs

    def discover(root):
        entered.append(root)
        return original_discover(root)

    monkeypatch.setattr(torch.compiler, "set_stance", set_stance)
    monkeypatch.setattr(mod, "discover_shared_inputs", discover)
    block = ToyBlock()
    grad_enabled = []
    block.solo.register_forward_pre_hook(lambda m, args: grad_enabled.append(torch.is_grad_enabled()))
    mod.install_delayed_scale_step_counter(block, discover_shared_input_scales=True)

    outputs = [block(torch.randn(2, 4), torch.randn(2, 4)) for _ in range(3)]

    assert entered == [block] and stances == ["force_eager"]
    assert grad_enabled == [False, True, True, True], "the discovery forward saves no activations"
    assert all(out.requires_grad for out in outputs)
    assert block.to_k._fp8_scale_x is block.to_q._fp8_scale_x
    assert mod._DELAYED_SCALE_STATE["step"] == 3


def test_without_shared_input_scales_the_step_counter_never_discovers(monkeypatch):
    monkeypatch.setattr(mod, "discover_shared_inputs", lambda root: pytest.fail("discovery ran"))
    block = ToyBlock()
    mod.install_delayed_scale_step_counter(block)

    with torch.no_grad():
        block(torch.randn(2, 4), torch.randn(2, 4))

    assert block.to_k._fp8_scale_x is not block.to_q._fp8_scale_x
    assert mod._DELAYED_SCALE_STATE["step"] == 1


def test_the_step_counter_is_installed_once():
    block = ToyBlock()
    mod.install_delayed_scale_step_counter(block)
    wrapped = block.forward

    mod.install_delayed_scale_step_counter(block)

    assert block.forward is wrapped


def test_an_eval_forward_neither_advances_the_schedule_nor_discovers(monkeypatch):
    """AutoModel validates in eval mode, and validation batches are not training state."""
    entered = []
    original_discover = mod.discover_shared_inputs

    def discover(root):
        entered.append(root)
        return original_discover(root)

    monkeypatch.setattr(mod, "discover_shared_inputs", discover)
    block = ToyBlock()
    mod.install_delayed_scale_step_counter(block, discover_shared_input_scales=True)
    inputs = (torch.randn(2, 4), torch.randn(2, 4))

    block.eval()
    with torch.no_grad():
        block(*inputs)

    assert mod._DELAYED_SCALE_STATE == {"step": 0, "refresh": True}
    assert entered == [] and block.to_k._fp8_scale_x is not block.to_q._fp8_scale_x

    block.train()
    block(*inputs)

    assert mod._DELAYED_SCALE_STATE["step"] == 1 and entered == [block]


# --------------------------------------------------------------------------- #
# The swap, the kernels and the compiler, on a GPU                            #
# --------------------------------------------------------------------------- #
@pytest.fixture
def every_linear_converts(monkeypatch):
    # AutoModel's predicate is not the subject here, and importing it pulls in diffusers.
    monkeypatch.setattr(fp8_linear, "should_convert", lambda name, linear: True)


def _swap(set_option, **settings):
    for key, value in settings.items():
        set_option(f"primus_turbo.{key}", value)
    model = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256))
    model = model.to("cuda", torch.bfloat16)
    fp8_linear.replace_linears(model, "toy")
    return model


@requires_turbo_on_a_gpu
class TestSwap:
    def test_the_traceable_class_is_swapped_in_when_requested(self, set_option, every_linear_converts):
        from primus_turbo.pytorch.modules import Float8Linear

        model = _swap(set_option, fp8_linear=True, fp8_linear_traceable=True)

        assert type(model[0]).__name__ == "TraceableFloat8Linear" and isinstance(model[0], Float8Linear)
        assert not hasattr(model[0], "_fp8_scale_x")
        assert not getattr(model, "_primus_fp8_step_counter", False)

    def test_delayed_scaling_adds_the_scale_buffers_and_the_step_counter(
        self, set_option, every_linear_converts
    ):
        model = _swap(set_option, fp8_linear=True, fp8_linear_traceable=True, fp8_linear_delayed_scaling=True)

        assert all(hasattr(model[0], name) for name in mod._SCALE_BUFFERS)
        assert not any(name in model[0].state_dict() for name in mod._SCALE_BUFFERS), "never checkpointed"
        assert model._primus_fp8_step_counter is True

    def test_without_the_setting_the_stock_class_stays(self, set_option, every_linear_converts):
        model = _swap(set_option, fp8_linear=True)

        assert type(model[0]).__name__ == "Float8Linear"

    def test_an_unsupported_granularity_keeps_the_stock_class_and_says_so(
        self, set_option, every_linear_converts, caplog
    ):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            model = _swap(
                set_option, fp8_linear=True, fp8_linear_granularity="rowwise", fp8_linear_traceable=True
            )

        assert type(model[0]).__name__ == "Float8Linear"
        assert "fp8_linear_traceable ignored: granularity=ROWWISE" in caplog.text


def _pair(in_features, out_features, *, delayed_scaling, fmt="E4M3"):
    """A stock Float8Linear and a traceable one holding the same weights."""
    from primus_turbo.pytorch.core.low_precision import (
        Float8QuantConfig,
        Format,
        ScalingGranularity,
    )
    from primus_turbo.pytorch.modules import Float8Linear

    from primus.backends.automodel import options

    sections = {"primus_turbo": {"fp8_linear_traceable": True, "fp8_linear_delayed_scaling": delayed_scaling}}
    options.load(sections)
    config = Float8QuantConfig(format=getattr(Format, fmt), granularity=ScalingGranularity.TENSORWISE)
    assert mod.resolve(config) is not None
    traceable_cls = mod.make_float8_linear(Float8Linear, delayed_scaling=delayed_scaling)
    kwargs = dict(bias=True, config=config, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(0)
    stock = Float8Linear(in_features, out_features, **kwargs)
    traceable = traceable_cls(in_features, out_features, **kwargs)
    traceable.load_state_dict(stock.state_dict())
    return stock, traceable


def _forward_backward(linear, x, grad_out):
    x = x.detach().clone().requires_grad_(True)
    linear.zero_grad(set_to_none=True)
    out = linear(x)
    out.backward(grad_out)
    return out.detach(), x.grad, linear.weight.grad, linear.bias.grad


def _assert_bit_identical(expected, got):
    for name, e, g in zip(("output", "input grad", "weight grad", "bias grad"), expected, got):
        assert e.shape == g.shape, f"{name}: {tuple(e.shape)} vs {tuple(g.shape)}"
        assert torch.equal(e, g), f"{name} differs"


# FLUX's x_embedder (K = 64), an attention projection, and an MLP-in.
SHAPES = [(1024, 64, 3072), (1024, 3072, 3072), (512, 3072, 12288)]


@requires_turbo_on_a_gpu
@pytest.mark.parametrize("tokens,in_features,out_features", SHAPES)
@pytest.mark.parametrize("fmt", ["E4M3", "HYBRID"])
def test_the_traceable_linear_is_bit_identical_to_float8linear(tokens, in_features, out_features, fmt):
    stock, traceable = _pair(in_features, out_features, delayed_scaling=False, fmt=fmt)
    x = torch.randn(4, tokens // 4, in_features, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(4, tokens // 4, out_features, device="cuda", dtype=torch.bfloat16)

    _assert_bit_identical(_forward_backward(stock, x, grad_out), _forward_backward(traceable, x, grad_out))


@requires_turbo_on_a_gpu
@pytest.mark.parametrize("tokens,in_features,out_features", SHAPES)
def test_a_refresh_forward_is_bit_identical_and_a_scaled_one_stays_close(tokens, in_features, out_features):
    """A refresh measures the amax as the dynamic path does; a scaled forward reuses it."""
    stock, traceable = _pair(in_features, out_features, delayed_scaling=True)
    x = torch.randn(tokens, in_features, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(tokens, out_features, device="cuda", dtype=torch.bfloat16)
    expected = _forward_backward(stock, x, grad_out)

    mod._DELAYED_SCALE_STATE["refresh"] = True
    _assert_bit_identical(expected, _forward_backward(traceable, x, grad_out))

    mod._DELAYED_SCALE_STATE["refresh"] = False
    scaled = _forward_backward(traceable, x, grad_out)
    for e, g in zip(expected, scaled):
        assert e.shape == g.shape
        assert (g.float() - e.float()).norm() <= 1e-2 * e.float().norm()


@requires_turbo_on_a_gpu
@pytest.mark.parametrize("refresh", [True, False])
def test_an_eval_forward_quantizes_dynamically_and_leaves_the_scales_alone(refresh):
    """A held-out batch must not become a training scale, nor be cast with a stale one."""
    stock, traceable = _pair(3072, 3072, delayed_scaling=True)
    traceable.eval()
    mod._DELAYED_SCALE_STATE["refresh"] = refresh
    x = torch.randn(512, 3072, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        want, got = stock(x), traceable(x)

    assert torch.equal(want, got)
    assert all(float(getattr(traceable, name)) == 1.0 for name in mod._SCALE_BUFFERS)


@requires_turbo_on_a_gpu
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("is_fwd_stage", [True, False])
@pytest.mark.parametrize("shape", [(512, 256), (1024, 3072), (129, 33)])
def test_the_inductor_cast_is_bit_identical_to_the_extension_cast(monkeypatch, compiled, is_fwd_stage, shape):
    """The GEMM reads raw bytes, so the plain-op cast must match exactly, saturation included."""
    from primus_turbo.pytorch.core.low_precision import Format

    _pair(16, 16, delayed_scaling=True)  # binds the quantize arguments for this Primus-Turbo
    fp8_dtype = mod.fp8_stage_dtype(Format.HYBRID, is_fwd_stage)
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda") * 3
    scale = (torch.finfo(fp8_dtype).max / x.abs().max().float() * 0.5).reshape(())
    x[0, 0] = 1e4

    monkeypatch.setattr(mod, "_INDUCTOR_CAST", False)
    expected = mod.cast_transpose_scaled(x, fp8_dtype, scale)

    monkeypatch.setattr(mod, "_INDUCTOR_CAST", True)
    fn = lambda t, s: mod.cast_transpose_scaled(t, fp8_dtype, s)  # noqa: E731
    if compiled:
        torch._dynamo.reset()
        fn = torch.compile(fn, fullgraph=True)
    got = fn(x, scale)

    assert got[0].dtype == expected[0].dtype and got[1].shape == expected[1].shape
    assert torch.equal(got[0].view(torch.uint8), expected[0].view(torch.uint8))
    assert torch.equal(got[1].view(torch.uint8), expected[1].view(torch.uint8))
    assert torch.equal(got[2], expected[2])


@requires_turbo_on_a_gpu
@pytest.mark.parametrize("delayed_scaling", [False, True])
def test_a_compiled_traceable_linear_has_no_graph_breaks(delayed_scaling):
    """``fullgraph=True`` raises on a graph break, which is what stock Float8Linear causes."""
    stock, traceable = _pair(3072, 3072, delayed_scaling=delayed_scaling)
    x = torch.randn(512, 3072, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(512, 3072, device="cuda", dtype=torch.bfloat16)
    expected = _forward_backward(stock, x, grad_out)
    torch._dynamo.reset()
    compiled = torch.compile(traceable, fullgraph=True)

    _assert_bit_identical(expected, _forward_backward(compiled, x, grad_out))


@requires_turbo_on_a_gpu
def test_a_compiled_delayed_linear_matches_eager_with_one_graph_per_kind_of_forward():
    """A compiled refresh that stopped writing the scales would leave every later cast at 1.0."""
    from torch._dynamo.testing import CompileCounterWithBackend

    _, eager = _pair(3072, 3072, delayed_scaling=True)
    _, traceable = _pair(3072, 3072, delayed_scaling=True)
    x = torch.randn(512, 3072, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(512, 3072, device="cuda", dtype=torch.bfloat16)
    torch._dynamo.reset()
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(traceable, backend=counter, fullgraph=True)

    for refresh in (True, False, True, False):
        mod._DELAYED_SCALE_STATE["refresh"] = refresh
        _assert_bit_identical(_forward_backward(eager, x, grad_out), _forward_backward(compiled, x, grad_out))
        for name in mod._SCALE_BUFFERS:
            assert torch.equal(getattr(traceable, name), getattr(eager, name)), name

    assert all(float(getattr(traceable, name)) != 1.0 for name in mod._SCALE_BUFFERS)
    assert counter.frame_count == 2, "one graph for refresh forwards, one for scaled forwards"
