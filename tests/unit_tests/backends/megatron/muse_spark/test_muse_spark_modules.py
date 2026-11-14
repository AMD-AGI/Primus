###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark module behaviour on CPU: fp32 islands through casts, the bda closure,
and the build_args patches (selective recompute, CP comm type)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")

from primus.backends.megatron.core.transformer.muse_spark.moe import (  # noqa: E402
    MuseSparkTopKRouter,
)
from primus.backends.megatron.core.transformer.muse_spark.norms import (  # noqa: E402
    MuseSparkCenteredRMSNorm,
    MuseSparkGatedResidual,
    MuseSparkInputScaledRMSNorm,
    MuseSparkRMSNorm,
    casts_to_half,
)
from primus.backends.megatron.patches.args import (  # noqa: E402
    muse_spark_arg_compat_patches as rc,
)


def _bare_router(weight):
    r = MuseSparkTopKRouter.__new__(MuseSparkTopKRouter)
    torch.nn.Module.__init__(r)
    r.weight = torch.nn.Parameter(weight)
    return r


def test_router_weight_survives_bf16_cast_exactly():
    w = torch.full((4, 8), 1.0 + 1e-3)  # not representable in bf16
    r = _bare_router(w.clone())
    r.bfloat16()
    assert r.weight.dtype == torch.float32
    assert torch.equal(r.weight.detach(), w)
    r.half()
    assert torch.equal(r.weight.detach(), w)


def test_router_weight_built_in_bf16_is_promoted():
    r = _bare_router(torch.ones(2, 3, dtype=torch.bfloat16))
    r.bfloat16()
    assert r.weight.dtype == torch.float32


def _gated_residual(kind="attn", init=0.0):
    cfg = SimpleNamespace(
        hidden_size=8,
        post_norm_eps=1e-8,
        residual_gate_temperature=0.3,
        residual_gate_init_value=init,
        sequence_parallel=False,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        memory_efficient_layer_norm=False,
        layernorm_zero_centered_gamma=False,
    )
    return MuseSparkGatedResidual(cfg, kind=kind)


def test_fp32_islands_are_marked_as_muse_spark_params(monkeypatch):
    # the optimizer checkpoint patches take their Muse Spark path only for an optimizer that
    # holds a marked param (muse_spark_optimizer_ckpt_patches.holds_muse_spark_params)
    from primus.backends.megatron.core.models.muse_spark.model_types import (
        MUSE_SPARK_PARAM_ATTR,
    )

    assert getattr(_gated_residual().gate, MUSE_SPARK_PARAM_ATTR) is True
    # the router marks its weight in __init__: run it on a stubbed base (a real build needs
    # the parallel state); the base creates the weight as TopKRouter does
    from primus.backends.megatron.core.transformer.muse_spark import moe as moe_mod

    def base_init(self, config, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.weight = torch.nn.Parameter(torch.zeros(4, 8))

    monkeypatch.setattr(moe_mod._RouterBase, "__init__", base_init)
    router = MuseSparkTopKRouter.__new__(MuseSparkTopKRouter)
    MuseSparkTopKRouter.__init__(router, SimpleNamespace(moe_router_output_multiplier=0.5))
    assert getattr(router.weight, MUSE_SPARK_PARAM_ATTR) is True
    assert router.output_multiplier == 0.5


def test_gate_init_value_starts_near_identity_residual():
    # 0.0: zero (HF-like) init
    assert torch.equal(_gated_residual().gate.detach(), torch.zeros(8))
    from primus.backends.megatron.core.models.muse_spark.muse_spark_transformer_config import (
        MuseSparkTransformerConfig,
    )

    # the config's default init (not a literal) must start near an identity residual
    init = MuseSparkTransformerConfig.__dataclass_fields__["residual_gate_init_value"].default
    cfg = SimpleNamespace(
        hidden_size=8,
        post_norm_eps=1e-8,
        residual_gate_temperature=0.3,
        residual_gate_init_value=init,
        sequence_parallel=False,
    )
    m = MuseSparkGatedResidual(cfg, kind="attn")
    assert torch.equal(m.gate.detach(), torch.full((8,), float(init)))
    beta = torch.sigmoid(m.gate / 0.3)
    alpha = torch.sqrt(torch.sigmoid(-m.gate / 0.3) * (1 + beta))
    assert float(beta.max()) < 0.05 and abs(float(alpha.mean()) - 1.0) < 0.01
    with torch.no_grad():
        m.gate.zero_()
    m.reset_parameters()
    assert torch.equal(m.gate.detach(), torch.full((8,), -0.9))


@pytest.mark.parametrize("kind", ["attn", "mlp"])
def test_gate_stays_fp32_and_bda_closure_reenters_module(kind):
    m = _gated_residual(kind)
    with torch.no_grad():
        m.gate.fill_(0.5 + 1e-4)  # not representable in bf16
    before = m.gate.detach().clone()
    m.bfloat16()
    assert m.gate.dtype == torch.float32 and torch.equal(m.gate.detach(), before)
    calls = []
    m.register_forward_pre_hook(lambda mod, args: calls.append(len(args)))
    bda = m(True, False)
    x, res = torch.randn(3, 8, dtype=torch.bfloat16), torch.randn(3, 8, dtype=torch.bfloat16)
    out = bda((x, None), res, 0.0)
    # hooks fire once per bda use: only around the tensor call that gates (FSDP2 unshards
    # the gate there); the bool-only factory call does not go through nn.Module.__call__
    assert calls == [4]
    assert callable(m.forward(True, False))
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.double(), _gated_reference(m, kind, x, res), **BF16_TOL)


# two bf16 ulps (2 * 2**-8) relative, plus an absolute floor for values near 0
BF16_TOL = dict(rtol=2 * 2**-8, atol=2 * 2**-8)


def _gated_reference(m, kind, branch, residual):
    """Independent fp64 reference written from the HF formulas, not from the module (kept
    in fp64: callers compare the module output as ``out.double()``):
    ``alpha * residual + beta * post_norm(branch)`` with ``w = gate / T``,
    ``beta = sigmoid(w)``, ``alpha = sqrt(max(sigmoid(-w) * (1 + beta), 1e-3))`` and
    post_norm the weightless RMSNorm (attn) or ``rms(x * (1 + scale))`` (mlp)."""
    b = branch.double()
    if kind == "mlp":
        b = b * (1.0 + m.post_norm.weight.detach().double())
    b = b / torch.sqrt(b.pow(2).mean(-1, keepdim=True) + 1e-8)
    w = m.gate.detach().double() / 0.3
    beta = 1.0 / (1.0 + torch.exp(-w))
    alpha = torch.sqrt(torch.clamp((1.0 / (1.0 + torch.exp(w))) * (1.0 + beta), min=1e-3))
    return alpha * residual.double() + beta * b


@pytest.mark.parametrize("kind", ["attn", "mlp"])
def test_gated_residual_matches_an_independent_fp64_reference(kind):
    torch.manual_seed(0)
    m = _gated_residual(kind)
    with torch.no_grad():
        m.gate.copy_(torch.linspace(-2.0, 2.0, 8))  # alphas / betas far from 0.5
        if kind == "mlp":
            m.post_norm.weight.copy_(torch.linspace(-0.5, 0.5, 8))
    m.bfloat16()
    x = torch.randn(5, 8, dtype=torch.bfloat16) * 3
    res = torch.randn(5, 8, dtype=torch.bfloat16)
    out = m(True, False)((x, None), res, 0.0)
    ref = _gated_reference(m, kind, x, res)
    # bf16 output vs the full-precision reference: within the bf16 rounding of the output
    # and of the bf16 post-norm (the module's only rounding points)
    torch.testing.assert_close(out.double(), ref, **BF16_TOL)
    # the reference is not trivially close: a wrong gate (residual * beta only) fails it
    wrong = torch.sigmoid(m.gate.detach().double() / 0.3) * res.double()
    assert not torch.allclose(wrong, ref, **BF16_TOL)


def _ctx(**args):
    return SimpleNamespace(extra={"backend_args": SimpleNamespace(**args)})


def test_missing_backend_args_raises():
    with pytest.raises(RuntimeError, match="backend_args"):
        rc.patch_muse_spark_cp_comm.__wrapped__(SimpleNamespace(extra={}))


@pytest.mark.parametrize(
    "comm, expected",
    [
        ("p2p", "all_gather"),  # language_model.yaml default
        (["p2p"], ["all_gather"]),  # Megatron argparse default
        ("allgather", "all_gather"),
        ("all_gather", "all_gather"),
        ("a2a", "a2a"),
        (["all_gather", "p2p", "a2a"], ["all_gather", "all_gather", "a2a"]),
        (None, "all_gather"),
        ([None], ["all_gather"]),
    ],
)
def test_cp_comm_resolution(comm, expected):
    assert rc.resolve_cp_comm_type(comm) == expected


@pytest.mark.parametrize(
    "comm, reason",
    [
        ("a2a+p2p", "no sliding-window backend"),
        (["all_gather", "a2a+p2p"], "no sliding-window backend"),
        ("ring", "unknown cp_comm_type entry 'ring'.*accepted"),
        ("allgather2", "unknown cp_comm_type entry 'allgather2'.*accepted"),
    ],
)
def test_cp_comm_rejected(comm, reason):
    with pytest.raises(ValueError, match=reason):
        rc.resolve_cp_comm_type(comm)


def test_cp_comm_patch_only_with_window_and_cp(monkeypatch):
    msgs = []
    monkeypatch.setattr(rc, "warning_rank_0", msgs.append)
    ctx = _ctx(context_parallel_size=1, window_size=(2048, 0), cp_comm_type=["p2p"])
    rc.patch_muse_spark_cp_comm(ctx)  # CP 1: nothing to check
    assert ctx.extra["backend_args"].cp_comm_type == ["p2p"] and msgs == []
    ctx = _ctx(context_parallel_size=2, window_size=None, cp_comm_type=["p2p"])
    rc.patch_muse_spark_cp_comm(ctx)  # no sliding window: p2p is fine
    assert ctx.extra["backend_args"].cp_comm_type == ["p2p"] and msgs == []
    ctx = _ctx(context_parallel_size=2, window_size=(2048, 0), cp_comm_type=["p2p"])
    rc.patch_muse_spark_cp_comm(ctx)
    assert ctx.extra["backend_args"].cp_comm_type == ["all_gather"] and len(msgs) == 1
    ctx = _ctx(context_parallel_size=2, window_size=(2048, 0), cp_comm_type=["all_gather"])
    rc.patch_muse_spark_cp_comm(ctx)
    assert len(msgs) == 1  # unchanged: no warning
    assert "no sliding-window attention backend" in msgs[0]
    # the TE-spelling rewrite says so, not that p2p lacks a backend
    rc.patch_muse_spark_cp_comm(
        _ctx(context_parallel_size=2, window_size=(2048, 0), cp_comm_type="allgather")
    )
    assert "spelled 'all_gather'" in msgs[1] and "p2p" not in msgs[1]
    with pytest.raises(ValueError):
        rc.patch_muse_spark_cp_comm.__wrapped__(
            _ctx(context_parallel_size=2, window_size=(2048, 0), cp_comm_type="a2a+p2p")
        )


def test_merged_args_reads_backend_args_first_then_params():
    from primus.backends.megatron.core.models.muse_spark.model_types import (
        merged_args_of,
    )

    params = SimpleNamespace(use_turbo_deepep=True, expert_model_parallel_size=8)
    ctx = _ctx(expert_model_parallel_size=1)
    ctx.extra["module_config"] = SimpleNamespace(params=params)
    m = merged_args_of(ctx)
    assert m.expert_model_parallel_size == 1 and m.use_turbo_deepep is True
    assert getattr(m, "missing", "dflt") == "dflt"
    m.use_turbo_deepep = False  # written where the key lives
    assert params.use_turbo_deepep is False and not hasattr(ctx.extra["backend_args"], "use_turbo_deepep")


def test_merged_args_refuses_a_context_without_backend_args():
    from primus.backends.megatron.core.models.muse_spark.model_types import (
        MergedArgs,
        merged_args_of,
    )

    ctx = _ctx(expert_model_parallel_size=1)
    ctx.extra.pop("backend_args")
    ctx.extra["module_config"] = SimpleNamespace(params=SimpleNamespace(use_turbo_deepep=True))
    with pytest.raises(RuntimeError, match="no backend_args"):
        merged_args_of(ctx)  # a write would reach params only
    with pytest.raises(ValueError, match="at least one"):
        MergedArgs(None, None)  # a write would go nowhere
    only = SimpleNamespace(a=1)
    m = MergedArgs(only, None)
    m.b = 2
    assert only.b == 2 and m.a == 1


def test_merged_args_does_not_let_a_none_default_hide_a_configured_value():
    import numpy as np

    from primus.backends.megatron.core.models.muse_spark.model_types import (
        MergedArgs,
        bool_flag,
    )

    m = MergedArgs(SimpleNamespace(k=None, n=None), SimpleNamespace(k=True))
    assert m.k is True and m.n is None
    assert bool_flag(np.bool_(True), "k") is True and bool_flag(np.bool_(False), "k") is False


def test_deepep_condition_sees_the_backend_args_ep_override(monkeypatch):
    # the yaml (params) says EP 8, the CLI (backend_args) says EP 1: DeepEP must go off
    monkeypatch.setattr(rc, "warning_rank_0", lambda m: None)
    params = SimpleNamespace(
        use_turbo_deepep=True,
        enable_primus_turbo=True,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        tensor_model_parallel_size=1,
    )
    ctx = _ctx(expert_model_parallel_size=1, expert_tensor_parallel_size=1, tensor_model_parallel_size=1)
    ctx.extra["module_config"] = SimpleNamespace(params=params)
    rc.patch_muse_spark_deepep_group_size(ctx)
    assert params.use_turbo_deepep is False


@pytest.mark.parametrize(
    "dispatcher, want", [("flex", "alltoall"), ("alltoall", "alltoall"), ("allgather", "allgather")]
)
def test_deepep_off_moves_a_flex_dispatcher_to_alltoall(monkeypatch, dispatcher, want):
    # flex needs DeepEP: when DeepEP goes off the dispatcher must not stay flex
    logs = []
    monkeypatch.setattr(rc, "warning_rank_0", logs.append)
    params = SimpleNamespace(
        use_turbo_deepep=True,
        enable_primus_turbo=True,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        tensor_model_parallel_size=1,
        moe_token_dispatcher_type=dispatcher,
        moe_enable_deepep=dispatcher == "flex",
    )
    ctx = _ctx(expert_model_parallel_size=1, expert_tensor_parallel_size=1, tensor_model_parallel_size=1)
    ctx.extra["module_config"] = SimpleNamespace(params=params)
    rc.patch_muse_spark_deepep_group_size(ctx)
    assert params.use_turbo_deepep is False and params.moe_token_dispatcher_type == want
    assert params.moe_enable_deepep is False
    assert f"moe_token_dispatcher_type: {want}" in logs[0]


@pytest.mark.parametrize(
    "params, backend, found",
    [
        (dict(turbo_sync_free_moe_stage=1), {}, ["turbo_sync_free_moe_stage=1"]),
        (dict(turbo_sync_free_moe_stage="3"), {}, ["turbo_sync_free_moe_stage=3"]),
        (
            dict(enable_primus_turbo=True, moe_use_fused_router_with_aux_score=True),
            {},
            ["moe_use_fused_router_with_aux_score=True"],
        ),
        ({}, dict(turbo_sync_free_moe_stage=2), ["turbo_sync_free_moe_stage=2"]),
        # split across namespaces: only the merged view sees both flags
        (
            dict(moe_use_fused_router_with_aux_score=True),
            dict(enable_primus_turbo=True),
            ["moe_use_fused_router_with_aux_score=True"],
        ),
        (
            dict(enable_primus_turbo=True),
            dict(moe_use_fused_router_with_aux_score=True),
            ["moe_use_fused_router_with_aux_score=True"],
        ),
    ],
)
def test_fused_router_is_rejected(params, backend, found):
    ctx = _ctx(**backend)
    ctx.extra["module_config"] = SimpleNamespace(params=SimpleNamespace(**params))
    with pytest.raises(ValueError, match="e_score_correction_bias") as err:
        rc.patch_muse_spark_no_fused_router.__wrapped__(ctx)
    for f in found:
        assert f in str(err.value)


@pytest.mark.parametrize(
    "params",
    [
        dict(
            turbo_sync_free_moe_stage=0, enable_primus_turbo=True, moe_use_fused_router_with_aux_score=False
        ),
        dict(turbo_sync_free_moe_stage=None),
        dict(enable_primus_turbo=False, moe_use_fused_router_with_aux_score=True),  # no Turbo router then
        {},
    ],
)
def test_stage_0_and_the_unfused_router_pass(params):
    ctx = _ctx()
    ctx.extra["module_config"] = SimpleNamespace(params=SimpleNamespace(**params))
    rc.patch_muse_spark_no_fused_router(ctx)
    rc.patch_muse_spark_no_fused_router(_ctx())  # no module_config: backend_args only


def test_backend_args_helper_is_public_in_model_types():
    from primus.backends.megatron.core.models.muse_spark import model_types
    from primus.backends.megatron.patches.args import (
        muse_spark_layer_patterns_patches as lpp,
    )

    assert rc.backend_args_of is model_types.backend_args_of is lpp.backend_args_of
    assert not hasattr(lpp, "_backend_args")


def test_island_opt_in_attribute():
    # the Muse Spark FSDP2 patch shards fp32 params as their own group only when opted in
    from primus.backends.megatron.core.models.muse_spark.fsdp2_islands import (
        ISLAND_ATTR,
    )

    assert ISLAND_ATTR == "fsdp2_dtype_island"
    assert MuseSparkTopKRouter.fsdp2_dtype_island is True
    assert MuseSparkGatedResidual.fsdp2_dtype_island is True


def test_casts_to_half_probe():
    assert casts_to_half(lambda t: t.bfloat16())
    assert casts_to_half(lambda t: t.half())
    assert not casts_to_half(lambda t: t.float())
    assert not casts_to_half(lambda t: t.double())
    assert not casts_to_half(lambda t: t.to("cpu"))


def test_router_weight_follows_device_and_fp64_casts():
    # R5-11: a pure device move or a non-half cast is not undone.
    r = _bare_router(torch.ones(2, 3))
    r.double()
    assert r.weight.dtype == torch.float64
    r.to("cpu")
    assert r.weight.dtype == torch.float64
    r.float()
    r.bfloat16()
    assert r.weight.dtype == torch.float32


@pytest.mark.parametrize("cls", [MuseSparkRMSNorm, MuseSparkCenteredRMSNorm, MuseSparkInputScaledRMSNorm])
def test_norms_build_without_hidden_size(cls):
    # R5-12: hidden_size=None falls back to config.hidden_size, then 1.
    assert cls(config=SimpleNamespace(hidden_size=6, sequence_parallel=False)).weight.shape == (6,)
    assert cls(config=None).weight.shape == (1,)


def test_expert_class_follows_the_provider_base():
    from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP

    from primus.backends.megatron.core.transformer.muse_spark import moe

    assert moe.muse_spark_expert_cls(TEGroupedMLP) is moe.MuseSparkGroupedMLP
    assert moe.muse_spark_expert_cls(SequentialMLP) is moe.MuseSparkSequentialMLP
    # needs no Turbo import: runs (and fails) whether or not Primus-Turbo is installed
    with pytest.raises(TypeError, match="no expert variant"):
        moe.muse_spark_expert_cls(torch.nn.Linear)


def test_expert_class_follows_the_turbo_provider_base():
    pytest.importorskip("primus_turbo")
    from primus.backends.megatron.core.transformer.experts import PrimusGroupedMLP
    from primus.backends.megatron.core.transformer.muse_spark import moe

    turbo = moe.muse_spark_expert_cls(PrimusGroupedMLP)
    assert issubclass(turbo, PrimusGroupedMLP) and issubclass(turbo, moe._PostExpertNormMixin)
    assert moe.muse_spark_expert_cls(PrimusGroupedMLP) is turbo


def test_layer_spec_looks_up_the_patched_provider_at_call_time(monkeypatch):
    from megatron.core.extensions import transformer_engine_spec_provider as tsp
    from megatron.core.transformer.moe.experts import SequentialMLP

    from primus.backends.megatron.core.models.muse_spark import (
        muse_spark_layer_specs as specs,
    )
    from primus.backends.megatron.core.transformer.muse_spark import moe

    class _Provider:
        def grouped_mlp_modules(self, grouped):
            return SequentialMLP, None

        def activation_func(self):
            return None

    monkeypatch.setattr(tsp, "TESpecProvider", _Provider)  # what the Turbo patch does
    spec = specs.get_muse_spark_moe_spec(grouped_gemm=True)
    assert spec.submodules.experts.module is moe.MuseSparkSequentialMLP


def test_latent_dispatch_config_is_a_live_latent_width_view():
    from types import SimpleNamespace

    from primus.backends.megatron.core.transformer.muse_spark.moe import (
        latent_dispatch_config,
    )

    cfg = SimpleNamespace(hidden_size=8192, moe_latent_size=2048, num_moe_experts=16, moe_router_topk=4)
    out = latent_dispatch_config(cfg)
    assert out.hidden_size == 2048 and out.num_moe_experts == 16 and cfg.hidden_size == 8192
    # a later change to the shared config is seen (a copy would go stale)
    cfg.moe_router_topk = 8
    cfg.moe_latent_size = 1024
    assert out.moe_router_topk == 8 and out.hidden_size == 1024
    with pytest.raises(AttributeError):
        out.hidden_size = 8192


def test_latent_dispatch_config_copies_and_pickles():
    import copy
    import pickle
    from types import SimpleNamespace

    from primus.backends.megatron.core.transformer.muse_spark.moe import (
        _LatentWidthConfig,
        latent_dispatch_config,
    )

    cfg = SimpleNamespace(hidden_size=8192, moe_latent_size=2048)
    out = latent_dispatch_config(cfg)
    deep = copy.deepcopy(out)
    assert deep.hidden_size == 2048 and deep._base is not cfg
    assert copy.copy(out)._base is cfg
    assert pickle.loads(pickle.dumps(out)).hidden_size == 2048
    bare = object.__new__(_LatentWidthConfig)  # no __init__: no recursion, a plain AttributeError
    with pytest.raises(AttributeError):
        bare.anything


def test_deepep_dispatcher_constructor_does_not_size_from_hidden_size():
    """The latent view is set after ``MoELayer.__init__`` builds the dispatcher. That is
    only right while the constructor sizes nothing from ``hidden_size``; this fails if
    it starts to."""
    import ast
    import pathlib

    import primus

    src = (
        pathlib.Path(primus.__file__).parent / "backends/megatron/core/extensions/primus_turbo.py"
    ).read_text()
    tree = ast.parse(src)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PrimusTurboDeepEPTokenDispatcher"
    )
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    used = {n.attr for n in ast.walk(init) if isinstance(n, ast.Attribute)}
    assert "hidden_size" not in used


def test_deepep_is_turned_off_only_at_tp_ep_1(monkeypatch):
    from types import SimpleNamespace

    def a(tp, ep, deepep=True, turbo=True, etp=1):
        return SimpleNamespace(
            tensor_model_parallel_size=tp,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=etp,
            use_turbo_deepep=deepep,
            enable_primus_turbo=turbo,
        )

    # local_ranks=8: an EP 8 group fits one node whatever LOCAL_WORLD_SIZE / GPUS_PER_NODE say
    assert _deepep_off(a(1, 1), 8)
    # the dispatcher's TP is the expert TP (Muse Spark: 1), so TP 2 with EP 1 is still 1
    assert _deepep_off(a(2, 1), 8)
    assert _deepep_on(a(1, 8), 8) and _deepep_on(a(2, 1, etp=None), 8)
    assert _deepep_on(a(1, 1, deepep=False), 8) and _deepep_on(a(1, 1, turbo=False), 8)


def test_deepep_is_turned_off_when_the_expert_group_spans_nodes(monkeypatch):
    from primus.backends.megatron.patches.args.muse_spark_arg_compat_patches import (
        deepep_disable_reason,
    )

    def a(ep, etp=1, internode=False):
        return SimpleNamespace(
            muse_spark_turbo_deepep_internode=internode,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=etp,
            use_turbo_deepep=True,
            enable_primus_turbo=True,
        )

    assert _deepep_on(a(8), local_ranks=8)
    assert "num_rdma_ranks" in deepep_disable_reason(a(64), local_ranks=8)
    assert "muse_spark_turbo_deepep_internode: true" in deepep_disable_reason(a(64), local_ranks=8)
    # a Primus-Turbo build with rocSHMEM: internode dispatch, keep DeepEP across nodes
    assert _deepep_on(a(64, internode=True), local_ranks=8)
    assert _deepep_off(a(1, internode=True), local_ranks=8)  # the group-1 rule still holds
    assert _deepep_off(a(8, etp=2), local_ranks=8)
    # a group that does not divide the node: consecutive ranks straddle a node boundary
    # (EP 3: ranks 6-8; EP 6: ranks 6-11), though the group is no larger than a node
    for ep in (3, 6, 5, 7):
        assert "spans nodes" in deepep_disable_reason(a(ep), local_ranks=8)
        assert _deepep_on(a(ep, internode=True), local_ranks=8)
    for ep in (2, 4, 8):
        assert _deepep_on(a(ep), local_ranks=8)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("NNODES", raising=False)
    monkeypatch.setenv("GPUS_PER_NODE", "4")
    assert _deepep_off(a(8)) and _deepep_on(a(4))
    # one process on an 8-GPU node (--nproc_per_node=1): the node still has 8 GPUs
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    assert _deepep_on(a(8))
    # a malformed GPUS_PER_NODE raises; it does not fall back to the process count (1)
    monkeypatch.setenv("GPUS_PER_NODE", "8x")
    with pytest.raises(ValueError, match="GPUS_PER_NODE='8x'"):
        deepep_disable_reason(a(8))


def test_turbo_grouped_gemm_is_turned_off_only_with_megatron_fsdp():
    from types import SimpleNamespace

    from primus.backends.megatron.patches.args.muse_spark_arg_compat_patches import (
        turbo_grouped_gemm_needs_disabling as f,
    )

    def a(mfsdp, gg=True, turbo=True):
        return SimpleNamespace(use_megatron_fsdp=mfsdp, use_turbo_grouped_gemm=gg, enable_primus_turbo=turbo)

    assert f(a(True))
    assert not f(a(False)) and not f(a(True, gg=False)) and not f(a(True, turbo=False))


def test_qk_norm_is_the_weightless_rms_norm():
    from primus.backends.megatron.core.transformer.muse_spark.attention import (
        MuseSparkQKNorm,
    )
    from primus.backends.megatron.core.transformer.muse_spark.norms import (
        MuseSparkRMSNorm,
    )

    q = MuseSparkQKNorm(config=SimpleNamespace(hidden_size=64), hidden_size=16, eps=1e-5)
    assert isinstance(q, MuseSparkRMSNorm) and q.weight is None
    assert not list(q.parameters())
    x = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    assert torch.equal(q(x), MuseSparkRMSNorm(None, 16, eps=1e-5, with_scale=False)(x))


def test_rms_helper_equals_torch_rms_norm():
    # the eager fp32 helper (Primus's shared eager_rms_norm) is the same function as
    # F.rms_norm: a later kernel swap only needs the GPU parity check
    from primus.backends.megatron.core.transformer.muse_spark.norms import _rms

    torch.manual_seed(0)
    x = (torch.randn(64, 256) * 3).bfloat16().float()
    ref = torch.nn.functional.rms_norm(x, (256,), eps=1e-8)
    torch.testing.assert_close(_rms(x, 1e-8), ref, rtol=1e-6, atol=1e-6)


def test_final_logit_softcapping_must_be_null_zero_or_positive():
    from primus.backends.megatron.patches.args.muse_spark_layer_patterns_patches import (
        normalize_softcap,
    )

    for ok in (None, 0, 20.0, "20"):
        normalize_softcap(ok)
    # the normalized float is returned (the patch writes it back to the args)
    assert normalize_softcap(None) is None
    assert normalize_softcap("20") == 20.0 and isinstance(normalize_softcap("20"), float)
    assert normalize_softcap("0") == 0.0
    for bad in (-1.0, float("nan"), float("inf"), "abc", True):
        with pytest.raises(ValueError, match="final_logit_softcapping"):
            normalize_softcap(bad)


def _deepep_off(args, local_ranks=None):
    from primus.backends.megatron.patches.args.muse_spark_arg_compat_patches import (
        deepep_disable_reason,
    )

    return deepep_disable_reason(args, local_ranks) is not None


def _deepep_on(args, local_ranks=None):
    return not _deepep_off(args, local_ranks)


def test_stacked_expert_factory_names_the_unsupported_optimizer_format():
    # the stacked Turbo expert weights: one factory that builds the per-expert entries; a
    # flattened range (only 'fully_sharded_model_space', a legacy load format, makes one)
    # fails with a message naming the formats that work
    from megatron.core.dist_checkpointing import ShardedTensor

    from primus.backends.megatron.core.transformer.muse_spark.moe import (
        stack_expert_sharded_state_dict,
    )

    weights = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    sd = {
        f"mlp.weight{i}": ShardedTensor.from_rank_offsets(
            "mlp.weight", weights[i], (0, i, 2), replica_id=(0, 0, 0)
        )
        for i in range(2)
    }
    linear = SimpleNamespace(num_gemms=2, weights=weights)
    stack_expert_sharded_state_dict(sd, "mlp.", linear)
    factory = sd.pop("mlp.weights")
    assert not sd
    built = factory.build_fn("opt.mlp.weight", weights, (0, 0, 0), None)
    assert [built[f"expert{i}"].key for i in range(2)] == ["opt.mlp.weight"] * 2
    assert torch.equal(factory.merge_fn({f"expert{i}": weights[i] for i in range(2)}), weights)
    with pytest.raises(NotImplementedError, match="fully_sharded_model_space.*dp_reshardable"):
        factory.build_fn("opt.mlp.weight", weights, (0, 0, 0), slice(0, 4))


def test_softcap_patch_writes_back_the_float_cap():
    from primus.backends.megatron.patches.args.muse_spark_layer_patterns_patches import (
        patch_muse_spark_layer_patterns,
    )
    from primus.core.patches import PatchContext

    args = SimpleNamespace(num_layers=4, moe_layer_freq=1, final_logit_softcapping="20")
    ctx = PatchContext(
        backend="megatron",
        phase="build_args",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace()), "backend_args": args},
    )
    patch_muse_spark_layer_patterns(ctx)
    assert args.final_logit_softcapping == 20.0 and isinstance(args.final_logit_softcapping, float)


def test_router_subclasses_primus_topk_router_with_the_layout_base_first():
    # R-10: the base is PrimusTopKRouter whatever the import order (no topk_router_patches here)
    from primus.backends.megatron.core.transformer.muse_spark import moe

    pytest.importorskip("triton")  # PrimusTopKRouter needs Triton; with it, it must import
    assert moe.PrimusTopKRouter is not None
    assert issubclass(MuseSparkTopKRouter, moe.PrimusTopKRouter)
    assert MuseSparkTopKRouter.__mro__[1] is moe._FSDP2LayoutBase


def test_router_uses_upstream_routing_without_primus_args(monkeypatch):
    # Megatron-only args (converter / parity harness): PrimusTopKRouter.routing would read
    # Primus-only keys from get_args(); the upstream TopKRouter.routing runs instead.
    from megatron.training import global_vars

    from primus.backends.megatron.core.transformer.muse_spark import moe

    pytest.importorskip("triton")  # PrimusTopKRouter needs Triton; with it, it must import
    assert moe.PrimusTopKRouter is not None
    upstream = [c for c in MuseSparkTopKRouter.__mro__ if c.__name__ == "TopKRouter"][-1]
    calls = []
    monkeypatch.setattr(upstream, "routing", lambda self, logits, **kw: calls.append("up") or "up")
    monkeypatch.setattr(
        moe.PrimusTopKRouter, "routing", lambda self, logits, **kw: calls.append("primus") or "primus"
    )
    # no global args yet: upstream routing, and the answer is not cached
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None, raising=False)
    r = _bare_router(torch.ones(2, 3))
    assert r.routing(torch.zeros(1, 2)) == "up"
    assert "_use_primus_routing" not in r.__dict__
    # Primus args: Primus routing, decided once and cached on the router
    full = SimpleNamespace(**{k: None for k in moe._PRIMUS_ROUTER_ARGS})
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", full, raising=False)
    assert r.routing(torch.zeros(1, 2)) == "primus"
    assert r.__dict__["_use_primus_routing"] is True
    monkeypatch.setattr(moe, "_primus_router_args_present", lambda: pytest.fail("not cached"))
    assert r.routing(torch.zeros(1, 2)) == "primus"
    # Megatron-only args (converter / parity harness): upstream routing, cached
    monkeypatch.undo()
    monkeypatch.setattr(upstream, "routing", lambda self, logits, **kw: calls.append("up") or "up")
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", SimpleNamespace(), raising=False)
    r2 = _bare_router(torch.ones(2, 3))
    assert r2.routing(torch.zeros(1, 2)) == "up"
    assert r2.__dict__["_use_primus_routing"] is False
    assert calls == ["up", "primus", "primus", "up"]


@pytest.mark.parametrize("router_dtype, held", [("fp64", torch.float32), (None, torch.bfloat16)])
def test_router_gating_does_not_move_the_parameter(router_dtype, held):
    # R-11: a weight in another dtype than the GEMM's is copied for the GEMM, never
    # reassigned in forward (fp64 gating of the fp32 weight; fp32 gating of a bf16-held one)
    r = _bare_router(torch.ones(2, 3))
    with torch.no_grad():
        r.weight.data = r.weight.data.to(held)
    r.output_multiplier = 2.0
    r.bias = None
    r.config = SimpleNamespace(moe_router_dtype=router_dtype)
    param = r.weight
    before = param.data_ptr()
    out = r.gating(torch.ones(4, 3, dtype=torch.bfloat16))
    want = torch.float64 if router_dtype == "fp64" else torch.float32
    assert out.dtype == want  # the GEMM ran on a copy in this dtype
    assert r.weight is param and param.data_ptr() == before and param.dtype == held
    assert torch.equal(out, torch.full((4, 2), 6.0, dtype=want))


def test_moe_preprocess_keeps_the_cudagraph_decorator():
    # R-13: upstream MoELayer.preprocess is wrapped by maybe_skip_or_early_return_by_cudagraph.
    # Behaviour: in a CUDA-graph replay (TE graphs, a store with hidden_states / probs and no
    # routing_map) the wrapper returns the stored tensors and never runs the body.
    from primus.backends.megatron.core.transformer.muse_spark.moe import (
        MuseSparkMoELayer,
    )

    pre = MuseSparkMoELayer.preprocess
    assert callable(getattr(pre, "__wrapped__", None))  # functools.wraps of the decorator

    class _Store:
        hidden_states, probs, routing_map = "stored-h", "stored-p", None

        def is_empty(self):
            return False

    layer = SimpleNamespace(
        config=SimpleNamespace(cuda_graph_impl="transformer_engine"),
        cudagraph_tensor_store=_Store(),
        training=False,
    )
    assert pre(layer, None, None, None) == ("stored-h", "stored-p")  # the body would fail on None


def test_the_pinned_megatron_passes_pg_collection_to_a_moe_subclass():
    # MuseSparkTransformerLayer overrides nothing: the pinned Megatron-core builds every mlp
    # (MuseSparkMoELayer included) with pg_collection / is_mtp_layer / name
    import inspect

    from megatron.core.transformer.transformer_layer import TransformerLayer

    from primus.backends.megatron.core.models.muse_spark import (
        muse_spark_layer_specs as specs,
    )

    assert "name" in inspect.signature(TransformerLayer.__init__).parameters
    assert specs.MuseSparkTransformerLayer.__init__ is TransformerLayer.__init__


# ---------------------------------------------------------------- MuseSparkMoELayer
def _moe_cfg(**kw):
    base = dict(
        moe_latent_size=8,
        layernorm_epsilon=1e-5,
        post_norm_eps=1e-8,
        hidden_size=16,
        sequence_parallel=False,
        cuda_graph_impl="none",
        muse_spark_fused_centered_norm=False,
        muse_spark_fused_post_expert_norm=False,
    )
    return SimpleNamespace(**{**base, **kw})


def _build_moe_layer(monkeypatch, experts, dispatcher, cfg=None):
    """MuseSparkMoELayer.__init__ over a stubbed MoELayer.__init__ (no parallel state / TE):
    the stub sets the experts and the dispatcher the subclass inspects."""
    from megatron.core.transformer.moe.moe_layer import MoELayer

    from primus.backends.megatron.core.transformer.muse_spark import moe

    def fake_init(self, config, submodules=None, layer_number=None, *a, **k):
        torch.nn.Module.__init__(self)
        self.config = config
        self.experts = experts
        self.token_dispatcher = dispatcher
        self.shared_expert_overlap = False

    monkeypatch.setattr(MoELayer, "__init__", fake_init)
    if not getattr(moe._turbo_deepep_dispatcher_cls, "_test_stub", False):
        # hermetic: no Primus-Turbo import (it queries the GPU at import time)
        stub = lambda: None  # noqa: E731
        stub._test_stub = True
        monkeypatch.setattr(moe, "_turbo_deepep_dispatcher_cls", stub)
    return moe.MuseSparkMoELayer(cfg or _moe_cfg())


def _muse_experts():
    from primus.backends.megatron.core.transformer.muse_spark import moe

    e = moe.MuseSparkSequentialMLP.__new__(moe.MuseSparkSequentialMLP)
    torch.nn.Module.__init__(e)
    e._init_post_expert_norm()
    return e


def test_the_moe_layer_wires_its_post_expert_norm_into_the_experts(monkeypatch):
    experts = _muse_experts()
    layer = _build_moe_layer(monkeypatch, experts, SimpleNamespace(config="orig"))
    assert experts._muse_post_expert_norm is layer.post_expert_norm
    assert layer.pre_expert_norm.weight.shape == (8,) and layer.post_expert_norm.eps == 1e-8
    assert layer.token_dispatcher.config == "orig"  # not a DeepEP dispatcher: untouched


def test_the_moe_layer_refuses_experts_that_are_not_muse_spark(monkeypatch):
    with pytest.raises(TypeError, match="needs Muse Spark experts"):
        _build_moe_layer(monkeypatch, torch.nn.Identity(), SimpleNamespace(config="orig"))


def test_the_moe_layer_gives_the_deepep_dispatcher_the_latent_width(monkeypatch):
    from primus.backends.megatron.core.transformer.muse_spark import moe

    class _DeepEP:
        def __init__(self):
            self.config = None

    stub = lambda: _DeepEP  # noqa: E731
    stub._test_stub = True
    monkeypatch.setattr(moe, "_turbo_deepep_dispatcher_cls", stub)
    cfg = _moe_cfg()
    dispatcher = _DeepEP()
    dispatcher.config = cfg
    _build_moe_layer(monkeypatch, _muse_experts(), dispatcher, cfg)
    assert isinstance(dispatcher.config, moe._LatentWidthConfig)
    assert dispatcher.config.hidden_size == 8 and dispatcher.config.layernorm_epsilon == 1e-5


def test_the_moe_preprocess_norms_the_latent_before_the_dispatch(monkeypatch):
    # preprocess: fc1_latent_proj -> pre_expert_norm -> dispatch_preprocess (HF order)
    from primus.backends.megatron.core.transformer.muse_spark.centered_norm import (
        centered_rms_norm_reference,
    )

    seen = {}

    class _Dispatcher:
        config = None

        def dispatch_preprocess(self, h, routing_map, probs):
            seen["h"] = h
            return h, probs

    layer = _build_moe_layer(monkeypatch, _muse_experts(), _Dispatcher())
    torch.manual_seed(0)
    with torch.no_grad():
        layer.pre_expert_norm.weight.normal_(0, 0.1)
    proj = torch.nn.Linear(16, 8, bias=False)
    layer.fc1_latent_proj = lambda x: (proj(x), None)
    x = torch.randn(5, 16)
    probs = torch.rand(5, 2)
    h, p = layer.preprocess(x, probs, torch.ones(5, 2, dtype=torch.bool))
    want = centered_rms_norm_reference(proj(x), layer.pre_expert_norm.weight, 1e-5)
    torch.testing.assert_close(seen["h"], want)
    torch.testing.assert_close(h, want)
    assert p is probs


def test_a_malformed_module_yaml_gives_none(tmp_path, monkeypatch):
    # yaml.YAMLError is not a ValueError: the docstring's "None when unreadable" holds for it
    from primus.backends.megatron.patches.args import (
        muse_spark_layer_patterns_patches as lp,
    )

    bad = tmp_path / "trainer_base.yaml"
    bad.write_text("make_vocab_size_divisible_by: [128\n")
    monkeypatch.setattr(lp, "_TRAINER_BASE_YAML", bad)
    lp.module_vocab_divisible_by.cache_clear()
    try:
        assert lp.module_vocab_divisible_by() is None
    finally:
        lp.module_vocab_divisible_by.cache_clear()


def test_the_ignored_vocab_multiple_reads_the_module_default_from_its_yaml():
    import yaml

    from primus.backends.megatron.patches.args import (
        muse_spark_layer_patterns_patches as lp,
    )

    default = yaml.safe_load(lp._TRAINER_BASE_YAML.read_text())["make_vocab_size_divisible_by"]
    assert lp.module_vocab_divisible_by() == default
    assert lp.ignored_vocab_multiple(SimpleNamespace(make_vocab_size_divisible_by=default)) is None
    assert lp.ignored_vocab_multiple(SimpleNamespace(make_vocab_size_divisible_by=64)) is None
    assert lp.ignored_vocab_multiple(SimpleNamespace(make_vocab_size_divisible_by=256)) == 256


def test_built_module_summary_reports_the_core_attention_window():
    # the trainer test checks "window=(511, 0)": the TE window the first layer was built with
    from types import SimpleNamespace

    from primus.backends.megatron.core.models.muse_spark.muse_spark_model import (
        built_module_summary,
    )

    def model(window):
        attn = SimpleNamespace(core_attention=SimpleNamespace(window_size=window))
        layer = SimpleNamespace(self_attention=attn, mlp=None, mlp_bda=None)
        return SimpleNamespace(embedding=None, decoder=SimpleNamespace(layers=[layer]))

    assert built_module_summary(model([511, 0])).endswith("window=(511, 0)")
    assert built_module_summary(model(None)).endswith("window=None")


@pytest.mark.parametrize(
    "world, nnodes, ep, kept",
    [
        ("3", None, 3, True),  # single node, 3 of 8 GPUs used: the EP 3 group is in the node
        ("6", "1", 6, True),
        ("8", "1", 8, True),
        ("24", "3", 6, False),  # 8 ranks per node: EP 6 groups straddle nodes
        ("24", "3", 8, True),
        ("16", "2", 16, False),  # larger than a node
        ("12", "2", 6, True),  # 6 ranks per node: EP 6 groups stay in a node
        ("16", None, 6, False),  # no NNODES, world > GPUs: the GPUs-per-node rule
    ],
)
def test_deepep_node_span_reads_the_ranks_placed_per_node(monkeypatch, world, nnodes, ep, kept):
    from primus.backends.megatron.patches.args.muse_spark_arg_compat_patches import (
        deepep_disable_reason,
        ranks_per_node,
    )

    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    monkeypatch.setenv("WORLD_SIZE", world)
    if nnodes is None:
        monkeypatch.delenv("NNODES", raising=False)
    else:
        monkeypatch.setenv("NNODES", nnodes)
    args = SimpleNamespace(
        muse_spark_turbo_deepep_internode=False,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=1,
        use_turbo_deepep=True,
        enable_primus_turbo=True,
    )
    assert (deepep_disable_reason(args) is None) is kept
    if not kept:
        assert f"{ranks_per_node()} ranks per node" in deepep_disable_reason(args)


@pytest.mark.parametrize("name", ["WORLD_SIZE", "NNODES"])
@pytest.mark.parametrize("bad", ["abc", "0", "-2", "1.5"])
def test_ranks_per_node_rejects_a_malformed_launcher_variable(monkeypatch, name, bad):
    # the same rule as the converter's NNODES check: a set but bad value names the variable
    from primus.backends.megatron.patches.args.muse_spark_arg_compat_patches import (
        ranks_per_node,
    )

    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("NNODES", "2")
    monkeypatch.setenv(name, bad)
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        ranks_per_node()


def test_a_rejected_setting_stops_the_run_through_the_patch_runner():
    # the training runtime calls run_patches without stop_on_error, which logs an Exception
    # and continues: the Muse Spark checks raise SystemExit (fail_fast) so the run stops
    from primus.core.patches import PatchContext
    from primus.core.patches.patch_registry import PatchRegistry

    patch = PatchRegistry.get("megatron.args.muse_spark_no_fused_router")
    params = SimpleNamespace(model_type="muse_spark", turbo_sync_free_moe_stage=1)
    ctx = PatchContext(
        backend="megatron",
        phase="build_args",
        extra={"backend_args": SimpleNamespace(), "module_config": SimpleNamespace(params=params)},
    )
    with pytest.raises(SystemExit, match="e_score_correction_bias") as err:
        patch.apply(ctx)
    assert isinstance(err.value.__cause__, ValueError)


@pytest.mark.parametrize(
    "name",
    [
        "megatron.args.muse_spark_deepep_group_size",
        "megatron.args.muse_spark_mfsdp_turbo_grouped_gemm",
        "megatron.args.muse_spark_cp_comm",
        "megatron.args.muse_spark_no_fused_router",
        "megatron.muse_spark.memory_check",
        "megatron.muse_spark.fp16_optimizer_fp32_param_ckpt",
        "megatron.muse_spark.chained_optimizer_relayout_load",
        "megatron.muse_spark.fsdp2_dtype_islands",
        "megatron.muse_spark.fsdp2_fp32_param_grads",
    ],
)
def test_muse_spark_patches_fail_fast(name):
    import primus.backends.megatron.patches.muse_spark_fsdp2_patches  # noqa: F401
    import primus.backends.megatron.patches.muse_spark_memory_check_patches  # noqa: F401
    import primus.backends.megatron.patches.muse_spark_optimizer_ckpt_patches  # noqa: F401
    from primus.core.patches.patch_registry import PatchRegistry

    patch = PatchRegistry.get(name)
    assert patch is not None
    # the registered handler is the fail_fast wrapper around the patch body
    assert hasattr(patch.handler, "__wrapped__")
