###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A Megatron MoE layer with replicas on 2 GPUs (EP 2), through the real Primus-Turbo DeepEP dispatcher and grouped
linears, against the same layer without replicas.

Two micro-batches of one step with per-micro-batch planning: the hot experts move from rank 0 to rank 1 between
them, so slots are reassigned and emptied mid-step. Outputs, input gradients and the experts' weight gradients right
after the step's backward (the step ends in the last micro-batch's backward, with no later forward) must match the
layer without replicas, with the transfers synchronous or overlapped and with the layer recomputed. When only the
first micro-batch has replicas, the slot gradients it left must still reach the owners. With a minimum gain no plan
reaches, replication is on but no micro-batch has replicas: the experts must then run the same grouped GEMM call as
without replication (own weights, ``L`` groups). In pipeline order (three micro-batches, F0 F1 B0 F2 B1 B2) each
backward runs after a later forward planned other slots, which the backward must restore. Wrapped in Megatron's DDP
with overlap_grad_reduce (expert data parallel size 1 and 2, and as the pipeline schedules reduce), the reduced expert
gradients must include the replica gradients.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD, E, TOPK, HIDDEN, FFN, TOKENS = 2, 8, 2, 256, 256, 512
TOL = 1e-2  # max |a - b| / max |b|: BF16 GEMMs over different row splits


def _skip_reason():
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
        return f"needs {WORLD} GPUs"
    try:
        from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (  # noqa: F401
            grouped_gemm_segments_impl,
        )
    except ImportError:
        return "needs a Primus-Turbo grouped_gemm that takes weight segments"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason()))

# Megatron global args the Turbo dispatcher and grouped linears read (see test_primus_turbo_grouped_checkpoint.py)
GLOBAL_ARGS = dict(
    enable_turbo_attention_float8=False,
    data_parallel_replicate_degree=1,
    offload=False,
    offload_ops=[],
    patch_primus_pipeline=False,
    pp_algorithm=None,
    patch_zero_bubble=False,
    enable_zero_bubble=False,
    use_turbo_fused_act_with_probs=False,
    moe_router_padding_for_quantization=False,
    turbo_sync_free_moe_stage=0,
    use_turbo_grouped_gemm=True,
    turbo_grouped_gemm_backend="default",
    turbo_grouped_gemm_wgrad_accum_fusion=None,
    turbo_deepep_use_comm_stream=False,
    turbo_deepep_num_cu=32,
    moe_router_force_load_balancing=False,
    moe_router_force_load_balancing_type="uniform",
    turbo_fp8_permute=False,
    moe_replica_placement="per_microbatch",
    moe_replica_slots=2,
    moe_replica_min_rows=1,
    moe_replica_min_gain=0.0,
    moe_replica_cost=0,
    moe_replica_block=1,
    moe_replica_replan_above=0.0,
    moe_replica_async_comm=False,
    moe_replica_log_interval=0,
    accumulate_allreduce_grads_in_fp32=True,
    params_dtype=torch.bfloat16,
)

# name -> (replica arguments, the sign added to each micro-batch's inputs, recompute the layer, pipeline order)
CASES = {
    "replicas": ({}, (1.0, -1.0), False, False),
    "replicas_async": ({"moe_replica_async_comm": True}, (1.0, -1.0), False, False),
    "replicas_recompute": ({}, (1.0, -1.0), True, False),
    # the second micro-batch is balanced: it plans no replica, so the step's end returns what the first left
    "last_microbatch_without_replicas": ({"moe_replica_min_gain": 0.2}, (1.0, 0.0), False, False),
    "no_replicas": ({"moe_replica_min_gain": 1e9}, (1.0, -1.0), False, False),
    # 1F1B order (F0 F1 B0 F2 B1 B2): each backward runs after a later forward planned other slots
    "pipelined": ({}, (1.0, -1.0, 1.0), False, True),
    "pipelined_async": ({"moe_replica_async_comm": True}, (1.0, -1.0, 1.0), False, True),
    "pipelined_recompute": ({}, (1.0, -1.0, 1.0), True, True),
    "pipelined_per_step": ({"moe_replica_placement": "per_step"}, (1.0, -1.0, 1.0), False, True),
}


def _moe_layer(planner, ep=WORLD, **args):
    from megatron.core.models.gpt import gpt_layer_specs
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.moe import moe_layer
    from megatron.core.transformer.spec_utils import build_module
    from megatron.training.global_vars import set_args

    from primus.backends.megatron.core.extensions import (
        transformer_engine_spec_provider as spec_provider,
    )
    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboDeepEPTokenDispatcher,
    )

    set_args(SimpleNamespace(**{**GLOBAL_ARGS, **args}, moe_replica_planner=planner))
    moe_layer.MoEFlexTokenDispatcher = PrimusTurboDeepEPTokenDispatcher
    cfg = SimpleNamespace(
        **{
            **vars(spec_provider._build_default_primus_args()),
            "enable_primus_turbo": True,
            "use_turbo_grouped_gemm": True,
        }
    )
    te_provider = gpt_layer_specs.TESpecProvider

    def turbo_provider(*args, **kwargs):
        provider = spec_provider.PrimusTurboSpecProvider(*args, **kwargs)
        provider.cfg = cfg
        return provider

    gpt_layer_specs.TESpecProvider = turbo_provider
    try:
        spec = gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec(
            num_experts=E, moe_grouped_gemm=True
        )
    finally:
        gpt_layer_specs.TESpecProvider = te_provider
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        ffn_hidden_size=FFN,
        num_attention_heads=4,
        num_moe_experts=E,
        moe_ffn_hidden_size=FFN,
        moe_router_topk=TOPK,
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="flex",
        moe_enable_deepep=True,
        moe_aux_loss_coeff=0.0,
        expert_model_parallel_size=ep,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        gradient_accumulation_fusion=False,
    )
    mlp = spec.submodules.mlp  # a ModuleSpec, or a functools.partial of the MoE layer class
    layer = build_module(mlp, config=config) if hasattr(mlp, "module") else mlp(config=config)
    return layer.cuda()


def _skew_router(*layers):
    # experts 0 and 1 (EP rank 0) are hot for positive inputs, 6 and 7 (the last EP rank) for negative ones
    for layer in layers:
        with torch.no_grad():
            w = layer.router.weight
            w.zero_()
            w[0], w[1], w[6], w[7] = 0.05, 0.04, -0.05, -0.04


def _worker(rank: int, init_file: str, case: str):
    import primus_turbo.pytorch as primus_turbo_torch
    from megatron.core import parallel_state as ps
    from torch.utils.checkpoint import checkpoint

    from primus.backends.megatron.core.transformer.moe.replica import scheduler as sch

    replica_args, signs, recompute, pipelined = CASES[case]
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
    try:
        ps.initialize_model_parallel(expert_model_parallel_size=WORLD)
        torch.manual_seed(0)
        ref = _moe_layer(None)
        rep = _moe_layer("ultraep", **replica_args)
        rep.load_state_dict(ref.state_dict())
        sch.attach(rep)  # what the megatron.moe.replica_attach patch does once the layer is built
        sch._num_microbatches = lambda: len(signs)
        _skew_router(ref, rep)
        gen = torch.Generator().manual_seed(1 + rank)
        inputs = [
            (torch.randn(TOKENS, 1, HIDDEN, generator=gen) + sign).cuda().to(torch.bfloat16) for sign in signs
        ]
        grads = [torch.randn(TOKENS, 1, HIDDEN, generator=gen).cuda().to(torch.bfloat16) for _ in inputs]

        def step(layer):
            # under Megatron's DDP the owners have a main_grad from the start; here a restore can return slot
            # gradients before the owners' first backward creates their grad
            for linear in (layer.experts.linear_fc1, layer.experts.linear_fc2):
                linear.weights.grad = torch.zeros_like(linear.weights)
            xs = [x.clone().requires_grad_() for x in inputs]
            outs = []
            for i, x in enumerate(xs):
                out, _ = checkpoint(layer, x, use_reentrant=True) if recompute else layer(x)
                outs.append(out)
                if not pipelined:
                    out.backward(grads[i])
                elif i > 0:
                    outs[i - 1].backward(grads[i - 1])
            if pipelined:
                outs[-1].backward(grads[-1])
            wgrads = [layer.experts.linear_fc1.weights.grad, layer.experts.linear_fc2.weights.grad]
            return [out.detach() for out in outs], [x.grad for x in xs], wgrads

        want = step(ref)
        # the weight operand and group count of every expert grouped GEMM of the replicated layer
        calls = []
        grouped_gemm = primus_turbo_torch.ops.grouped_gemm

        def spy(a, b, m_splits, *args, **kwargs):
            calls.append((isinstance(b, torch.Tensor), m_splits.numel()))
            return grouped_gemm(a, b, m_splits, *args, **kwargs)

        primus_turbo_torch.ops.grouped_gemm = spy
        try:
            got = step(rep)
        finally:
            primus_turbo_torch.ops.grouped_gemm = grouped_gemm
        sched = rep.token_dispatcher.replica
        per_step = replica_args.get("moe_replica_placement") == "per_step"
        assert sched.num_plans == (1 if per_step else len(signs)), sched.num_plans
        assert sched.new_step and sched.mb_in_step == 0, "the step did not end in the last backward"
        plain = (True, E // WORLD)
        # micro-batches x 2 expert linears (the recompute runs the forward again)
        assert len(calls) == 2 * len(signs) * (2 if recompute else 1), calls
        if case == "no_replicas":
            assert sched.plan.num_replicas() == 0, sched.plan.num_replicas()
            assert calls == [plain] * 4, calls
        elif case == "last_microbatch_without_replicas":
            assert sched.plan.num_replicas() == 0, sched.plan.num_replicas()
            assert plain not in calls[:2] and calls[2:] == [plain] * 2, calls
        else:
            assert any(not p for p, _ in calls), "the skewed routing must build replicas"

        def rel(a, b):
            return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()

        for name, a, b in zip(("output", "input grad", "weight grad"), got, want):
            for i, (x, y) in enumerate(zip(a, b)):
                assert rel(x, y) <= TOL, (name, i, rel(x, y))
        for linear in (rep.experts.linear_fc1, rep.experts.linear_fc2):
            assert not bool(linear._replica_slots.main_grad.any()), "slot gradients left after the step"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("has_replicas", [False, True])
def test_slot_accumulation_decides_fusion_only_with_replicas(monkeypatch, has_replicas):
    from primus.backends.megatron.core.extensions import primus_turbo as pt

    monkeypatch.setattr(pt, "_bf16_grouped_wgrad_can_accumulate", lambda main_grad=None: False)
    seen = []
    slots = SimpleNamespace(
        sched=SimpleNamespace(has_replicas=has_replicas),
        main_grad=torch.zeros(1),
        weight_segments=lambda weights, m_splits, fused: seen.append(fused) or (weights, m_splits),
    )
    linear = SimpleNamespace(_replica_slots=slots)
    _, _, pattern = pt.PrimusTurboGroupedLinear._with_replica_segments(
        linear, object(), torch.zeros(6), "megatron"
    )
    # without replicas the operand is the own weights only, so the slots cannot turn fusion off
    assert pattern == (None if has_replicas else "megatron") and seen == [not has_replicas]


def test_replica_segments_need_bf16(monkeypatch):
    from primus.backends.megatron.core.extensions import primus_turbo as pt

    linear = SimpleNamespace(_replica_slots=SimpleNamespace(sched=SimpleNamespace(has_replicas=False)))
    monkeypatch.setattr(
        pt.PrimusTurboLowPrecisionGlobalStateManager, "is_turbo_fp8_enabled", staticmethod(lambda: True)
    )
    with pytest.raises(NotImplementedError, match="BF16 experts only"):
        pt.PrimusTurboGroupedLinear._with_replica_segments(linear, object(), torch.zeros(6), None)


@pytest.mark.parametrize("case", list(CASES))
def test_moe_layer_with_replicas_matches_without(tmp_path, case):
    mp.spawn(_worker, args=(str(tmp_path / "pg_init"), case), nprocs=WORLD, join=True)


def _ddp_worker(rank: int, init_file: str, world: int, ep: int, aligned: bool):
    """The layer with and without replicas, each wrapped in Megatron's DDP with overlap_grad_reduce: two steps of
    two micro-batches (the first in no_sync); the experts' main_grad after finish_grad_sync of the second step must
    match, and the expert buckets' reduction must have started in the backward (overlapped). The second step is the
    test: in the first one DDP starts every reduction in finish_grad_sync, after the replica gradients are back, even
    without the megatron.moe.replica_ddp patch. ``aligned``: as Megatron's pipeline schedules with
    align_grad_reduce, both micro-batches in flight (F0 F1 B0 B1), every backward in no_sync and the reduction started
    by an explicit start_grad_sync."""
    import contextlib

    from megatron.core import parallel_state as ps
    from megatron.core.distributed import (
        DistributedDataParallel,
        DistributedDataParallelConfig,
    )

    from primus.backends.megatron.core.transformer.moe.replica import scheduler as sch
    from primus.backends.megatron.patches.moe_patches import moe_replica_patches

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        ps.initialize_model_parallel(expert_model_parallel_size=ep)
        moe_replica_patches.patch_moe_replica_ddp(None)  # applied before Megatron wraps the model in DDP
        torch.manual_seed(0)
        ref = _moe_layer(None, ep=ep)
        rep = _moe_layer("ultraep", ep=ep)
        rep.load_state_dict(ref.state_dict())
        sch.attach(rep)
        sch._num_microbatches = lambda: 2
        _skew_router(ref, rep)
        # the slots' main_grad follows accumulate_allreduce_grads_in_fp32 (set in GLOBAL_ARGS), as DDP's does
        ddp_config = DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=True)
        models = [DistributedDataParallel(layer.config, ddp_config, layer) for layer in (ref, rep)]
        gen = torch.Generator().manual_seed(1 + rank)
        inputs = [
            (torch.randn(TOKENS, 1, HIDDEN, generator=gen) + sign).cuda().to(torch.bfloat16)
            for sign in (1.0, -1.0)
        ]
        grads = [torch.randn(TOKENS, 1, HIDDEN, generator=gen).cuda().to(torch.bfloat16) for _ in inputs]
        sched = rep.token_dispatcher.replica

        def train_step(model):
            model.zero_grad_buffer()
            if aligned:
                with model.no_sync():
                    outs = [model(x.clone().requires_grad_())[0] for x in inputs]
                    for out, g in zip(outs, grads):
                        out.backward(g)
                model.start_grad_sync()
            else:
                for i, (x, g) in enumerate(zip(inputs, grads)):
                    with model.no_sync() if i < len(inputs) - 1 else contextlib.nullcontext():
                        out, _ = model(x.clone().requires_grad_())
                        out.backward(g)
            # overlapped: the expert buckets' reduction started in the backward (from the second step on; without
            # the owners' grad-ready marking DDP would fall back to reducing them in finish_grad_sync)
            dispatched = all(g.grad_reduce_handle is not None for g in model.expert_parallel_bucket_groups)
            model.finish_grad_sync()
            experts = model.module.experts
            wgrads = [
                experts.linear_fc1.weights.main_grad.clone(),
                experts.linear_fc2.weights.main_grad.clone(),
            ]
            return wgrads, dispatched

        for step in range(2):
            (want, ref_dispatched), (got, dispatched) = train_step(models[0]), train_step(models[1])
            assert sched.plan.num_replicas() > 0, "the skewed routing must build replicas in every step"
        assert ref_dispatched and dispatched, (ref_dispatched, dispatched)
        for name, a, b in zip(("fc1", "fc2"), got, want):
            err = ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()
            assert err <= TOL, (name, err)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("world", "ep", "aligned"),
    [(2, 2, False), (4, 2, False), (2, 2, True)],
    ids=["expert_dp1", "expert_dp2", "pipeline_aligned"],
)
def test_overlapped_ddp_gradient_reduction_includes_the_replica_gradients(tmp_path, world, ep, aligned):
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    mp.spawn(_ddp_worker, args=(str(tmp_path / "pg_init"), world, ep, aligned), nprocs=world, join=True)
