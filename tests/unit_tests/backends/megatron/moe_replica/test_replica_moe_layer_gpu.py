###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A Megatron MoE layer with replicas on 2 GPUs (EP 2), through the real Primus-Turbo DeepEP dispatcher and grouped
linears, against the same layer without replicas.

Two micro-batches of one step with per-micro-batch planning: the hot experts move from rank 0 to rank 1 between
them, so slots are reassigned and emptied mid-step. Outputs, input gradients and the experts' weight gradients after
the end-of-step return must match the layer without replicas.
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
)


def _moe_layer(planner):
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

    set_args(SimpleNamespace(**GLOBAL_ARGS, moe_replica_planner=planner))
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
        expert_model_parallel_size=WORLD,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        gradient_accumulation_fusion=False,
    )
    mlp = spec.submodules.mlp  # a ModuleSpec, or a functools.partial of the MoE layer class
    layer = build_module(mlp, config=config) if hasattr(mlp, "module") else mlp(config=config)
    return layer.cuda()


def _worker(rank: int, init_file: str):
    from megatron.core import parallel_state as ps

    from primus.backends.megatron.core.transformer.moe.replica import scheduler as sch

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
    try:
        ps.initialize_model_parallel(expert_model_parallel_size=WORLD)
        torch.manual_seed(0)
        ref = _moe_layer(None)
        rep = _moe_layer("ultraep")
        rep.load_state_dict(ref.state_dict())
        sch.attach(rep)
        assert sch.allocate_slots([rep]) == [rep.token_dispatcher.replica]
        sch._num_microbatches = lambda: 2
        # experts 0 and 1 (rank 0) are hot for positive inputs, 6 and 7 (rank 1) for negative ones
        for layer in (ref, rep):
            with torch.no_grad():
                w = layer.router.weight
                w.zero_()
                w[0], w[1], w[6], w[7] = 0.05, 0.04, -0.05, -0.04
        gen = torch.Generator().manual_seed(1 + rank)
        inputs = [
            (torch.randn(TOKENS, 1, HIDDEN, generator=gen) + sign).cuda().to(torch.bfloat16)
            for sign in (1.0, -1.0)
        ]
        grads = [torch.randn(TOKENS, 1, HIDDEN, generator=gen).cuda().to(torch.bfloat16) for _ in inputs]

        def step(layer):
            outs, dxs = [], []
            for x, g in zip(inputs, grads):
                x = x.clone().requires_grad_()
                out, _ = layer(x)
                out.backward(g)
                outs.append(out.detach())
                dxs.append(x.grad)
            sch.end_step_all([layer])
            wgrads = [layer.experts.linear_fc1.weights.grad, layer.experts.linear_fc2.weights.grad]
            return outs, dxs, wgrads

        want, got = step(ref), step(rep)
        plans = rep.token_dispatcher.replica.num_plans
        assert plans == 2, plans

        def rel(a, b):
            return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()

        for name, a, b in zip(("output", "input grad", "weight grad"), got, want):
            for i, (x, y) in enumerate(zip(a, b)):
                assert rel(x, y) <= TOL, (name, i, rel(x, y))
        assert not bool(
            rep.experts.linear_fc1._replica_slots.main_grad.any()
        ), "slot gradients left after the step"
    finally:
        dist.destroy_process_group()


def test_moe_layer_with_replicas_matches_without(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "pg_init"),), nprocs=WORLD, join=True)
