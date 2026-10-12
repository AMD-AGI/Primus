###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Accuracy parity for ``PrimusGroupedMLP``'s ``turbo_fused_grouped_gemm`` switch.

The flag replaces the ``fc1 -> weighted SwiGLU -> fc2`` chain with a single
Primus-Turbo ``grouped_mlp_fp8`` call that also owns WGRAD. Each test runs the
same weights, tokens and probs through the flag on and off and compares the
forward output plus every gradient (dx, dW1, dW2, dprobs).
"""

from types import SimpleNamespace
from typing import Optional, Sequence

import pytest
import torch
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F
from megatron.core import parallel_state  # noqa: E402
from megatron.core.num_microbatches_calculator import (  # noqa: E402
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
    reconfigure_num_microbatches_calculator,
)
from megatron.core.process_groups_config import ProcessGroupCollection  # noqa: E402
from megatron.core.transformer.moe.experts import TEGroupedMLPSubmodules  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402
from megatron.training.global_vars import set_args  # noqa: E402
from primus_turbo.pytorch.core.low_precision import (  # noqa: E402
    Format,
    ScaleDtype,
    ScalingGranularity,
    check_mxfp8_support,
)
from primus_turbo.pytorch.core.utils import is_gfx950  # noqa: E402

import primus.backends.megatron.core.transformer.experts as experts_mod  # noqa: E402
from primus.backends.megatron.core.extensions.primus_turbo import (  # noqa: E402
    PrimusTurboColumnParallelGroupedLinear,
    PrimusTurboQuantConfig,
    PrimusTurboRowParallelGroupedLinear,
    primus_turbo_fp8_autocast,
)
from primus.backends.megatron.core.fp8_utils import (  # noqa: E402
    MXFP8_SCALING_BLOCK_SIZE,
    SCALING_BLOCK_SIZE,
)

# `grouped_mlp_fp8` routes both recipes through `grouped_gemm_fp8_glu_impl` /
# `grouped_gemm_fp8_dglu_impl`, which assert `is_gfx950()`. On gfx942 the fused
# path is unreachable, so there is nothing to compare against.
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx950()),
    reason="turbo_fused_grouped_gemm requires gfx950 (MI350X/MI355X)",
)

PrimusGroupedMLP = experts_mod.PrimusGroupedMLP

_NUM_EXPERTS = 4
_HIDDEN = 1024
_FFN = 512

# "aligned" keeps every group a multiple of 128; "ragged" is the case
# `turbo_grouped_gemm_without_padding` exists for.
_GROUP_LENS = {
    "aligned": (256, 128, 384, 128),
    "ragged": (203, 77, 411, 41),
}

_SNR_VS_REFERENCE_DB = 20.0

# The two paths differ only in quantization order, so they track each other far
# more closely than either tracks fp32: measured 32-37 dB on the GEMM-carried
# tensors and 50-52 dB on dprobs. Semantic divergence (swapped GLU halves, probs
# on the wrong operand) lands under 10 dB.
_SNR_FUSED_VS_UNFUSED_DB = {
    "forward": 26.0,
    "dx": 26.0,
    "dw1": 26.0,
    "dw2": 26.0,
    "dprobs": 40.0,
}


def _make_args(*, fused: bool) -> SimpleNamespace:
    """The ``get_args()`` namespace the Turbo grouped-MLP stack reads.

    ``turbo_fused_grouped_gemm`` is cached onto the instance in ``__init__``, so
    flipping it here and rebuilding is what selects the fused path.
    """
    return SimpleNamespace(
        # PrimusGroupedMLP
        turbo_fused_grouped_gemm=fused,
        use_turbo_fused_act_with_probs=False,
        moe_router_padding_for_quantization=False,
        # Raw expert token counts on both sides; TE's zero-padding would
        # otherwise wrap only the unfused one.
        turbo_grouped_gemm_without_padding=True,
        # PrimusTurboGroupedLinear
        turbo_grouped_gemm_backend="default",
        offload=False,
        offload_ops=[],
        # read by the shared split-wgrad helper in the same module
        patch_primus_pipeline=False,
        pp_algorithm=None,
        patch_zero_bubble=False,
        enable_zero_bubble=False,
        enable_primus_turbo=True,
        use_turbo_grouped_gemm=True,
        sequence_parallel=False,
        context_parallel_size=1,
        micro_batch_size=1,
    )


def _build_config(fp8_recipe: str) -> TransformerConfig:
    """Single-rank MoE config matching the production Turbo grouped-GEMM recipe.

    ``bias_activation_fusion`` stays off so the unfused side's activation is the
    same torch expression the fp32 reference computes, keeping any disagreement
    attributable to the GEMMs.
    """
    return TransformerConfig(
        num_layers=1,
        hidden_size=_HIDDEN,
        num_attention_heads=8,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        num_moe_experts=_NUM_EXPERTS,
        moe_ffn_hidden_size=_FFN,
        moe_router_topk=2,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bias_dropout_fusion=False,
        bias_activation_fusion=False,
        # main_grad is only wired up by DDP, which these tests do not build.
        gradient_accumulation_fusion=False,
        fp8="e4m3",
        fp8_recipe=fp8_recipe,
        params_dtype=torch.bfloat16,
        bf16=True,
    )


def _quant_config(fp8_recipe: str) -> PrimusTurboQuantConfig:
    """Mirror ``fp8_utils.get_fp8_quant_config`` for the recipes under test."""
    if fp8_recipe == "tensorwise":
        return PrimusTurboQuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
    if fp8_recipe == "mxfp8":
        return PrimusTurboQuantConfig(
            format=Format.E4M3,
            granularity=ScalingGranularity.MX_BLOCKWISE,
            block_size=MXFP8_SCALING_BLOCK_SIZE,
            scale_dtype=ScaleDtype.E8M0,
        )
    if fp8_recipe == "blockwise":
        return PrimusTurboQuantConfig(
            format=Format.E4M3,
            granularity=ScalingGranularity.BLOCKWISE,
            block_size=SCALING_BLOCK_SIZE,
        )
    raise ValueError(f"unhandled fp8 recipe {fp8_recipe!r}")


def _build_mlp(config: TransformerConfig, pg_collection, *, fused: bool) -> PrimusGroupedMLP:
    set_args(_make_args(fused=fused))
    submodules = TEGroupedMLPSubmodules(
        linear_fc1=PrimusTurboColumnParallelGroupedLinear,
        linear_fc2=PrimusTurboRowParallelGroupedLinear,
    )
    # Built straight onto the CUDA device: the per-expert weights are views into
    # a single `weights` Parameter, which a later .cuda() would detach.
    return PrimusGroupedMLP(_NUM_EXPERTS, config, submodules, pg_collection=pg_collection)


def _copy_weights(dst: PrimusGroupedMLP, src: PrimusGroupedMLP) -> None:
    with torch.no_grad():
        dst.linear_fc1.weights.copy_(src.linear_fc1.weights)
        dst.linear_fc2.weights.copy_(src.linear_fc2.weights)


def _attach_main_grad(mlp: PrimusGroupedMLP) -> None:
    """Stand in for DDP's grad buffer.

    Above one microbatch ``_WeightGradBridge`` writes WGRAD into
    ``weight.main_grad`` and hands autograd a dummy; without DDP those
    attributes do not exist and its backward asserts.
    """
    for linear in (mlp.linear_fc1, mlp.linear_fc2):
        weights = linear.weights
        weights.main_grad = torch.zeros_like(weights, dtype=torch.float32)
        weights.grad_added_to_main_grad = False


def _weight_grad(param: torch.nn.Parameter, *, from_main_grad: bool) -> torch.Tensor:
    return param.main_grad if from_main_grad else param.grad


def _reference_mlp_fp32(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_lens: Sequence[int],
    probs: torch.Tensor,
) -> torch.Tensor:
    """Per-expert fp32 SwiGLU MLP: the ground truth both Turbo paths are checked against.

    ``w1`` is ``[experts, 2 * ffn, hidden]`` with the gate half first, matching
    the ``torch.chunk(x, 2, dim=-1)`` split in ``PrimusGroupedMLP.bias_act_func``.
    ``w1``/``w2`` must be fp32 *copies* of the module weights, or this backward
    accumulates a second gradient into the buffers under test.
    """
    outputs = []
    offset = 0
    for expert, num_tokens in enumerate(group_lens):
        tokens = x[offset : offset + num_tokens]
        fc1_out = tokens @ w1[expert].t()
        gate, linear = torch.chunk(fc1_out, 2, dim=-1)
        act = F.silu(gate) * linear
        act = act * probs[offset : offset + num_tokens].unsqueeze(-1)
        outputs.append(act @ w2[expert].t())
        offset += num_tokens
    return torch.cat(outputs)


def _compute_snr(reference: torch.Tensor, actual: torch.Tensor) -> float:
    """SNR in dB, kept identical to Primus-Turbo's ``tests.pytorch.test_utils``
    so the thresholds carried over from there keep meaning the same thing."""
    reference = reference.detach().float()
    actual = actual.detach().float()
    signal_power = reference.norm().pow(2)
    noise_power = (reference - actual).norm().pow(2)
    return (10 * torch.log10(signal_power / (noise_power + 1e-12))).item()


@pytest.fixture(scope="module")
def pg_collection(tmp_path_factory):
    """Single-rank EP1/TP1 process groups; Turbo grouped linears require TP=1."""
    torch.cuda.set_device(0)
    created_pg = not dist.is_initialized()
    if created_pg:
        store_path = tmp_path_factory.mktemp("fused_grouped_gemm") / "store"
        dist.init_process_group(
            backend="nccl", store=dist.FileStore(str(store_path), 1), world_size=1, rank=0
        )
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )
    # init_ asserts on a live calculator an earlier module may have left behind.
    destroy_num_microbatches_calculator()
    init_num_microbatches_calculator(
        rank=0,
        rampup_batch_size=None,
        global_batch_size=1,
        micro_batch_size=1,
        data_parallel_size=1,
    )
    try:
        yield ProcessGroupCollection.use_mpu_process_groups()
    finally:
        destroy_num_microbatches_calculator()
        parallel_state.destroy_model_parallel()
        if created_pg:
            dist.destroy_process_group()


def _make_inputs(group_lens: Sequence[int], seed: int):
    total_tokens = sum(group_lens)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=generator, device="cuda", dtype=dtype)

    x = randn(total_tokens, _HIDDEN)
    grad_out = randn(total_tokens, _HIDDEN)
    # The fused GLU kernel asserts `probs.dtype == torch.float32`; the unfused
    # chain accepts fp32 too. Kept off zero so dprobs stays well conditioned.
    probs = randn(total_tokens, dtype=torch.float32).abs() + 0.1
    tokens_per_expert = torch.tensor(list(group_lens), dtype=torch.long, device="cuda")
    return x, probs, grad_out, tokens_per_expert


def _run_grouped_mlp(
    mlp: PrimusGroupedMLP,
    x: torch.Tensor,
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    use_main_grad: bool,
):
    """One fwd+bwd, returning the output and every gradient it produced."""
    x = x.clone().requires_grad_(True)
    probs = probs.clone().requires_grad_(True)

    output, output_bias = mlp(x, tokens_per_expert, probs)
    assert output_bias is None, "add_bias_linear is off, so the MLP must not return a bias"
    output.backward(grad_out)

    # Cloned, not aliased: `.grad` / `main_grad` are live accumulation buffers.
    def snapshot(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return None if tensor is None else tensor.detach().clone()

    return SimpleNamespace(
        output=output.detach().clone(),
        dx=snapshot(x.grad),
        dprobs=snapshot(probs.grad),
        dw1=snapshot(_weight_grad(mlp.linear_fc1.weights, from_main_grad=use_main_grad)),
        dw2=snapshot(_weight_grad(mlp.linear_fc2.weights, from_main_grad=use_main_grad)),
    )


@pytest.mark.parametrize("fp8_recipe", ["tensorwise", "mxfp8"])
@pytest.mark.parametrize("group_lens_name", sorted(_GROUP_LENS))
@pytest.mark.parametrize("num_microbatches", [1, 2])
def test_fused_grouped_gemm_matches_unfused(pg_collection, fp8_recipe, group_lens_name, num_microbatches):
    """``turbo_fused_grouped_gemm`` on and off must both compute the naive MLP.

    Both sides run under the same FP8 autocast over identical inputs, so the
    only difference is fused vs unfused. Each is held to Turbo's 20 dB floor
    against the fp32 reference, and the two to a tighter floor against each
    other -- so an error the unfused path happens to share still fails.
    """
    if fp8_recipe == "mxfp8":
        supported, reason = check_mxfp8_support()
        if not supported:
            pytest.skip(f"MXFP8 unsupported on this device: {reason}")

    group_lens = _GROUP_LENS[group_lens_name]
    config = _build_config(fp8_recipe)

    unfused_mlp = _build_mlp(config, pg_collection, fused=False)
    fused_mlp = _build_mlp(config, pg_collection, fused=True)
    assert unfused_mlp.turbo_fused_grouped_gemm is False
    assert fused_mlp.turbo_fused_grouped_gemm is True
    _copy_weights(fused_mlp, unfused_mlp)

    reconfigure_num_microbatches_calculator(
        rank=0,
        rampup_batch_size=None,
        global_batch_size=num_microbatches,
        micro_batch_size=1,
        data_parallel_size=1,
    )
    # Above one microbatch the real weight gradients live in main_grad.
    use_main_grad = num_microbatches > 1
    if use_main_grad:
        _attach_main_grad(unfused_mlp)
        _attach_main_grad(fused_mlp)

    x, probs, grad_out, tokens_per_expert = _make_inputs(group_lens, seed=1234)

    with primus_turbo_fp8_autocast(
        enabled=False, enabled_turbo=True, turbo_quant_config=_quant_config(fp8_recipe)
    ):
        supported, reason = experts_mod.is_turbo_fused_grouped_mlp_supported()
        assert supported, f"{fp8_recipe} should reach the fused grouped MLP, got: {reason}"

        unfused = _run_grouped_mlp(
            unfused_mlp, x, probs, tokens_per_expert, grad_out, use_main_grad=use_main_grad
        )
        fused = _run_grouped_mlp(
            fused_mlp, x, probs, tokens_per_expert, grad_out, use_main_grad=use_main_grad
        )

    # Independent fp32 leaves, so this backward yields reference wgrads without
    # touching the module weights it was copied from.
    ref_x = x.detach().float().requires_grad_(True)
    ref_probs = probs.detach().float().requires_grad_(True)
    ref_w1 = unfused_mlp.linear_fc1.weights.detach().float().requires_grad_(True)
    ref_w2 = unfused_mlp.linear_fc2.weights.detach().float().requires_grad_(True)
    ref_out = _reference_mlp_fp32(ref_x, ref_w1, ref_w2, group_lens, ref_probs)
    ref_out.backward(grad_out.float())
    reference = {
        "forward": ref_out.detach(),
        "dx": ref_x.grad,
        "dprobs": ref_probs.grad,
        "dw1": ref_w1.grad,
        "dw2": ref_w2.grad,
    }

    case = f"[{fp8_recipe}/{group_lens_name}/mb{num_microbatches}]"
    failures = []
    for name, attr in (
        ("forward", "output"),
        ("dx", "dx"),
        ("dw1", "dw1"),
        ("dw2", "dw2"),
        ("dprobs", "dprobs"),
    ):
        report = []
        for label, run in (("unfused", unfused), ("fused", fused)):
            tensor = getattr(run, attr)
            assert tensor is not None, f"{label} run produced no {name}"

            snr = _compute_snr(reference[name], tensor)
            report.append(f"{label}={snr:.2f}dB")
            # Negated compare so a NaN SNR fails instead of slipping through.
            if not snr > _SNR_VS_REFERENCE_DB:
                failures.append(
                    f"{name}/{label}: {snr:.2f} dB against the fp32 reference, "
                    f"below the {_SNR_VS_REFERENCE_DB} dB floor"
                )

        paths_floor = _SNR_FUSED_VS_UNFUSED_DB[name]
        paths_snr = _compute_snr(getattr(unfused, attr), getattr(fused, attr))
        report.append(f"fused-vs-unfused={paths_snr:.2f}dB")
        if not paths_snr > paths_floor:
            failures.append(
                f"{name}: fused vs unfused {paths_snr:.2f} dB, below the {paths_floor} dB floor -- "
                "the fused op is not computing what the unfused chain computes"
            )

        print(f"{case} {name} SNR: " + ", ".join(report), flush=True)

    assert not failures, f"{case} grouped MLP accuracy check failed: " + "; ".join(failures)
