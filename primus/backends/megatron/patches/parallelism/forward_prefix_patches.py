# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Optional prefetches for Megatron's non-interleaved training schedule."""

import inspect
import textwrap
from functools import partial

from primus.backends.megatron.patches._source_patch_utils import patch_function_source
from primus.core.patches import PatchContext, get_param, register_patch
from primus.core.utils.module_utils import log_rank_0


def validate_pipeline_prefetch(args):
    depth = getattr(args, "pipeline_forward_prefix_layers", 0)
    early_gather = getattr(args, "early_expert_param_gather", False)
    if not (depth or early_gather):
        return
    if (
        args.pipeline_model_parallel_size < 2
        or getattr(args, "cuda_graph_impl", "none") != "none"
        or any(
            getattr(args, name, False)
            for name in (
                "virtual_pipeline_model_parallel_size",
                "num_layers_per_virtual_pipeline_stage",
                "enable_cuda_graph",
                "external_cuda_graph",
                "patch_zero_bubble",
                "patch_primus_pipeline",
                "overlap_moe_expert_parallel_comm",
                "use_megatron_fsdp",
                "use_torch_fsdp2",
            )
        )
    ):
        raise ValueError("Pipeline prefetches require eager, non-interleaved Megatron DDP training.")
    if early_gather and not (
        args.num_experts
        and args.use_distributed_optimizer
        and args.overlap_param_gather
        and not args.align_param_gather
        and not args.overlap_param_gather_with_optimizer_step
        and args.num_distributed_optimizer_instances == 1
    ):
        raise ValueError(
            "early_expert_param_gather requires native, unaligned distributed-optimizer parameter hooks."
        )
    if depth and not (
        type(depth) is int
        and depth > 0
        and args.pipeline_model_parallel_size == 2
        and args.tensor_model_parallel_size == 1
        and (
            getattr(args, "decoder_first_pipeline_num_layers", None) is None
            or depth < args.decoder_first_pipeline_num_layers
        )
        and args.context_parallel_size == 1
        and args.num_experts
        and args.enable_primus_turbo
        and not args.sequence_parallel
        and args.fp8 == "hybrid"
        and getattr(args.fp8_recipe, "value", args.fp8_recipe) == "tensorwise"
        and args.recompute_granularity == "full"
        and args.recompute_method == "block"
        and args.recompute_num_layers == 0
        and getattr(args, "recompute_layer_ids", None) is None
        and not getattr(args, "mtp_num_layers", None)
        and args.hidden_dropout == args.attention_dropout == 0
        and getattr(args, "num_microbatches_with_partial_activation_checkpoints", None) is None
        and not any(
            getattr(args, key, False)
            for key in (
                "cpu_offloading",
                "fine_grained_activation_offloading",
                "moe_paged_stash",
                "enable_autocast",
            )
        )
    ):
        raise ValueError(
            "pipeline_forward_prefix_layers requires PP2, TP1, CP1, hybrid tensorwise FP8, "
            "full/block recompute with zero checkpointed layers, no dropout, offloading or MTP."
        )


def _replace_once(source, anchor, replacement):
    if source.count(anchor) != 1:
        raise RuntimeError(f"Unexpected Megatron non-interleaved schedule near {anchor.strip()!r}")
    return source.replace(anchor, replacement, 1)


def _schedule_source(source, early_gather, depth):
    if early_gather:
        anchor = "    # Before running 1F1B, need to receive first forward tensor."
        source = _replace_once(
            source,
            anchor,
            """    if not forward_only and p2p_communicator.is_pp_last_stage:
        _primus_prefetch_expert_parameters(model)

"""
            + anchor,
        )
    if not depth:
        return source
    loop = "    # Run 1F1B in steady state."
    before, steady = source.split(loop)
    # Reuse Megatron's entire steady-state exchange/backward body, including its
    # deallocation, FIFO bookkeeping and DDP synchronization, at either call site.
    start = "            output_tensor_grad = p2p_communicator.send_forward_recv_backward("
    end = "\n    # Run cooldown backward passes."
    tail = steady[steady.index(start) : steady.index(end)]
    if "input_tensor = p2p_communicator.send_backward_recv_forward(" not in tail:
        raise RuntimeError("Missing native steady-state receive boundary")
    before += (
        """    prefix = None
    if not forward_only and p2p_communicator.is_pp_first_stage and num_microbatches_remaining >= 2:
        prefix = _primus_forward_prefix(model, forward_step=forward_step, timers=config.timers)

    def complete_steady_backward(input_tensor, output_tensor, last_iteration):
"""
        + textwrap.indent(textwrap.dedent(tail), "        ")
        + "        return input_tensor\n\n"
    )
    call = "        output_tensor, num_tokens = forward_step("
    steady = _replace_once(
        steady,
        call,
        """        run_forward = prefix.forward if prefix is not None and i == 1 else forward_step
        output_tensor, num_tokens = run_forward(""",
    )
    steady = _replace_once(
        steady,
        tail,
        """            if prefix is not None and i == 0:
                prefix.defer(_primus_partial(complete_steady_backward, input_tensor, output_tensor, last_iteration))
            else:
                input_tensor = complete_steady_backward(input_tensor, output_tensor, last_iteration)
""",
    )
    return before + loop + steady


@register_patch(
    "megatron.pp.pipeline_prefetch",
    backend="megatron",
    phase="before_train",
    condition=lambda ctx: bool(
        get_param(ctx, "early_expert_param_gather", False)
        or get_param(ctx, "pipeline_forward_prefix_layers", 0)
    ),
    description="Prefetch the last-stage expert parameters or a prefix of the first stage's third forward",
)
def patch_pipeline_prefetch(ctx: PatchContext):
    from megatron.core import recompute
    from megatron.core.pipeline_parallel import schedules
    from megatron.core.transformer import transformer_block

    from primus.backends.megatron.core.pipeline_parallel.forward_prefix import (
        ForwardPrefix,
        after_decoder_layer,
        prefetch_expert_parameters,
    )

    depth = get_param(ctx, "pipeline_forward_prefix_layers", 0)
    gather = get_param(ctx, "early_expert_param_gather", False)
    name = "forward_backward_pipelining_without_interleaving"
    original = getattr(schedules, name)
    if getattr(original, "_primus_pipeline_prefetch", False):
        return
    source = inspect.getsource(original)
    replacement = _schedule_source(source, gather, depth)
    if depth:
        old = recompute.checkpointed_forward
        if transformer_block.checkpointed_forward is not old:
            raise ValueError("Forward prefix requires the native decoder layer loop.")
        anchor = "            chunk_runner(layer_idx, layer_idx + 1, use_checkpoint)\n"
        changed = _replace_once(
            inspect.getsource(old),
            anchor,
            anchor + "            _primus_after_decoder_layer(self, layer_idx + 1)\n",
        )
        recompute._primus_after_decoder_layer = after_decoder_layer
        transformer_block.checkpointed_forward = patch_function_source(
            recompute, "checkpointed_forward", inspect.getsource(old), changed
        )
        schedules._primus_forward_prefix = partial(ForwardPrefix, depth=depth)
        schedules._primus_partial = partial
    schedules._primus_prefetch_expert_parameters = prefetch_expert_parameters
    patched = patch_function_source(schedules, name, source, replacement)
    patched._primus_pipeline_prefetch = True
    log_rank_0(f"Enabled pipeline prefetches: early expert gather={gather}, forward prefix layers={depth}")
