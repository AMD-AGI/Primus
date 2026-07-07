# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

###############################################################################
# LB-Mini (sequence-length load balancing) for Megatron's FSDP2 + ODC path.
#
# ONE switch, TWO code paths:
#   * enable_odc_lb_mini=false (DEFAULT) -> this patch is a complete no-op;
#     Megatron runs its stock fixed-num_microbatches, all-ranks-in-lockstep
#     schedule. Byte-for-byte unchanged.
#   * enable_odc_lb_mini=true (and ODC_ENABLE=1) -> data is served variable
#     length and Karmarkar-Karp balanced across DP ranks; each rank runs its OWN
#     (possibly different) number of micro-batches. Only ODC's point-to-point
#     comm can drive ranks out of lockstep without a collective deadlock, hence
#     the ODC_ENABLE=1 requirement.
#
# All wiring is monkey-patch in the Primus layer; the third-party Megatron-LM
# source is NOT modified.
#
# Stage-1 scope (this file): make "different micro-batch count per rank" run end
# to end without deadlocking. Numerical normalization (loss_scale / consumed
# samples by real tokens) is Stage-2.
###############################################################################

import os

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0, warning_rank_0

# Global handle so the schedule patch can reach the LB-Mini iterator that the
# dataloader patch created. Stage-1 only drives the TRAIN iterator (eval_iters=0
# in the aligned config), so a single handle is sufficient.
_LB_MINI_TRAIN_ITER = None
_FB_PATCHED = False


def _lb_mini_enabled(args) -> bool:
    """LB-Mini requires BOTH the explicit switch AND ODC comm to be on.

    The switch can come from the YAML arg ``enable_odc_lb_mini`` or, as a
    convenience for experiments, the env var ``ODC_LB_MINI=1``.
    """
    switch_on = bool(getattr(args, "enable_odc_lb_mini", False)) or (
        os.environ.get("ODC_LB_MINI", "0") == "1"
    )
    if not (switch_on and bool(getattr(args, "use_torch_fsdp2", False))):
        return False
    # Normal case: LB-Mini needs ODC point-to-point comm (KK can give ranks
    # different micro-batch counts -> NCCL collectives would deadlock).
    if os.environ.get("ODC_ENABLE", "0") == "1":
        return True
    # A/B exception: LB_MINI_FORCE_DATA=1 lets the variable-length data layer run
    # under NCCL too -- ONLY safe with round_robin packing (all ranks keep equal
    # micro-batch counts). Used to benchmark ODC vs NCCL on the SAME uneven data.
    if os.environ.get("LB_MINI_FORCE_DATA", "0") == "1":
        return True
    return False


def _build_lb_mini_train_iterator(args):
    """Build the variable-length, KK-balanced LB-Mini train iterator."""
    from megatron.core import mpu
    from megatron.training import get_tokenizer
    from primus.backends.megatron.sft.lb_mini_dataset import (
        LBMiniDataIterator,
        build_varlen_samples,
    )
    from primus.backends.megatron.sft.packing import _resolve_pad_token_id

    tokenizer = get_tokenizer()
    samples = build_varlen_samples(
        dataset_name=getattr(args, "sft_dataset_name", "tatsu-lab/alpaca"),
        tokenizer=tokenizer,
        max_seq_length=args.seq_length,
        # env-gated: datasets like SWE-bench/SWE-smith-trajectories have no "train" split
        # (splits: tool/xml/ticks). SFT_DATASET_SPLIT overrides; default "train" preserved.
        split=os.environ.get("SFT_DATASET_SPLIT", "train"),
        formatter=getattr(args, "sft_conversation_format", "alpaca"),
        seed=args.seed,
        bridge_compat_inline_bos=bool(getattr(args, "sft_bridge_compat_inline_bos", False)),
    )
    # Optional length filter: drop samples longer than LB_MINI_FILTER_MAXLEN.
    # Lets us REUSE a long-seq cache (e.g. seq65536, key 334087) while keeping
    # every micro-batch small enough that STANDARD (non-fused) CE logits do not
    # OOM. Used to validate ODC integration WITHOUT fused CE: filter to <=16384
    # keeps per-micro-batch logits ~= 16384 * vocab * 4B ~= 10GB (safe).
    filter_maxlen = int(os.environ.get("LB_MINI_FILTER_MAXLEN", "0") or 0)
    if filter_maxlen > 0:
        _before = len(samples)
        samples = [s for s in samples if int(s["length"]) <= filter_maxlen]
        log_rank_0(
            f"[LB-Mini] LB_MINI_FILTER_MAXLEN={filter_maxlen}: kept "
            f"{len(samples)}/{_before} samples (dropped len > {filter_maxlen})"
        )
    # Optional LOWER bound: drop samples shorter than LB_MINI_FILTER_MINLEN.
    # Combined with LB_MINI_FILTER_MAXLEN this carves a NARROW length window out
    # of an existing dataset to construct a controlled SMALL-VARIANCE tier (same
    # source as the medium/large-variance runs, only the length spread changes),
    # for the "length variance vs ODC load-balance advantage" study. Default 0 ->
    # disabled, so every other dataset/experiment keeps its exact prior behavior.
    filter_minlen = int(os.environ.get("LB_MINI_FILTER_MINLEN", "0") or 0)
    if filter_minlen > 0:
        _before = len(samples)
        samples = [s for s in samples if int(s["length"]) >= filter_minlen]
        log_rank_0(
            f"[LB-Mini] LB_MINI_FILTER_MINLEN={filter_minlen}: kept "
            f"{len(samples)}/{_before} samples (dropped len < {filter_minlen})"
        )
    # Per-micro-batch token cap. Larger than a single sample lets short samples
    # pack together and long samples stand alone -> creates per-rank micro-batch
    # count differences (where DiffMicro saves comm rounds). Priority:
    # env LB_MINI_MAX_TOKEN > yaml lb_mini_max_token_len > seq_length.
    # NOTE: int() each candidate BEFORE `or` -- the env var is a string and the
    # string "0" is truthy, so `os.environ.get(..., "0") or ...` would wrongly
    # pick "0". Convert to int first so 0 is correctly falsy.
    max_token_len = (
        int(os.environ.get("LB_MINI_MAX_TOKEN", "0") or 0)
        or int(getattr(args, "lb_mini_max_token_len", 0) or 0)
        or int(args.seq_length)
    )
    # A/B knob for the perf study: LB_MINI_SAME_MICRO=1 -> aligned baseline (all
    # ranks same micro-batch count, no LB-Mini decoupling). Default 0 -> LB-Mini.
    same_micro = os.environ.get("LB_MINI_SAME_MICRO", "0") == "1"
    packing_method = os.environ.get("LB_MINI_PACKING", "kk")
    it = LBMiniDataIterator(
        samples=samples,
        global_batch_size=args.global_batch_size,
        max_token_len=max_token_len,
        dp_rank=mpu.get_data_parallel_rank(),
        dp_size=mpu.get_data_parallel_world_size(),
        pad_id=_resolve_pad_token_id(tokenizer),
        cost_model=str(getattr(args, "lb_mini_cost_model", "linear")),
        seed=args.seed,
        shuffle=True,
        same_micro_num=same_micro,
        packing_method=packing_method,
    )
    log_rank_0(
        f"[ODC.lb_mini] built LB-Mini train iterator: {len(samples)} varlen samples, "
        f"global_batch_size={args.global_batch_size}, max_token_len={max_token_len}, "
        f"dp_size={mpu.get_data_parallel_world_size()}, cost_model={getattr(args, 'lb_mini_cost_model', 'linear')}, "
        f"same_micro_num={same_micro} ({'ALIGNED baseline' if same_micro else 'LB-Mini decoupled'})"
    )
    return it


def _install_dataloader_patch():
    """Patch build_pretraining_data_loader so the TRAIN loader is LB-Mini.

    Valid/test loaders (if any) fall through to the stock builder unchanged.
    We tag the first (train) request via a module flag because the stock
    signature does not carry the split explicitly.
    """
    import megatron.training.training as mt_training
    from megatron.training.datasets import data_samplers

    if getattr(data_samplers.build_pretraining_data_loader, "_lb_mini_hooked", False):
        return

    orig_builder = data_samplers.build_pretraining_data_loader

    def lb_mini_builder(dataset, consumed_samples):
        global _LB_MINI_TRAIN_ITER
        # Runtime call: use Megatron's get_args() (no ctx). Primus' get_args(ctx)
        # is only valid inside register_patch conditions / patch bodies.
        from megatron.training import get_args as _mt_get_args

        args = _mt_get_args()
        # Only replace the TRAIN loader, and only once (the first non-zero-len
        # build). Identify train by: not yet built + dataset present.
        if _lb_mini_enabled(args) and _LB_MINI_TRAIN_ITER is None and dataset is not None:
            try:
                _LB_MINI_TRAIN_ITER = _build_lb_mini_train_iterator(args)
                log_rank_0("[ODC.lb_mini] TRAIN dataloader replaced by LB-Mini iterator")
                return _LB_MINI_TRAIN_ITER
            except Exception as e:  # noqa: BLE001
                warning_rank_0(
                    f"[ODC.lb_mini] failed to build LB-Mini iterator, "
                    f"falling back to stock loader: {type(e).__name__}: {e}"
                )
        return orig_builder(dataset, consumed_samples)

    lb_mini_builder._lb_mini_hooked = True
    data_samplers.build_pretraining_data_loader = lb_mini_builder
    # training.py imported the symbol into its own namespace; rebind there too.
    if hasattr(mt_training, "build_pretraining_data_loader"):
        mt_training.build_pretraining_data_loader = lb_mini_builder
    log_rank_0("[ODC.lb_mini] hooked build_pretraining_data_loader")


def _install_schedule_patch():
    """Patch forward_backward_no_pipelining to use THIS rank's micro-batch count.

    At the top of every train_step the LB-Mini iterator plans one global
    minibatch (KK balance across ranks); we read this rank's micro-batch count
    and override the (globally-identical) ``num_microbatches`` argument so the
    schedule's forward/backward loop runs the right rank-local number of steps.
    """
    global _FB_PATCHED
    import megatron.core.pipeline_parallel.schedules as sched

    if _FB_PATCHED or getattr(sched.forward_backward_no_pipelining, "_lb_mini_hooked", False):
        return

    orig_fb = sched.forward_backward_no_pipelining

    def lb_mini_fb(*args, **kwargs):
        it = _LB_MINI_TRAIN_ITER
        forward_only = kwargs.get("forward_only", False)
        # Only re-plan for the train path (an LB-Mini iterator exists) and when
        # actually training (forward_only=False is the train_step path).
        if it is not None and not forward_only:
            try:
                local_nmb = it.begin_minibatch()
                if local_nmb > 0:
                    kwargs["num_microbatches"] = local_nmb
            except Exception as e:  # noqa: BLE001
                warning_rank_0(
                    f"[ODC.lb_mini] begin_minibatch failed, using stock "
                    f"num_microbatches: {type(e).__name__}: {e}"
                )
        return orig_fb(*args, **kwargs)

    lb_mini_fb._lb_mini_hooked = True
    sched.forward_backward_no_pipelining = lb_mini_fb

    # get_forward_backward_func returns the module-global by name; rebinding the
    # module attribute is enough as long as it is fetched AFTER this patch. Also
    # patch the function it returns defensively if it caches a reference.
    _FB_PATCHED = True
    log_rank_0("[ODC.lb_mini] hooked forward_backward_no_pipelining (rank-local num_microbatches)")


@register_patch(
    "megatron.fsdp.odc_lb_mini",
    backend="megatron",
    phase="before_train",
    description="LB-Mini sequence-length load balancing for Megatron FSDP2+ODC (one switch, two paths).",
    condition=lambda ctx: _lb_mini_enabled(get_args(ctx)),
)
def patch_odc_lb_mini(ctx: PatchContext):
    log_rank_0(
        "[ODC.lb_mini] enable_odc_lb_mini=true + ODC_ENABLE=1 -> installing LB-Mini "
        "(variable-length KK-balanced data, rank-local micro-batch counts)."
    )
    # Loss normalization under torch FSDP2: we KEEP calculate_per_token_loss at
    # its default (False). Each micro-batch loss is mean-reduced (/=num_tokens)
    # then /=num_microbatches (this rank's count) -- the same per-minibatch mean
    # ODC's own example uses, and it keeps gradients at the right magnitude.
    #
    # We deliberately do NOT force calculate_per_token_loss=True. Under
    # use_torch_fsdp2 the per-token grad rescale lives in Megatron's
    # finalize_model_grads (scale by the GLOBAL all-reduced token count), but
    # FSDP2 does its OWN reduce-scatter and BYPASSES that path, so per-token
    # leaves the summed (un-normalized) loss and gradients explode ~1000x
    # (measured: grad norm ~45000 vs ~55). KK balancing keeps per-rank
    # micro-batch counts nearly equal, so the residual per-minibatch-mean
    # weighting difference (e.g. a rank with 3 vs 4 micro-batches) is negligible.
    _install_dataloader_patch()
    _install_schedule_patch()


__all__ = ["patch_odc_lb_mini"]
