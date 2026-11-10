###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark: fix up or reject arg combinations that Megatron / TE would mis-handle.

Each case either rewrites the args with a rank-0 warning that names the old and new
values (so an env-driven example yaml can change one knob alone), or raises. Rewritten:
p2p CP, DeepEP at expert TP x EP == 1 or spanning nodes
(without ``muse_spark_turbo_deepep_internode``), Turbo grouped GEMM with Megatron FSDP. Rejected: ``a2a+p2p`` CP and the fused Turbo router
(``turbo_sync_free_moe_stage`` > 0 / ``moe_use_fused_router_with_aux_score``).

The DeepEP and Turbo grouped-GEMM checks read ``backend_args`` and
``module_config.params`` merged (``model_types.merged_args_of``: ``backend_args`` wins on a
key both carry; a Primus-only key such as a Turbo flag is read from ``params``). The
fused-router check is stricter: it rejects a conflict in ``params``, in ``backend_args`` or
in the merged view, so a fused-router key in either place stops the run. The
CP check reads ``backend_args`` only (the Megatron args it fixes up).

* context parallel with sliding-window attention: TE's p2p CP has no SWA backend.
  Megatron's default (``['p2p']``, and ``p2p`` in Primus's language_model.yaml) is
  rewritten to ``all_gather`` with a warning, per entry of the list; ``a2a+p2p`` is
  an error. ``muse_spark/muse_spark_base.yaml`` already sets ``all_gather``, so the rewrite
  only fires when a config or the CLI asks for p2p, leaves cp_comm_type unset, or uses
  the ``allgather`` spelling (respelled to ``all_gather``).
* Primus-Turbo DeepEP (``use_turbo_deepep``, opt-in in every Muse Spark example) with expert
  TP x EP == 1 (the dispatcher requires > 1) or an expert group that spans nodes (larger
  than a node, or the GPUs per node not a multiple of it, e.g. EP 3 or 6 on 8-GPU nodes;
  a Primus-Turbo build without rocSHMEM, as in the Primus image, asserts
  ``num_rdma_ranks == 1``): it is turned
  off with a warning and the alltoall dispatcher runs. ``muse_spark_turbo_deepep_internode: true``
  keeps it across nodes for a build with rocSHMEM.
* Primus-Turbo grouped GEMM with Megatron FSDP (``use_megatron_fsdp``): turned off with a
  warning (TE grouped GEMM), see ``turbo_grouped_gemm_needs_disabling``.
* fused linear cross entropy (``enable_fused_linear_ce`` / ``FUSED_LINEAR_CE=1``) is
  allowed: its chunked loss calls ``MuseSparkModel._scale_logits`` (multiplier and soft-cap)
  in forward and backward.
* the fused Turbo router: ``turbo_sync_free_moe_stage`` >= 1 turns on
  ``moe_use_fused_router_with_aux_score`` (stage 0 turns on nothing). That path
  (``PrimusTopKRouter`` -> ``fused_group_topk_routing_with_aux_score``) takes no
  ``expert_bias``, so it drops Muse Spark's ``e_score_correction_bias`` from the expert
  selection and the routing differs from HF. Both settings are rejected.
"""

from primus.backends.megatron.core.models.muse_spark.memory_check import gpus_per_node
from primus.backends.megatron.core.models.muse_spark.model_types import (
    backend_args_of,
    bool_flag,
    config_flag,
    is_muse_spark_model_type,
    merged_args_of,
)
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import warning_rank_0

INTERNODE_ARG = "muse_spark_turbo_deepep_internode"  # config / args key (muse_spark_base.yaml: false)


def internode_deepep_opted_in(args) -> bool:
    """``args.muse_spark_turbo_deepep_internode`` is true: the Primus-Turbo build has rocSHMEM
    (internode dispatch / combine), so DeepEP may span nodes. Parsed by
    :func:`model_types.bool_flag` (an unrecognised value raises ValueError)."""
    value = getattr(
        args, "muse_spark_turbo_deepep_internode", False
    )  # INTERNODE_ARG; a literal for the config-schema scan
    return bool_flag(value, INTERNODE_ARG)


def deepep_disable_reason(args, local_gpus=None):
    """Why Turbo DeepEP must be turned off for these args, or None to keep it.

    * expert TP x EP == 1: the DeepEP dispatcher raises at build (its tp_size is the
      expert TP, which Muse Spark keeps at 1);
    * expert TP x EP > GPUs per node, or GPUs per node not a multiple of it: some expert
      group (consecutive ranks) spans nodes. A Primus-Turbo build
      without rocSHMEM (the Primus image's: its ``_C`` library holds the "rocSHMEM is
      disabled during compilation" assert) asserts ``num_rdma_ranks == 1`` there. Whether
      a build has internode dispatch is known only from a live ``deep_ep.Buffer``
      (``is_internode_available()``), not at args time, so a build with rocSHMEM opts in
      with ``muse_spark_turbo_deepep_internode: true`` and keeps DeepEP across nodes.
    """
    if not (config_flag(args, "use_turbo_deepep", False) and config_flag(args, "enable_primus_turbo", False)):
        return None
    etp = getattr(args, "expert_tensor_parallel_size", None)
    if etp is None:  # Megatron: expert TP defaults to TP
        etp = getattr(args, "tensor_model_parallel_size", 1)
    group = int(etp or 1) * int(getattr(args, "expert_model_parallel_size", 1) or 1)
    if group == 1:
        return "it needs expert_tensor_parallel_size x expert_model_parallel_size > 1; it is 1"
    local = gpus_per_node() if local_gpus is None else int(local_gpus)
    # expert groups are runs of consecutive ranks: one stays inside a node only when it is no
    # larger than a node and the node size is a multiple of it (EP 3 or 6 on 8-GPU nodes:
    # ranks 6-8 or 6-11 straddle two nodes)
    spans_nodes = group > local or local % group != 0
    if spans_nodes and not internode_deepep_opted_in(args):
        return (
            f"the expert group (expert TP x EP = {group}) spans nodes ({local} GPUs per node, "
            "not a multiple of the group) and "
            "a Primus-Turbo build without rocSHMEM is intranode only (it asserts num_rdma_ranks == 1); "
            f"with a rocSHMEM build set {INTERNODE_ARG}: true to keep DeepEP"
        )
    return None


@register_patch(
    "megatron.args.muse_spark_deepep_group_size",
    backend="megatron",
    phase="build_args",
    condition=is_muse_spark_model_type,
    description=(
        "Muse Spark: Turbo DeepEP needs expert TP x EP > 1 and, unless "
        "muse_spark_turbo_deepep_internode, an expert group inside one node (<= GPUs per node "
        "and dividing it); else turn it off (alltoall)"
    ),
)
def patch_muse_spark_deepep_group_size(ctx: PatchContext):
    # The Turbo flags live in module_config.params (what the Turbo dispatcher patch reads
    # through get_args); the parallel sizes may be overridden in backend_args: read both.
    args = merged_args_of(ctx)
    reason = deepep_disable_reason(args)
    if reason is None:
        return
    args.use_turbo_deepep = False
    dispatcher = getattr(args, "moe_token_dispatcher_type", None)
    if dispatcher == "flex":  # flex needs DeepEP (or HybridEP): without it, use alltoall
        args.moe_token_dispatcher_type = "alltoall"
        args.moe_enable_deepep = False
        dispatcher = "alltoall (was flex)"
    warning_rank_0(
        f"[Patch:megatron.args.muse_spark_deepep_group_size] use_turbo_deepep: {reason}; "
        f"DeepEP is off; moe_token_dispatcher_type: {dispatcher}"
    )


def turbo_grouped_gemm_needs_disabling(args) -> bool:
    """True when Turbo grouped GEMM is requested with Megatron FSDP.

    PrimusTurboGroupedLinear merges the expert weights into one ``weights`` Parameter and
    registers the per-expert ``weight{i}`` views at the first forward, after Megatron FSDP
    has built its param -> bucket map, so FSDP's post-forward release hits an unknown param.
    """
    return (
        config_flag(args, "use_megatron_fsdp", False)
        and config_flag(args, "enable_primus_turbo", False)
        and config_flag(args, "use_turbo_grouped_gemm", False)
    )


@register_patch(
    "megatron.args.muse_spark_mfsdp_turbo_grouped_gemm",
    backend="megatron",
    phase="build_args",
    condition=is_muse_spark_model_type,
    description="Muse Spark: Megatron FSDP cannot shard Turbo grouped-GEMM experts; use TE grouped GEMM",
)
def patch_muse_spark_mfsdp_turbo_grouped_gemm(ctx: PatchContext):
    args = merged_args_of(ctx)
    if not turbo_grouped_gemm_needs_disabling(args):
        return
    args.use_turbo_grouped_gemm = False
    warning_rank_0(
        "[Patch:megatron.args.muse_spark_mfsdp_turbo_grouped_gemm] use_megatron_fsdp with "
        "use_turbo_grouped_gemm: Turbo's grouped linear registers its per-expert weight views "
        "after FSDP wrapping, which Megatron FSDP cannot track; using the TE grouped GEMM"
    )


def resolve_cp_comm_type(comm):
    """The cp_comm_type to use with sliding-window attention under CP (> 1).

    Each entry (a str, or the per-layer list Megatron's ``--cp-comm-type`` gives,
    default ``['p2p']``; Primus's language_model.yaml sets ``p2p``) maps as:
    ``p2p`` / None (unset) -> ``all_gather``; ``allgather`` -> ``all_gather`` (TE's
    spelling); ``all_gather`` / ``a2a`` unchanged; ``a2a+p2p`` -> ValueError (its p2p
    ring has no sliding-window backend and there is no equal replacement). A list
    stays a list.
    """

    def one(c):
        if c in (None, "p2p", "allgather"):
            return "all_gather"
        if c in ("all_gather", "a2a"):
            return c
        if c == "a2a+p2p":
            raise ValueError(
                "Muse Spark: cp_comm_type 'a2a+p2p' with sliding-window attention and context "
                "parallelism: its p2p ring has no sliding-window backend in TE; use all_gather or a2a"
            )
        raise ValueError(
            f"Muse Spark: unknown cp_comm_type entry {c!r}; accepted: p2p / allgather (both run "
            "as all_gather with sliding-window attention), all_gather, a2a"
        )

    if isinstance(comm, (list, tuple)):
        return [one(c) for c in comm]
    return one(comm)


@register_patch(
    "megatron.args.muse_spark_cp_comm",
    backend="megatron",
    phase="build_args",
    condition=is_muse_spark_model_type,
    description="Muse Spark: sliding-window attention with CP needs TE's all_gather CP (p2p has no SWA backend)",
)
def patch_muse_spark_cp_comm(ctx: PatchContext):
    args = backend_args_of(ctx)
    cp = getattr(args, "context_parallel_size", 1) or 1
    if getattr(args, "window_size", None) is None or cp <= 1:
        return
    comm = getattr(args, "cp_comm_type", None)
    new = resolve_cp_comm_type(comm)
    if new != (list(comm) if isinstance(comm, tuple) else comm):
        args.cp_comm_type = new
        entries = comm if isinstance(comm, (list, tuple)) else [comm]
        reasons = []
        if any(c in (None, "p2p") for c in entries):
            reasons.append(
                "TE's p2p context parallelism (the default when unset) has no sliding-window attention backend"
            )
        if "allgather" in entries:
            reasons.append("'allgather' is spelled 'all_gather' in TE")
        warning_rank_0(
            f"[Patch:megatron.args.muse_spark_cp_comm] cp_comm_type {comm!r} -> {new!r}: "
            + "; ".join(reasons)
        )


def fused_router_conflicts(args) -> list:
    """The settings that would run the fused Turbo router (no expert bias): a
    ``turbo_sync_free_moe_stage`` >= 1 (every stage sets ``moe_use_fused_router_with_aux_score``)
    or that flag itself with Primus-Turbo on. An empty list means none."""
    found = []
    stage = getattr(args, "turbo_sync_free_moe_stage", 0)
    try:
        stage = int(stage or 0)
    except (TypeError, ValueError):
        stage = 0  # rocm_arg_validation rejects a malformed stage
    if stage > 0:
        found.append(f"turbo_sync_free_moe_stage={stage}")
    if config_flag(args, "enable_primus_turbo", False) and config_flag(
        args, "moe_use_fused_router_with_aux_score", False
    ):
        found.append("moe_use_fused_router_with_aux_score=True")
    return found


@register_patch(
    "megatron.args.muse_spark_no_fused_router",
    backend="megatron",
    phase="build_args",
    condition=is_muse_spark_model_type,
    description="Muse Spark: the fused Turbo router drops e_score_correction_bias; reject sync-free MoE stages >= 1",
)
def patch_muse_spark_no_fused_router(ctx: PatchContext):
    # the Turbo keys live in module_config.params (Primus-only); backend_args has the rest.
    # A rejection checks each namespace and the merged view (a conflict in any one counts).
    params = getattr(ctx.extra.get("module_config"), "params", None)
    found = []
    for args in (params, ctx.extra.get("backend_args"), merged_args_of(ctx)):
        if args is not None:
            found += [f for f in fused_router_conflicts(args) if f not in found]
    if found:
        raise ValueError(
            f"Muse Spark: {', '.join(found)} selects the fused Primus-Turbo router "
            "(fused_group_topk_routing_with_aux_score), which takes no expert bias: it drops "
            "e_score_correction_bias from the expert selection, so the routing differs from HF. "
            "Use turbo_sync_free_moe_stage: 0 and moe_use_fused_router_with_aux_score: false "
            "(set moe_permute_fusion / use_turbo_deepep / use_turbo_grouped_gemm one by one instead)."
        )
