###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ROCm-specific Megatron argument validation.

``validate_args_on_rocm`` is invoked by the ``megatron.validate_args`` patch
after Megatron's own ``validate_args`` runs. It enforces ROCm/Primus-Turbo
constraints (deterministic-mode env vars, Turbo FP8/FP4 recipes, sync-free MoE
auto-config, DeepEP restrictions, ...).
"""

import importlib.util
import inspect
import os

import torch

from primus.core.utils import logger
from primus.core.utils.module_utils import warning_rank_0

# Environment variables read by the PyTorch caching allocator.
ALLOC_CONF_ENVS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")


def is_last_rank():
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def print_rank_last(msg):
    """If distributed is initialized, print only on last rank."""
    log_func = logger.info_with_caller

    caller = inspect.stack()[1]
    caller_frame = caller.frame
    function_name = caller_frame.f_code.co_name
    module_name = caller_frame.f_globals["__name__"].split(".")[-1]
    line = caller.lineno

    if torch.distributed.is_initialized():
        if is_last_rank():
            log_func(msg, module_name, function_name, line)
    else:
        log_func(msg, module_name, function_name, line)


def _get_sync_free_moe_options(args) -> dict:
    stage = args.turbo_sync_free_moe_stage

    if stage > 3 or stage < 0:
        raise ValueError("turbo_sync_free_moe_stage only support [0-3]")

    sync_free_moe = {
        1: {
            "moe_use_fused_router_with_aux_score": True,
            "moe_permute_fusion": True,
            "moe_router_padding_for_quantization": True if args.fp8 or args.fp4 else False,
        },
        2: {
            "moe_use_fused_router_with_aux_score": True,
            "use_turbo_deepep": True,
            "moe_permute_fusion": True,
            "use_turbo_grouped_gemm": True,
            "moe_router_padding_for_quantization": True if args.fp8 or args.fp4 else False,
        },
        3: {
            "moe_use_fused_router_with_aux_score": True,
            "use_turbo_deepep": True,
            "moe_permute_fusion": True,
            "use_turbo_grouped_gemm": True,
            "moe_router_padding_for_quantization": True if args.fp8 or args.fp4 else False,
            "use_turbo_fused_act_with_probs": True,
        },
    }

    return sync_free_moe[stage]


# FSDP2 custom optimizer selection flags. Each one monkeypatches
# get_megatron_optimizer at the same priority (50), so at most one may be set.
_FSDP2_OPTIMIZER_FLAGS = (
    "use_fsdp2_fp32_param_optimizer",
    "use_fsdp2_bf16_master_weight_optimizer",
)


def validate_fsdp2_optimizer_exclusivity(args) -> None:
    """Ensure at most one FSDP2 custom optimizer flag is enabled.

    Enabling more than one would silently let whichever optimizer patch applies
    last win the monkeypatch, so raise ValueError to fail loudly at
    arg-validation time (before training starts).
    """
    enabled = [flag for flag in _FSDP2_OPTIMIZER_FLAGS if getattr(args, flag, False)]
    if len(enabled) > 1:
        raise ValueError(
            "Conflicting FSDP2 optimizer selection: at most one of "
            f"{list(_FSDP2_OPTIMIZER_FLAGS)} may be enabled, but got {enabled}. "
            "Enable exactly one."
        )


def validate_cross_entropy_exclusivity(args) -> None:
    """Reject Turbo cross entropy together with the deferred TE cross entropy.

    Both replace Megatron's ``te_parallel_cross_entropy``, and patch failures are only
    logged, so the check has to stop the run here rather than in either patch.
    """
    if getattr(args, "deferred_cross_entropy", False) and getattr(args, "use_turbo_cross_entropy", False):
        raise ValueError(
            "deferred_cross_entropy and use_turbo_cross_entropy are mutually exclusive; enable at most one."
        )


def validate_turbo_grouped_gemm_without_padding(args) -> None:
    """Validate the no-padding PrimusTurbo grouped-GEMM path."""
    option = "turbo_grouped_gemm_without_padding"
    if not getattr(args, option, False):
        return
    if not getattr(args, "enable_primus_turbo", False) or not getattr(args, "use_turbo_grouped_gemm", False):
        raise ValueError(f"{option}=True requires enable_primus_turbo=True and use_turbo_grouped_gemm=True.")
    if getattr(args, "moe_router_padding_for_quantization", False):
        raise ValueError(f"{option}=True requires moe_router_padding_for_quantization=False.")


def _primus_turbo_installed() -> bool:
    return importlib.util.find_spec("primus_turbo") is not None


def _has_moe_layers(args) -> bool:
    if not getattr(args, "num_experts", None):
        return False
    # Pattern strings are already lists here: the build_args moe_layer_freq patch evaluates them.
    freq = getattr(args, "moe_layer_freq", 1)
    if isinstance(freq, int):
        return freq > 0
    return any(freq)


def validate_turbo_grouped_gemm_with_muon(args) -> None:
    """Muon orthogonalizes only 2-D weights and hands the rest to its scalar optimizer.

    PrimusTurboGroupedLinear trains each expert linear as one 3-D ``weights`` tensor, so
    under Muon every routed expert would silently train with Adam (and ``dist_muon``
    rejects the resulting expert-parallel Adam groups).
    """
    optimizer = str(getattr(args, "optimizer", ""))
    if "muon" not in optimizer:
        return
    turbo_grouped_experts = (
        getattr(args, "enable_primus_turbo", False)
        and getattr(args, "use_turbo_grouped_gemm", False)
        and getattr(args, "moe_grouped_gemm", False)
        and _has_moe_layers(args)
        # Without primus_turbo, the spec provider patch is skipped and the experts are TE's.
        and _primus_turbo_installed()
    )
    if turbo_grouped_experts:
        advice = "Set use_turbo_grouped_gemm=false"
        if (getattr(args, "turbo_sync_free_moe_stage", 0) or 0) >= 2:
            advice += " and turbo_sync_free_moe_stage below 2 (stages 2 and 3 require Turbo grouped GEMM)"
        raise ValueError(
            f"optimizer={optimizer} is incompatible with use_turbo_grouped_gemm=True: Turbo's stacked "
            "3-D expert weights are not orthogonalized by Muon and would train with "
            f"{getattr(args, 'muon_scalar_optimizer', 'adam')} instead. {advice}."
        )


def validate_turbo_fp8_permute(args) -> None:
    """The FP8 permute handoff only has one consumer: the fused grouped MLP behind Turbo DeepEP."""
    option = "turbo_fp8_permute"
    if not getattr(args, option, False):
        return
    for required in ("enable_primus_turbo", "use_turbo_deepep", "turbo_fused_grouped_gemm"):
        if not getattr(args, required, False):
            raise ValueError(f"{option}=True requires {required}=True.")
    recompute_modules = getattr(args, "recompute_modules", None) or []
    if getattr(args, "recompute_granularity", None) == "selective" and "moe_act" in recompute_modules:
        raise ValueError(f"{option}=True does not support recompute of moe_act.")


def _bf16_grouped_gemm_backend(value) -> str:
    """The backend ``turbo_grouped_gemm_backend`` picks for BF16, in Primus-Turbo's format (``triton``, or
    ``bf16:triton,fp8:ck`` / ``other:...`` per precision; BF16, FP16 and FP32 share one entry), lower case;
    ``default`` when none applies."""
    value = str(value or "default").strip().lower()
    if ":" not in value:
        return value
    pairs = dict(p.split(":", 1) for p in (s.strip() for s in value.split(",")) if p)
    for key in ("bf16", "fp16", "fp32", "other"):
        if key in pairs:
            return pairs[key].strip()
    return "default"


def _num_moe_layers(args) -> int:
    freq = getattr(args, "moe_layer_freq", 1)
    num_layers = getattr(args, "num_layers", 0) or 0
    if isinstance(freq, int):
        n = sum(1 for i in range(num_layers) if i % freq == 0) if freq > 0 else 0
    else:
        n = sum(1 for x in freq if x)
    return n + (getattr(args, "mtp_num_layers", None) or 0)


def moe_replica_slot_bytes(args) -> int:
    """Memory the replica slots take on every rank: per MoE layer, ``moe_replica_slots`` copies of one expert's
    fc1 and fc2 weights (``params_dtype``) plus their gradients (``main_grads_dtype``)."""
    hidden = getattr(args, "hidden_size", 0) or 0
    ffn = getattr(args, "moe_ffn_hidden_size", None) or getattr(args, "ffn_hidden_size", 0) or 0
    fc1 = (2 if getattr(args, "swiglu", False) else 1) * ffn * hidden
    weight_bytes = 4 if str(getattr(args, "params_dtype", "")).endswith("float32") else 2
    grad_dtype = str(getattr(args, "main_grads_dtype", "fp32") or "fp32")
    grad_bytes = 2 if "bf16" in grad_dtype or "bfloat16" in grad_dtype else 4
    slots = getattr(args, "moe_replica_slots", 4)
    return _num_moe_layers(args) * slots * (fc1 + ffn * hidden) * (weight_bytes + grad_bytes)


def _check_moe_replica_memory(args, option: str) -> None:
    need = moe_replica_slot_bytes(args)
    if not need:
        return
    gib = need / 2**30
    total = (
        torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        if torch.cuda.is_available()
        else 0
    )
    msg = (
        f"{option}: the replica slots take {gib:.1f} GiB per GPU ({_num_moe_layers(args)} MoE layers x "
        f"{getattr(args, 'moe_replica_slots', 4)} slots x one expert's weights and gradients)"
    )
    if total and need > total:
        raise ValueError(
            f"{msg}, more than the {total / 2**30:.0f} GiB of this GPU. Lower moe_replica_slots."
        )
    warning_rank_0(msg)


def _cuda_graphs_enabled(args) -> bool:
    return bool(
        getattr(args, "enable_cuda_graph", False)
        or getattr(args, "external_cuda_graph", False)
        or getattr(args, "cuda_graph_impl", None) not in (None, "none")
    )


def validate_moe_replica(args) -> None:
    """Dynamic expert replication (``moe_replica_planner``): BF16 Turbo DeepEP and Turbo grouped GEMM, TP 1, PP 1.

    The replicas live in ``PrimusTurboDeepEPTokenDispatcher`` and ``PrimusTurboGroupedLinear``; a combination
    they do not handle would silently run without replicas or lose replica gradients, so it is rejected here.
    """
    planner = getattr(args, "moe_replica_planner", None)
    if not planner:
        return
    option = f"moe_replica_planner={planner}"
    if planner not in ("ultraep", "moonep"):
        raise ValueError(f"moe_replica_planner must be ultraep or moonep, got {planner!r}.")
    placement = getattr(args, "moe_replica_placement", "per_step")
    if placement not in ("per_step", "per_microbatch"):
        raise ValueError(f"moe_replica_placement must be per_step or per_microbatch, got {placement!r}.")
    slots = getattr(args, "moe_replica_slots", 4)
    min_rows = getattr(args, "moe_replica_min_rows", 1024)
    min_gain = getattr(args, "moe_replica_min_gain", 0.1)
    cost = getattr(args, "moe_replica_cost", 0)
    block = getattr(args, "moe_replica_block", 1)
    replan_above = getattr(args, "moe_replica_replan_above", 0.0)
    log_interval = getattr(args, "moe_replica_log_interval", 0)
    for name, value, ok, want in (
        ("moe_replica_slots", slots, slots >= 1, ">= 1"),
        ("moe_replica_min_rows", min_rows, min_rows >= 0, ">= 0"),
        ("moe_replica_min_gain", min_gain, 0 <= min_gain < 1, "in [0, 1)"),
        ("moe_replica_cost", cost, cost >= 0, ">= 0"),
        ("moe_replica_block", block, block >= 1, ">= 1"),
        ("moe_replica_replan_above", replan_above, replan_above == 0 or replan_above >= 1, "0 or >= 1"),
        ("moe_replica_log_interval", log_interval, log_interval >= 0, ">= 0"),
    ):
        if not ok:
            raise ValueError(f"{name} must be {want}, got {value}.")
    if replan_above > 0 and placement != "per_microbatch":
        raise ValueError("moe_replica_replan_above > 0 requires moe_replica_placement=per_microbatch.")

    for required in ("enable_primus_turbo", "use_turbo_deepep", "use_turbo_grouped_gemm", "moe_grouped_gemm"):
        if not getattr(args, required, False):
            raise ValueError(f"{option} requires {required}=True.")
    ep = getattr(args, "expert_model_parallel_size", 1)
    if ep <= 1:
        raise ValueError(f"{option} requires expert_model_parallel_size > 1.")
    num_experts = getattr(args, "num_experts", None) or 0
    if planner == "moonep" and slots < num_experts // ep:
        raise ValueError(
            f"moe_replica_planner=moonep requires moe_replica_slots >= num_experts / "
            f"expert_model_parallel_size = {num_experts // ep}, got {slots}."
        )
    backend = getattr(args, "turbo_grouped_gemm_backend", "default")
    if _bf16_grouped_gemm_backend(backend) not in ("default", "triton"):
        raise ValueError(
            f"{option} runs the experts and their replicas in one Triton grouped GEMM; "
            f"turbo_grouped_gemm_backend={backend} picks another BF16 backend, use default or triton for BF16."
        )

    tp = getattr(args, "tensor_model_parallel_size", 1)
    etp = getattr(args, "expert_tensor_parallel_size", None) or tp
    pp = getattr(args, "pipeline_model_parallel_size", 1)
    for name, value in (
        ("tensor_model_parallel_size", tp),
        ("expert_tensor_parallel_size", etp),
        ("pipeline_model_parallel_size", pp),
    ):
        if value != 1:
            raise ValueError(f"{option} requires {name}=1, got {value}.")
    unsupported = {
        "use_turbo_mega_moe": "MegaMoE replaces the DeepEP dispatcher",
        "turbo_fused_grouped_gemm": "replicas run on the BF16 grouped GEMM only",
        "fp8": "replicas run on the BF16 grouped GEMM only",
        "fp4": "replicas run on the BF16 grouped GEMM only",
        "overlap_moe_expert_parallel_comm": "the replica scheduler assumes one micro-batch in flight",
        "delay_wgrad_compute": "the replica gradients are returned with the weight gradients",
        "use_megatron_fsdp": "the replica gradients go into the owners' DDP main_grad",
        "use_torch_fsdp2": "the replica gradients go into the owners' DDP main_grad",
        "use_deprecated_20241209_moe_layer": "the replicas attach to Megatron's MoELayer",
    }
    for name, why in unsupported.items():
        if getattr(args, name, None):
            raise ValueError(f"{option} does not support {name} ({why}).")
    if getattr(args, "moe_expert_capacity_factor", None) is not None:
        raise ValueError(f"{option} does not support token dropping (moe_expert_capacity_factor).")
    if (
        getattr(args, "moe_router_force_load_balancing", False)
        and getattr(args, "moe_router_force_load_balancing_type", "uniform") == "even"
    ):
        raise ValueError(f"{option} does not support moe_router_force_load_balancing_type=even.")
    if _cuda_graphs_enabled(args):
        raise ValueError(
            f"{option} does not support CUDA graphs: planning reads the routed counts on the host."
        )
    if getattr(args, "overlap_grad_reduce", False):
        raise ValueError(
            f"{option} does not support overlap_grad_reduce=True: the expert gradient buckets are reduced during "
            "the backward, before the replica gradients reach the owners' main_grad. Set overlap_grad_reduce=False."
        )
    _check_moe_replica_memory(args, option)


def expandable_segments_requested() -> bool:
    """Return True if any allocator config env var sets ``expandable_segments:True``."""
    for env in ALLOC_CONF_ENVS:
        for item in os.environ.get(env, "").split(","):
            key, _, value = item.partition(":")
            if key.strip() == "expandable_segments" and value.strip() == "True":
                return True
    return False


def warn_turbo_grouped_gemm_without_expandable_segments(args) -> None:
    """Recommend expandable segments when PrimusTurbo grouped GEMM is enabled.

    PrimusTurboGroupedLinear stores all experts of a layer in a single weight, so the
    optimizer's fp32 grad/state buffers become few, very large blocks. Without
    expandable segments the caching allocator cannot fit them into the fragmented
    free pool and keeps reserving new segments, inflating peak reserved memory.
    """
    if not getattr(args, "use_turbo_grouped_gemm", False) or expandable_segments_requested():
        return
    warning_rank_0(
        "use_turbo_grouped_gemm=True without expandable segments: the per-layer stacked expert "
        "weights produce large optimizer buffers that fragment the caching allocator and increase "
        "peak reserved memory. Set PYTORCH_HIP_ALLOC_CONF=expandable_segments:True to reduce "
        "memory fragmentation."
    )


def validate_args_on_rocm(args):
    # Primus-Turbo auto-tuning
    use_turbo_autotune = getattr(args, "use_turbo_autotune", False)
    if use_turbo_autotune:
        # NOTE: Set PRIMUS_TURBO_AUTO_TUNE to 1 to enable turbo auto-tuning.
        os.environ["PRIMUS_TURBO_AUTO_TUNE"] = "1"

    # Deterministic mode
    if args.deterministic_mode:
        # NOTE: Some environment variables affect deterministic mode on ROCm. Need to do extra check.
        NON_DETERMINISTIC_ENVS = {
            "TORCH_COMPILE_DISABLE": "1",
            "ROCBLAS_DEFAULT_ATOMICS_MODE": "0",
            "PRIMUS_TURBO_AUTO_TUNE": "0",
            "PRIMUS_DETERMINISTIC": "1",
        }
        # NOTE: Some version triton compile exist potential racing condition issue.
        for env, value in NON_DETERMINISTIC_ENVS.items():
            assert (
                os.environ.get(env, None) == value
            ), f"{env} must be set to {value} in deterministic mode but got {os.environ.get(env, None)} instead."

        # Set fill_uninitialized_memory to False to avoid calling extra fill kernel in deterministic mode.
        torch.utils.deterministic.fill_uninitialized_memory = False

    assert not getattr(
        args, "use_turbo_parallel_linear", False
    ), "use_turbo_parallel_linear has been removed; please use use_turbo_gemm instead."

    validate_fsdp2_optimizer_exclusivity(args)
    validate_cross_entropy_exclusivity(args)

    use_turbo_gemm = getattr(args, "use_turbo_gemm", False)
    # Turbo FP8 linear check
    if args.fp8 and use_turbo_gemm:
        support_fp8_recipe = ["tensorwise", "blockwise", "mxfp8"]
        assert (
            args.fp8_recipe in support_fp8_recipe
        ), f"{args.fp8_recipe} recipe is not support when enable `use_turbo_gemm`."

    # Turbo FP4 linear check
    if args.fp4 and use_turbo_gemm:
        support_fp4_recipe = ["mxfp4"]
        assert (
            args.fp4_recipe in support_fp4_recipe
        ), f"{args.fp4_recipe} recipe is not support when enable `use_turbo_gemm`."

    # Turbo FP4 autocast check: PrimusTurboLinear quantizes only under the Turbo
    # autocast, which the TE-native path never enters. Local specs quantize
    # in-module rather than off that state, so they are exempt.
    if (
        args.fp4
        and use_turbo_gemm
        and getattr(args, "fp4_use_native_te_autocast", False)
        and getattr(args, "transformer_impl", "transformer_engine") != "local"
    ):
        raise ValueError(
            "fp4_use_native_te_autocast=True is incompatible with use_turbo_gemm=True. "
            "Native TE FP4 autocast does not set the Primus-Turbo FP4 state that "
            "PrimusTurboLinear reads, so Turbo GEMM linears would silently run BF16 "
            "instead of FP4. Set use_turbo_gemm=false to use native TE MXFP4 linears, "
            "or set fp4_use_native_te_autocast=false to use the Primus-Turbo FP4 path."
        )

    # NOTE: mxfp8 environment variable must be set to 1 to enable mxfp8 recipe on ROCm.
    if args.fp8_recipe == "mxfp8":
        assert (
            os.getenv("NVTE_ROCM_ENABLE_MXFP8", "0") == "1"
        ), "Please set `NVTE_ROCM_ENABLE_MXFP8=1` to enable `mxfp8` recipe."

    # dump pp data
    if args.dump_pp_data and args.pipeline_model_parallel_size == 1:
        args.dump_pp_data = False
        print_rank_last(f"Disable args.dump_pp_data since args.pipeline_model_parallel_size=1")

    # PrimusTurboGroupedMLP no longer depends on legacy GroupedMLP; the two
    # flags are mutually exclusive when turbo is enabled.
    assert not getattr(
        args, "use_turbo_grouped_mlp", False
    ), "use_turbo_grouped_mlp has been removed; please use use_turbo_grouped_gemm instead."
    use_turbo_grouped_gemm = getattr(args, "use_turbo_grouped_gemm", False)
    if use_turbo_grouped_gemm:
        if getattr(args, "moe_use_legacy_grouped_gemm", False):
            raise ValueError(
                "use_turbo_grouped_gemm=True is incompatible with moe_use_legacy_grouped_gemm=True. "
                "please set moe_use_legacy_grouped_gemm=False."
            )

    # sync-free MoE
    if args.turbo_sync_free_moe_stage > 0:
        assert args.enable_primus_turbo, "Please set `enable_primus_turbo=True` to enable sync-free MoE."

        if args.turbo_sync_free_moe_stage > 1 and not use_turbo_grouped_gemm:
            raise ValueError(
                "Sync-Free MoE stage 2 or 3 require PrimusTurboGroupedLinear, please set `use_turbo_grouped_gemm=True`"
            )
        options = _get_sync_free_moe_options(args)
        print_rank_last(
            f"========== Enable Sync-Free MoE Stage {args.turbo_sync_free_moe_stage} (Auto-Enabled Options) =========="
        )
        for flag, value in options.items():
            dots = "." * (73 - len(flag) - len(str(value)))
            print_rank_last(f"{flag}{dots}{value}")
            setattr(args, flag, value)
        print_rank_last(
            f"========== Enable Sync-Free MoE Stage {args.turbo_sync_free_moe_stage} (Auto-Enabled Options) =========="
        )

    validate_turbo_grouped_gemm_without_padding(args)
    validate_turbo_grouped_gemm_with_muon(args)
    validate_turbo_fp8_permute(args)
    # After sync-free MoE, which may auto-enable use_turbo_grouped_gemm.
    warn_turbo_grouped_gemm_without_expandable_segments(args)
    validate_moe_replica(args)

    # turbo deepep
    if args.use_turbo_deepep:
        assert (
            not args.moe_shared_expert_overlap
        ), "DeepEP not support moe_shared_expert_overlap, please set `moe_shared_expert_overlap=False`."
        assert (
            args.moe_router_dtype == "fp32"
        ), "DeepEP only supports float32 probs, please set `moe_router_dtype=fp32`"
        if (
            args.expert_model_parallel_size >= 16
            and os.getenv("PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND", "TURBO") == "TURBO"
        ):
            # Turbo DeepEP is not supported for CUs > 32 when using internode dispatch/combine.
            assert args.turbo_deepep_num_cu <= 32, "Set `turbo_deepep_num_cu<=32` when using ep_size >= 16."
