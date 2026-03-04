###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import json
import os
from datetime import datetime

import torch.distributed as dist

from primus.moe_core.autotune import grid_search
from primus.moe_core.config import DispatchRuntimeConfig, RouterRuntimeConfig


def build_router_runtime_config(config) -> RouterRuntimeConfig:
    return RouterRuntimeConfig(
        num_experts=int(config.num_moe_experts),
        router_topk=int(config.moe_router_topk),
        router_num_groups=(
            int(config.moe_router_num_groups)
            if getattr(config, "moe_router_num_groups", None) is not None
            else None
        ),
        router_group_topk=(
            int(config.moe_router_group_topk)
            if getattr(config, "moe_router_group_topk", None) is not None
            else None
        ),
        router_score_function=str(config.moe_router_score_function),
        router_topk_scaling_factor=float(config.moe_router_topk_scaling_factor),
    )


def build_dispatch_runtime_config(
    args, config, *, tp_size: int, tp_ep_group_size: int
) -> DispatchRuntimeConfig:
    default_stage = int(getattr(args, "turbo_sync_free_moe_stage", 0))
    effective_stage = default_stage
    use_comm_stream = bool(getattr(args, "turbo_deepep_use_comm_stream", False))
    num_cu = int(getattr(args, "turbo_deepep_num_cu", 32))

    use_autotune = str(os.environ.get("PRIMUS_MOE_AUTOTUNE", "0")).lower() in ("1", "true", "yes", "on")
    if use_autotune:
        tuned = _autotune_deepep_runtime(
            args=args,
            ep_size=int(getattr(args, "expert_model_parallel_size", 1)),
            default_num_cu=num_cu,
            default_use_comm_stream=use_comm_stream,
            default_stage=default_stage,
        )
        num_cu = int(tuned["num_cu"])
        effective_stage = int(tuned["stage"])
        use_comm_stream = bool(tuned["use_comm_stream"])
        setattr(args, "turbo_sync_free_moe_stage", effective_stage)
        setattr(args, "turbo_deepep_num_cu", num_cu)
        setattr(args, "turbo_deepep_use_comm_stream", use_comm_stream)
        _maybe_dump_tuned_config(tuned)

    # Enable sync-free moe to remove CPU busy-wait in deepep.
    num_worst_tokens = 0
    permute_max_token_num = 0
    if effective_stage > 1:
        if bool(getattr(args, "sequence_parallel", False)):
            seq_length = int(getattr(args, "seq_length", 0)) // max(tp_size, 1)
        else:
            seq_length = int(getattr(args, "seq_length", 0))
        num_tokens = (
            seq_length
            // max(int(getattr(args, "context_parallel_size", 1)), 1)
            * int(getattr(args, "micro_batch_size", 1))
        )
        num_worst_tokens = num_tokens * max(tp_ep_group_size, 1)
        if effective_stage > 2:
            permute_max_token_num = num_worst_tokens * int(config.moe_router_topk)

    use_cuda_num_tokens_per_expert = bool(
        getattr(args, "use_turbo_grouped_mlp", False) and getattr(args, "moe_use_legacy_grouped_gemm", False)
    )
    return DispatchRuntimeConfig(
        num_experts=int(config.num_moe_experts),
        router_topk=int(config.moe_router_topk),
        expert_capacity_factor=config.moe_expert_capacity_factor,
        permute_fusion=bool(config.moe_permute_fusion),
        permute_max_token_num=int(permute_max_token_num),
        num_worst_tokens=int(num_worst_tokens),
        use_comm_stream=use_comm_stream,
        num_cu=num_cu,
        use_cuda_num_tokens_per_expert=use_cuda_num_tokens_per_expert,
        async_finish=True,
        allocate_on_comm_stream=True,
    )


def _parse_candidates_from_env(name: str, default_values: list[int]) -> list[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default_values
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(int(tok))
        except ValueError:
            continue
    return vals or default_values


def _autotune_deepep_runtime(
    *,
    args,
    ep_size: int,
    default_num_cu: int,
    default_use_comm_stream: bool,
    default_stage: int,
) -> dict[str, int | bool | float]:
    cu_candidates = _parse_candidates_from_env("PRIMUS_MOE_AUTOTUNE_CU_CANDIDATES", [32, 64, 80])
    stage_candidates = _parse_candidates_from_env("PRIMUS_MOE_AUTOTUNE_STAGE_CANDIDATES", [1, 2, 3])
    stream_candidates = [False, True]

    candidates = []
    for cu in cu_candidates:
        for stage in stage_candidates:
            for use_stream in stream_candidates:
                candidates.append(
                    {
                        "num_cu": int(cu),
                        "stage": int(stage),
                        "use_comm_stream": bool(use_stream),
                        "ep_size": ep_size,
                    }
                )

    def evaluate(cfg: dict[str, int | bool]) -> float:
        # Lower is better; this is a heuristic for startup auto-tuning.
        score = 0.0
        num_cu = int(cfg["num_cu"])
        stage = int(cfg["stage"])
        use_stream = bool(cfg["use_comm_stream"])

        if ep_size <= 8:
            # Existing guidance in docs: 64/80 for EP=8.
            score += abs(num_cu - 64) * 1.0
        elif ep_size <= 64:
            # Existing guidance in docs: 32 for EP=16~64.
            score += abs(num_cu - 32) * 1.0
        else:
            score += abs(num_cu - 32) * 0.5

        # Stage-2 is current default recommendation.
        score += abs(stage - 2) * 5.0

        # Comm stream more likely helpful for larger EP.
        if ep_size >= 16:
            score += 0.0 if use_stream else 1.0
        else:
            score += 0.5 if use_stream else 0.0

        # Keep near existing defaults to avoid aggressive drift.
        score += abs(num_cu - default_num_cu) * 0.05
        score += abs(stage - default_stage) * 0.2
        if use_stream != default_use_comm_stream:
            score += 0.1
        return score

    tuned = grid_search(candidates, evaluate)
    cfg = tuned.config
    return {
        "num_cu": int(cfg["num_cu"]),
        "stage": int(cfg["stage"]),
        "use_comm_stream": bool(cfg["use_comm_stream"]),
        "score": float(tuned.score),
    }


def _maybe_dump_tuned_config(tuned_cfg: dict[str, int | bool | float]) -> None:
    dump_path = os.environ.get("PRIMUS_MOE_AUTOTUNE_DUMP", "").strip()
    if not dump_path:
        return
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    dump_dir = os.path.dirname(dump_path)
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
    payload = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "deepep": tuned_cfg}
    with open(dump_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
