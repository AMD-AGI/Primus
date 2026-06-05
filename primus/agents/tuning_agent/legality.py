"""Per-axis legal-value derivation + candidate validation.

This is the *deterministic* knowledge base of the agent. The LLM is told
the legal axis values and the validation rules; configs that violate them
are rejected before reaching the projection tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import OptimizationConfig, TargetCluster
from .workload import ArchitectureRecord


# ---------------------------------------------------------------------------
# Trial config record (what the LLM proposes / the evaluator consumes)
# ---------------------------------------------------------------------------

@dataclass
class TrialConfig:
    """All knobs the agent sweeps.

    Categorisation follows the Primus Projection skill
    (``Primus/.skills/primus-projection.md``):

    * **Parallelism**: ``tp``, ``pp``, ``ep``, ``cp``, ``vpp``
    * **Pipeline schedule**: ``pp_schedule``, ``enable_zero_bubble``
    * **MoE comm (Tier-A)**: ``use_turbo_deepep``, ``sync_free_stage``,
      ``target_ep_size``
    * **Precision (Tier-A)**: ``fp8`` (None or "hybrid")
    * **Memory (Tier-B/C)**: ``recompute_granularity``,
      ``recompute_num_layers``, ``cross_entropy_loss_fusion``,
      ``use_torch_fsdp2``, ``use_distributed_optimizer``
    * **Batch**: ``mbs``, ``gbs``, ``overlap_grad_reduce``

    Fields with default ``None`` mean "inherit from workload yaml" — the
    evaluator only writes overrides for fields that are explicitly set.
    """
    # parallelism
    tp: int = 1
    pp: int = 1
    ep: int = 1
    cp: int = 1
    mbs: int = 1
    gbs: int = 1
    vpp: int | None = None
    # pipeline schedule
    pp_schedule: str = "auto"
    enable_zero_bubble: bool | None = None
    # memory levers
    recompute_granularity: str | None = None     # None / "selective" / "full"
    recompute_num_layers: int = 0
    cross_entropy_loss_fusion: bool | None = None
    use_torch_fsdp2: bool | None = None
    use_distributed_optimizer: bool | None = None
    # MoE comm (Tier-A; high impact for MoE)
    use_turbo_deepep: bool | None = None
    sync_free_stage: int | None = None        # 0=off, 1, 2, 3
    target_ep_size: int | None = None
    # precision (Tier-A; high impact)
    fp8: str | None = None                    # None / "hybrid"
    # batching
    overlap_grad_reduce: bool = True

    def as_dict(self) -> dict:
        return {
            "tp": self.tp, "pp": self.pp, "ep": self.ep, "cp": self.cp,
            "mbs": self.mbs, "gbs": self.gbs, "vpp": self.vpp,
            "pp_schedule": self.pp_schedule,
            "enable_zero_bubble": self.enable_zero_bubble,
            "recompute_granularity": self.recompute_granularity,
            "recompute_num_layers": self.recompute_num_layers,
            "cross_entropy_loss_fusion": self.cross_entropy_loss_fusion,
            "use_torch_fsdp2": self.use_torch_fsdp2,
            "use_distributed_optimizer": self.use_distributed_optimizer,
            "use_turbo_deepep": self.use_turbo_deepep,
            "sync_free_stage": self.sync_free_stage,
            "target_ep_size": self.target_ep_size,
            "fp8": self.fp8,
            "overlap_grad_reduce": self.overlap_grad_reduce,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrialConfig":
        def _opt_int(v):
            return int(v) if v not in (None, "null", "") else None

        def _opt_bool(v):
            if v in (None, "null", ""):
                return None
            if isinstance(v, str):
                return v.lower() in ("true", "1", "yes", "on")
            return bool(v)

        def _opt_str(v):
            if v in (None, "null", ""):
                return None
            return str(v)

        return cls(
            tp=int(d.get("tp", 1)),
            pp=int(d.get("pp", 1)),
            ep=int(d.get("ep", 1)),
            cp=int(d.get("cp", 1)),
            mbs=int(d.get("mbs", 1)),
            gbs=int(d.get("gbs", 1)),
            vpp=_opt_int(d.get("vpp")),
            pp_schedule=str(d.get("pp_schedule", "auto")),
            enable_zero_bubble=_opt_bool(d.get("enable_zero_bubble")),
            recompute_granularity=d.get("recompute_granularity"),
            recompute_num_layers=int(d.get("recompute_num_layers", 0)),
            cross_entropy_loss_fusion=_opt_bool(d.get("cross_entropy_loss_fusion")),
            use_torch_fsdp2=_opt_bool(d.get("use_torch_fsdp2")),
            use_distributed_optimizer=_opt_bool(d.get("use_distributed_optimizer")),
            use_turbo_deepep=_opt_bool(d.get("use_turbo_deepep")),
            sync_free_stage=_opt_int(d.get("sync_free_stage")),
            target_ep_size=_opt_int(d.get("target_ep_size")),
            fp8=_opt_str(d.get("fp8")),
            overlap_grad_reduce=bool(d.get("overlap_grad_reduce", True)),
        )

    def signature(self) -> str:
        d = self.as_dict()
        return ",".join(f"{k}={d[k]}" for k in sorted(d))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _divisors(n: int, max_val: int | None = None) -> list[int]:
    if n <= 0:
        return [1]
    out = [d for d in range(1, n + 1) if n % d == 0]
    if max_val is not None:
        out = [d for d in out if d <= max_val]
    return out


def _powers_of_two(max_val: int) -> list[int]:
    out = [1]
    while out[-1] * 2 <= max_val:
        out.append(out[-1] * 2)
    return out


# ---------------------------------------------------------------------------
# Per-axis legal sets
# ---------------------------------------------------------------------------

@dataclass
class AxisLegality:
    tp: list[int]
    pp: list[int]
    ep: list[int]
    cp: list[int]
    mbs: list[int]
    vpp: list[int]
    pp_schedules_by_vpp: dict[int, list[str]]
    recompute_granularity: list[str]

    def to_prompt_dict(self) -> dict:
        return {
            "tp": self.tp, "pp": self.pp, "ep": self.ep, "cp": self.cp,
            "mbs": self.mbs, "vpp": self.vpp,
            "pp_schedules_by_vpp": {str(k): v for k, v in self.pp_schedules_by_vpp.items()},
            "recompute_granularity": self.recompute_granularity,
            # Tier-A/B optimisation axes from the Primus Projection skill.
            # Boolean flags accept true/false/null (null = inherit yaml).
            "use_turbo_deepep": [None, False, True],          # MoE only; +35% per skill
            "sync_free_stage": [None, 0, 1, 2, 3],            # MoE only; 2/3 auto-enables DeepEP
            "fp8": [None, "hybrid"],                          # ~2x compute on linear layers
            "enable_zero_bubble": [None, False, True],        # pairs with VPP=1 zerobubble schedules
            "cross_entropy_loss_fusion": [None, True, False], # large-vocab memory + compute win
            "use_distributed_optimizer": [None, True, False], # ZeRO-1 optimizer sharding
            "use_torch_fsdp2": [None, True, False],           # full FSDP2; mutually exclusive with distributed_optimizer
            "target_ep_size": [None],                         # int override; agent fills as needed
        }


def derive_legality(arch: ArchitectureRecord, cluster: TargetCluster) -> AxisLegality:
    """Compute the (per-axis) legal value sets for this (model, cluster) pair."""
    world = cluster.num_nodes * cluster.gpus_per_node
    gpn = cluster.gpus_per_node

    # TP: divides num_attention_heads AND hidden_size, ≤ gpus_per_node
    tp_candidates = sorted(set(_divisors(arch.num_attention_heads, gpn))
                           & set(_divisors(arch.hidden_size, gpn)))
    if not tp_candidates:
        tp_candidates = [1]

    # PP: divides num_layers, ≤ world.
    # NOTE: workloads like DeepSeek V3 (61 layers, prime) ship a non-divisor
    # PP=16 backed by ``pipeline_model_parallel_layout`` (per-stage layer
    # counts). We always include the workload's own PP so the published
    # baseline is reachable; for layout-aware workloads we *also* allow the
    # standard non-divisor PP depths (2, 4, 8) so the agent can reach
    # configurations like the published DSv3 ``PP=8 / VPP=2`` reference,
    # whose 16 stages each carry 3-4 layers via an uneven layout. The
    # evaluator regenerates a per-stage layout for the new (PP, VPP) so this
    # is always representable.
    pp = _divisors(arch.num_layers, world) if arch.num_layers else [1]
    if arch.pipeline_model_parallel_size and arch.pipeline_model_parallel_size <= world:
        if arch.pipeline_model_parallel_size not in pp:
            pp = sorted(set(pp) | {arch.pipeline_model_parallel_size})
    layout_aware = bool(getattr(arch, "pipeline_model_parallel_layout", None))
    if layout_aware and arch.num_layers:
        for cand in (2, 4, 8, 16):
            if cand <= world and cand <= arch.num_layers:
                pp = sorted(set(pp) | {cand})

    # EP: only meaningful for MoE; divides num_experts; ≤ world
    if arch.is_moe:
        ep = _divisors(arch.num_experts, world) or [1]
    else:
        ep = [1]

    # CP: dense → free divisor of seq_length up to gpus_per_node
    #     MoE  → CP ≤ EP and EP % CP == 0 (parallel folding)
    if arch.is_moe:
        cp = [c for c in ep if c <= max(ep)]
    else:
        cp = _divisors(arch.seq_length, gpn) if arch.seq_length else [1]
    if not cp:
        cp = [1]

    # MBS: small powers of two; we cap at 16
    mbs = _powers_of_two(16)

    # VPP: a candidate is legal for *some* PP if it divides num_layers/PP.
    # The flat union here is for prompt/seed exploration; the per-trial check
    # in validate() enforces that the chosen (PP, VPP) pair is coherent.
    # NOTE: using ``max(pp)`` here is wrong for models like Mixtral 8x22B
    # (PP=56 ⇒ layers/stage=1 ⇒ only VPP=1 ever proposed, masking the
    # published PP=4/VPP=2 reference config).
    vpp_set: set[int] = {1}
    if arch.num_layers:
        for p in pp:
            stage_layers = arch.num_layers // p
            if stage_layers <= 0:
                continue
            for d in _divisors(stage_layers):
                if d <= 8:
                    vpp_set.add(d)
    # Workload may ship a non-divisor VPP backed by ``pipeline_model_parallel_layout``
    # (e.g. DeepSeek V3 PP=16, VPP=2 with 61 layers). Always include it.
    if arch.virtual_pipeline_model_parallel_size:
        vpp_set.add(int(arch.virtual_pipeline_model_parallel_size))
    # For layout-aware workloads, also expose VPP ∈ {2, 3, 4} as long as
    # PP×VPP ≤ num_layers — the evaluator regenerates a layout per trial.
    if layout_aware and arch.num_layers:
        for v in (2, 3, 4):
            for p in pp:
                if p * v <= arch.num_layers:
                    vpp_set.add(v)
                    break
    vpp = sorted(vpp_set)

    # `"auto"` means "do not pass --pipeline-schedule-algorithm and let
    # Megatron pick its default" (1F1B for VPP=1, 1F1B-interleaved for VPP>1).
    # That's always legal, so it must appear in every per-VPP set — otherwise
    # the seed planner can't reach VPP=2 / VPP=4 with a default schedule.
    pp_schedules_by_vpp = {
        1: ["auto", "zerobubble", "zerobubble-heuristic", "seaailab-ilp"],
        2: ["auto", "zbv-formatted", "zbv-greedy-half", "zbv-greedy-min"],
    }
    for v in vpp:
        if v not in pp_schedules_by_vpp:
            pp_schedules_by_vpp[v] = ["auto"]

    recompute_granularity = ["none", "selective", "full"]

    return AxisLegality(
        tp=tp_candidates, pp=pp, ep=ep, cp=cp, mbs=mbs, vpp=vpp,
        pp_schedules_by_vpp=pp_schedules_by_vpp,
        recompute_granularity=recompute_granularity,
    )


# ---------------------------------------------------------------------------
# Candidate validation
# ---------------------------------------------------------------------------

def derived_dp(cfg: TrialConfig, arch: ArchitectureRecord, cluster: TargetCluster) -> int:
    world = cluster.num_nodes * cluster.gpus_per_node
    if arch.is_moe:
        denom = cfg.tp * cfg.pp * cfg.ep
    else:
        denom = cfg.tp * cfg.pp * cfg.cp
    return world // denom if denom > 0 else 0


def validate(cfg: TrialConfig, arch: ArchitectureRecord, cluster: TargetCluster,
             legality: AxisLegality) -> tuple[bool, str]:
    """Check a trial config against (model, cluster) legality. Returns
    (ok, reason). reason is empty when ok is True."""
    world = cluster.num_nodes * cluster.gpus_per_node

    if cfg.tp not in legality.tp:
        return False, f"TP={cfg.tp} not in legal set {legality.tp}"
    if cfg.pp not in legality.pp:
        return False, f"PP={cfg.pp} not in legal set {legality.pp}"
    if cfg.ep not in legality.ep:
        return False, f"EP={cfg.ep} not in legal set {legality.ep}"
    if cfg.cp not in legality.cp:
        return False, f"CP={cfg.cp} not in legal set {legality.cp}"
    if cfg.vpp is not None and cfg.vpp not in legality.vpp:
        return False, f"VPP={cfg.vpp} not in legal set {legality.vpp}"

    # Per-trial coherence: VPP must divide num_layers / PP — but only when
    # the workload uses *even* PP layer distribution. Workloads like
    # DeepSeek V3 ship a per-stage ``pipeline_model_parallel_layout`` so PP
    # and VPP can be arbitrary divisors of the layout's stage count. When
    #   (a) the trial's (PP, VPP) match the workload's, OR
    #   (b) the workload ships a layout (so the evaluator will regenerate
    #       one for the trial's PP, VPP),
    # we accept any (PP, VPP) whose product is ≤ num_layers (so each stage
    # can host at least one layer).
    if cfg.vpp is not None and cfg.vpp != 1 and arch.num_layers:
        keeps_layout = (
            cfg.pp == arch.pipeline_model_parallel_size
            and cfg.vpp == arch.virtual_pipeline_model_parallel_size
        )
        layout_aware = bool(getattr(arch, "pipeline_model_parallel_layout", None))
        if keeps_layout or layout_aware:
            n_stages = (cfg.pp or 1) * cfg.vpp
            if n_stages > arch.num_layers:
                return False, (
                    f"PP×VPP = {cfg.pp}×{cfg.vpp} = {n_stages} > num_layers="
                    f"{arch.num_layers}; cannot give every stage at least one layer"
                )
        else:
            stage_layers = arch.num_layers // cfg.pp if cfg.pp else 0
            if stage_layers <= 0 or stage_layers % cfg.vpp != 0:
                return False, (
                    f"VPP={cfg.vpp} does not divide num_layers/PP "
                    f"= {arch.num_layers}/{cfg.pp} = {stage_layers}"
                )

    if arch.is_moe:
        if cfg.ep % cfg.cp != 0:
            return False, f"MoE parallel folding requires EP({cfg.ep}) % CP({cfg.cp}) == 0"
        if cfg.cp > cfg.ep:
            return False, f"MoE parallel folding requires CP({cfg.cp}) ≤ EP({cfg.ep})"
        gpus_per_replica = cfg.tp * cfg.pp * cfg.ep
    else:
        gpus_per_replica = cfg.tp * cfg.pp * cfg.cp

    if gpus_per_replica > world:
        return False, (
            f"requires {gpus_per_replica} GPUs per replica but only {world} available "
            f"({cluster.num_nodes} nodes × {cluster.gpus_per_node} gpus)"
        )

    dp = world // gpus_per_replica
    if dp <= 0:
        return False, f"derived DP={dp} (gpus_per_replica={gpus_per_replica})"

    if cfg.gbs % (cfg.mbs * dp) != 0:
        return False, (
            f"GBS({cfg.gbs}) must be divisible by MBS({cfg.mbs}) × DP({dp}) = {cfg.mbs*dp}"
        )

    # schedule × vpp coherence
    vpp_for_lookup = cfg.vpp or 1
    sched_set = legality.pp_schedules_by_vpp.get(vpp_for_lookup, ["auto"])
    if cfg.pp_schedule not in sched_set:
        return False, (
            f"schedule '{cfg.pp_schedule}' not legal for VPP={vpp_for_lookup}; "
            f"legal: {sched_set}"
        )

    if cfg.recompute_granularity not in (None, "none", "selective", "full"):
        return False, f"unknown recompute_granularity '{cfg.recompute_granularity}'"

    from primus.core.projection.config_validation import check_recompute_pipeline_compat

    rec_ok, rec_reason = check_recompute_pipeline_compat(
        recompute_granularity=cfg.recompute_granularity,
        recompute_num_layers=cfg.recompute_num_layers or 0,
        pipeline_schedule=cfg.pp_schedule,
        enable_zero_bubble=cfg.enable_zero_bubble,
    )
    if not rec_ok:
        return False, rec_reason

    # New axes from the Primus Projection skill ----------------------------
    if cfg.sync_free_stage is not None and cfg.sync_free_stage not in (0, 1, 2, 3):
        return False, f"sync_free_stage must be in {{0,1,2,3}}, got {cfg.sync_free_stage}"

    if cfg.fp8 is not None and cfg.fp8 not in ("hybrid", "e4m3", "delayed"):
        return False, f"fp8 must be None or one of (hybrid, e4m3, delayed); got {cfg.fp8!r}"

    # MoE-only optimisations: DeepEP / SyncFree only make sense for MoE workloads.
    if not arch.is_moe:
        if cfg.use_turbo_deepep:
            return False, "use_turbo_deepep is only meaningful for MoE workloads"
        if cfg.sync_free_stage and cfg.sync_free_stage > 0:
            return False, "sync_free_stage is only meaningful for MoE workloads"

    # SyncFree stages 2/3 implicitly enable DeepEP per the projection CLI;
    # explicitly setting use_turbo_deepep=False with stage>=2 is contradictory.
    if (cfg.sync_free_stage and cfg.sync_free_stage >= 2
            and cfg.use_turbo_deepep is False):
        return False, (
            f"sync_free_stage={cfg.sync_free_stage} auto-enables DeepEP; "
            f"setting use_turbo_deepep=False is contradictory"
        )

    # FSDP2 and distributed_optimizer are alternative DP-sharding strategies;
    # exposing both at once is a config-level mistake — pick one.
    if cfg.use_torch_fsdp2 and cfg.use_distributed_optimizer:
        return False, (
            "use_torch_fsdp2 and use_distributed_optimizer are mutually "
            "exclusive (FSDP2 already shards the optimizer state)"
        )

    if cfg.target_ep_size is not None:
        if cfg.target_ep_size <= 0:
            return False, f"target_ep_size must be positive, got {cfg.target_ep_size}"
        if not arch.is_moe:
            return False, "target_ep_size is only meaningful for MoE workloads"

    return True, ""


def fill_defaults_from_baseline(cfg: TrialConfig, arch: ArchitectureRecord) -> TrialConfig:
    """If the LLM omitted GBS/MBS, fall back to the workload baseline."""
    if cfg.gbs <= 1:
        cfg.gbs = arch.global_batch_size
    if cfg.mbs <= 0:
        cfg.mbs = arch.micro_batch_size
    return cfg


def axis_value_set(name: str, legality: AxisLegality) -> Iterable:
    return getattr(legality, name)
