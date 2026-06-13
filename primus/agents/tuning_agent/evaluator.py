"""Evaluator — runs `projection memory` + `projection performance --profiling-mode
{simulate,benchmark}` against a per-trial workload YAML and parses the stdout
for the metrics the agent needs.

The interface is uniform regardless of mode:

    EvalResult = {
      legal: bool, reason: str|None,
      memory_per_gpu_gb: float|None,
      param_optimizer_gb: float|None,
      activation_gb: float|None,
      iteration_ms: float|None,
      tokens_per_s_per_gpu: float|None,
      source: "memory_only"|"simulate"|"benchmark",
      stdout_tail: str, returncode: int,
      duration_s: float,
      cmd: list[str],
    }
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import AgentConfig
from .legality import TrialConfig, derived_dp, validate
from .workload import ArchitectureRecord

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    legal: bool = True
    reason: str | None = None
    memory_per_gpu_gb: float | None = None
    memory_per_gpu_gb_adjusted: float | None = None
    memory_per_gpu_gb_upper: float | None = None  # bench-mode 5%-margin upper bound
    memory_source: str | None = None  # "benchmark_point" | "simulate" | …
    memory_warning: str | None = None
    param_optimizer_gb: float | None = None
    activation_gb: float | None = None
    iteration_ms: float | None = None
    tokens_per_s_per_gpu: float | None = None
    tflops_per_s_per_gpu: float | None = None
    mfu: float | None = None
    source: str = "memory_only"
    stdout_tail: str = ""
    returncode: int = 0
    duration_s: float = 0.0
    cmd: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    derived_dp: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-trial YAML generation
# ---------------------------------------------------------------------------


def _build_pipeline_layout(num_layers: int, pp: int, vpp: int) -> str | None:
    """Build a Primus ``pipeline_model_parallel_layout`` string for the
    given (pp, vpp) split of ``num_layers``.

    Output format matches the published DeepSeek V3 reference:
    ``Et*N0|t*N1|...|t*Nlast,L`` where:
      * The first stage carries the embedding (``E`` prefix).
      * The last stage carries the output / loss head (``,L`` suffix).
      * ``N0..Nlast`` sums to ``num_layers`` and lengths are as balanced as
        possible, with extras allocated to *middle* stages so the first and
        last stages (which already host embedding + output/loss) stay light.

    Returns None if PP*VPP > num_layers (would force empty stages).
    """
    if num_layers <= 0 or pp <= 0 or vpp <= 0:
        return None
    n_stages = pp * vpp
    if n_stages > num_layers:
        return None

    base = num_layers // n_stages
    extra = num_layers - base * n_stages
    counts = [base] * n_stages

    # Distribute extras to *middle* stages first (skipping stage 0 and the
    # last stage), wrapping around if there's more extra than middles. This
    # mirrors the load-balancing pattern the published DSv3 reference uses
    # (Et*3 | t*4 × 13 | t*3 | t*3,L for 61 layers / 16 stages).
    if n_stages >= 3:
        middles = list(range(1, n_stages - 1))
    else:
        middles = list(range(n_stages))
    for i in range(extra):
        counts[middles[i % len(middles)]] += 1

    parts: list[str] = []
    for i, c in enumerate(counts):
        prefix = "E" if i == 0 else ""
        parts.append(f"{prefix}t*{c}")
    return "|".join(parts) + ",L"


def write_trial_yaml(arch: ArchitectureRecord, cfg: TrialConfig, out_dir: Path, tag: str) -> Path:
    """Write a copy of the workload YAML with cfg's overrides applied.

    Knobs are applied in roughly the priority order from the Projection
    skill's Optimization Exploration Guidelines (Step 1 → Step 6):
      1. Parallelism (TP/PP/EP/CP/VPP)
      2. Pipeline schedule (zero-bubble + algorithm)
      3. MoE comm (DeepEP, SyncFree)
      4. Precision (FP8)
      5. Memory (recompute, loss-fusion, FSDP2/distributed-optimizer)
      6. Batch (MBS, GBS, overlap-grad-reduce)

    We deliberately do NOT mutate the original YAML — we copy & overlay so
    multiple trials can run concurrently and so the user's source is
    untouched.
    """
    src = Path(arch.workload_path)
    raw = yaml.safe_load(src.read_text())

    pre_trainer = (raw.get("modules") or {}).get("pre_trainer") or {}
    overrides = dict(pre_trainer.get("overrides") or {})

    # 1) parallelism ------------------------------------------------------
    workload_pp = arch.pipeline_model_parallel_size or 1
    workload_vpp = arch.virtual_pipeline_model_parallel_size or 1
    overrides["tensor_model_parallel_size"] = cfg.tp
    overrides["pipeline_model_parallel_size"] = cfg.pp
    overrides["expert_model_parallel_size"] = cfg.ep
    overrides["context_parallel_size"] = cfg.cp
    if cfg.vpp is not None:
        # Primus's projection reads ``virtual_pipeline_model_parallel_size``
        # (Megatron-LM core API), while the runtime trainer reads
        # ``num_virtual_stages_per_pipeline_rank`` (Primus's own override).
        # Set both to keep them coherent across projection + benchmark.
        overrides["num_virtual_stages_per_pipeline_rank"] = cfg.vpp
        overrides["virtual_pipeline_model_parallel_size"] = cfg.vpp
    # When the trial overrides PP or VPP, the workload's published per-stage
    # layout no longer matches PP×VPP. For models with a *non-divisor* layer
    # count (DeepSeek V3 = 61, prime; Kimi-K2 patterns) we regenerate a
    # balanced layout for the new (PP, VPP) — distributing num_layers across
    # PP×VPP stages as evenly as possible, with extras going to *middle*
    # stages so the embedding (first stage) and output/loss (last stages)
    # stay light. Without this the agent could never reach the published
    # ``PP=8 / VPP=2`` reference for DSv3, which encodes 16 stages of 3-4
    # layers via ``Et*3|t*4|...|t*3|t*3,L'``.
    trial_vpp = cfg.vpp if cfg.vpp is not None else workload_vpp
    workload_layout = arch.pipeline_model_parallel_layout
    if cfg.pp != workload_pp or trial_vpp != workload_vpp:
        # Always remove the workload's stage-count helpers; they're tied to
        # the original PP and would conflict with the regenerated layout.
        for layout_key in (
            "decoder_first_pipeline_num_layers",
            "decoder_last_pipeline_num_layers",
            "num_layers_in_first_pipeline_stage",
            "num_layers_in_last_pipeline_stage",
        ):
            overrides.pop(layout_key, None)
            pre_trainer.pop(layout_key, None)

        if workload_layout and arch.num_layers:
            # Regenerate a layout for the new (PP, VPP).
            new_vpp = trial_vpp or 1
            new_layout = _build_pipeline_layout(arch.num_layers, cfg.pp, new_vpp)
            if new_layout is not None:
                overrides["pipeline_model_parallel_layout"] = new_layout
            else:
                # Trial is too sliced (PP*VPP > num_layers) — let validate()
                # have caught this; fall back to clearing the layout.
                overrides.pop("pipeline_model_parallel_layout", None)
                pre_trainer.pop("pipeline_model_parallel_layout", None)
        else:
            overrides.pop("pipeline_model_parallel_layout", None)
            pre_trainer.pop("pipeline_model_parallel_layout", None)

    # 2) pipeline schedule ------------------------------------------------
    if cfg.enable_zero_bubble is not None:
        overrides["enable_zero_bubble"] = cfg.enable_zero_bubble
    if cfg.pp_schedule and cfg.pp_schedule != "auto":
        from primus.core.projection.config_validation import (
            PP_SCHEDULE_TO_RUNTIME_ALGORITHM,
        )

        overrides["pp_algorithm"] = PP_SCHEDULE_TO_RUNTIME_ALGORITHM.get(cfg.pp_schedule, cfg.pp_schedule)

    # 3) MoE communication (Tier-A — biggest single MoE win per skill) ----
    # Required-by-runtime couplings for the Primus Turbo MoE stack:
    #   * DeepEP / SyncFree stages 2-3 require ``moe_router_dtype=fp32`` —
    #     otherwise:
    #       AssertionError: DeepEP only supports float32 probs, please set
    #       `moe_router_dtype=fp32`
    #   * SyncFree stages 2-3 require ``moe_use_legacy_grouped_gemm=True`` —
    #     otherwise:
    #       ValueError: Sync-Free MoE stage 2 or 3 require PrimusTurboGroupedMLP,
    #       please set `moe_use_legacy_grouped_gemm=True`
    # We force these couplings here so the planner doesn't have to know about them.
    if cfg.use_turbo_deepep is not None:
        overrides["use_turbo_deepep"] = cfg.use_turbo_deepep
    if cfg.sync_free_stage is not None:
        overrides["sync_free_stage"] = cfg.sync_free_stage
    sync_free_high = cfg.sync_free_stage is not None and cfg.sync_free_stage >= 2
    deepep_active = bool(cfg.use_turbo_deepep) or sync_free_high
    if deepep_active:
        overrides["moe_router_dtype"] = "fp32"
    if sync_free_high:
        overrides["moe_use_legacy_grouped_gemm"] = True

    # 4) precision (Tier-A — ~2× compute for linear layers when "hybrid")  -
    if cfg.fp8 is not None:
        overrides["fp8"] = cfg.fp8

    # Multi-Latent-Attention models (DeepSeek V2/V3, Kimi-K2, Qwen-V2) auto-
    # enable ``use_turbo_parallel_linear`` inside Primus's projection. Two
    # things go wrong on the v26.2 container ``primus_turbo==0.2.0``:
    #   * The default ``fp8_recipe: delayed`` is incompatible with that path
    #     (``primus/modules/trainer/megatron/utils.py:464`` asserts).
    #   * The dense FP8 GEMM op (``primus_turbo.pytorch.ops.gemm_fp8``) on
    #     this version raises ``ValueError: Unsupported FP8 format: HYBRID``
    #     for ``fp8: hybrid`` (a common DSv3 / Kimi-K2 configuration).
    # We disable the MLA-specific turbo dense linear path while keeping
    # ``use_turbo_grouped_mlp`` (which DOES support HYBRID) so MoE grouped
    # GEMMs still benefit from FP8. This costs a few % of dense-linear FP8
    # speedup vs. a fully turbo'd path, but lets the agent actually
    # benchmark MLA + FP8 workloads on v26.2.
    fp8_active = (overrides.get("fp8") or arch.fp8) is not None
    if fp8_active and getattr(arch, "attention_type", "standard") == "mla":
        overrides.setdefault("fp8_recipe", "tensorwise")
        # v26.2's ``PrimusTurboGroupedMLP.forward`` references
        # ``self.bias_act_func`` which is set by the parent class only when
        # an older Megatron-LM lifecycle is honored. On v26.2 it is missing,
        # so route through the fused activation+probs kernel instead — which
        # is the recommended path anyway for SiLU/GeLU GLU models (= every
        # model we tune here).
        overrides.setdefault("use_turbo_fused_act_with_probs", True)

    # 5) memory levers ----------------------------------------------------
    if cfg.recompute_granularity in ("selective", "full"):
        overrides["recompute_granularity"] = cfg.recompute_granularity
        overrides["recompute_method"] = overrides.get("recompute_method", "block")
        overrides["recompute_num_layers"] = cfg.recompute_num_layers or 1
        # Some workload yamls (DeepSeek V3) ship a more granular
        # ``recompute_layer_ids`` (a comma-separated string of layer
        # indices) which Primus's arg parser then asserts must be a real
        # YAML list. The agent's recompute_num_layers replaces that
        # finer-grained list, so drop it when we're driving the recompute
        # strategy.
        overrides.pop("recompute_layer_ids", None)
        pre_trainer.pop("recompute_layer_ids", None)
    elif cfg.recompute_granularity == "none":
        overrides["recompute_granularity"] = None
        overrides["recompute_num_layers"] = 0
        overrides.pop("recompute_layer_ids", None)
        pre_trainer.pop("recompute_layer_ids", None)
    if cfg.cross_entropy_loss_fusion is not None:
        overrides["cross_entropy_loss_fusion"] = cfg.cross_entropy_loss_fusion
    if cfg.use_torch_fsdp2 is not None:
        overrides["use_torch_fsdp2"] = cfg.use_torch_fsdp2
    if cfg.use_distributed_optimizer is not None:
        overrides["use_distributed_optimizer"] = cfg.use_distributed_optimizer

    # 6) batch ------------------------------------------------------------
    overrides["micro_batch_size"] = cfg.mbs
    overrides["global_batch_size"] = cfg.gbs
    overrides["overlap_grad_reduce"] = cfg.overlap_grad_reduce

    pre_trainer["overrides"] = overrides
    raw.setdefault("modules", {})["pre_trainer"] = pre_trainer

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trial_{tag}.yaml"
    out_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return out_path


# ---------------------------------------------------------------------------
# Subprocess invocation
# ---------------------------------------------------------------------------

_FLOAT = r"([\-+]?\d+(?:[\.,]\d+)*(?:\.\d+)?)"

_RE_PARAM_OPT = re.compile(rf"Param\+Optimizer Memory:\s*{_FLOAT}\s*GB", re.IGNORECASE)
_RE_ACT = re.compile(rf"Activation Memory[^:]*:\s*{_FLOAT}\s*GB", re.IGNORECASE)
_RE_TOTAL_MEM = re.compile(rf"Projected Total Memory:\s*{_FLOAT}\s*GB", re.IGNORECASE)
# Bench-anchored memory projection (memory_projection.benchmark) prints:
#   "Point estimate (per rank): X GB"   ← OOM-accurate per-rank peak
#   "Upper bound    (per rank): Y GB"   ← point + safety_margin × residual_reserved
# Prefer these over the simulate-mode "Projected Total Memory" when both
# are present (e.g. in --memory-mode both output).
_RE_BENCH_POINT = re.compile(rf"Point estimate \(per rank\):\s*{_FLOAT}\s*GB", re.IGNORECASE)
_RE_BENCH_UPPER = re.compile(rf"Upper bound\s*\(per rank\):\s*{_FLOAT}\s*GB", re.IGNORECASE)
_RE_ITER_MS = re.compile(rf"Iteration Time:\s*{_FLOAT}\s*ms", re.IGNORECASE)
_RE_TPS = re.compile(rf"Tokens/s/GPU:\s*{_FLOAT}", re.IGNORECASE)
# Primus's projection prints both "TFLOPs/s/GPU" and "TFLOP/s/GPU" depending
# on version and output source.
_RE_TFLOPS = re.compile(rf"TFLOPs?/s/GPU:\s*{_FLOAT}", re.IGNORECASE)
_RE_MFU = re.compile(rf"\bMFU(?:\s*\(.*?\))?:\s*{_FLOAT}", re.IGNORECASE)


def _to_float(s: str) -> float | None:
    if s is None:
        return None
    try:
        return float(s.replace(",", ""))
    except (TypeError, ValueError):
        return None


def _parse_metrics(stdout: str) -> dict:
    out: dict[str, Any] = {}
    if m := _RE_PARAM_OPT.search(stdout):
        out["param_optimizer_gb"] = _to_float(m.group(1))
    if m := _RE_ACT.search(stdout):
        out["activation_gb"] = _to_float(m.group(1))
    # Prefer the bench-anchored point estimate when present — it is
    # OOM-accurate (real per-rank peak measured + analytical extrapolation
    # to target).  Fall back to simulate-mode "Projected Total Memory" or
    # the param+act sum when the bench section isn't in this output.
    if m := _RE_BENCH_POINT.search(stdout):
        out["memory_per_gpu_gb"] = _to_float(m.group(1))
        out["memory_source"] = "benchmark_point"
        if mu := _RE_BENCH_UPPER.search(stdout):
            out["memory_per_gpu_gb_upper"] = _to_float(mu.group(1))
    elif m := _RE_TOTAL_MEM.search(stdout):
        out["memory_per_gpu_gb"] = _to_float(m.group(1))
        out["memory_source"] = "simulate"
    elif "param_optimizer_gb" in out and "activation_gb" in out:
        out["memory_per_gpu_gb"] = out["param_optimizer_gb"] + out["activation_gb"]
        out["memory_source"] = "simulate_summed"
    if m := _RE_ITER_MS.search(stdout):
        out["iteration_ms"] = _to_float(m.group(1))
    if m := _RE_TPS.search(stdout):
        out["tokens_per_s_per_gpu"] = _to_float(m.group(1))
    if m := _RE_TFLOPS.search(stdout):
        out["tflops_per_s_per_gpu"] = _to_float(m.group(1))
    if m := _RE_MFU.search(stdout):
        out["mfu"] = _to_float(m.group(1))
    return out


def _python_invoker(
    primus_root: Path, profiling_mode: str | None = None, nproc_per_node: int | None = None
) -> list[str]:
    """Pick how to invoke `primus/cli/main.py`.

    For ``simulate`` (and ``memory``) we use ``sys.executable`` directly so
    Origami / memory analytics run in a single process. For ``benchmark``
    we route through ``python -m torch.distributed.run --standalone
    --nproc-per-node=N`` because the projection's micro-train step expects
    the Megatron launcher env (NODE_RANK / RANK / WORLD_SIZE / MASTER_*).
    """
    import sys as _sys

    main_py = str(primus_root / "primus" / "cli" / "main.py")
    if profiling_mode == "benchmark" and nproc_per_node and nproc_per_node > 0:
        return [
            _sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={nproc_per_node}",
            main_py,
        ]
    return [_sys.executable, main_py]


def _build_perf_cmd(
    yaml_path: Path,
    cfg: TrialConfig,
    agent_cfg: AgentConfig,
    profiling_mode: str,
    primus_root: Path,
    *,
    save_benchmark: Path | None = None,
) -> list[str]:
    bench_gpus = (
        agent_cfg.benchmark_host.benchmark_gpus
        if agent_cfg.benchmark_host.has_gpu and agent_cfg.benchmark_host.benchmark_gpus > 0
        else agent_cfg.target_cluster.gpus_per_node
    )
    cmd: list[str] = [
        *_python_invoker(primus_root, profiling_mode=profiling_mode, nproc_per_node=bench_gpus),
        "projection",
        "performance",
        "--config",
        str(yaml_path),
        "--target-nodes",
        str(agent_cfg.target_cluster.num_nodes),
        "--profiling-mode",
        profiling_mode,
        "--gpu-arch",
        agent_cfg.target_cluster.gpu_arch,
    ]
    if profiling_mode == "benchmark":
        cmd += ["--benchmark-gpus", str(bench_gpus)]
        if save_benchmark is not None:
            # Persist the bench artifact so the memory projection can
            # consume it via --load-benchmark (one bench, two projections).
            cmd += ["--save-benchmark", str(save_benchmark)]
    if agent_cfg.target_cluster.hardware_config:
        cmd += ["--hardware-config", agent_cfg.target_cluster.hardware_config]
    if agent_cfg.target_cluster.gpu_clock_mhz:
        cmd += ["--gpu-clock-mhz", str(agent_cfg.target_cluster.gpu_clock_mhz)]
    if cfg.pp_schedule and cfg.pp_schedule != "auto":
        cmd += ["--pipeline-schedule-algorithm", cfg.pp_schedule]
    if cfg.vpp is not None:
        cmd += ["--num-virtual-stages-per-pipeline-rank", str(cfg.vpp)]
    cmd += ["--micro-batch-size", str(cfg.mbs)]
    cmd += ["--global-batch-size", str(cfg.gbs)]

    # Tier-A optimisations from the Primus Projection skill — these have
    # CLI flags on `projection performance` that take precedence over yaml
    # overrides (so we pass them explicitly even when the yaml carries the
    # corresponding key, to make the agent's signature self-describing).
    if cfg.use_turbo_deepep:
        cmd += ["--enable-deepep"]
    if cfg.sync_free_stage and cfg.sync_free_stage > 0:
        cmd += ["--sync-free-stage", str(cfg.sync_free_stage)]
    if cfg.enable_zero_bubble:
        cmd += ["--enable-zero-bubble"]
    if cfg.target_ep_size is not None:
        cmd += ["--target-ep-size", str(cfg.target_ep_size)]
    return cmd


def _build_memory_cmd(
    yaml_path: Path,
    cfg: TrialConfig,
    agent_cfg: AgentConfig,
    primus_root: Path,
    *,
    memory_mode: str = "simulate",
    load_benchmark: Path | None = None,
    safety_margin: float | None = None,
) -> list[str]:
    """Build the ``projection memory`` argv.

    When ``memory_mode='benchmark'`` is paired with ``load_benchmark=<path>``,
    the call skips its own bench step and projects from the artifact saved
    by an earlier ``projection performance --save-benchmark`` (this is how
    the agent gets *bench-anchored* memory without paying for two benches).
    """
    cmd: list[str] = [
        *_python_invoker(primus_root),
        "projection",
        "memory",
        "--config",
        str(yaml_path),
    ]
    if memory_mode:
        # Pass the memory mode explicitly so the agent does not rely on the
        # CLI's default (which is ``benchmark`` and would launch a GPU
        # sub-node layer benchmark for the simulate/memory-only path).
        cmd += ["--memory-mode", memory_mode]
    if load_benchmark is not None:
        cmd += ["--load-benchmark", str(load_benchmark)]
    if safety_margin is not None:
        cmd += ["--memory-safety-margin", f"{float(safety_margin):.4f}"]
    # Project to the real target cluster so the bench-anchored
    # extrapolator scales params/comm/DeepEP buffers correctly.
    cmd += ["--target-nodes", str(agent_cfg.target_cluster.num_nodes)]
    if cfg.pp_schedule and cfg.pp_schedule != "auto":
        cmd += ["--pipeline-schedule-algorithm", cfg.pp_schedule]
    if cfg.mbs is not None:
        cmd += ["--micro-batch-size", str(cfg.mbs)]
    if cfg.gbs is not None:
        cmd += ["--global-batch-size", str(cfg.gbs)]
    if cfg.vpp is not None:
        cmd += ["--num-virtual-stages-per-pipeline-rank", str(cfg.vpp)]
    return cmd


def _build_env(agent_cfg: AgentConfig, primus_root: Path, profiling_mode: str | None = None) -> dict:
    env = os.environ.copy()
    # For memory + simulate: pass the real target-cluster shape so
    # `projection memory`'s ``get_dp_size`` doesn't enter the "recompute min
    # nodes" branch where ``TP*CP*PP*EP // GPUS_PER_NODE`` underflows to zero
    # (=> ZeroDivisionError) when ``NNODES==1``.
    #
    # For benchmark: we are physically launching on this single host (via
    # torch.distributed.run --nnodes=1), so NNODES=1 / NODE_RANK=0; the CLI
    # carries ``--target-nodes`` for the projection target.
    if profiling_mode == "benchmark":
        env["NNODES"] = "1"
        env["NODE_RANK"] = "0"
    else:
        env["NNODES"] = str(agent_cfg.target_cluster.num_nodes)
    env["GPUS_PER_NODE"] = str(agent_cfg.target_cluster.gpus_per_node)
    env.setdefault("HSA_NO_SCRATCH_RECLAIM", "1")
    env["PRIMUS_GPU_ARCH"] = agent_cfg.target_cluster.gpu_arch

    # PYTHONPATH discipline: the subprocess runs the OUTER ``primus/cli/
    # main.py`` and only needs ``primus_root`` on the path.  When the agent
    # is launched with the tuning-agent subtree on PYTHONPATH (so it can
    # import its own modules), that subtree contains a *nested* ``primus/``
    # package — without forcing primus_root to the front of PYTHONPATH and
    # stripping any nested "primus/agents/tuning-agent" entries, the inner
    # nested copy shadows the outer one and the subprocess imports stale
    # CLI modules (e.g. one that doesn't know ``--save-benchmark``).  Fix:
    # always put primus_root first; drop any path under the tuning-agent
    # subtree.
    sep = os.pathsep
    py_path = env.get("PYTHONPATH", "")
    parts = [p for p in py_path.split(sep) if p]
    inner_marker = str(Path("primus") / "agents" / "tuning-agent")
    parts = [p for p in parts if inner_marker not in p]
    parts = [p for p in parts if Path(p) != Path(primus_root)]
    parts.insert(0, str(primus_root))
    env["PYTHONPATH"] = sep.join(parts)
    return env


def _run(cmd: list[str], cwd: Path, env: dict, timeout: int) -> tuple[int, str, float]:
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        return proc.returncode, proc.stdout or "", time.time() - started
    except subprocess.TimeoutExpired as e:
        return (
            124,
            (e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")),
            time.time() - started,
        )


# ---------------------------------------------------------------------------
# Public evaluator class
# ---------------------------------------------------------------------------


class Evaluator:
    """Stateful evaluator. Tracks call counts so the agent stays within budget.

    ``mode`` selects how candidates are scored:

    * ``"dry"`` — never call ``primus-cli``; both memory and tps are
      synthesised. Useful for testing the agent loop on hosts without a Primus
      install.
    * ``"memory-real"`` — call ``projection memory`` for real (no GPU
      required), then synthesise a tokens/s/GPU heuristic from the measured
      memory + parallelism axes. Useful when ``projection performance``
      cannot run because Origami / FAv3 simulator deps are missing.
    * ``"full"`` — call ``projection memory`` and ``projection performance``
      (``--profiling-mode {simulate,benchmark}``). Default. Requires Origami
      for simulate and a GPU for benchmark.
    """

    def __init__(
        self,
        agent_cfg: AgentConfig,
        arch: ArchitectureRecord,
        primus_root: Path,
        dry_run: bool = False,
        mode: str = "full",
    ):
        self.cfg = agent_cfg
        self.arch = arch
        self.primus_root = primus_root
        # ``dry_run`` is kept for back-compat with earlier callers.
        if dry_run:
            mode = "dry"
        if mode not in ("dry", "memory-real", "full"):
            raise ValueError(f"unknown evaluator mode: {mode!r}")
        self.mode = mode
        self.dry_run = mode == "dry"
        self.trials_dir = agent_cfg.out_dir / "trials"
        self.trials_dir.mkdir(parents=True, exist_ok=True)
        self.n_memory_calls = 0
        self.n_simulate_calls = 0
        self.n_benchmark_calls = 0

    # ── public API ────────────────────────────────────────────────────────

    def evaluate_memory_only(self, cfg: TrialConfig, tag: str) -> EvalResult:
        return self._evaluate(cfg, tag, mode="memory_only")

    def evaluate_simulate(self, cfg: TrialConfig, tag: str) -> EvalResult:
        return self._evaluate(cfg, tag, mode="simulate")

    def evaluate_benchmark(self, cfg: TrialConfig, tag: str) -> EvalResult:
        if not self.cfg.benchmark_host.has_gpu:
            return EvalResult(
                legal=False,
                reason="benchmark requested but has_gpu=false",
                source="benchmark",
                config=cfg.as_dict(),
            )
        return self._evaluate(cfg, tag, mode="benchmark")

    # ── internal ──────────────────────────────────────────────────────────

    def _evaluate(self, cfg: TrialConfig, tag: str, mode: str) -> EvalResult:
        from .legality import derive_legality

        legality = derive_legality(self.arch, self.cfg.target_cluster)
        ok, reason = validate(cfg, self.arch, self.cfg.target_cluster, legality)
        result = EvalResult(
            legal=ok,
            reason=reason,
            source=mode,
            config=cfg.as_dict(),
            derived_dp=derived_dp(cfg, self.arch, self.cfg.target_cluster),
        )
        if not ok:
            return result

        if self.mode == "dry":
            # synthesise a metric so the loop is testable without primus-cli
            result.memory_per_gpu_gb = 100.0
            result.tokens_per_s_per_gpu = 1000.0 + (cfg.tp + cfg.pp + cfg.ep)
            result.iteration_ms = 1000.0
            return result

        yaml_path = write_trial_yaml(self.arch, cfg, self.trials_dir, tag)

        # In benchmark mode, defer memory until after perf so the bench
        # artifact is shared (one GPU bench feeds both projections).  The
        # simulate-mode memory call is skipped because its formula
        # disagrees with the bench-anchored one anyway, and we want OOM
        # decisions to use the measured peak.
        if mode == "benchmark":
            return self._evaluate_benchmark(cfg, tag, yaml_path, result)

        # 1. memory (cheap, always run for legality + headroom check)
        mem_cmd = _build_memory_cmd(yaml_path, cfg, self.cfg, self.primus_root)
        rc, out, dur = _run(
            mem_cmd, cwd=self.primus_root, env=_build_env(self.cfg, self.primus_root), timeout=300
        )
        self.n_memory_calls += 1
        result.cmd = mem_cmd
        result.returncode = rc
        result.duration_s = dur
        result.stdout_tail = _tail(out, 4000)
        result_metrics = _parse_metrics(out)
        for k, v in result_metrics.items():
            setattr(result, k, v)

        # `projection memory` is non-deterministic in its failure mode: it can
        # crash mid-print and still leave some inner metrics parsed but no
        # final summary. Require BOTH a clean exit AND a parsed total to
        # accept the trial.
        if rc != 0 or result.memory_per_gpu_gb is None:
            result.legal = False
            result.reason = (
                f"projection memory failed (rc={rc}, "
                f"total_parsed={result.memory_per_gpu_gb}); see stdout_tail"
            )
            return result

        # memory cap enforcement
        #
        # Caveat for `recompute_granularity in {"selective", "none", None}`:
        # Primus's analytic memory model only models ``full`` recompute
        # (see core/projection/module_profilers/language_model.py:504). For
        # ``selective`` and ``none`` it stores **all** per-layer activations,
        # which over-counts vs. real Megatron runs (selective saves ~50%
        # because attention activations are recomputed). To avoid falsely
        # pre-rejecting these in *benchmark* mode (where the actual GPU run
        # is the ground truth and will OOM if it can't fit), we apply a
        # heuristic correction:
        #   - "selective": multiply projection by 0.55  (saves attention)
        #   - "none"/None: leave as-is, trust the projection
        # When in benchmark mode, even a still-over-cap result is treated
        # as ADVISORY: log it but proceed to benchmark, because real OOM is
        # the only authoritative answer. In simulate / memory-only modes
        # the projection remains authoritative.
        cap = self.cfg.optimization.hbm_capacity_gb * (1 - self.cfg.optimization.memory_safety_margin)
        adjusted_mem = result.memory_per_gpu_gb
        if cfg.recompute_granularity == "selective" and adjusted_mem is not None:
            adjusted_mem = adjusted_mem * 0.55
        result.memory_per_gpu_gb_adjusted = adjusted_mem

        # Strict memory enforcement: any trial whose projected per-GPU
        # memory exceeds the safety-margined HBM cap is rejected up-front,
        # *regardless* of profiling mode. This enforces the user's rule:
        # "if solution memory is more than projected HBM capacity, that
        # solution should not be considered". Sub-node benchmark profiling
        # may have succeeded in earlier runs because the local 8-GPU
        # measurement avoided the projected bottleneck — but that's a
        # measurement artefact, not a green light to deploy the config to
        # the target N-node cluster.
        if adjusted_mem is not None and adjusted_mem > cap:
            result.legal = False
            result.reason = (
                f"memory {adjusted_mem:.1f} GB > cap {cap:.1f} GB "
                f"(HBM={self.cfg.optimization.hbm_capacity_gb} GB, margin="
                f"{self.cfg.optimization.memory_safety_margin})"
            )
            return result

        if mode == "memory_only":
            return result

        # If we are not allowed to call `projection performance` (e.g. Origami
        # not installed and no GPU on this host), synthesise a tps from the
        # measured memory + parallelism axes so the LLM still has a signal.
        if self.mode == "memory-real":
            result.tokens_per_s_per_gpu = _synth_tps_from_memory(cfg, result, self.arch)
            result.iteration_ms = 1000.0
            return result

        # 2. performance (simulate path only — benchmark goes through
        # ``_evaluate_benchmark`` via the early-return above).
        profiling_mode = "simulate"
        perf_cmd = _build_perf_cmd(yaml_path, cfg, self.cfg, profiling_mode, self.primus_root)
        rc2, out2, dur2 = _run(
            perf_cmd,
            cwd=self.primus_root,
            env=_build_env(self.cfg, self.primus_root, profiling_mode=profiling_mode),
            timeout=600,
        )
        self.n_simulate_calls += 1
        result.cmd = perf_cmd
        result.returncode = rc2
        result.duration_s += dur2
        result.stdout_tail = (result.stdout_tail + "\n--- perf ---\n" + _tail(out2, 6000))[-12000:]
        for k, v in _parse_metrics(out2).items():
            setattr(result, k, v)
        if rc2 != 0 and result.tokens_per_s_per_gpu is None:
            result.legal = False
            result.reason = f"projection performance failed (rc={rc2}); see stdout_tail"
        return result

    def _evaluate_benchmark(
        self, cfg: TrialConfig, tag: str, yaml_path: Path, result: EvalResult
    ) -> EvalResult:
        """Benchmark-mode flow with shared perf↔memory artifact.

        Runs:

          1. ``projection performance --profiling-mode benchmark
             --save-benchmark <tmp>`` → writes the bench artifact and
             prints tokens/s/GPU + iter time.
          2. ``projection memory --memory-mode benchmark --load-benchmark
             <tmp> --target-nodes N --memory-safety-margin M`` → projects
             OOM-accurate per-rank peak (point + 5 % upper bound) without
             re-running the bench.

        Both projections see the same measured memory and timing.
        """
        import tempfile

        cap = self.cfg.optimization.hbm_capacity_gb * (1 - self.cfg.optimization.memory_safety_margin)

        fd, tmp_path = tempfile.mkstemp(prefix="tuning_agent_bench_", suffix=".json")
        os.close(fd)
        artifact = Path(tmp_path)
        try:
            # ---- 1. perf bench (writes artifact) -------------------------
            perf_cmd = _build_perf_cmd(
                yaml_path,
                cfg,
                self.cfg,
                "benchmark",
                self.primus_root,
                save_benchmark=artifact,
            )
            rc1, out1, dur1 = _run(
                perf_cmd,
                cwd=self.primus_root,
                env=_build_env(self.cfg, self.primus_root, profiling_mode="benchmark"),
                timeout=3600,
            )
            self.n_benchmark_calls += 1
            result.cmd = perf_cmd
            result.returncode = rc1
            result.duration_s = dur1
            result.stdout_tail = _tail(out1, 6000)
            for k, v in _parse_metrics(out1).items():
                setattr(result, k, v)

            if rc1 != 0:
                result.legal = False
                result.reason = f"projection performance --benchmark failed (rc={rc1}); " f"see stdout_tail"
                return result

            if not artifact.exists() or artifact.stat().st_size == 0:
                result.legal = False
                result.reason = (
                    "perf bench did not produce a --save-benchmark artifact; "
                    "memory cannot be projected from it"
                )
                return result

            # ---- 2. memory in benchmark/load mode -----------------------
            mem_cmd = _build_memory_cmd(
                yaml_path,
                cfg,
                self.cfg,
                self.primus_root,
                memory_mode="benchmark",
                load_benchmark=artifact,
                safety_margin=self.cfg.optimization.memory_safety_margin,
            )
            rc2, out2, dur2 = _run(
                mem_cmd,
                cwd=self.primus_root,
                # Memory-from-artifact path doesn't need megatron; pass
                # default env (NNODES=target so projection sees the right
                # cluster shape for analytical extrapolation).
                env=_build_env(self.cfg, self.primus_root),
                timeout=300,
            )
            self.n_memory_calls += 1
            result.duration_s += dur2
            result.stdout_tail = (result.stdout_tail + "\n--- mem (bench/load) ---\n" + _tail(out2, 6000))[
                -14000:
            ]
            mem_metrics = _parse_metrics(out2)
            for k, v in mem_metrics.items():
                setattr(result, k, v)

            if rc2 != 0 or result.memory_per_gpu_gb is None:
                result.legal = False
                result.reason = (
                    f"projection memory --benchmark/load failed "
                    f"(rc={rc2}, total_parsed={result.memory_per_gpu_gb}); "
                    f"see stdout_tail"
                )
                return result

            # OOM-fits decision uses the bench-anchored *upper bound*
            # (point + 5 % safety margin × residual_reserved) when present;
            # otherwise the point estimate.  This is the OOM-accurate
            # signal the memory benchmark mode was designed for.
            decision_mem = (
                result.memory_per_gpu_gb_upper
                if result.memory_per_gpu_gb_upper is not None
                else result.memory_per_gpu_gb
            )
            result.memory_per_gpu_gb_adjusted = decision_mem
            if decision_mem is not None and decision_mem > cap:
                result.legal = False
                result.reason = (
                    f"benchmark memory {decision_mem:.1f} GB > cap {cap:.1f} GB "
                    f"(HBM={self.cfg.optimization.hbm_capacity_gb} GB, "
                    f"margin={self.cfg.optimization.memory_safety_margin}, "
                    f"source={result.memory_source or 'benchmark'})"
                )
                return result

            return result
        finally:
            try:
                artifact.unlink(missing_ok=True)
            except OSError:
                pass


def _tail(s: str, n: int) -> str:
    if not s:
        return ""
    return s if len(s) <= n else s[-n:]


def _synth_tps_from_memory(cfg: TrialConfig, result: EvalResult, arch: ArchitectureRecord) -> float:
    """A *very* coarse heuristic so the agent has a relative signal in
    ``memory-real`` mode (no Origami, no GPU). Higher is better.

    Intentionally simple — the LLM is told this is a heuristic, not a real
    perf number, and is used only to rank candidates relative to each other.
    """
    base = 1000.0
    # Reward larger MBS (better GEMM utilisation) up to a saturating point.
    base += min(cfg.mbs, 8) * 25.0
    # Penalise TP and PP communication; EP is cheap *if* it stays intra-node.
    base -= (cfg.tp - 1) * 30.0
    base -= (cfg.pp - 1) * 20.0
    base -= max(cfg.ep - 8, 0) * 40.0  # EP > one node hurts (A2A across nodes)
    # Penalise CP > 1 lightly (extra ring comms).
    base -= (cfg.cp - 1) * 10.0
    # Penalise full recompute (extra forward) lightly.
    if cfg.recompute_granularity == "full":
        base -= 40.0
    elif cfg.recompute_granularity == "selective":
        base -= 10.0
    # Reward staying well within HBM headroom (less paging, more activations).
    if result.memory_per_gpu_gb is not None:
        base += max(0.0, 50.0 - result.memory_per_gpu_gb * 0.05)
    return round(max(100.0, base), 1)
