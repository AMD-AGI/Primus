"""Command-line entry point.

Usage:
    python -m primus.agents.tuning_agent \
        --workload examples/megatron/configs/MI355X/mixtral_8x22B_v0.1-BF16-pretrain.yaml \
        --target-cluster examples/agents/tuning_agent/target_cluster.yaml \
        [--out-dir tuning_runs/mi355x-2nodes] \
        [--dry-run] [--seed-only] [--no-agent]

Sub-modes:
    --mode dry           synthesise metrics; never call primus-cli (good
                         for testing the agent loop on hosts without a
                         Primus / torch / origami install).
    --mode memory-real   call ``projection memory`` for real (no GPU
                         needed) and synthesise tps from a memory + axes
                         heuristic. Use when ``projection performance``
                         can't run (no Origami / no GPU).
    --mode full          call ``projection memory`` and ``projection
                         performance`` (simulate or benchmark). Default.
                         Requires Origami for simulate; a GPU for benchmark.
    --dry-run            shorthand for --mode dry.
    --seed-only          evaluate only the systematic seed plan, then exit
                         (no LLM stage).
    --no-agent           alias for --seed-only.
    --resume             reuse an existing trials.jsonl in --out-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .config import load_config
from .evaluator import Evaluator
from .history import History
from .legality import derive_legality
from .plan import build_seed_plan
from .plotting import plot_history
from .scratchpad import Scratchpad
from .workload import resolve_workload, _find_primus_root


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="primus.agents.tuning_agent",
        description="LLM-driven search for an optimal Primus parallelization configuration.",
    )
    p.add_argument("--workload", required=True, type=Path,
                   help="Path to a Primus pretrain YAML.")
    p.add_argument("--target-cluster", required=True, type=Path,
                   help="Path to target_cluster.yaml.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory for trials, plot, scratchpad.")
    p.add_argument("--mode", choices=("dry", "memory-real", "full"),
                   default="full",
                   help="Evaluator mode (default: full). See module docstring.")
    p.add_argument("--profiling-mode", choices=("simulate", "benchmark"),
                   default="simulate",
                   help=("Profiling backend for `projection performance` in --mode full. "
                         "`simulate` uses Origami (no GPU); `benchmark` runs real GPUs "
                         "via torch.distributed.run on the local node and projects to "
                         "target_cluster.num_nodes (requires has_gpu=true)."))
    p.add_argument("--dry-run", action="store_true",
                   help="Shorthand for --mode dry.")
    p.add_argument("--seed-only", action="store_true",
                   help="Evaluate the systematic seed plan only; skip LLM.")
    p.add_argument("--no-agent", action="store_true",
                   help="Alias for --seed-only.")
    p.add_argument("--agent-only", action="store_true",
                   help="Skip seed evaluation; run LLM agent on existing history.")
    p.add_argument("--resume", action="store_true",
                   help="Reuse an existing trials.jsonl in --out-dir.")
    p.add_argument("--seed-budget", type=int, default=12,
                   help="Max seed candidates evaluated before the LLM takes over.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    workload_yaml = args.workload.resolve()
    target_cluster_yaml = args.target_cluster.resolve()

    if not workload_yaml.is_file():
        print(f"ERROR: workload not found: {workload_yaml}", file=sys.stderr)
        return 2
    if not target_cluster_yaml.is_file():
        print(f"ERROR: target cluster YAML not found: {target_cluster_yaml}", file=sys.stderr)
        return 2

    agent_cfg = load_config(target_cluster_yaml, workload_yaml, out_dir=args.out_dir)
    primus_root = _find_primus_root(workload_yaml)

    # ── resolve workload ────────────────────────────────────────────────
    print(f"[tuning-agent] workload:        {workload_yaml}")
    print(f"[tuning-agent] target cluster:  {target_cluster_yaml}")
    print(f"[tuning-agent] out dir:         {agent_cfg.out_dir}")
    print(f"[tuning-agent] primus root:     {primus_root}")
    arch = resolve_workload(workload_yaml, primus_root=primus_root)
    print(f"[tuning-agent] resolved model:  {arch.model_name} "
          f"(layers={arch.num_layers}, hidden={arch.hidden_size}, "
          f"moe={arch.is_moe} experts={arch.num_experts} topk={arch.moe_router_topk})")

    legality = derive_legality(arch, agent_cfg.target_cluster)
    print(f"[tuning-agent] legal axes: TP={legality.tp} PP={legality.pp} "
          f"EP={legality.ep} CP={legality.cp} MBS={legality.mbs} VPP={legality.vpp}")

    # ── output dirs ─────────────────────────────────────────────────────
    agent_cfg.out_dir.mkdir(parents=True, exist_ok=True)
    history_path = agent_cfg.out_dir / "trials.jsonl"
    scratchpad_path = agent_cfg.out_dir / "scratchpad.txt"
    plot_path = agent_cfg.out_dir / "trials.png"
    summary_path = agent_cfg.out_dir / "summary.json"

    if not args.resume and history_path.exists():
        print(f"[tuning-agent] WARN: {history_path} exists; re-using "
              f"(pass --resume to suppress this warning).")

    history = History.load(history_path)
    scratchpad = Scratchpad(scratchpad_path)
    eval_mode = "dry" if args.dry_run else args.mode
    evaluator = Evaluator(agent_cfg, arch, primus_root, mode=eval_mode)
    print(f"[tuning-agent] evaluator mode:  {eval_mode}")
    print(f"[tuning-agent] profiling mode:  {args.profiling_mode}")
    if args.profiling_mode == "benchmark" and not agent_cfg.benchmark_host.has_gpu:
        print("[tuning-agent] WARN: --profiling-mode benchmark but "
              "available_for_benchmark.has_gpu=false; benchmark trials will be "
              "marked illegal. Edit the target-cluster YAML to enable.",
              file=sys.stderr)

    # ── seed evaluation ─────────────────────────────────────────────────
    if args.agent_only:
        print(f"\n[tuning-agent] --agent-only: skipping seeds ({len(history.trials)} trials loaded)")
    seed = build_seed_plan(arch, agent_cfg, max_candidates=args.seed_budget)
    if not args.agent_only:
        print(f"\n[tuning-agent] seed plan: {seed.rationale}")
    for cfg in ([] if args.agent_only else seed.candidates):
        sig = cfg.signature()
        if history.already_evaluated(cfg.as_dict()):
            print(f"    [seed] skip (already evaluated): {sig}")
            continue
        idx = len(history.trials)
        tag = f"seed_{idx:03d}_tp{cfg.tp}_pp{cfg.pp}_ep{cfg.ep}_cp{cfg.cp}_mbs{cfg.mbs}"
        if eval_mode == "full" and args.profiling_mode == "benchmark":
            # Benchmark mode: skip the strict simulate-mode memory pre-filter
            # and let `evaluate_benchmark` apply its advisory check (which
            # accounts for Primus's analytic memory model NOT honouring
            # selective recompute — see evaluator.py for details). The actual
            # GPU run is the ground truth: real OOM => FAIL, real success
            # => trustworthy tps even if the analytic model said no.
            r = evaluator.evaluate_benchmark(cfg, tag)
            note = "seed[benchmark]"
        else:
            r = evaluator.evaluate_memory_only(cfg, tag)
            if not r.legal:
                history.add(cfg.as_dict(), r, notes="seed[mem-only]")
                print(f"    [seed #{idx}] REJECT: {r.reason}")
                continue
            r = evaluator.evaluate_simulate(cfg, tag)
            note = "seed[simulate]"
        history.add(cfg.as_dict(), r, notes=note)
        tps = r.tokens_per_s_per_gpu
        tflops = getattr(r, "tflops_per_s_per_gpu", None)
        mem = r.memory_per_gpu_gb
        ok = "OK" if r.legal else "FAIL"
        warn = (" WARN: " + r.memory_warning) if getattr(r, "memory_warning", None) else ""
        tflops_str = f" tflops={tflops}" if tflops is not None else ""
        print(f"    [seed #{idx}] {ok} tps={tps}{tflops_str} mem={mem} cfg={sig}{warn}")

    # ── LLM agent stage ─────────────────────────────────────────────────
    if not (args.seed_only or args.no_agent):
        try:
            from .agent import run_agent
        except ImportError as e:
            print(f"[tuning-agent] WARNING: cannot import agent ({e}); skipping LLM stage. "
                  f"Install dspy + python-dotenv to enable.", file=sys.stderr)
        else:
            try:
                summary = run_agent(
                    agent_cfg=agent_cfg,
                    arch=arch,
                    history=history,
                    evaluator=evaluator,
                    scratchpad=scratchpad,
                    workspace=agent_cfg.out_dir,
                )
                summary_path.write_text(json.dumps(summary, indent=2, default=str))
            except Exception as e:
                print(f"[tuning-agent] agent stage failed: {e}", file=sys.stderr)
                traceback.print_exc()

    # ── final report ────────────────────────────────────────────────────
    plot = plot_history(history, plot_path, agent_cfg.optimization.objective)
    if plot:
        print(f"\n[tuning-agent] wrote plot: {plot}")

    best = history.best(agent_cfg.optimization.objective)
    if best is None:
        print("\n[tuning-agent] no legal trial found.")
        return 1

    print("\n=== BEST CONFIGURATION ===")
    print(f"  trial #{best.idx}  source={best.source}")
    cfg = best.config
    print(f"  TP={cfg.get('tp')} PP={cfg.get('pp')} EP={cfg.get('ep')} CP={cfg.get('cp')}")
    print(f"  MBS={cfg.get('mbs')} GBS={cfg.get('gbs')} VPP={cfg.get('vpp')} schedule={cfg.get('pp_schedule')}")
    print(f"  recompute={cfg.get('recompute_granularity')}/{cfg.get('recompute_num_layers')}")
    print(f"  → tokens/s/GPU = {best.result.get('tokens_per_s_per_gpu')}")
    if best.result.get("tflops_per_s_per_gpu") is not None:
        print(f"  → TFLOPs/s/GPU = {best.result.get('tflops_per_s_per_gpu')}")
    if best.result.get("mfu") is not None:
        print(f"  → MFU          = {best.result.get('mfu')}")
    print(f"  → memory/GPU   = {best.result.get('memory_per_gpu_gb')} GB")

    print("\n=== EXPORTS for primus-cli ===")
    print(f"  export PRIMUS_TP={cfg.get('tp')}")
    print(f"  export PRIMUS_PP={cfg.get('pp')}")
    print(f"  export PRIMUS_EP={cfg.get('ep')}")
    if cfg.get("cp"):
        print(f"  export PRIMUS_CP={cfg.get('cp')}   # not all configs honour this")
    print(f"  # then pass --micro-batch-size {cfg.get('mbs')} --global-batch-size {cfg.get('gbs')}")

    if not summary_path.exists():
        summary_path.write_text(json.dumps({
            "best": best.as_dict(),
            "trials_total": len(history.trials),
        }, indent=2, default=str))

    print(f"\n[tuning-agent] artifacts in {agent_cfg.out_dir}/")
    print(f"  trials.jsonl  trials.png  scratchpad.txt  summary.json  trials/*.yaml")
    return 0
