"""Tool functions exposed to the DSPy.RLM REPL.

Mirrors iterative_fix's `build_tools()` shape: state is held in shared
mutable containers passed by reference; tools log their calls and append
structured records so the orchestrator can inspect them after the RLM run.

Tools provided to the LLM:

  evaluate_simulate(config_json) -> JSON metrics
  evaluate_memory_only(config_json) -> JSON metrics  (cheap pre-filter)
  evaluate_with_benchmark(config_json) -> JSON metrics  (gated by has_gpu)
  get_history(k=20) -> str  (last k trials, compact summary)
  get_best() -> JSON       (current incumbent)
  get_legal_axes() -> JSON  (per-axis legal sets)
  get_architecture() -> JSON  (model fields)
  get_cluster() -> JSON
  note_to_scratchpad(text) -> str
  read_scratchpad() -> str
  query_llm(prompt, system?) -> str  (extra LLM consultation tool)
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from .config import AgentConfig
from .evaluator import Evaluator, EvalResult
from .history import History
from .legality import (
    AxisLegality, TrialConfig, derive_legality, fill_defaults_from_baseline,
    derived_dp,
)
from .scratchpad import Scratchpad
from .workload import ArchitectureRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_load_json(s: str) -> dict | None:
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s.startswith("```"):
        # strip fenced
        s = s.strip("`")
        if s.lower().startswith("json\n"):
            s = s[5:]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _result_for_llm(r: EvalResult) -> dict:
    return {
        "legal": r.legal,
        "reason": r.reason,
        "memory_per_gpu_gb": r.memory_per_gpu_gb,
        "param_optimizer_gb": r.param_optimizer_gb,
        "activation_gb": r.activation_gb,
        "iteration_ms": r.iteration_ms,
        "tokens_per_s_per_gpu": r.tokens_per_s_per_gpu,
        "source": r.source,
        "derived_dp": r.derived_dp,
        "config": r.config,
    }


def _tag(cfg: TrialConfig, idx: int) -> str:
    return f"{idx:03d}_tp{cfg.tp}_pp{cfg.pp}_ep{cfg.ep}_cp{cfg.cp}_mbs{cfg.mbs}"


# ---------------------------------------------------------------------------
# Tool-belt builder
# ---------------------------------------------------------------------------

def build_tools(
    agent_cfg: AgentConfig,
    arch: ArchitectureRecord,
    legality: AxisLegality,
    history: History,
    evaluator: Evaluator,
    scratchpad: Scratchpad,
    session_log: list[dict],
    quiet: bool = False,
) -> list[Callable]:
    """Build the tool list passed to dspy.RLM(...).

    All shared state (history, evaluator counters, session_log) lives on the
    closure so the caller can inspect it after the RLM finishes.
    """
    budget = agent_cfg.optimization.budget

    def _log(kind: str, **payload):
        session_log.append({
            "ts": datetime.now().isoformat(),
            "kind": kind,
            **payload,
        })
        if not quiet:
            print(f"    [tool:{kind}] {payload.get('summary', '')[:160]}")

    # -----------------------------------------------------------------
    # Evaluation tools
    # -----------------------------------------------------------------
    def _evaluate(cfg_dict: dict, mode: str) -> dict:
        d = _safe_load_json(cfg_dict)
        if d is None:
            return {"error": "config_json must be a JSON object"}
        cfg = TrialConfig.from_dict(d)
        cfg = fill_defaults_from_baseline(cfg, arch)
        if history.already_evaluated(cfg.as_dict()):
            for t in reversed(history.trials):
                if t.config == cfg.as_dict():
                    return {"already_evaluated": True, **t.result}
        if mode == "simulate" and evaluator.n_simulate_calls >= budget.max_perf_calls:
            return {"error": f"simulate budget exhausted ({evaluator.n_simulate_calls}/{budget.max_perf_calls})"}
        if mode == "benchmark" and evaluator.n_benchmark_calls >= budget.max_benchmark_calls:
            return {"error": f"benchmark budget exhausted ({evaluator.n_benchmark_calls}/{budget.max_benchmark_calls})"}

        idx = len(history.trials)
        tag = _tag(cfg, idx)
        if mode == "memory_only":
            r = evaluator.evaluate_memory_only(cfg, tag)
        elif mode == "simulate":
            r = evaluator.evaluate_simulate(cfg, tag)
        else:
            r = evaluator.evaluate_benchmark(cfg, tag)
        history.add(cfg.as_dict(), r, notes=f"mode={mode}")
        out = _result_for_llm(r)
        _log("evaluate",
             summary=f"{mode} #{idx} legal={r.legal} tps={r.tokens_per_s_per_gpu} mem={r.memory_per_gpu_gb}",
             config=cfg.as_dict(), result=out)
        return out

    def evaluate_simulate(config_json: str) -> str:
        """Evaluate a configuration using the analytical Origami+SDPA simulator
        (no GPU). Use this as the primary search tool.

        Args:
            config_json: JSON object with keys tp, pp, ep, cp, mbs, gbs,
                vpp (optional), pp_schedule (optional), recompute_granularity
                (optional 'none'|'selective'|'full'), recompute_num_layers,
                overlap_grad_reduce.

        Returns:
            JSON string with the trial metrics. Always check `legal`; if False,
            see `reason` and adjust the next proposal accordingly.
        """
        return json.dumps(_evaluate(config_json, "simulate"))

    def evaluate_memory_only(config_json: str) -> str:
        """Cheap pre-filter: only runs `projection memory` (no performance).
        Use this when you suspect a config may OOM and you want to confirm
        before paying for a perf call.
        """
        return json.dumps(_evaluate(config_json, "memory_only"))

    def evaluate_with_benchmark(config_json: str) -> str:
        """Run the hybrid benchmark (`projection performance`, no
        --profiling-mode override). Requires has_gpu=True. Reserve for the
        top-k candidates because it is much slower than simulate.
        """
        if not agent_cfg.benchmark_host.has_gpu:
            return json.dumps({"error": "has_gpu=false; benchmark not available"})
        return json.dumps(_evaluate(config_json, "benchmark"))

    # -----------------------------------------------------------------
    # Inspection tools
    # -----------------------------------------------------------------
    def get_history(k: int = 30) -> str:
        """Return the last k trials as a compact, line-per-trial table."""
        return history.summary_for_llm(k=k)

    def get_best() -> str:
        """Return the current incumbent (best legal trial) as JSON."""
        b = history.best(agent_cfg.optimization.objective)
        if b is None:
            return json.dumps({"none": True})
        return json.dumps({
            "idx": b.idx,
            "config": b.config,
            "result": b.result,
        })

    def get_legal_axes() -> str:
        """Per-axis legal value sets for this (model, cluster)."""
        return json.dumps(legality.to_prompt_dict())

    def get_architecture() -> str:
        """Resolved model architecture record."""
        return json.dumps(arch.as_prompt_dict())

    def get_cluster() -> str:
        """Target cluster spec."""
        return json.dumps({
            "name": agent_cfg.target_cluster.name,
            "num_nodes": agent_cfg.target_cluster.num_nodes,
            "gpus_per_node": agent_cfg.target_cluster.gpus_per_node,
            "gpu_arch": agent_cfg.target_cluster.gpu_arch,
            "world_size": agent_cfg.target_cluster.num_nodes * agent_cfg.target_cluster.gpus_per_node,
            "hbm_capacity_gb": agent_cfg.optimization.hbm_capacity_gb,
            "memory_safety_margin": agent_cfg.optimization.memory_safety_margin,
            "has_gpu_for_benchmark": agent_cfg.benchmark_host.has_gpu,
            "benchmark_gpus": agent_cfg.benchmark_host.benchmark_gpus,
        })

    def get_budget_status() -> str:
        """Remaining call budget."""
        return json.dumps({
            "simulate_used": evaluator.n_simulate_calls,
            "simulate_max": budget.max_perf_calls,
            "benchmark_used": evaluator.n_benchmark_calls,
            "benchmark_max": budget.max_benchmark_calls,
            "memory_only_used": evaluator.n_memory_calls,
            "trials_total": len(history.trials),
        })

    # -----------------------------------------------------------------
    # Scratchpad tools
    # -----------------------------------------------------------------
    def note_to_scratchpad(note: str) -> str:
        """Write a free-form note (plan, hypothesis, observation) to the
        durable scratchpad. The scratchpad survives across iterations and
        rounds and is included in every prompt."""
        return scratchpad.append(note)

    def read_scratchpad() -> str:
        """Read the current scratchpad contents."""
        return scratchpad.read() or "(empty)"

    # -----------------------------------------------------------------
    # Sub-LLM tool — extra "LLM-inside-the-LLM" consultation
    # -----------------------------------------------------------------
    def query_llm(prompt: str, system: str = "") -> str:
        """Ask a one-shot question to a fresh LLM call (no tools, no history
        cluttering its context). Useful for quick architectural reasoning,
        sanity checks, or summaries. The fresh call uses the same backing
        model as the agent.
        """
        try:
            import dspy
            lm = dspy.settings.lm
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            out = lm(messages=messages)
            if isinstance(out, list) and out:
                first = out[0]
                if isinstance(first, dict):
                    return first.get("text", "") or json.dumps(first)
                return str(first)
            return str(out)
        except Exception as e:
            return f"ERROR: query_llm failed: {e}"

    return [
        evaluate_simulate,
        evaluate_memory_only,
        evaluate_with_benchmark,
        get_history,
        get_best,
        get_legal_axes,
        get_architecture,
        get_cluster,
        get_budget_status,
        note_to_scratchpad,
        read_scratchpad,
        query_llm,
    ]
