"""DSPy planner + RLM driver for the tuning agent.

Two-stage flow:

  1. Planner (ChainOfThought) — produces an investigation plan and the
     initial set of high-priority axes / hypotheses, given the architecture,
     cluster, axis legality, and prior history.
  2. RLM — actually exercises the tool belt: evaluates configs, reads the
     scratchpad, queries a sub-LLM, etc.

LLM configuration uses DSPy's built-in LiteLLM integration — no external
proxy is required. Set the model string and credentials via environment
variables (see config.py) or via the target-cluster YAML ``agent.llm``
section.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import dspy

from .config import AgentConfig
from .evaluator import Evaluator
from .history import History
from .legality import derive_legality
from .scratchpad import Scratchpad
from .tools import build_tools
from .workload import ArchitectureRecord

# ---------------------------------------------------------------------------
# DSPy LM configuration
# ---------------------------------------------------------------------------


def configure_dspy(agent_cfg: AgentConfig) -> None:
    """Configure the global DSPy LM from AgentConfig.

    DSPy uses LiteLLM under the hood, so ``model`` follows LiteLLM's
    provider-prefixed naming convention (e.g. ``openai/gpt-4o``,
    ``anthropic/claude-opus-4-5``, ``ollama/llama3``).  Any provider
    supported by LiteLLM works without an additional proxy.

    Credentials are taken from the environment as usual for each provider
    (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, etc.) or from the
    ``api_key`` / ``base_url`` fields in the target-cluster YAML.
    """
    model = agent_cfg.llm.model
    api_key = agent_cfg.llm.api_key or None
    base_url = agent_cfg.llm.base_url or None

    lm_kwargs: dict = {
        "cache": False,
        "timeout": agent_cfg.llm.timeout,
        "max_tokens": agent_cfg.llm.max_tokens,
    }
    if api_key:
        lm_kwargs["api_key"] = api_key
    if base_url:
        lm_kwargs["api_base"] = base_url

    # ChatAdapter avoids JSONAdapter failures when the model wraps output in ```json.
    lm = dspy.LM(model, **lm_kwargs)
    try:
        dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
    except TypeError:
        dspy.configure(lm=lm)


def patch_dspy_python_interpreter(workspace: Path) -> None:
    """Mirror iterative_fix: ensure DSPy's deno-based PythonInterpreter works
    behind a corporate TLS-inspecting network and with a deno installed in
    ~/.deno/bin. No-op if deno is on PATH and no interpreter exists.
    """
    try:
        import dspy.primitives.python_interpreter as _pi
    except ImportError:
        return  # older dspy, no REPL to patch

    if shutil.which("deno") is None:
        candidates = [
            Path.home() / ".deno" / "bin",
            Path("/usr/local/bin"),
        ]
        for c in candidates:
            if (c / "deno").exists():
                os.environ["PATH"] = f"{c}{os.pathsep}{os.environ.get('PATH', '')}"
                break
        else:
            print("[tuning-agent] WARNING: deno not found; dspy.RLM REPL may fail", file=sys.stderr)
            return

    # Some hosts (e.g. this WSL setup) have a root-owned ~/.cache, which makes
    # Deno's default cache unwritable. Always route DENO_DIR to a per-run dir
    # under the workspace so RLM's PythonInterpreter can fetch its runner.
    deno_dir = workspace / ".deno_cache"
    deno_dir.mkdir(parents=True, exist_ok=True)
    os.environ["DENO_DIR"] = str(deno_dir)

    importmap = workspace / "deno_importmap.json"
    importmap.write_text(
        json.dumps(
            {
                "imports": {
                    "https://deno.land/std@0.186.0/": "https://raw.githubusercontent.com/denoland/std/0.186.0/"
                }
            }
        )
    )
    runner = Path(_pi.__file__).parent / "runner.js"
    custom_cmd = [
        "deno",
        "run",
        "--unsafely-ignore-certificate-errors",
        f"--import-map={importmap}",
        "--allow-all",
        str(runner),
    ]
    _orig_init = _pi.PythonInterpreter.__init__

    def _patched_init(self, deno_command=None, **kw):
        _orig_init(self, deno_command=deno_command or custom_cmd, **kw)

    _pi.PythonInterpreter.__init__ = _patched_init  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# DSPy signatures
# ---------------------------------------------------------------------------


class TuningPlanSignature(dspy.Signature):
    """Before touching any tools, design a structured search plan for tuning a
    distributed-training configuration on a target cluster.

    You receive:
      - architecture: model fields (num_layers, hidden_size, MoE config, …)
      - cluster: target cluster spec
      - legal_axes: per-axis legal value sets
      - history_summary: results of seed evaluations already executed
      - extra_guidance: free-form notes from the user
      - scratchpad: durable notes carried across rounds

    Produce a concrete, numbered plan: which axes are most likely to matter
    for this (model, cluster), which configurations should be tried in what
    order, and what to do if they fail. Account for memory headroom,
    intra/inter-node boundaries, and known trade-offs (e.g. MoE topk
    dominating activations, EP-spans-nodes hurting A2A).
    """

    architecture: str = dspy.InputField(desc="JSON architecture record")
    cluster: str = dspy.InputField(desc="JSON target cluster spec")
    legal_axes: str = dspy.InputField(desc="JSON per-axis legal sets")
    history_summary: str = dspy.InputField(desc="Compact text summary of prior trials")
    extra_guidance: str = dspy.InputField(desc="User-provided guidance (may be empty)")
    scratchpad: str = dspy.InputField(desc="Durable notes from previous rounds (may be empty)")

    rationale: str = dspy.OutputField(
        desc="2-4 sentences: which axes matter most for THIS (model, cluster) and why."
    )
    hypotheses: str = dspy.OutputField(
        desc="Numbered list (max 5) of testable hypotheses about high-throughput configs."
    )
    candidate_configs: str = dspy.OutputField(
        desc=(
            "JSON array of 5-8 candidate configurations to evaluate next (compact). "
            "Each must be a JSON object whose keys are a subset of: "
            "tp, pp, ep, cp, mbs, gbs, vpp, pp_schedule, enable_zero_bubble, "
            "recompute_granularity, recompute_num_layers, overlap_grad_reduce, "
            "use_turbo_deepep, sync_free_stage, target_ep_size, fp8, "
            "cross_entropy_loss_fusion, use_distributed_optimizer, use_torch_fsdp2. "
            "Per the Primus Projection skill (.skills/primus-projection.md), "
            "prioritise in this order for MoE: "
            "(1) DeepEP (use_turbo_deepep=true) — biggest single win (~35%), "
            "(2) SyncFree stage 2 then 3 (auto-enables DeepEP), "
            "(3) FP8 hybrid — ~2x compute on linear layers, "
            "(4) recompute=none/selective (only if memory fits), "
            "(5) ZB / ZBV pipeline schedules (with matching VPP), "
            "(6) coarse parallelism reshapes (TP/PP/EP/CP). "
            "Order from highest expected throughput to lowest."
        )
    )
    polish_plan: str = dspy.OutputField(
        desc=(
            "One short paragraph (2-4 sentences): polish-pass axes to sweep on top-k. "
            "Plain text only, no markdown fences."
        )
    )


class TuningSearchSignature(dspy.Signature):
    """Drive an iterative search for the best parallelization configuration
    of a distributed training workload, using the available tools.

    Tools available (call them from Python):

      evaluate_simulate(config_json) -> JSON metrics
          Primary tool. Always use this first for new candidates.
      evaluate_memory_only(config_json) -> JSON metrics
          Cheap pre-filter when you suspect OOM.
      evaluate_with_benchmark(config_json) -> JSON metrics
          ONLY if has_gpu_for_benchmark=True. Reserve for the top-k.
      get_history(k=30) -> str
          Compact table of past trials. ALWAYS read at least once per round.
      get_best() -> JSON
          The current incumbent.
      get_legal_axes() -> JSON
          Per-axis legal sets. Configurations outside these are auto-rejected.
      get_architecture() -> JSON
          Model fields (num_layers, hidden_size, num_experts, topk, …).
      get_cluster() -> JSON
          Target cluster (num_nodes, gpus_per_node, gpu_arch, hbm_capacity_gb).
      get_budget_status() -> JSON
          Tells you how many calls remain.
      note_to_scratchpad(text) -> str
          Write a durable note; survives across rounds. Use it to record
          "tried EP=16 across 2 nodes — A2A dominates" so you don't repeat.
      read_scratchpad() -> str
          Read all notes carried over.
      query_llm(prompt, system?) -> str
          Ask a fresh, tool-less LLM a focused question (e.g. "given this
          MoE shape, would CP help?"). Use sparingly.

    WORKFLOW for each iteration:

      1. read_scratchpad() and get_history() to ground yourself.
      2. Look at the candidate_configs from the plan; pick the next one
         that is NOT already in history.
      3. evaluate_simulate(config_json). If `legal=False`, read the
         `reason` and adjust — never propose the same illegal pattern twice.
      4. After 3-5 evaluations, write a short note_to_scratchpad summarising
         what you have learned (e.g. "MBS=4 OOMs at TP=1; need recompute or TP≥2").
      5. When simulate budget is mostly spent, propose a *polish pass* on the
         top-k (vary MBS, recompute, schedule). If has_gpu, validate the top-1
         with evaluate_with_benchmark.
      6. STOP when get_budget_status reports the simulate budget is exhausted
         OR you have evaluated >= max_proposals candidates without improving
         the incumbent in the last 5 trials.

    Output:
      - best_config: the winning trial's JSON config
      - summary: 5-8 sentences on what worked, what didn't, and confidence
      - next_steps: optional advice for a follow-up round
    """

    plan: str = dspy.InputField(desc="The investigation plan from the planner stage")
    architecture: str = dspy.InputField(desc="JSON architecture record")
    cluster: str = dspy.InputField(desc="JSON target cluster spec")
    legal_axes: str = dspy.InputField(desc="JSON per-axis legal sets")
    history_summary: str = dspy.InputField(desc="Compact text summary of prior trials")
    extra_guidance: str = dspy.InputField(desc="User-provided guidance")

    best_config: str = dspy.OutputField(desc="JSON object: the winning trial config (or empty {} if none).")
    summary: str = dspy.OutputField(
        desc="5-8 sentences on the search: what was tried, what worked, what didn't."
    )
    next_steps: str = dspy.OutputField(
        desc="Suggested follow-up runs / axes to explore further. May be empty."
    )


# ---------------------------------------------------------------------------
# Progress callback (mirrors iterative_fix._RLMProgressCallback)
# ---------------------------------------------------------------------------


class _RLMProgressCallback(dspy.utils.callback.BaseCallback):
    def __init__(self, session_log: list[dict]):
        self._iter = 0
        self._session_log = session_log

    def on_lm_end(self, call_id, outputs, exception=None):  # noqa: ARG002
        if exception or not outputs:
            return
        try:
            text = ""
            if isinstance(outputs, list) and outputs:
                item = outputs[0]
                if isinstance(item, dict):
                    text = item.get("text", "") or ""
                elif isinstance(item, str):
                    text = item
            elif isinstance(outputs, str):
                text = outputs
            if text and len(text.strip()) > 20:
                self._iter += 1
                preview = "\n    │  ".join(text.strip().splitlines()[:20])
                print(f"\n    [RLM iter {self._iter}] LM output:\n    │  {preview}\n")
                self._session_log.append(
                    {
                        "ts": datetime.now().isoformat(),
                        "kind": "lm_output",
                        "iter": self._iter,
                        "text": text[:3000],
                        "text_len": len(text),
                    }
                )
        except Exception as e:
            # Best-effort callback: never break agent execution on progress/log formatting failures.
            self._session_log.append(
                {
                    "ts": datetime.now().isoformat(),
                    "kind": "lm_output_callback_error",
                    "error": str(e),
                }
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_agent(
    agent_cfg: AgentConfig,
    arch: ArchitectureRecord,
    history: History,
    evaluator: Evaluator,
    scratchpad: Scratchpad,
    workspace: Path,
) -> dict:
    """Run one full agent loop (plan + RLM rounds). Returns a dict summary."""
    configure_dspy(agent_cfg)
    patch_dspy_python_interpreter(workspace)

    legality = derive_legality(arch, agent_cfg.target_cluster)
    session_log: list[dict] = []
    cb = _RLMProgressCallback(session_log)
    dspy.settings.callbacks.append(cb)
    try:
        # ── planner ──────────────────────────────────────────────────
        planner = dspy.Predict(TuningPlanSignature)
        plan_inputs = dict(
            architecture=json.dumps(arch.as_prompt_dict()),
            cluster=_cluster_blob(agent_cfg),
            legal_axes=json.dumps(legality.to_prompt_dict()),
            history_summary=history.summary_for_llm(k=30),
            extra_guidance=agent_cfg.extra_prompt or "(none)",
            scratchpad=scratchpad.read() or "(empty)",
        )
        try:
            plan = planner(**plan_inputs)
        except Exception as e:
            print(
                f"[tuning-agent] planner parse failed ({e}); retrying with ChatAdapter fallback",
                file=sys.stderr,
            )
            dspy.configure(adapter=dspy.ChatAdapter())
            plan = planner(**plan_inputs)
        if not getattr(plan, "polish_plan", None):
            plan.polish_plan = "(none)"
        plan_blob = (
            f"RATIONALE:\n{plan.rationale}\n\n"
            f"HYPOTHESES:\n{plan.hypotheses}\n\n"
            f"CANDIDATE CONFIGS:\n{plan.candidate_configs}\n\n"
            f"POLISH PLAN:\n{plan.polish_plan}\n"
        )
        scratchpad.append("PLAN: " + plan.rationale[:400])
        print("\n=== INVESTIGATION PLAN ===")
        print(plan_blob)

        # ── RLM rounds ───────────────────────────────────────────────
        tools = build_tools(
            agent_cfg=agent_cfg,
            arch=arch,
            legality=legality,
            history=history,
            evaluator=evaluator,
            scratchpad=scratchpad,
            session_log=session_log,
        )

        budget = agent_cfg.optimization.budget
        rounds_run = 0
        last_best_idx: int | None = None
        for round_idx in range(budget.max_rounds):
            print(f"\n=== ROUND {round_idx + 1}/{budget.max_rounds} ===")
            try:
                rlm = dspy.RLM(
                    TuningSearchSignature,
                    tools=tools,
                    max_iterations=budget.max_rlm_iterations,
                )
                result = rlm(
                    plan=plan_blob,
                    architecture=json.dumps(arch.as_prompt_dict()),
                    cluster=_cluster_blob(agent_cfg),
                    legal_axes=json.dumps(legality.to_prompt_dict()),
                    history_summary=history.summary_for_llm(k=50),
                    extra_guidance=agent_cfg.extra_prompt or "(none)",
                )
            except Exception as e:
                print(f"    RLM error in round {round_idx + 1}: {e}", file=sys.stderr)
                break

            rounds_run += 1
            best = history.best(agent_cfg.optimization.objective)
            cur_best_idx = best.idx if best else None
            print(
                f"\n    [round summary] best so far: trial #{cur_best_idx} "
                f"tps={best.result.get('tokens_per_s_per_gpu') if best else None}"
            )

            # early stop if no improvement and budget mostly used
            if last_best_idx == cur_best_idx and evaluator.n_simulate_calls >= int(
                budget.max_perf_calls * 0.7
            ):
                print("    [round summary] no improvement and budget mostly spent — stopping.")
                break
            last_best_idx = cur_best_idx

            if evaluator.n_simulate_calls >= budget.max_perf_calls:
                print("    [round summary] simulate budget exhausted — stopping.")
                break

        best = history.best(agent_cfg.optimization.objective)
        return {
            "best": (best.as_dict() if best else None),
            "rounds_run": rounds_run,
            "session_log_len": len(session_log),
            "trials_total": len(history.trials),
        }
    finally:
        try:
            dspy.settings.callbacks.remove(cb)
        except ValueError:
            pass


def _cluster_blob(agent_cfg: AgentConfig) -> str:
    return json.dumps(
        {
            "name": agent_cfg.target_cluster.name,
            "num_nodes": agent_cfg.target_cluster.num_nodes,
            "gpus_per_node": agent_cfg.target_cluster.gpus_per_node,
            "world_size": agent_cfg.target_cluster.num_nodes * agent_cfg.target_cluster.gpus_per_node,
            "gpu_arch": agent_cfg.target_cluster.gpu_arch,
            "hbm_capacity_gb": agent_cfg.optimization.hbm_capacity_gb,
            "memory_safety_margin": agent_cfg.optimization.memory_safety_margin,
            "has_gpu_for_benchmark": agent_cfg.benchmark_host.has_gpu,
            "benchmark_gpus": agent_cfg.benchmark_host.benchmark_gpus,
        }
    )
