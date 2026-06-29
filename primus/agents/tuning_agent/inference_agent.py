"""DSPy planner + RLM driver for the *inference / serving* tuning agent.

This is the serving-mode counterpart to ``agent.run_agent`` (which tunes
distributed *training*).  It drives an LLM-guided search over the serving
search space defined in ``inference_tuning.py`` — request/batching profile,
serving parallelism (TP/EP/PP/CP), KV/weight quantization, chunked prefill,
speculative decoding (+ draft cost), CUDA-graph capture, paged-KV block size,
scheduler token budget, MoE routing imbalance / DeepEP, prefill/decode
disaggregation, offered-load (request-rate) queueing, kernel-backend selection,
native sparse attention, MoE expert dtype, fused kernels, and custom collective
ops (quick-reduce / fused RMSNorm+AllReduce).

Two-stage flow mirrors training:

  1. Planner — produces an investigation plan + initial candidate serving
     configs, given the architecture, cluster, legal axes, objective, and the
     seed-sweep history.
  2. RLM — exercises the tool belt: proposes configs, evaluates them via
     ``projection inference`` (analytical, no GPU), reads history/scratchpad,
     and converges on the best config for the configured objective.

The seed sweep (``inference_tuning.build_inference_seed_plan``) is still run
first by the CLI as a warm start; this module continues the search with the
LLM, sharing the same ``History`` + ``Evaluator``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import dspy

from .agent import (
    _RLMProgressCallback,
    configure_dspy,
    patch_dspy_python_interpreter,
)
from .config import AgentConfig
from .evaluator import EvalResult, Evaluator
from .history import History
from .inference_tuning import (
    derive_inference_legality,
    inference_trial_from_dict,
    objective_is_minimize,
    resolve_objective,
    score_result,
    validate_inference,
)
from .scratchpad import Scratchpad
from .workload import ArchitectureRecord

# ---------------------------------------------------------------------------
# DSPy signatures
# ---------------------------------------------------------------------------

# The set of keys the LLM may put in a proposal (a subset is fine — unspecified
# fields inherit the profile-anchored baseline). Kept in one place so the plan
# and search signatures stay in sync with InferenceTrialConfig.
_PROPOSAL_KEYS = (
    "tp, pp, ep, cp, batch_size, input_len, output_len, max_concurrency, "
    "weight_dtype, kv_cache_dtype, chunked_prefill_size, speculative_num_tokens, "
    "speculative_acceptance_rate, speculative_draft_cost_factor, tp_allreduce_algo, "
    "ep_a2a_algo, use_turbo_deepep, cudagraph_mode, kv_cache_memory_fraction, "
    "kv_block_size, max_num_batched_tokens, ep_load_balance, redundant_experts, "
    "disaggregate, prefill_tp, decode_tp, decode_replicas, transfer_backend, "
    "request_rate, arrival_model, attention_backend, sparse_attention_topk, "
    "moe_expert_dtype, fused_kernels, quick_reduce, fuse_rmsnorm_allreduce"
)


class InferenceTuningPlanSignature(dspy.Signature):
    """Before touching any tools, design a structured search plan for tuning an
    LLM *inference / serving* configuration on a target cluster.

    You receive:
      - architecture: model fields (num_layers, hidden_size, MoE config, …)
      - cluster: target cluster spec (nodes, GPUs/node, HBM)
      - legal_axes: per-axis legal value sets for the serving knobs
      - objective: the metric to optimize (and its direction)
      - history_summary: results of the seed sweep already executed
      - extra_guidance: free-form notes from the user
      - scratchpad: durable notes carried across rounds

    Produce a concrete, numbered plan: which serving knobs are most likely to
    move the objective for THIS (model, cluster, request profile), in what
    order to try them, and what to do if they fail. Account for: the
    latency↔throughput trade-off (batch_size / max_concurrency / request_rate),
    KV-cache capacity (kv_cache_dtype, kv_cache_memory_fraction, kv_block_size),
    prefill cost (chunked_prefill_size, max_num_batched_tokens), MoE comm
    (ep, use_turbo_deepep, ep_load_balance), disaggregation, and the ROCm
    kernel knobs (attention_backend, moe_expert_dtype, fused_kernels,
    quick_reduce, fuse_rmsnorm_allreduce, sparse_attention_topk).
    """

    architecture: str = dspy.InputField(desc="JSON architecture record")
    cluster: str = dspy.InputField(desc="JSON target cluster spec")
    legal_axes: str = dspy.InputField(desc="JSON per-axis legal sets for serving knobs")
    objective: str = dspy.InputField(desc="Objective metric + direction (max/min)")
    history_summary: str = dspy.InputField(desc="Compact text summary of seed trials")
    extra_guidance: str = dspy.InputField(desc="User-provided guidance (may be empty)")
    scratchpad: str = dspy.InputField(desc="Durable notes from previous rounds (may be empty)")

    rationale: str = dspy.OutputField(
        desc="2-4 sentences: which serving knobs matter most for THIS objective and why."
    )
    hypotheses: str = dspy.OutputField(
        desc="Numbered list (max 5) of testable hypotheses about the best serving config."
    )
    candidate_configs: str = dspy.OutputField(
        desc=(
            "JSON array of 5-8 candidate serving configurations to evaluate next "
            "(compact). Each is a JSON object whose keys are a subset of: "
            f"{_PROPOSAL_KEYS}. Unspecified keys inherit the profile baseline. "
            "Order from most to least promising for the objective."
        )
    )
    polish_plan: str = dspy.OutputField(
        desc=(
            "One short paragraph (2-4 sentences): polish-pass knobs to sweep on the "
            "top candidates. Plain text only, no markdown fences."
        )
    )


class InferenceSearchSignature(dspy.Signature):
    """Drive an iterative search for the best LLM serving configuration of a
    workload — request/batching profile, serving parallelism, KV/weight
    quantization, chunked prefill, speculative decoding, CUDA-graph capture,
    scheduler/KV knobs, disaggregation, offered-load, and ROCm kernel knobs —
    using the available tools, optimizing the given objective.

    Tools available (call them from Python):

      evaluate_inference(config_json) -> JSON metrics
          Primary tool. Runs `projection inference` (analytical, no GPU) and
          returns TTFT/ITL/throughput/KV/memory + a signed `score` (higher is
          always better). Always check `legal`; if False, read `reason` and
          adjust — never repeat the same illegal pattern.
      get_history(k=30) -> str
          Compact table of past serving trials. Read at least once per round.
      get_best() -> JSON
          The current incumbent for the objective.
      get_legal_axes() -> JSON
          Per-axis legal sets; proposals outside these are auto-rejected.
      get_architecture() -> JSON
      get_cluster() -> JSON
      get_objective() -> JSON     (metric + direction)
      get_budget_status() -> JSON (how many evaluate calls remain)
      note_to_scratchpad(text) -> str / read_scratchpad() -> str
      query_llm(prompt, system?) -> str

    WORKFLOW per iteration:
      1. read_scratchpad() + get_history() to ground yourself.
      2. Pick the next promising config NOT already in history (you may specify
         only the knobs you want to change; the rest inherit the baseline).
      3. evaluate_inference(config_json). If illegal, read `reason` and fix.
      4. Every 3-5 evals, note_to_scratchpad() what you learned.
      5. STOP when get_budget_status says the budget is exhausted, or you have
         stopped improving the incumbent.

    Output:
      - best_config: the winning trial's JSON config (or {} if none)
      - summary: 5-8 sentences on what worked / didn't
      - next_steps: optional follow-up advice
    """

    plan: str = dspy.InputField(desc="The investigation plan from the planner stage")
    architecture: str = dspy.InputField(desc="JSON architecture record")
    cluster: str = dspy.InputField(desc="JSON target cluster spec")
    legal_axes: str = dspy.InputField(desc="JSON per-axis legal sets")
    objective: str = dspy.InputField(desc="Objective metric + direction")
    history_summary: str = dspy.InputField(desc="Compact text summary of prior trials")
    extra_guidance: str = dspy.InputField(desc="User-provided guidance")

    best_config: str = dspy.OutputField(desc="JSON object: winning serving config (or {} if none).")
    summary: str = dspy.OutputField(
        desc="5-8 sentences on the search: what was tried, what worked, what didn't."
    )
    next_steps: str = dspy.OutputField(desc="Suggested follow-up runs / knobs. May be empty.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_load_json(s) -> dict | None:
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json\n"):
            s = s[5:]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _result_for_llm(r: EvalResult, score: float | None) -> dict:
    return {
        "legal": r.legal,
        "reason": r.reason,
        "score": score,
        "ttft_ms": r.ttft_ms,
        "itl_ms": r.itl_ms,
        "request_latency_ms": r.request_latency_ms,
        "decode_throughput_tps": r.decode_throughput_tps,
        "decode_throughput_tps_per_gpu": r.decode_throughput_tps_per_gpu,
        "prefill_throughput_tps": r.prefill_throughput_tps,
        "memory_per_gpu_gb": r.memory_per_gpu_gb,
        "kv_cache_gb": r.kv_cache_gb,
        "max_concurrent_sequences": r.max_concurrent_sequences,
        "config": r.config,
    }


def _best_by_score(history: History, objective: str):
    """Objective-aware incumbent (score_result handles minimize metrics)."""
    best = None
    best_sc = None
    for t in history.trials:
        if not t.result.get("legal"):
            continue
        sc = score_result(t.result, objective)
        if sc is None:
            continue
        if best_sc is None or sc > best_sc:
            best_sc, best = sc, t
    return best


def _inference_history_summary(history: History, objective: str, k: int = 30) -> str:
    rows = history.trials[-k:] if k else history.trials
    obj = resolve_objective(objective)
    lines = []
    for t in rows:
        r = t.result
        c = t.config

        def _g(key, fmt="{}"):
            v = r.get(key)
            return fmt.format(v) if isinstance(v, (int, float)) else "—"

        objval = r.get(obj)
        obj_s = f"{objval:,.2f}" if isinstance(objval, (int, float)) else "—"
        tag = "OK" if r.get("legal") else f"REJECT({str(r.get('reason',''))[:48]})"
        cfg_s = (
            f"tp{c.get('tp')} pp{c.get('pp')} ep{c.get('ep')} bs{c.get('batch_size')} "
            f"w{c.get('weight_dtype')} kv{c.get('kv_cache_dtype')}"
        )
        extra = []
        for key, short in (
            ("cudagraph_mode", "cg"),
            ("attention_backend", "attn"),
            ("moe_expert_dtype", "moe"),
            ("request_rate", "rate"),
            ("sparse_attention_topk", "spk"),
        ):
            v = c.get(key)
            if v:
                extra.append(f"{short}={v}")
        cfg_s = cfg_s + ((" " + " ".join(extra)) if extra else "")
        lines.append(
            f"#{t.idx:03d} {tag:38s} {obj}={obj_s:>10s} "
            f"ttft={_g('ttft_ms','{:.1f}')} itl={_g('itl_ms','{:.2f}')} "
            f"dtps/gpu={_g('decode_throughput_tps_per_gpu','{:.0f}')} "
            f"mem={_g('memory_per_gpu_gb','{:.1f}')}GB | {cfg_s}"
        )
    return "\n".join(lines) if lines else "(no trials yet)"


# ---------------------------------------------------------------------------
# Tool belt
# ---------------------------------------------------------------------------


def build_inference_tools(
    agent_cfg: AgentConfig,
    arch: ArchitectureRecord,
    legality,
    history: History,
    evaluator: Evaluator,
    scratchpad: Scratchpad,
    session_log: list[dict],
    objective: str,
    quiet: bool = False,
) -> list:
    """Tool list for the inference RLM. Shared mutable state lives on the
    closure so the caller can inspect it after the RLM finishes."""
    budget = agent_cfg.optimization.budget
    cluster = agent_cfg.target_cluster
    opt = agent_cfg.optimization

    def _log(kind: str, **payload):
        session_log.append({"ts": datetime.now().isoformat(), "kind": kind, **payload})
        if not quiet:
            print(f"    [tool:{kind}] {str(payload.get('summary', ''))[:160]}")

    def evaluate_inference(config_json: str) -> str:
        """Evaluate a serving config via `projection inference` (analytical).

        Args:
            config_json: JSON object; keys are a subset of the serving knobs
                (unspecified ones inherit the profile baseline).

        Returns:
            JSON string of metrics incl. a signed `score` (higher = better for
            the objective). Always check `legal`; on False, read `reason`.
        """
        d = _safe_load_json(config_json)
        if d is None:
            return json.dumps({"error": "config_json must be a JSON object"})
        cfg = inference_trial_from_dict(d, arch, cluster, opt)
        if history.already_evaluated(cfg.as_dict()):
            for t in reversed(history.trials):
                if t.config == cfg.as_dict():
                    return json.dumps({"already_evaluated": True, **t.result})
        if evaluator.n_simulate_calls >= budget.max_perf_calls:
            return json.dumps(
                {"error": f"eval budget exhausted ({evaluator.n_simulate_calls}/{budget.max_perf_calls})"}
            )
        idx = len(history.trials)
        tag = f"inf_{idx:03d}_tp{cfg.tp}_pp{cfg.pp}_ep{cfg.ep}_bs{cfg.batch_size}"
        r = evaluator.evaluate_inference(cfg, tag)
        history.add(cfg.as_dict(), r, notes="inference[agent]")
        score = score_result(r.as_dict(), objective) if r.legal else None
        out = _result_for_llm(r, score)
        _log(
            "evaluate_inference",
            summary=f"#{idx} legal={r.legal} score={score} ttft={r.ttft_ms} "
            f"dtps/gpu={r.decode_throughput_tps_per_gpu}",
            config=cfg.as_dict(),
            result=out,
        )
        return json.dumps(out, default=str)

    def get_history(k: int = 30) -> str:
        """Return the last k serving trials as a compact, line-per-trial table."""
        return _inference_history_summary(history, objective, k=k)

    def get_best() -> str:
        """Return the current incumbent (best legal trial for the objective)."""
        b = _best_by_score(history, objective)
        if b is None:
            return json.dumps({"none": True})
        return json.dumps({"idx": b.idx, "config": b.config, "result": b.result}, default=str)

    def get_legal_axes() -> str:
        """Per-axis legal value sets for the serving knobs."""
        return json.dumps(legality.to_prompt_dict())

    def get_architecture() -> str:
        """Resolved model architecture record."""
        return json.dumps(arch.as_prompt_dict())

    def get_cluster() -> str:
        """Target cluster spec."""
        return json.dumps(
            {
                "name": cluster.name,
                "num_nodes": cluster.num_nodes,
                "gpus_per_node": cluster.gpus_per_node,
                "gpu_arch": cluster.gpu_arch,
                "world_size": cluster.num_nodes * cluster.gpus_per_node,
                "hbm_capacity_gb": opt.hbm_capacity_gb,
                "memory_safety_margin": opt.memory_safety_margin,
            }
        )

    def get_objective() -> str:
        """The objective metric being optimized and its direction."""
        obj = resolve_objective(objective)
        return json.dumps(
            {"objective": obj, "direction": "minimize" if objective_is_minimize(obj) else "maximize"}
        )

    def get_budget_status() -> str:
        """Remaining evaluation budget."""
        return json.dumps(
            {
                "eval_used": evaluator.n_simulate_calls,
                "eval_max": budget.max_perf_calls,
                "trials_total": len(history.trials),
            }
        )

    def note_to_scratchpad(note: str) -> str:
        """Write a durable note (plan/hypothesis/observation). Survives rounds."""
        return scratchpad.append(note)

    def read_scratchpad() -> str:
        """Read the current scratchpad contents."""
        return scratchpad.read() or "(empty)"

    def query_llm(prompt: str, system: str = "") -> str:
        """Ask a one-shot question to a fresh, tool-less LLM call."""
        try:
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
        evaluate_inference,
        get_history,
        get_best,
        get_legal_axes,
        get_architecture,
        get_cluster,
        get_objective,
        get_budget_status,
        note_to_scratchpad,
        read_scratchpad,
        query_llm,
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_inference_agent(
    agent_cfg: AgentConfig,
    arch: ArchitectureRecord,
    history: History,
    evaluator: Evaluator,
    scratchpad: Scratchpad,
    workspace: Path,
    objective: str,
) -> dict:
    """Run one full inference-tuning agent loop (planner + RLM rounds).

    Shares ``history`` + ``evaluator`` with the seed sweep, so the LLM
    continues from the warm-started incumbent. Returns a dict summary.
    """
    configure_dspy(agent_cfg)
    patch_dspy_python_interpreter(workspace)

    legality = derive_inference_legality(arch, agent_cfg.target_cluster)
    obj_resolved = resolve_objective(objective)
    direction = "minimize" if objective_is_minimize(obj_resolved) else "maximize"
    obj_blob = f"{obj_resolved} ({direction})"

    session_log: list[dict] = []
    cb = _RLMProgressCallback(session_log)
    dspy.settings.callbacks.append(cb)
    try:
        # ── planner ──────────────────────────────────────────────────
        planner = dspy.Predict(InferenceTuningPlanSignature)
        plan_inputs = dict(
            architecture=json.dumps(arch.as_prompt_dict()),
            cluster=_cluster_blob(agent_cfg),
            legal_axes=json.dumps(legality.to_prompt_dict()),
            objective=obj_blob,
            history_summary=_inference_history_summary(history, objective, k=40),
            extra_guidance=agent_cfg.extra_prompt or "(none)",
            scratchpad=scratchpad.read() or "(empty)",
        )
        try:
            plan = planner(**plan_inputs)
        except Exception as e:
            print(
                f"[tuning-agent] inference planner parse failed ({e}); "
                f"retrying with ChatAdapter fallback",
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
        scratchpad.append("INFERENCE PLAN: " + str(plan.rationale)[:400])
        print("\n=== INFERENCE INVESTIGATION PLAN ===")
        print(plan_blob)

        # ── RLM rounds ───────────────────────────────────────────────
        tools = build_inference_tools(
            agent_cfg=agent_cfg,
            arch=arch,
            legality=legality,
            history=history,
            evaluator=evaluator,
            scratchpad=scratchpad,
            session_log=session_log,
            objective=objective,
        )

        budget = agent_cfg.optimization.budget
        rounds_run = 0
        last_best_idx: int | None = None
        for round_idx in range(budget.max_rounds):
            print(f"\n=== INFERENCE ROUND {round_idx + 1}/{budget.max_rounds} ===")
            try:
                rlm = dspy.RLM(
                    InferenceSearchSignature,
                    tools=tools,
                    max_iterations=budget.max_rlm_iterations,
                )
                rlm(
                    plan=plan_blob,
                    architecture=json.dumps(arch.as_prompt_dict()),
                    cluster=_cluster_blob(agent_cfg),
                    legal_axes=json.dumps(legality.to_prompt_dict()),
                    objective=obj_blob,
                    history_summary=_inference_history_summary(history, objective, k=50),
                    extra_guidance=agent_cfg.extra_prompt or "(none)",
                )
            except Exception as e:
                print(f"    RLM error in inference round {round_idx + 1}: {e}", file=sys.stderr)
                break

            rounds_run += 1
            best = _best_by_score(history, objective)
            cur_best_idx = best.idx if best else None
            best_val = best.result.get(obj_resolved) if best else None
            print(
                f"\n    [round summary] best so far: trial #{cur_best_idx} "
                f"{obj_resolved}={best_val}"
            )

            if last_best_idx == cur_best_idx and evaluator.n_simulate_calls >= int(
                budget.max_perf_calls * 0.7
            ):
                print("    [round summary] no improvement and budget mostly spent — stopping.")
                break
            last_best_idx = cur_best_idx
            if evaluator.n_simulate_calls >= budget.max_perf_calls:
                print("    [round summary] eval budget exhausted — stopping.")
                break

        best = _best_by_score(history, objective)
        return {
            "mode": "inference",
            "objective": obj_resolved,
            "direction": direction,
            "best": (best.as_dict() if best else None),
            "rounds_run": rounds_run,
            "session_log_len": len(session_log),
            "trials_total": len(history.trials),
        }
    finally:
        try:
            dspy.settings.callbacks.remove(cb)
        except (ValueError, AttributeError):
            pass


def _cluster_blob(agent_cfg: AgentConfig) -> str:
    c = agent_cfg.target_cluster
    return json.dumps(
        {
            "name": c.name,
            "num_nodes": c.num_nodes,
            "gpus_per_node": c.gpus_per_node,
            "gpu_arch": c.gpu_arch,
            "world_size": c.num_nodes * c.gpus_per_node,
            "hbm_capacity_gb": agent_cfg.optimization.hbm_capacity_gb,
        }
    )
