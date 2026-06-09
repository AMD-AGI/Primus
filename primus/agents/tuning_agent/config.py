"""Agent configuration: .env, target_cluster.yaml, optimization knobs.

LLM credentials are resolved from standard provider environment variables,
exactly as DSPy/LiteLLM expect them.  Common examples:

    # OpenAI
    OPENAI_API_KEY=sk-...

    # Anthropic
    ANTHROPIC_API_KEY=sk-ant-...

    # Any OpenAI-compatible endpoint (local Ollama, vLLM, LiteLLM proxy, …)
    OPENAI_API_KEY=<key-or-dummy>
    OPENAI_API_BASE=http://localhost:11434/v1

    # Override the model from the environment (takes precedence over YAML)
    LLM_MODEL=openai/gpt-4o

The ``model`` field follows LiteLLM's provider-prefixed convention:
``openai/gpt-4o``, ``anthropic/claude-opus-4-5``, ``ollama/llama3``, etc.
See https://docs.litellm.ai/docs/providers for the full list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]


DEFAULT_MODEL = "openai/gpt-4o"


def _resolve_api_key() -> str:
    """Return the first populated credential env var, in priority order."""
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_API_KEY"):
        val = os.environ.get(key)
        if val:
            return val
    return ""


_ENV_CANDIDATES = (
    Path.cwd() / ".env",
    Path(__file__).resolve().parents[3] / ".env",  # repo root
    Path.home() / ".env",
)


def load_env(extra: Path | None = None) -> Path | None:
    """Load .env from the first candidate that exists. Returns the loaded path."""
    if load_dotenv is None:
        return None
    candidates = list(_ENV_CANDIDATES)
    if extra:
        candidates.insert(0, extra)
    for cand in candidates:
        if cand.is_file():
            load_dotenv(cand, override=False)
            return cand
    return None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TargetCluster:
    name: str = "unnamed"
    num_nodes: int = 1
    gpus_per_node: int = 8
    gpu_arch: str = "mi355x"
    hardware_config: str | None = None  # path to existing primus hardware yaml
    gpu_clock_mhz: int | None = None


@dataclass
class BenchmarkHost:
    has_gpu: bool = False
    benchmark_gpus: int = 0
    benchmark_arch: str | None = None


@dataclass
class Budget:
    max_proposals: int = 30  # LLM-proposed candidates
    max_perf_calls: int = 30  # simulate calls
    max_benchmark_calls: int = 0  # GPU benchmark calls
    max_rounds: int = 8  # outer agent rounds
    max_rlm_iterations: int = 30  # inner RLM iterations per round


@dataclass
class OptimizationConfig:
    objective: str = "tokens_per_s_per_gpu"
    memory_safety_margin: float = 0.10
    hbm_capacity_gb: float = 192.0  # MI300X/MI325X/MI355X = 192/256/288
    budget: Budget = field(default_factory=Budget)
    axes: dict[str, bool] = field(
        default_factory=lambda: {
            "tp": True,
            "pp": True,
            "ep": True,
            "cp": True,
            "mbs": True,
            "gbs": False,
            "vpp": True,
            "pp_schedule": True,
            "recompute": True,
            "overlap_grad_reduce": False,
        }
    )


@dataclass
class LLMConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = DEFAULT_MODEL
    timeout: int = 300
    max_tokens: int = 16000


@dataclass
class AgentConfig:
    target_cluster: TargetCluster
    benchmark_host: BenchmarkHost
    optimization: OptimizationConfig
    llm: LLMConfig
    out_dir: Path = field(default_factory=lambda: Path("./tuning_runs"))
    workload_yaml: Path = field(default_factory=lambda: Path("."))
    extra_prompt: str = ""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _from_env(key: str, default: Any = None) -> Any:
    val = os.environ.get(key)
    return val if val not in (None, "") else default


def load_config(target_cluster_yaml: Path, workload_yaml: Path, out_dir: Path | None = None) -> AgentConfig:
    """Build the agent config from a target-cluster YAML and the env."""
    load_env()

    raw = yaml.safe_load(target_cluster_yaml.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{target_cluster_yaml} is not a mapping")

    tc_raw = raw.get("target_cluster", {})
    bh_raw = raw.get("available_for_benchmark", {})
    opt_raw = raw.get("optimization", {})
    agent_raw = raw.get("agent", {})
    llm_raw = (agent_raw.get("llm") or {}) if isinstance(agent_raw, dict) else {}

    target_cluster = TargetCluster(
        name=tc_raw.get("name", "unnamed"),
        num_nodes=int(tc_raw.get("num_nodes", 1)),
        gpus_per_node=int(tc_raw.get("gpus_per_node", 8)),
        gpu_arch=str(tc_raw.get("gpu_arch", "mi355x")),
        hardware_config=tc_raw.get("hardware_config"),
        gpu_clock_mhz=tc_raw.get("gpu_clock_mhz"),
    )

    benchmark_host = BenchmarkHost(
        has_gpu=bool(bh_raw.get("has_gpu", False)),
        benchmark_gpus=int(bh_raw.get("benchmark_gpus", 0)),
        benchmark_arch=bh_raw.get("benchmark_arch"),
    )

    budget_raw = (opt_raw.get("budget") or {}) if isinstance(opt_raw, dict) else {}
    optimization = OptimizationConfig(
        objective=opt_raw.get("objective", "tokens_per_s_per_gpu"),
        memory_safety_margin=float(opt_raw.get("memory_safety_margin", 0.10)),
        hbm_capacity_gb=float(opt_raw.get("hbm_capacity_gb", 192.0)),
        budget=Budget(
            max_proposals=int(budget_raw.get("max_proposals", 30)),
            max_perf_calls=int(budget_raw.get("max_perf_calls", 30)),
            max_benchmark_calls=int(budget_raw.get("max_benchmark_calls", 0)),
            max_rounds=int(budget_raw.get("max_rounds", 8)),
            max_rlm_iterations=int(budget_raw.get("max_rlm_iterations", 30)),
        ),
        axes={**OptimizationConfig().axes, **(opt_raw.get("axes") or {})},
    )

    llm = LLMConfig(
        base_url=str(llm_raw.get("base_url") or _from_env("OPENAI_API_BASE") or ""),
        api_key=str(llm_raw.get("api_key") or _resolve_api_key()),
        model=str(llm_raw.get("model") or _from_env("LLM_MODEL", DEFAULT_MODEL)),
        timeout=int(llm_raw.get("timeout", 300)),
        max_tokens=int(llm_raw.get("max_tokens", 16000)),
    )

    return AgentConfig(
        target_cluster=target_cluster,
        benchmark_host=benchmark_host,
        optimization=optimization,
        llm=llm,
        out_dir=Path(out_dir) if out_dir else Path("./tuning_runs") / target_cluster.name,
        workload_yaml=Path(workload_yaml).resolve(),
        extra_prompt="\n".join(agent_raw.get("prompt_extras") or []) if isinstance(agent_raw, dict) else "",
    )
