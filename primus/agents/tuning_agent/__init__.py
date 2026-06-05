"""Tuning agent: LLM-driven search for an optimal parallelization
configuration of a Primus workload on a target cluster.

The agent uses the Primus Projection tool as an oracle (memory + simulate +
optional benchmark) and a DSPy.RLM loop with planner + scratchpad + history
to navigate the space efficiently.

Entry point:
    python -m primus.agents.tuning_agent --workload <yaml> --target-cluster <yaml>
"""

from .config import AgentConfig, TargetCluster, OptimizationConfig, load_config
from .workload import ArchitectureRecord, resolve_workload

__all__ = [
    "AgentConfig",
    "TargetCluster",
    "OptimizationConfig",
    "load_config",
    "ArchitectureRecord",
    "resolve_workload",
]
