"""Configuration: models, budgets, paths."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .schemas import Budget


# Model slugs — swap to whatever is available in your Anthropic account.
# Orchestrator is thin; a smaller/faster model is fine.
# Stage Workers do the heavy Skill reasoning; give them the strongest model.
DEFAULT_ORCHESTRATOR_MODEL = "claude-opus-4-5-20250929"
DEFAULT_WORKER_MODEL = "claude-opus-4-5-20250929"


@dataclass
class OrchestratorConfig:
    session_id: str
    session_dir: Path
    skills_dir: Path

    budget_total: Budget

    orchestrator_model: str = DEFAULT_ORCHESTRATOR_MODEL
    worker_model: str = DEFAULT_WORKER_MODEL

    # Context hygiene thresholds (§13.4 Context budget)
    orchestrator_response_tokens: int = 2_048
    worker_max_tokens: int = 30_000
    handoff_token_ratio: float = 0.5  # trigger handoff when budget_used.total_tokens > ratio * budget_total
    max_rounds_before_handoff: int = 15

    # Dry-run: no Anthropic API calls, Workers return canned SubagentResult
    dry_run: bool = False

    def ensure_dirs(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "checkpoints").mkdir(exist_ok=True)
        (self.session_dir / "handoffs").mkdir(exist_ok=True)
        (self.session_dir / "artifacts").mkdir(exist_ok=True)
