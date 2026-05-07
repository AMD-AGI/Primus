"""Skill loader.

Orchestrator only reads:
  - workflow/state_machine.md
  - workflow/orchestration.md

Stage Workers read their scoped glob (e.g. ["workflow/diagnose.md",
"execution-model/*", "optimization/comm/*"]).

If a requested Skill is missing, return a placeholder rather than raising
so the skeleton is runnable before Skills are authored.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.root = Path(skills_dir)

    def read(self, rel_path: str) -> str:
        p = self.root / rel_path
        if not p.exists():
            log.warning("skill missing: %s (using placeholder)", p)
            return f"<!-- SKILL PLACEHOLDER: {rel_path} not yet authored -->"
        return p.read_text(encoding="utf-8")

    def read_scope(self, scope: list[str]) -> str:
        """Concatenate all .md files matching the glob patterns in scope.

        Each pattern is interpreted relative to self.root. Example:
            scope=["workflow/diagnose.md", "execution-model/*", "optimization/comm/*"]
        """
        chunks: list[str] = []
        for pattern in scope:
            matches = sorted(self.root.glob(pattern))
            if not matches:
                log.warning("skill scope glob empty: %s", pattern)
                chunks.append(f"<!-- NO MATCH for scope: {pattern} -->")
                continue
            for p in matches:
                if p.is_file() and p.suffix == ".md":
                    rel = p.relative_to(self.root)
                    chunks.append(f"### {rel}\n\n{p.read_text(encoding='utf-8')}")
        return "\n\n".join(chunks)
