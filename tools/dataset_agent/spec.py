# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Run specification passed from the phase machine into the Claude SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Task:
    """The inputs the operator supplies, plus run limits."""

    backend: str
    model: str
    dataset: str
    stage: str = "posttrain"
    split: str = "train"
    samples: int = 4
    max_retries: int = 3
    llm_model: str = ""
    dry_run: bool = False
    no_branch: bool = False


@dataclass
class PathPolicy:
    """Which relative paths a phase may create or modify."""

    write_roots: tuple[str, ...] = ()
    write_files: tuple[str, ...] = ()
    allow_bash: bool = False


@dataclass
class AgentRunSpec:
    """One SDK session."""

    name: str
    system_prompt: str
    user_prompt: str
    cwd: str
    model: str
    max_turns: int
    allowed_tools: list[str]
    policy: PathPolicy
    permission_mode: str = "acceptEdits"


@dataclass
class AgentRunResult:
    """Normalized stream from one SDK query."""

    text: str = ""
    session_id: str = ""
    subtype: str = ""
    end_reason: str = ""
    num_turns: int | None = None
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    stderr_tail: str = ""


Hook = Callable[..., Any]
