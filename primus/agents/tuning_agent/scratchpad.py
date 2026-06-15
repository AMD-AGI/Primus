"""Plain-text scratchpad shared between agent rounds.

The scratchpad survives across RLM iterations and across rounds. The LLM
uses it to write down its current plan, hypotheses, and what it has ruled
out — mirroring the iterative_fix pattern but here used to organise the
multi-round search rather than a single bug fix.
"""

from __future__ import annotations

import time
from pathlib import Path


class Scratchpad:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    def read(self) -> str:
        return self.path.read_text() if self.path.is_file() else ""

    def append(self, note: str) -> str:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {note.strip()}\n"
        with self.path.open("a") as f:
            f.write(line)
        return f"OK: scratchpad updated ({len(note)} chars)"

    def reset(self) -> None:
        self.path.write_text("")
