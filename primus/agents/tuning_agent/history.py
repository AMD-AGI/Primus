"""Trial history (JSONL) + lightweight in-memory store.

Each trial appended on disk so a run can be resumed / inspected.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .evaluator import EvalResult


@dataclass
class TrialRecord:
    idx: int
    timestamp: float
    config: dict
    result: dict
    source: str
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "idx": self.idx,
            "timestamp": self.timestamp,
            "config": self.config,
            "result": self.result,
            "source": self.source,
            "notes": self.notes,
        }


@dataclass
class History:
    path: Path
    trials: list[TrialRecord] = field(default_factory=list)
    seen_signatures: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> "History":
        h = cls(path=path)
        if path.is_file():
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    h.trials.append(
                        TrialRecord(
                            idx=d["idx"],
                            timestamp=d["timestamp"],
                            config=d["config"],
                            result=d["result"],
                            source=d.get("source", "unknown"),
                            notes=d.get("notes", ""),
                        )
                    )
                    sig = _sig(d["config"])
                    h.seen_signatures.add(sig)
                except (json.JSONDecodeError, KeyError):
                    continue
        return h

    def add(self, config: dict, result: EvalResult, notes: str = "") -> TrialRecord:
        rec = TrialRecord(
            idx=len(self.trials),
            timestamp=time.time(),
            config=config,
            result=result.as_dict(),
            source=result.source,
            notes=notes,
        )
        self.trials.append(rec)
        self.seen_signatures.add(_sig(config))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(rec.as_dict(), default=str) + "\n")
        return rec

    def already_evaluated(self, config: dict) -> bool:
        return _sig(config) in self.seen_signatures

    def best(self, objective: str = "tokens_per_s_per_gpu") -> TrialRecord | None:
        legal = [t for t in self.trials if t.result.get("legal") and t.result.get(objective) is not None]
        if not legal:
            return None
        return max(legal, key=lambda t: t.result.get(objective) or 0.0)

    def summary_for_llm(self, k: int | None = None) -> str:
        rows = self.trials if k is None else self.trials[-k:]
        lines = []
        for t in rows:
            r = t.result
            tps = r.get("tokens_per_s_per_gpu")
            mem = r.get("memory_per_gpu_gb")
            tps_s = f"{tps:,.0f}" if isinstance(tps, (int, float)) else "—"
            mem_s = f"{mem:.1f}GB" if isinstance(mem, (int, float)) else "—"
            cfg = t.config
            cfg_s = (
                f"TP={cfg.get('tp')} PP={cfg.get('pp')} EP={cfg.get('ep')} "
                f"CP={cfg.get('cp')} MBS={cfg.get('mbs')} GBS={cfg.get('gbs')} "
                f"VPP={cfg.get('vpp')} sched={cfg.get('pp_schedule')} "
                f"recompute={cfg.get('recompute_granularity')}"
            )
            tag = "OK" if r.get("legal") else f"REJECT({r.get('reason','')[:60]})"
            lines.append(f"#{t.idx:03d} [{t.source:9s}] {tag:40s} tps={tps_s:>10s} mem={mem_s:>8s} | {cfg_s}")
        return "\n".join(lines) if lines else "(no trials yet)"


def _sig(config: dict) -> str:
    return ",".join(f"{k}={config[k]}" for k in sorted(config))
