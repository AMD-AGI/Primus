###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""The projection agent: keeps the projection able to price the model it is asked about.

Before a projection runs, the agent checks the workload's model config against
what the built-in profilers and registered op models price.  Every op it finds
unmodeled goes through op bring-up (:mod:`.bringup`) -- context pack, coder,
validator, registration -- so the projection that follows prices it with its
own validated model instead of a built-in stand-in.

It is driven three ways:

* ``primus projection <suite> --config <yaml> --op-bringup auto ...`` -- bring
  the workload's missing ops up, then run the projection as usual;
* ``python -m primus.agents.projection_agent --workload <yaml>`` -- bring the
  ops up without projecting;
* the tuning agent, which calls :meth:`ProjectionAgent.ensure_ops` before its
  search and :meth:`ProjectionAgent.handle_gaps` when a trial reports an op
  the YAML did not reveal.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from .bringup import BringupResult, Coder, bring_up_gaps

# <root>/primus/agents/projection_agent/agent.py
PRIMUS_ROOT = Path(__file__).resolve().parents[3]

POLICIES = ("auto", "report", "off")


class ProjectionAgent:
    """Brings up the ops a workload needs and the projection does not model.

    ``policy`` is ``auto`` (write, validate and register op models with
    *coder*; without a coder it behaves as ``report``), ``report`` (write the
    context packs only) or ``off``.  An op id is brought up at most once per
    agent, however many times it is reported.
    """

    def __init__(
        self,
        *,
        primus_root: Optional[Path] = None,
        out_dir: Path = Path("op_bringup_runs"),
        gpu_arch: Optional[str] = None,
        policy: str = "auto",
        coder: Optional[Coder] = None,
        papers: Sequence[str] = (),
        hf_config: Optional[str] = None,
        max_attempts: int = 3,
        register: bool = True,
        env: Optional[dict] = None,
        log: Callable[[str], None] = print,
    ):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
        self.primus_root = Path(primus_root or PRIMUS_ROOT)
        self.out_dir = Path(out_dir)
        self.gpu_arch = gpu_arch
        self.policy = policy if coder is not None or policy != "auto" else "report"
        self.coder = coder
        self.papers = list(papers)
        self.hf_config = hf_config
        self.max_attempts = max_attempts
        self.register = register
        self.env = env
        self.log = log
        self.results: List[BringupResult] = []
        self.gaps_seen: Dict[str, dict] = {}
        self._archs: Dict[Path, object] = {}

    # ---- workload ----
    def _arch(self, workload: Path, arch=None):
        workload = Path(workload).resolve()
        if arch is not None:
            self._archs[workload] = arch
        if workload not in self._archs:
            from primus.agents.tuning_agent.workload import resolve_workload

            self._archs[workload] = resolve_workload(workload, primus_root=self.primus_root)
        return self._archs[workload]

    def detect_gaps(self, workload: Path, *, arch=None) -> list:
        """Blocking gaps of *workload*'s model config, before any projection runs."""
        from primus.core.projection.op_models.coverage import detect_op_gaps
        from primus.core.projection.op_models.registry import discover_op_models

        a = self._arch(workload, arch)
        discover_op_models(force=True)
        flat = {
            **(a.raw_model or {}),
            **{k: v for k, v in (a.raw_overrides or {}).items() if not isinstance(v, dict)},
        }
        return detect_op_gaps(flat, framework=getattr(a, "framework", "megatron")).blocking

    # ---- bring-up ----
    def ensure_ops(self, workload: Path, *, arch=None, only: Iterable[str] = ()) -> List[BringupResult]:
        """Bring up every op of *workload* the projection does not model yet."""
        if self.policy == "off":
            return []
        try:
            gaps = self.detect_gaps(workload, arch=arch)
        except Exception as exc:
            self.log(f"[projection-agent] op coverage check failed: {exc}")
            return []
        only = set(only)
        if only:
            gaps = [g for g in gaps if g.op_id in only]
        if not gaps:
            self.log("[projection-agent] op coverage: every op in the workload is modeled")
            return []
        self.log(
            f"[projection-agent] op coverage: {len(gaps)} unmodeled op(s): "
            f"{[g.op_id for g in gaps]} (policy={self.policy})"
        )
        return self._bring_up(gaps, workload, arch)

    def handle_gaps(self, gaps: Sequence, workload: Path, *, arch=None) -> bool:
        """Bring up gaps a projection run reported; True when the projection changed.

        *gaps* are :class:`OpGap` objects or their dicts (as parsed from
        ``PRIMUS_OP_GAP`` markers); ones already handled are skipped.
        """
        from primus.core.projection.op_models.coverage import OpGap

        parsed = [
            (
                g
                if isinstance(g, OpGap)
                else OpGap(**{k: g[k] for k in ("op_id", "severity", "slot", "reason", "keys") if k in g})
            )
            for g in gaps
        ]
        return any(r.changed_projection for r in self._bring_up(parsed, workload, arch))

    def _bring_up(self, gaps: Sequence, workload: Path, arch=None) -> List[BringupResult]:
        new = [g for g in gaps if g.op_id not in self.gaps_seen]
        if self.policy == "off" or not new:
            return []
        for g in new:
            self.gaps_seen[g.op_id] = asdict(g)
        a = self._arch(workload, arch)
        res = bring_up_gaps(
            new,
            workload=Path(workload).resolve(),
            primus_root=self.primus_root,
            out_dir=self.out_dir,
            coder=self.coder if self.policy == "auto" else None,
            gpu_arch=self.gpu_arch,
            model_path=Path(a.model_path) if getattr(a, "model_path", None) else None,
            raw_model=getattr(a, "raw_model", None),
            hf_config=self.hf_config,
            papers=self.papers,
            max_attempts=self.max_attempts,
            register=self.register,
            env=self.env,
            log=lambda s: self.log(f"[projection-agent] {s}"),
        )
        if any(r.status == "registered" for r in res):
            from primus.core.projection.op_models.registry import discover_op_models

            discover_op_models(force=True)
        self.results.extend(res)
        return res

    def summary(self) -> dict:
        return {
            "policy": self.policy,
            "gaps_seen": self.gaps_seen,
            "results": [{k: v for k, v in r.as_dict().items() if k != "validation"} for r in self.results],
        }
