###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tier 1 -- G: foreign / leaked process detection on each GPU.

The single most common reason a "healthy" cluster fails to launch a large
training job is that a previous job's Python ranks are still attached to
the GPUs (held HBM, half-torn-down NCCL communicators, or just stuck in
__del__). Symptoms in the new training job: torch.cuda.OutOfMemoryError
at model init with a misleading "free=Y" message, NCCL/RCCL bootstrap
hang, or random ranks failing the first all-reduce due to compute
contention. node-smoke catches these BEFORE the operator launches the
real job by enumerating PIDs that hold each GPU and FAILing the node
unless the operator explicitly opted in via --allow-foreign-procs.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, List, Optional

from ..shell_utils import _parse_size_with_unit, _which
from .rocm_smi import _rocm_smi_processes

# Process-name placeholders that some versions of `amd-smi process` and
# `rocm-smi --showpids` emit when they cannot read the real name from
# /proc/<pid>/comm (typically for kernel/system-owned PIDs like
# `gpuagent`). We treat these as "missing" and fall back to /proc.
_MISSING_NAME_TOKENS = frozenset({"", "n/a", "na", "none", "null", "-", "unknown", "?"})


def _is_missing_name(name: str) -> bool:
    return (name or "").strip().lower() in _MISSING_NAME_TOKENS


def _resolve_proc_name(pid: int) -> str:
    """Best-effort recover the process name for ``pid`` from /proc.

    Tries ``/proc/<pid>/comm`` first (15-char kernel name, world-readable
    on default Linux), then falls back to the ``Name:`` line in
    ``/proc/<pid>/status``. Returns ``""`` when both reads fail (PID gone,
    hidepid mount, ptrace_scope, etc.) so the caller can decide whether
    to keep the original placeholder.
    """
    try:
        with open(f"/proc/{int(pid)}/comm", "r", encoding="utf-8") as f:
            name = f.read().strip()
            if name:
                return name
    except OSError:
        pass
    try:
        with open(f"/proc/{int(pid)}/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Name:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def _resolve_self_pid_view(self_pid: int) -> Dict[str, Any]:
    """Resolve our own PID as the host kernel sees it, even from inside a
    private PID namespace (e.g. a Docker / Kubernetes container).

    ``amd-smi process``, ``rocm-smi --showpids`` and ``lsof /dev/kfd``
    all report PIDs in the **root (host) PID namespace** because KFD is
    a host-kernel resource that knows nothing about user namespaces.
    ``os.getpid()``, by contrast, returns the PID *as our own namespace
    sees it*. In a private PID namespace these are different numbers,
    and the naive ``reported_pid == os.getpid()`` test would always
    return False -- causing our own training rank to be flagged
    ``is_foreign=True`` and (with the default policy) failing the node.

    The kernel exposes the full mapping in ``/proc/self/status``:

        NSpid:  2005679  42

    where the first entry is the deepest-namespace PID (= the host PID
    on bare metal) and the LAST entry is the most-deeply-nested PID
    (= what ``os.getpid()`` returns inside a private namespace). On a
    bare-metal host or a non-namespaced container, the line has a
    single field equal to ``os.getpid()``.

    Returns::

        {
            "host_pid": int,           # what amd-smi/rocm-smi report
            "container_pid": int,      # == self_pid (passed in)
            "pid_namespaced": bool,    # True if the two differ
            "ns_chain": [int, ...],    # full NSpid chain (for the report)
        }

    Best-effort: if ``/proc/self/status`` cannot be read or has no
    ``NSpid`` line, we assume bare-metal and fall back to ``self_pid``.
    """
    out: Dict[str, Any] = {
        "host_pid": int(self_pid),
        "container_pid": int(self_pid),
        "pid_namespaced": False,
        "ns_chain": [int(self_pid)],
    }
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("NSpid:"):
                    continue
                parts = line.split()[1:]  # drop "NSpid:"
                chain: List[int] = []
                for tok in parts:
                    try:
                        chain.append(int(tok))
                    except ValueError:
                        continue
                if not chain:
                    break
                out["ns_chain"] = chain
                # Convention: NSpid lists outermost (host) namespace first
                # and the current namespace last; matches /proc man page.
                out["host_pid"] = chain[0]
                out["container_pid"] = chain[-1]
                out["pid_namespaced"] = chain[0] != chain[-1]
                break
    except Exception:
        pass
    return out


def _collect_gpu_processes(
    self_pid: int,
    allowed_proc_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Enumerate processes currently holding each GPU on this node.

    Tries, in order:
      1. ``amd-smi process --json``  (preferred -- structured)
      2. ``amd-smi process``         (text fallback -- best-effort parse)
      3. ``lsof /dev/kfd /dev/dri/renderD*``  (last resort: just openers)

    Output shape::

        {
            "ok": bool,
            "tool": "amd-smi process --json" | "amd-smi process" | "lsof" | None,
            "self_pid": int,                # container/local view (legacy)
            "self_pgid": int,               # pgid in our own namespace
            "self_host_pid": int,           # PID amd-smi/rocm-smi report for us
            "pid_namespaced": bool,         # True inside a private PID ns
            "ns_pid_chain": [int, ...],     # full NSpid chain (root..ours)
            "allowed_proc_names": [str, ...],   # passthrough for aggregator
            "per_gpu": [
                {"gpu": 0, "processes": [
                    {"pid": int, "name": str, "hbm_bytes": int|None,
                     "is_self": bool, "is_allowed": bool, "is_foreign": bool},
                    ...
                ]},
                ...
            ],
            "foreign_count": int,   # PIDs not us and not allowed
            "error": str            # only when ok is False
        }

    Filtering: a PID is treated as "ours" (and thus not foreign) if it
    matches our **host-namespace PID** (because amd-smi / rocm-smi /
    lsof always report root-ns PIDs) or, when we are NOT inside a
    private PID namespace, our pgid (which catches per-GPU subprocesses
    we may have spawned). Inside a private PID namespace the pgid match
    is intentionally skipped: we cannot ``os.getpgid()`` PIDs we cannot
    see, and any spawned subprocess will appear as a sibling host PID
    that the operator should treat the same way as a leaked rank.

    ``allowed_proc_names`` is a case-insensitive name allow-list for
    known node-resident agents (``rocm-smi-daemon``, ``amd-smi``,
    ``dcgm-exporter``, ``gpuagent``, ...).
    """
    allowed = {n.strip().lower() for n in (allowed_proc_names or []) if n.strip()}

    # Resolve host-side PID. Inside a private PID namespace this is the
    # number amd-smi / rocm-smi / lsof will report for us; on bare metal
    # or a shared-PID-ns container (the SLURM + pyxis/enroot default) it
    # equals self_pid.
    pid_view = _resolve_self_pid_view(int(self_pid))
    host_self_pid = int(pid_view["host_pid"])
    pid_namespaced = bool(pid_view["pid_namespaced"])

    # pgid is only meaningful within OUR own PID namespace -- attempting
    # os.getpgid() on a host-side PID we cannot see would raise ESRCH.
    try:
        self_pgid = os.getpgid(int(self_pid))
    except OSError:
        self_pgid = int(self_pid)

    out: Dict[str, Any] = {
        "ok": False,
        "tool": None,
        "self_pid": int(self_pid),
        "self_pgid": int(self_pgid),
        "self_host_pid": host_self_pid,
        "pid_namespaced": pid_namespaced,
        "ns_pid_chain": list(pid_view.get("ns_chain") or [int(self_pid)]),
        "allowed_proc_names": sorted(allowed),
        "per_gpu": [],
        "foreign_count": 0,
    }

    def _annotate(pid: int, name: str, hbm_bytes: Optional[int]) -> Dict[str, Any]:
        # Direct PID match against our HOST-side PID (works in every
        # mode: bare metal, shared PID ns, private PID ns).
        is_self = int(pid) == host_self_pid
        # pgid match is only safe outside a private PID namespace; inside
        # one we cannot see host PIDs at all and os.getpgid() would
        # ESRCH for every reported pid.
        if not is_self and not pid_namespaced:
            try:
                pgid = os.getpgid(int(pid))
            except OSError:
                pgid = -1
            if pgid == self_pgid:
                is_self = True

        # Some amd-smi / rocm-smi builds report `name="N/A"` (or empty)
        # for kernel/system PIDs like `gpuagent` because they cannot read
        # /proc/<pid>/comm themselves. We can: do it inline so the
        # allowlist actually matches and the report shows the real name.
        raw_name = name or ""
        name_resolved_from_proc = False
        if _is_missing_name(raw_name):
            recovered = _resolve_proc_name(int(pid))
            if recovered:
                name = recovered
                name_resolved_from_proc = True

        is_allowed = (name or "").strip().lower() in allowed
        return {
            "pid": int(pid),
            "name": name or "",
            "name_raw": raw_name,
            "name_resolved_from_proc": bool(name_resolved_from_proc),
            "hbm_bytes": int(hbm_bytes) if isinstance(hbm_bytes, (int, float)) else None,
            "is_self": bool(is_self),
            "is_allowed": bool(is_allowed),
            "is_foreign": bool(not is_self and not is_allowed),
        }

    if _which("amd-smi") is not None:
        # 1) amd-smi process --json
        try:
            cp = subprocess.run(
                ["amd-smi", "process", "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
                check=False,
            )
            if cp.returncode == 0 and cp.stdout.strip():
                try:
                    doc = json.loads(cp.stdout)
                    out["per_gpu"] = _flatten_amd_smi_process_json(doc, _annotate)
                    out["tool"] = "amd-smi process --json"
                    out["ok"] = True
                except Exception as e:
                    out["json_parse_error"] = str(e)
            else:
                out["json_rc"] = cp.returncode
                out["json_stderr"] = (cp.stderr or "").strip()[:200]
        except subprocess.TimeoutExpired:
            out["json_error"] = "amd-smi process --json timed out"
        except Exception as e:
            out["json_error"] = str(e)

        # 2) amd-smi process (text)
        if not out["ok"]:
            try:
                cp = subprocess.run(
                    ["amd-smi", "process"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=15,
                    check=False,
                )
                if cp.returncode == 0:
                    parsed = _parse_amd_smi_process_text(cp.stdout, _annotate)
                    out["per_gpu"] = parsed
                    out["tool"] = "amd-smi process"
                    out["ok"] = True
                    out["raw"] = cp.stdout[:4000]
                else:
                    out["text_rc"] = cp.returncode
                    out["text_stderr"] = (cp.stderr or "").strip()[:200]
            except subprocess.TimeoutExpired:
                out["text_error"] = "amd-smi process timed out"
            except Exception as e:
                out["text_error"] = str(e)

    # 3) rocm-smi --showpids --json. Doesn't give us per-GPU mapping
    # (--showpidgpus emits "WARNING: No JSON data to report"), so all
    # PIDs go into the gpu=-1 bucket -- same convention as the lsof
    # fallback. This is still a HUGE win over lsof: rocm-smi reports
    # the actual KFD process name which lets the operator decide if
    # it's a leaked rank vs a known agent.
    if not out["ok"]:
        rocm = _rocm_smi_processes(_annotate)
        if rocm.get("ok"):
            out["per_gpu"] = rocm.get("per_gpu") or []
            out["tool"] = rocm.get("tool")
            out["ok"] = True
        else:
            out["rocm_smi_error"] = rocm.get("error")

    # 4) lsof on /dev/kfd + /dev/dri/renderD* (we cannot map back to specific
    # GPUs reliably this way, but at least we surface foreign openers).
    if not out["ok"]:
        try:
            import glob as _glob

            paths = ["/dev/kfd"] + sorted(_glob.glob("/dev/dri/renderD*"))
            existing = [p for p in paths if os.path.exists(p)]
            if existing and _which("lsof") is not None:
                cp = subprocess.run(
                    ["lsof", "-Fpcn", "--", *existing],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if cp.returncode in (0, 1):  # lsof exits 1 when nothing open
                    procs = _parse_lsof_pcn(cp.stdout, _annotate)
                    # Without per-GPU mapping, surface as a single bucket
                    # under gpu=-1 so the aggregator still flags foreigners.
                    out["per_gpu"] = [{"gpu": -1, "processes": procs}] if procs else []
                    out["tool"] = "lsof"
                    out["ok"] = True
                else:
                    out["lsof_rc"] = cp.returncode
        except Exception as e:
            out["lsof_error"] = str(e)

    if not out["ok"] and "error" not in out:
        out["error"] = "no working enumeration tool " "(amd-smi process / rocm-smi --showpids / lsof)"

    out["foreign_count"] = sum(
        1 for g in out["per_gpu"] for p in (g.get("processes") or []) if p.get("is_foreign")
    )
    return out


def _flatten_amd_smi_process_json(
    doc: Any,
    annotate: Any,
) -> List[Dict[str, Any]]:
    """Coerce the amd-smi process JSON into a stable per-GPU shape.

    The schema varies across releases; we tolerate missing fields silently.
    Three top-level shapes are seen in the wild:

      A) Per-GPU dicts each carrying ``process_list``, where each list
         item wraps the actual process under ``process_info`` and reports
         memory as ``{"value": N, "unit": "B"}`` -- this is the modern
         ``amd-smi`` (>= 6.x) layout::

           [{"gpu": 0, "process_list": [
               {"process_info": {
                   "pid": 12345, "name": "python",
                   "memory_usage": {"vram_mem": {"value": 256, "unit": "MB"}}
               }}
           ]}]

      A') Same as A but each ``process_list`` item is the process dict
          directly (older amd-smi releases), with memory as a flat int or
          formatted string::

           [{"gpu": 0, "process_list": [
               {"pid": 12345, "name": "python",
                "memory_usage": {"vram_mem": "256 MB"}}
           ]}]

      B) Top-level list of per-process dicts each carrying explicit
         ``gpu`` / ``gpus`` (uncommon, but seen on a couple of branches)::

           [{"pid": ..., "name": ..., "gpu": 0, "memory_usage": {...}}]
    """
    out_by_gpu: Dict[int, List[Dict[str, Any]]] = {}

    def _push(gpu: int, pid: Any, name: Any, hbm: Any) -> None:
        if not isinstance(pid, (int, float)):
            return
        ann = annotate(int(pid), str(name or ""), hbm)
        out_by_gpu.setdefault(int(gpu), []).append(ann)

    def _value_unit_to_bytes(v: Any) -> Optional[int]:
        """Resolve a size value that may be int, formatted string, or
        ``{"value": N, "unit": "B|KB|MB|GB|..."}`` (modern amd-smi shape)
        into bytes."""
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str):
            return _parse_size_with_unit(v)
        if isinstance(v, dict):
            val = v.get("value")
            unit = v.get("unit")
            if isinstance(val, (int, float)):
                if isinstance(unit, str) and unit.strip():
                    return _parse_size_with_unit(f"{val} {unit.strip()}")
                return int(val)
            if isinstance(val, str):
                if isinstance(unit, str) and unit.strip():
                    return _parse_size_with_unit(f"{val} {unit.strip()}")
                return _parse_size_with_unit(val)
        return None

    def _hbm_of(d: Dict[str, Any]) -> Optional[int]:
        # Preferred path: memory_usage.vram_mem (covers shapes A, A', B).
        mu = d.get("memory_usage") if isinstance(d.get("memory_usage"), dict) else None
        if mu is not None:
            for k in ("vram_mem", "vram_memory", "vram"):
                if k in mu:
                    n = _value_unit_to_bytes(mu.get(k))
                    if n is not None:
                        return n
        # Fallback: mem_usage at the same level (older shapes -- usually
        # mirrors vram_mem on AMD GPUs since GTT/CPU are negligible).
        if "mem_usage" in d:
            n = _value_unit_to_bytes(d.get("mem_usage"))
            if n is not None:
                return n
        # Last resort: a flat "vram" key directly under d.
        if "vram" in d:
            n = _value_unit_to_bytes(d.get("vram"))
            if n is not None:
                return n
        return None

    def _unwrap_proc(p: Dict[str, Any]) -> Dict[str, Any]:
        """Modern amd-smi wraps each process under ``process_info``;
        unwrap so the rest of the parser can read pid/name/memory at
        the top level uniformly."""
        if isinstance(p.get("process_info"), dict):
            return p["process_info"]
        return p

    items = (
        doc
        if isinstance(doc, list)
        else (doc.get("processes") or doc.get("gpus") or [] if isinstance(doc, dict) else [])
    )
    if not isinstance(items, list):
        items = []

    for item in items:
        if not isinstance(item, dict):
            continue
        # Shape A / A': per-GPU dict with process_list
        plist = item.get("process_list") or item.get("processes")
        if isinstance(plist, list):
            gpu_idx = item.get("gpu")
            if not isinstance(gpu_idx, int):
                gpu_idx = -1
            for p in plist:
                if not isinstance(p, dict):
                    continue
                proc = _unwrap_proc(p)
                _push(
                    gpu_idx,
                    proc.get("pid") if proc.get("pid") is not None else proc.get("process_id"),
                    proc.get("name") or proc.get("process_name"),
                    _hbm_of(proc),
                )
            continue
        # Shape B: per-process dict with explicit gpu
        proc = _unwrap_proc(item)
        if "pid" in proc or "process_id" in proc:
            g = proc.get("gpu")
            gpus = (
                proc.get("gpus")
                if isinstance(proc.get("gpus"), list)
                else ([g] if isinstance(g, int) else [-1])
            )
            for gpu_idx in gpus:
                if not isinstance(gpu_idx, int):
                    gpu_idx = -1
                _push(
                    gpu_idx,
                    proc.get("pid") if proc.get("pid") is not None else proc.get("process_id"),
                    proc.get("name") or proc.get("process_name"),
                    _hbm_of(proc),
                )

    return [{"gpu": k, "processes": v} for k, v in sorted(out_by_gpu.items())]


def _parse_amd_smi_process_text(text: str, annotate: Any) -> List[Dict[str, Any]]:
    """Best-effort parser for ``amd-smi process`` plain-text output.

    Format varies, but typical layout is one record per process with lines
    like ``GPU: 0`` / ``PID: 12345`` / ``NAME: python`` / ``VRAM_MEM: 256 MB``.
    We tokenise key-value pairs case-insensitively and group records by
    blank-line separators.
    """
    out_by_gpu: Dict[int, List[Dict[str, Any]]] = {}
    cur: Dict[str, Any] = {}
    blocks: List[Dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if cur:
                blocks.append(cur)
                cur = {}
            continue
        if ":" in line:
            k, _, v = line.partition(":")
            cur[k.strip().lower()] = v.strip()
    if cur:
        blocks.append(cur)
    for b in blocks:
        gpu_s = b.get("gpu") or b.get("gpu_id") or b.get("device") or "-1"
        try:
            gpu = int(gpu_s.split()[0])
        except Exception:
            gpu = -1
        try:
            pid = int(b.get("pid", "").split()[0])
        except Exception:
            continue
        name = b.get("name") or b.get("process") or b.get("process_name") or ""
        hbm_raw = b.get("vram_mem") or b.get("vram") or b.get("memory_usage") or b.get("mem_usage") or ""
        hbm = _parse_size_with_unit(hbm_raw) if hbm_raw else None
        out_by_gpu.setdefault(gpu, []).append(annotate(pid, name, hbm))
    return [{"gpu": k, "processes": v} for k, v in sorted(out_by_gpu.items())]


def _parse_lsof_pcn(text: str, annotate: Any) -> List[Dict[str, Any]]:
    """Parse ``lsof -Fpcn`` output (one field per line, prefixed by f-code).

    We only need ``p<pid>`` and ``c<command>``. Returns one annotated
    record per unique PID; HBM bytes unknown (lsof can't measure that).
    """
    out: Dict[int, Dict[str, Any]] = {}
    cur_pid: Optional[int] = None
    cur_name = ""
    for line in text.splitlines():
        if not line:
            continue
        tag, val = line[0], line[1:]
        if tag == "p":
            try:
                cur_pid = int(val)
            except ValueError:
                cur_pid = None
            cur_name = ""
        elif tag == "c" and cur_pid is not None:
            cur_name = val
            out[cur_pid] = annotate(cur_pid, cur_name, None)
    return list(out.values())
