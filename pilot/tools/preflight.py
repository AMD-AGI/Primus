"""preflight tools: hardware baseline + cluster-level env baseline.

Authoritative protocol: ``skills/workflow/preflight.md``.
Universal input contract:  ``AGENTS.md`` §4 (every tool requires ``cluster.yaml``).

CLI surface (stable contract; the measurement layer is staged for follow-up):

    # Universal mandatory env input: cluster.yaml (one of:
    #   --cluster-config <path>, $PILOT_CLUSTER_CONFIG, ./cluster.yaml)

    python -m pilot.tools.preflight run [--cluster-config <path>] \\
        [--reason bootstrap|reentry_hang|reentry_cluster|reentry_stale|force] \\
        [--target-version <v>] [--force] [--delta-only] \\
        [--max-wallclock-s 1800] \\
        [--blacklist state/blacklist.yaml] \\
        [--out-dir state/cluster_profiles]

    python -m pilot.tools.preflight env_probe [--cluster-config <path>] \\
        [--candidate-file <path-to-yaml>] \\
        [--tier 1|2|3|all] \\
        [--out-dir state/env_probe_results]

    python -m pilot.tools.preflight env_sweep [--cluster-config <path>] \\
        --base-plan <ref> --candidates <json-or-path> \\
        [--max-steps 50] [--out-dir state/round_<N>]

All commands:
  - run the three universal fast-fail checks (cluster.yaml valid; SLURM job
    alive when mode=slurm; ≥1 GPU visible) BEFORE any measurement.
  - emit a single JSON document on stdout matching the corresponding schema.
  - log progress to stderr.
  - exit code 0 on success, 1 on stage-failure (still emits JSON with
    ``status=failed``), 2 on usage error, 3 on TOOL_ERROR
    (NotImplementedError / unmet dep), 4 on CLUSTER (cluster.yaml /
    SLURM / GPU visibility failure).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pilot.tools._cluster_config import (
    ClusterConfig,
    ClusterConfigError,
    LaunchPlan,
    cluster_config_failure,
    preflight_check,
)


# ---------------------------------------------------------------------------
# Internal error type — measurement-layer specific, separate from
# ClusterConfigError which is used for fast-fail contract violations.
# ---------------------------------------------------------------------------


class _PreflightError(Exception):
    """Raised by Step 1-5 measurement helpers.

    The ``kind`` attribute maps directly to ``failure.kind`` in the SubagentResult.
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


# ---------------------------------------------------------------------------
# Path anchoring
# ---------------------------------------------------------------------------

# Repo-relative anchor for state directories. Resolves to `<repo>/pilot`
# regardless of the user's cwd when invoking the CLI. Centralised here so all
# default --out-dir / blacklist paths land under `pilot/state/...`.
_PILOT_ROOT: Path = Path(__file__).resolve().parent.parent  # tools/ -> pilot/


def _resolve_pilot_path(p: str | Path) -> Path:
    """Resolve a path relative to the pilot package root.

    Absolute paths are returned unchanged; relative paths are anchored at
    ``_PILOT_ROOT`` so that, e.g., ``state/cluster_profiles`` always lands at
    ``<repo>/pilot/state/cluster_profiles`` regardless of where the user
    invokes the CLI from. This avoids the surprise of artifacts landing at
    ``<cwd>/state/...`` when the user runs from a sibling directory.
    """
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return _PILOT_ROOT / pp


# ---------------------------------------------------------------------------
# Public API surface (load-bearing signatures; impl returns NotImplementedError
# until measurement layer is wired up)
# ---------------------------------------------------------------------------


def run(
    cfg: ClusterConfig,
    plan: LaunchPlan,
    *,
    reason: str = "bootstrap",
    target_version: str | None = None,
    force: bool = False,
    delta_only: bool = False,
    max_wallclock_s: int = 1800,
    blacklist_path: str = "state/blacklist.yaml",
    out_dir: str = "state/cluster_profiles",
) -> dict[str, Any]:
    """Collect hardware baseline → ClusterProfile (§8.1).

    Caller MUST have already invoked ``preflight_check()`` to obtain ``cfg`` and
    ``plan``; this function trusts both as already-validated.

    Dispatches by ``plan.mode``:
      - ``single``: in-process torchrun on the local node.
      - ``slurm``:  ``srun --jobid=<id> -N <k> --ntasks-per-node=1
                    python -m pilot.tools._preflight_node_entry ...``

    Args:
        cfg: parsed cluster.yaml.
        plan: derived LaunchPlan (mode, nnodes, rdzv_*, etc.).
        reason: trigger origin (state machine context).
        target_version: if set, attempt to match (e.g. continuation of a
            specific calibration epoch); else auto-increment.
        force: full re-collect even if cache is fresh.
        delta_only: re-run only steps 4 (RCCL) and 5 (env_probe).
        max_wallclock_s: hard cap; Worker aborts on overrun.
        blacklist_path: where to read existing blacklist; nodes excluded.
        out_dir: where to write `<cluster_id>_<version>.yaml` and update
            `_index.yaml` atomically.

    Returns:
        dict matching ``schemas/cluster_profile.schema.json``. On failure,
        a partial dict with required ``status: failed`` and ``failure`` block.

    Raises:
        NotImplementedError: until the measurement layer (Step 1-5) is wired
            up. The CLI maps this to exit code 3 + JSON
            ``status=failed, failure.kind=TOOL_ERROR``.
    """
    if plan.mode == "single":
        return _dispatch_run_single(
            cfg, plan,
            reason=reason, target_version=target_version,
            force=force, delta_only=delta_only,
            max_wallclock_s=max_wallclock_s,
            blacklist_path=blacklist_path, out_dir=out_dir,
        )
    if plan.mode == "slurm":
        return _dispatch_run_slurm(
            cfg, plan,
            reason=reason, target_version=target_version,
            force=force, delta_only=delta_only,
            max_wallclock_s=max_wallclock_s,
            blacklist_path=blacklist_path, out_dir=out_dir,
        )
    raise NotImplementedError(f"unsupported plan.mode={plan.mode!r}")


def env_probe(
    cfg: ClusterConfig,
    plan: LaunchPlan,
    *,
    candidate: dict[str, Any] | None = None,
    tier: str = "all",
    out_dir: str = "state/env_probe_results",
) -> dict[str, Any]:
    """Validate an env_baseline candidate via the 3-tier safe-probe (`profiling/env_probe.md`).

    Args:
        cfg: parsed cluster.yaml.
        plan: derived LaunchPlan.
        candidate: env block to validate; if None, falls back to
            ``skills/env/presets.md`` lookup by cluster_class.
        tier: which tiers to run (``"1"`` connectivity / ``"2"`` micro-bench /
            ``"3"`` multi-node short / ``"all"``).
        out_dir: where to write the probe result yaml.

    Returns:
        dict with fields ``{cluster_id, candidate, tier_results: [{tier, pass,
        details, wallclock_s}], status: validated|tentative|unsafe_fallback,
        recommended_baseline}``.

    Raises:
        NotImplementedError.
    """
    raise NotImplementedError("pilot.tools.preflight.env_probe")


def env_sweep(
    cfg: ClusterConfig,
    plan: LaunchPlan,
    base_plan_ref: str,
    candidates: list[dict[str, Any]],
    *,
    max_steps: int = 50,
    out_dir: str | None = None,
) -> dict[str, Any]:
    """Inner-loop env sweep → EnvSweepResult (§8.5).

    Lock the structure of ``base_plan_ref``, scan ``candidates`` (≤ 8 combos,
    ≤ 5 flag axes, ≤ 50 steps each), return best diff to merge.

    Returns:
        dict matching ``schemas/env_sweep_result.schema.json``.

    Raises:
        NotImplementedError.
    """
    raise NotImplementedError("pilot.tools.preflight.env_sweep")


# ---------------------------------------------------------------------------
# rccl_baseline schema-2.0 helpers
# ---------------------------------------------------------------------------

# Schema 2.0 collective name list, in canonical order.
_COLLECTIVES: tuple[str, ...] = (
    "allreduce", "allgather", "reduce_scatter", "broadcast", "alltoall",
)


def _legacy_curve_to_axes(
    curve: list[dict[str, Any]],
) -> tuple[list[float], list[float], list[float]]:
    """Split a list-of-dicts curve [{size_mb, bw_gbs, latency_us}, ...] into
    three aligned arrays. Entries flagged with `error` are kept as 0.0 / -1.0."""
    sizes: list[float] = []
    bws: list[float] = []
    lats: list[float] = []
    for entry in curve:
        sizes.append(float(entry.get("size_mb", 0)))
        bws.append(float(entry.get("bw_gbs", 0.0)))
        lats.append(float(entry.get("latency_us", -1.0)))
    return sizes, bws, lats


def _roll_up_stats(
    per_node_bw: dict[str, list[float]],
) -> dict[str, list[float] | list[str]]:
    """Compute median/min/max/stddev across nodes, aligned to sizes_mb axis."""
    if not per_node_bw:
        return {
            "median_bw_gbs": [], "min_bw_gbs": [], "max_bw_gbs": [],
            "stddev_pct": [], "slow_nodes_at_max_size": [],
        }
    nodes = list(per_node_bw.keys())
    n_sizes = len(next(iter(per_node_bw.values())))
    medians: list[float] = []
    mins: list[float] = []
    maxs: list[float] = []
    stds: list[float] = []
    for i in range(n_sizes):
        vals = [per_node_bw[n][i] for n in nodes if i < len(per_node_bw[n])]
        if not vals:
            medians.append(0.0); mins.append(0.0); maxs.append(0.0); stds.append(0.0)
            continue
        vals_sorted = sorted(vals)
        med = vals_sorted[len(vals_sorted) // 2]
        mn = vals_sorted[0]
        mx = vals_sorted[-1]
        # population stddev expressed as percent of mean
        mean = sum(vals) / len(vals)
        if mean > 0:
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std_pct = (var ** 0.5) / mean * 100.0
        else:
            std_pct = 0.0
        medians.append(round(med, 2))
        mins.append(round(mn, 2))
        maxs.append(round(mx, 2))
        stds.append(round(std_pct, 2))

    # Slow-node detection at the largest size only (most discriminative).
    slow_nodes: list[str] = []
    if n_sizes > 0:
        last_idx = n_sizes - 1
        last_vals = [(n, per_node_bw[n][last_idx]) for n in nodes if last_idx < len(per_node_bw[n])]
        if last_vals:
            mean_last = sum(v for _, v in last_vals) / len(last_vals)
            std_last = (
                sum((v - mean_last) ** 2 for _, v in last_vals) / len(last_vals)
            ) ** 0.5
            threshold = mean_last - 2 * std_last
            slow_nodes = sorted(n for n, v in last_vals if v < threshold and v < mean_last * 0.85)

    return {
        "median_bw_gbs": medians,
        "min_bw_gbs": mins,
        "max_bw_gbs": maxs,
        "stddev_pct": stds,
        "slow_nodes_at_max_size": slow_nodes,
    }


def _to_intra_node_block(
    per_node_payloads: dict[str, dict[str, Any]],
    *,
    world_size: int,
) -> dict[str, Any]:
    """Convert {hostname: legacy_worker_payload} -> schema-2.0 intra_node block.

    Each worker payload has ``rccl_baseline.<collective>`` as a list of
    {size_mb, bw_gbs, latency_us} dicts. We pivot to per-collective
    {sizes_mb, per_node_bw_gbs, per_node_latency_us, roll_up}.
    """
    collectives_out: dict[str, Any] = {}

    for coll in _COLLECTIVES:
        # Find a reference sizes_mb axis: first node that has non-empty curve.
        ref_sizes: list[float] = []
        for host, payload in per_node_payloads.items():
            curve = payload.get("rccl_baseline", {}).get(coll, []) or []
            if curve:
                ref_sizes, _, _ = _legacy_curve_to_axes(curve)
                break
        if not ref_sizes:
            # Nothing to record; skip this collective entirely.
            continue

        per_node_bw: dict[str, list[float]] = {}
        per_node_lat: dict[str, list[float]] = {}
        for host, payload in per_node_payloads.items():
            curve = payload.get("rccl_baseline", {}).get(coll, []) or []
            sizes, bws, lats = _legacy_curve_to_axes(curve)
            # Align to ref_sizes; pad with 0.0/-1.0 if a node skipped a size.
            host_bw: list[float] = []
            host_lat: list[float] = []
            size_to_bw = dict(zip(sizes, bws))
            size_to_lat = dict(zip(sizes, lats))
            for s in ref_sizes:
                host_bw.append(round(size_to_bw.get(s, 0.0), 2))
                host_lat.append(round(size_to_lat.get(s, -1.0), 1))
            per_node_bw[host] = host_bw
            per_node_lat[host] = host_lat

        collectives_out[coll] = {
            "sizes_mb": [int(s) if s.is_integer() else s for s in ref_sizes],
            "per_node_bw_gbs": per_node_bw,
            "per_node_latency_us": per_node_lat,
            "roll_up": _roll_up_stats(per_node_bw),
        }

    return {
        "world_size": world_size,
        "nnodes_measured": len(per_node_payloads),
        "collectives": collectives_out,
    }


# ---------------------------------------------------------------------------
# Internal mode dispatchers (measurement bodies live here once implemented)
# ---------------------------------------------------------------------------


def _dispatch_run_single(
    cfg: ClusterConfig,
    plan: LaunchPlan,
    *,
    reason: str = "bootstrap",
    target_version: str | None = None,
    force: bool = False,
    delta_only: bool = False,
    max_wallclock_s: int = 1800,
    blacklist_path: str = "state/blacklist.yaml",
    out_dir: str = "state/cluster_profiles",
) -> dict[str, Any]:
    """Single-node measurement path. Implements the 5-step protocol locally.

    Pipeline (all in-process, no SLURM):
      Step 1: topology discovery (rocm-smi / torch.cuda.get_device_properties)
      Step 2: BF16 GEMM peak per GPU (sequential, M=N=K=8192)
      Steps 3-5: spawn ``torchrun --nnodes=1 --nproc-per-node=<n>
                  -m pilot.tools._preflight_workers`` for AllReduce / AllToAll
                  curves and T1 connectivity sanity check.

    Output: writes ``state/cluster_profiles/<cluster_id>_<version>.yaml``
    atomically and returns a ``SubagentResult`` referencing it.

    See ``skills/workflow/preflight.md`` §4 for full protocol details.
    """
    started = time.time()

    # === Step 1 ===
    topo = _step1_topology(cfg)

    # === Step 2 ===
    compute = _step2_compute_peak(topo["gpus_per_node"])
    compute["hbm_capacity_gb"] = topo["hbm_capacity_gb_per_gpu"]
    cluster_class = _derive_cluster_class(topo["gfx_version"], topo["gpus_per_node"])
    spec_bf16 = _VENDOR_PEAK_BF16.get(cluster_class.split("_")[0])
    if spec_bf16 and compute["peak_tflops_bf16"]:
        compute["peak_pct_of_spec_bf16"] = round(compute["peak_tflops_bf16"] / spec_bf16, 3)

    # === Steps 3-5 (T1 connectivity + AR/A2A curves) ===
    dist = _steps_345_distributed(topo["gpus_per_node"])

    # === Compose ClusterProfile ===
    version = target_version or f"{cluster_class}-1node-v1"
    if version and not version.endswith(tuple(f"-v{n}" for n in range(1, 100))):
        # Schema requires `<...>-v<digits>` suffix; default if not present
        if "-v" not in version:
            version = f"{version}-v1"

    intra_type = "xgmi" if topo["gfx_version"].startswith(("gfx9", "gfx94", "gfx95")) else "nvlink"
    intra_bw = dist.get("intra_node_bw_gbs")
    t1 = dist.get("t1_connectivity", {"pass": False, "msg": "no result"})

    # Schema 2.0: rccl_baseline is scoped to {intra_node, inter_node, world}.
    # In single-node mode we measure intra_node only; inter_node and world
    # require a SLURM allocation (handled by _dispatch_run_slurm).
    hostname = socket.gethostname()
    intra_node_block = _to_intra_node_block(
        {hostname: dist},
        world_size=topo["gpus_per_node"],
    )

    env_status = "tentative"  # T2/T3 not implemented in this revision
    overall_status = "tentative"

    notes: list[str] = []
    notes.append(f"T1 connectivity: {'ok' if t1.get('pass') else 'FAIL: ' + str(t1.get('msg'))}")
    notes.append("T2 (env candidate micro-bench) and T3 (multi-node short run) not implemented.")

    warnings: list[str] = []
    if not t1.get("pass"):
        warnings.append(f"T1 connectivity failed: {t1.get('msg')}")
        env_status = "unsafe_fallback"
        overall_status = "tentative"
    warnings.append("env_probe T2/T3 not implemented (single-node demo)")
    warnings.append("single-node only: inter-node bandwidth not measured")
    if compute.get("per_node_variance_pct", 0) > 5:
        warnings.append(
            f"compute peak variance across {topo['gpus_per_node']} GPUs: "
            f"{compute['per_node_variance_pct']}%"
        )
    elif compute.get("per_node_variance_pct") is not None:
        warnings.append(
            f"compute peak variance across {topo['gpus_per_node']} GPUs: "
            f"{compute['per_node_variance_pct']}%"
        )

    profile: dict[str, Any] = {
        "schema_version": "2.0",
        "cluster_id": cfg.cluster_id,
        "version": version,
        "cluster_class": cluster_class,
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": overall_status,
        "supersedes": None,
        "nodes_total": 1,
        "nodes_healthy": 1,
        "nodes_blacklisted": [],
        "gpus_per_node": topo["gpus_per_node"],
        "compute": {
            "peak_tflops_bf16": compute["peak_tflops_bf16"],
            "peak_tflops_fp8": None,
            "peak_pct_of_spec_bf16": compute.get("peak_pct_of_spec_bf16"),
            "hbm_bandwidth_gbs": None,  # not measured in this revision
            "hbm_capacity_gb": compute["hbm_capacity_gb"],
            "per_node_variance_pct": compute["per_node_variance_pct"],
            "per_gpu_tflops_bf16": compute["per_gpu_tflops_bf16"],
        },
        "interconnect": {
            "intra_node": {
                "type": intra_type,
                "bandwidth_gbs": intra_bw,
                "topology": f"{topo['gpus_per_node']}-gpu-ring",
            },
            "inter_node": {
                "type": "ib",
                "bandwidth_gbs": None,
                "uniformity": "unknown",
            },
        },
        "rccl_baseline": {
            "intra_node": intra_node_block,
            "inter_node": None,  # populated by _dispatch_run_slurm
            "world": None,       # populated by _dispatch_run_slurm
        },
        "env_baseline": {
            "version": version,
            "status": env_status,
            "source": "vendor_default",
            "rccl": {},
            "hsa": {},
            "alloc": {},
            "threading": {
                "OMP_NUM_THREADS": str(
                    max(1, (os.cpu_count() or 1) // max(topo["gpus_per_node"], 1))
                ),
            },
            "notes": notes,
            "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "warnings": warnings,
        "metadata": {
            "rocm_version": topo.get("rocm_version"),
            "rccl_version": None,
            "torch_version": str(compute.get("_torch_version", "unknown")),
            "preflight_wallclock_s": round(time.time() - started, 2),
            "preflight_gpu_h": round(
                (time.time() - started) / 3600 * topo["gpus_per_node"], 5
            ),
            "hostname": socket.gethostname(),
            "image_label": plan.image_label,
        },
    }

    # === Persist atomically ===
    # Anchor relative paths at the pilot package root so artifacts always
    # land under `<repo>/pilot/state/cluster_profiles/` regardless of cwd.
    out_path = _resolve_pilot_path(out_dir) / f"{cfg.cluster_id}_{version}.yaml"
    _atomic_write_yaml(out_path, profile, schema_name="cluster_profile")

    return _build_subagent_result(profile, out_path, started)


# ---------------------------------------------------------------------------
# Step 1 — Topology discovery
# ---------------------------------------------------------------------------


_GFX_TO_MODEL = {
    "gfx950": "mi355x",
    "gfx942": "mi300x",
    "gfx940": "mi300x",
    "gfx90a": "mi250x",
    "gfx908": "mi100",
}

_VENDOR_PEAK_BF16 = {
    # Approximate vendor BF16 dense peaks per GPU (TFLOPs).
    # Used only for `peak_pct_of_spec_bf16`; not load-bearing.
    "mi355x": 2500.0,
    "mi300x": 1300.0,
    "mi250x": 380.0,
    "mi100": 184.6,
    "h100": 989.0,
    "a100": 312.0,
    "unknown": None,  # type: ignore[dict-item]
}


def _step1_topology(cfg: ClusterConfig) -> dict[str, Any]:
    """Probe local GPU topology.

    Uses ``torch.cuda.get_device_properties`` for the authoritative count and
    HBM size, then ``rocm-smi --version`` (best-effort) for the ROCm version
    string in metadata.

    Returns: ``{gpus_per_node, gfx_version, hbm_capacity_gb_per_gpu, rocm_version}``.
    """
    try:
        import torch
    except ImportError as exc:
        raise _PreflightError("TOOL_ERROR", f"torch not available: {exc}") from exc

    n = torch.cuda.device_count()
    if n == 0:
        raise _PreflightError("CLUSTER", "no GPUs visible to torch.cuda")

    if cfg.single and cfg.single.get("max_local_gpus") is not None:
        n = min(n, int(cfg.single["max_local_gpus"]))

    props0 = torch.cuda.get_device_properties(0)
    raw_gfx = getattr(props0, "gcnArchName", None)
    if raw_gfx:
        # Modern ROCm packs variant flags into gcnArchName, e.g.
        # 'gfx950:sramecc+:xnack-'. Strip everything after the first ':' so
        # downstream maps (gfx950 -> mi355x) work.
        gfx_version = raw_gfx.split(":", 1)[0]
    else:
        gfx_version = f"sm_{props0.major}{props0.minor}"
    hbm_gb = round(props0.total_memory / (1024**3), 1)

    rocm_version: str | None = None
    if shutil.which("rocm-smi"):
        try:
            r = subprocess.run(
                ["rocm-smi", "--version"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                for ln in r.stdout.splitlines():
                    if ":" in ln and "version" in ln.lower():
                        rocm_version = ln.split(":", 1)[1].strip()
                        break
        except subprocess.TimeoutExpired:
            pass

    return {
        "gpus_per_node": n,
        "gfx_version": gfx_version,
        "hbm_capacity_gb_per_gpu": hbm_gb,
        "rocm_version": rocm_version,
    }


def _derive_cluster_class(gfx_version: str, gpus_per_node: int) -> str:
    """Map gfx version -> coarse model name -> cluster_class.

    Examples:
        gfx950 + 8 -> 'mi355x_8gpu'
        gfx942 + 8 -> 'mi300x_8gpu'
        sm_90 + 8  -> 'unknown_8gpu'   (CUDA hosts not specially classified yet)
    """
    base = _GFX_TO_MODEL.get(gfx_version, "unknown")
    return f"{base}_{gpus_per_node}gpu"


# ---------------------------------------------------------------------------
# Step 2 — BF16 GEMM peak per GPU
# ---------------------------------------------------------------------------


def _step2_compute_peak(n_gpus: int, *, dim: int = 8192, runs: int = 8) -> dict[str, Any]:
    """Sequentially measure BF16 GEMM peak on each visible GPU.

    Single-stream matmul, M=N=K=`dim` (default 8192 → 1.1 TFLOP per matmul),
    median of `runs` after 3-iter warmup. Returns per-GPU TFLOPs plus
    aggregated stats.
    """
    try:
        import torch
    except ImportError as exc:
        raise _PreflightError("TOOL_ERROR", f"torch not available: {exc}") from exc

    flops_per_matmul = 2.0 * dim * dim * dim
    per_gpu: list[float] = []

    for i in range(n_gpus):
        try:
            torch.cuda.set_device(i)
            a = torch.randn(dim, dim, dtype=torch.bfloat16, device=f"cuda:{i}")
            b = torch.randn(dim, dim, dtype=torch.bfloat16, device=f"cuda:{i}")

            for _ in range(3):  # warmup
                c = torch.matmul(a, b)
            torch.cuda.synchronize()

            times: list[float] = []
            for _ in range(runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

            times.sort()
            median = times[len(times) // 2]
            tflops = flops_per_matmul / median / 1e12
            per_gpu.append(round(tflops, 1))

            del a, b, c
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            print(f"[preflight] step2 GEMM on GPU {i} failed: {exc}", file=sys.stderr)
            per_gpu.append(0.0)

    valid = [v for v in per_gpu if v > 0]
    if valid:
        valid_sorted = sorted(valid)
        median_tflops = valid_sorted[len(valid_sorted) // 2]
        mean = sum(valid) / len(valid)
        std = (sum((v - mean) ** 2 for v in valid) / len(valid)) ** 0.5
        cv_pct = round(std / mean * 100, 2) if mean else 0.0
    else:
        median_tflops, cv_pct = 0.0, 0.0

    torch_version = str(getattr(__import__("torch"), "__version__", "unknown"))

    return {
        "peak_tflops_bf16": median_tflops,
        "per_gpu_tflops_bf16": per_gpu,
        "per_node_variance_pct": cv_pct,
        "_torch_version": torch_version,
    }


# ---------------------------------------------------------------------------
# Steps 3-5 — distributed AR / A2A baseline
# ---------------------------------------------------------------------------


def _steps_345_distributed(n_gpus: int, *, timeout_s: int = 300) -> dict[str, Any]:
    """Spawn ``torchrun -m pilot.tools._preflight_workers`` for the local node.

    The worker module emits a single JSON line on rank-0 stdout containing
    AR/A2A curves and a T1 connectivity verdict. We capture stdout, find the
    last JSON-shaped line, and return the parsed dict.

    Returns the parsed worker JSON, or a fallback dict with
    ``t1_connectivity.pass = False`` on any failure.
    """
    if n_gpus < 2:
        return {
            "world_size": 1,
            "device_count": n_gpus,
            "intra_node_bw_gbs": None,
            "rccl_baseline": {
                "allreduce": [], "allgather": [], "reduce_scatter": [],
                "broadcast": [], "alltoall": [],
            },
            "t1_connectivity": {
                "pass": False,
                "msg": f"only {n_gpus} GPU; need ≥ 2 for AllReduce",
            },
        }

    cmd = [
        sys.executable,
        "-m", "torch.distributed.run",
        "--nnodes=1",
        f"--nproc-per-node={n_gpus}",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=127.0.0.1:0",
        "-m", "pilot.tools._preflight_workers",
    ]

    env = os.environ.copy()
    env.setdefault("PILOT_AR_SIZES_MB", "1,16,64,256")
    env.setdefault("PILOT_A2A_SIZES_MB", "1,16,64")
    env.setdefault("OMP_NUM_THREADS", "8")

    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, env=env,
        )
    except subprocess.TimeoutExpired:
        return _dist_fallback(f"torchrun timeout > {timeout_s}s")
    except FileNotFoundError as exc:
        return _dist_fallback(f"torchrun not found: {exc}")

    if r.returncode != 0:
        return _dist_fallback(
            f"torchrun rc={r.returncode}; stderr tail: {r.stderr[-300:]}"
        )

    for ln in reversed(r.stdout.splitlines()):
        ln = ln.strip()
        if ln.startswith("{") and ln.endswith("}"):
            try:
                return json.loads(ln)
            except json.JSONDecodeError:
                continue

    return _dist_fallback(f"no JSON output from rank 0; stdout tail: {r.stdout[-200:]}")


def _dist_fallback(msg: str) -> dict[str, Any]:
    return {
        "intra_node_bw_gbs": None,
        "rccl_baseline": {
            "allreduce": [], "allgather": [], "reduce_scatter": [],
            "broadcast": [], "alltoall": [],
        },
        "t1_connectivity": {"pass": False, "msg": msg},
    }


# ---------------------------------------------------------------------------
# Persistence + SubagentResult
# ---------------------------------------------------------------------------


def _atomic_write_yaml(
    path: Path,
    data: dict[str, Any],
    *,
    schema_name: str | None = None,
) -> None:
    """Write YAML atomically: validate → write tmp → rename. Requires PyYAML.

    When ``schema_name`` is provided, the payload is validated against the
    matching JSON Schema before any bytes hit disk. Validation errors are
    raised as :class:`pilot.tools._schema.SchemaValidationError`, which the
    CLI dispatcher maps to ``failure.kind=TOOL_ERROR``.
    """
    import yaml
    if schema_name is not None:
        from pilot.tools._schema import validate
        validate(data, schema_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    tmp.replace(path)


def _build_subagent_result(
    profile: dict[str, Any],
    out_path: Path,
    started: float,
) -> dict[str, Any]:
    """Assemble the SubagentResult JSON returned to the Orchestrator."""
    compute = profile["compute"]
    rccl = profile["rccl_baseline"]
    # Headline AR@256MB comes from intra_node roll_up median.
    ar_256 = None
    intra = rccl.get("intra_node") or {}
    ar_block = intra.get("collectives", {}).get("allreduce", {})
    sizes = ar_block.get("sizes_mb", [])
    medians = ar_block.get("roll_up", {}).get("median_bw_gbs", [])
    for s, bw in zip(sizes, medians):
        if s == 256:
            ar_256 = bw
            break

    pct = compute.get("peak_pct_of_spec_bf16")
    pct_str = f" ({pct * 100:.0f}% of spec)" if pct else ""
    cluster_short = profile["cluster_class"].split("_")[0]
    headline = (
        f"{profile['gpus_per_node']}x{cluster_short} on 1 node, "
        f"peak BF16 {compute['peak_tflops_bf16']:.0f} TFLOPs{pct_str}, "
        f"env_baseline={profile['env_baseline']['version']} "
        f"{profile['env_baseline']['status']}"
    )

    return {
        "stage": "PREFLIGHT",
        "status": profile["status"],
        "artifacts": [{"kind": "ClusterProfile", "ref": str(out_path)}],
        "summary": {
            "headline": headline,
            "key_metrics": {
                "nodes_healthy": profile["nodes_healthy"],
                "nodes_total": profile["nodes_total"],
                "peak_tflops_bf16": compute["peak_tflops_bf16"],
                "peak_pct_of_spec": pct,
                "ib_bw_gbs": None,
                "rccl_ar_256mb_gbs": ar_256,
                "env_baseline_version": profile["env_baseline"]["version"],
                "env_baseline_status": profile["env_baseline"]["status"],
                "blacklist_proposals": 0,
            },
            "warnings": profile.get("warnings", []),
        },
        "suggested_transition": {
            "to": "PROJECTION",
            "reason": (
                "cluster ready"
                if profile["status"] == "validated"
                else "tentative; T2/T3 staged for follow-up"
            ),
        },
        "cost": {
            "gpu_h": profile["metadata"]["preflight_gpu_h"],
            "wallclock_s": profile["metadata"]["preflight_wallclock_s"],
            "tool_calls": 1 + profile["gpus_per_node"],  # smi + per-gpu gemm
        },
        "failure": None,
    }


def _dispatch_run_slurm(
    cfg: ClusterConfig,
    plan: LaunchPlan,
    **kwargs: Any,
) -> dict[str, Any]:
    """Multi-node measurement path. Fans out via srun --jobid=<id>.

    Schema 2.0 requires three independent collective measurements; each maps
    to a dedicated `srun` invocation that this dispatcher orchestrates:

      Run 1 (intra_node, parallel across all nodes):
        srun --jobid=<jid> -N <nnodes> --ntasks-per-node=1 --export=ALL \\
             python -m torch.distributed.run \\
                  --nnodes=1 --nproc-per-node=<gpus_per_node> \\
                  --rdzv-endpoint=127.0.0.1:0 \\
                  -m pilot.tools._preflight_workers
        Each node runs an independent local 8-GPU PG; per-node JSON
        captured via stdout (`SLURM_NODEID` / hostname tag) and pivoted
        into the columnar `intra_node.collectives.<coll>` block.

      Run 2 (inter_node, single ring):
        srun --jobid=<jid> -N <nnodes> --ntasks-per-node=1 --export=ALL \\
             python -m pilot.tools._preflight_node_entry \\
                  --nnodes=<nnodes> --nproc-per-node=1 \\
                  --rdzv-endpoint=<plan.rdzv_endpoint> \\
                  --rdzv-id=<plan.rdzv_id>
        Rank 0 emits a single JSON line; consumed verbatim into the
        `inter_node` block (single_ring_collective form).

      Run 3 (world, full N x gpus_per_node ring):
        srun --jobid=<jid> -N <nnodes> --ntasks-per-node=1 --export=ALL \\
             python -m pilot.tools._preflight_node_entry \\
                  --nnodes=<nnodes> --nproc-per-node=<gpus_per_node> \\
                  --rdzv-endpoint=<plan.rdzv_endpoint> \\
                  --rdzv-id=<plan.rdzv_id>
        Rank 0 emits a single JSON line consumed into the `world` block.

    Steps 1 (topology) and 2 (compute peak) reuse the intra_node fan-out:
    each node reports its local GPU count + GFX + per-GPU TFLOPs along
    with the RCCL curves, and this dispatcher aggregates per-node data
    into compute.per_gpu_tflops_bf16 / per_node_variance_pct.

    Status: SCAFFOLDED. The cluster.yaml contract + LaunchPlan derivation
    are fully validated by `_cluster_config.preflight_check`; the actual
    srun fan-out is staged for follow-up. See:
        TODO followup-slurm-dispatch
    """
    raise NotImplementedError(
        "pilot.tools.preflight.run [mode=slurm]: 3-srun fan-out staged for "
        "follow-up; cluster.yaml contract validated successfully "
        f"(cluster_id={cfg.cluster_id}, slurm_job_id={plan.slurm_job_id}, "
        f"nnodes={plan.nnodes}, head={plan.head_host}, "
        f"rdzv_endpoint={plan.rdzv_endpoint}, image_label={plan.image_label})"
    )


# ---------------------------------------------------------------------------
# CLI dispatcher
# ---------------------------------------------------------------------------


_EXIT_OK = 0
_EXIT_STAGE_FAILED = 1
_EXIT_USAGE = 2
_EXIT_TOOL_ERROR = 3
_EXIT_CLUSTER = 4


def _emit(payload: dict[str, Any]) -> None:
    json.dump(payload, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def _failure(kind: str, message: str, *, stage: str = "PREFLIGHT") -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "failed",
        "failure": {
            "kind": kind,
            "message": message,
            "escalate_to_orchestrator": True,
        },
    }


def _add_cluster_config_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--cluster-config",
        default=None,
        help=(
            "Path to cluster.yaml. Resolution priority: this flag > "
            "$PILOT_CLUSTER_CONFIG > ./cluster.yaml. See pilot/SETUP.md."
        ),
    )


def _cli() -> int:
    p = argparse.ArgumentParser(prog="pilot.tools.preflight")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Collect ClusterProfile (5-step protocol)")
    _add_cluster_config_arg(p_run)
    p_run.add_argument(
        "--reason",
        choices=["bootstrap", "reentry_hang", "reentry_cluster", "reentry_stale", "force"],
        default="bootstrap",
    )
    p_run.add_argument("--target-version", default=None)
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--delta-only", action="store_true")
    p_run.add_argument("--max-wallclock-s", type=int, default=1800)
    # Default paths are anchored at the pilot package root; `state/...` therefore
    # resolves to `<repo>/pilot/state/...` regardless of cwd. Pass an absolute
    # path to override, or another relative path to anchor it against pilot/.
    p_run.add_argument("--blacklist", default="state/blacklist.yaml")
    p_run.add_argument("--out-dir", default="state/cluster_profiles")

    p_probe = sub.add_parser("env_probe", help="Validate env_baseline candidate (3-tier)")
    _add_cluster_config_arg(p_probe)
    p_probe.add_argument("--candidate-file", default=None)
    p_probe.add_argument("--tier", choices=["1", "2", "3", "all"], default="all")
    p_probe.add_argument("--out-dir", default="state/env_probe_results")

    p_sweep = sub.add_parser("env_sweep", help="Inner-loop env sweep")
    _add_cluster_config_arg(p_sweep)
    p_sweep.add_argument("--base-plan", required=True)
    p_sweep.add_argument(
        "--candidates",
        required=True,
        help="JSON literal or path to a YAML/JSON file listing candidate env diffs.",
    )
    p_sweep.add_argument("--max-steps", type=int, default=50)
    p_sweep.add_argument("--out-dir", default=None)

    args = p.parse_args()

    # Universal step: load cluster.yaml + LaunchPlan + 3 fast-fail checks BEFORE
    # any subcommand-specific work.
    try:
        cfg, plan = preflight_check(args.cluster_config)
    except ClusterConfigError as exc:
        _emit(cluster_config_failure(exc))
        return _EXIT_CLUSTER

    try:
        if args.cmd == "run":
            result = run(
                cfg, plan,
                reason=args.reason,
                target_version=args.target_version,
                force=args.force,
                delta_only=args.delta_only,
                max_wallclock_s=args.max_wallclock_s,
                blacklist_path=args.blacklist,
                out_dir=args.out_dir,
            )
        elif args.cmd == "env_probe":
            candidate = None
            if args.candidate_file:
                import yaml  # local import — only required when invoked
                with open(args.candidate_file) as f:
                    candidate = yaml.safe_load(f)
            result = env_probe(
                cfg, plan,
                candidate=candidate,
                tier=args.tier,
                out_dir=args.out_dir,
            )
        elif args.cmd == "env_sweep":
            try:
                cands = json.loads(args.candidates)
            except json.JSONDecodeError:
                import yaml  # local import
                with open(args.candidates) as f:
                    cands = yaml.safe_load(f)
            result = env_sweep(
                cfg, plan,
                base_plan_ref=args.base_plan,
                candidates=cands,
                max_steps=args.max_steps,
                out_dir=args.out_dir,
            )
        else:
            _emit(_failure("UNKNOWN", f"unknown subcommand {args.cmd!r}"))
            return _EXIT_USAGE

        _emit(result)
        return _EXIT_OK if result.get("status") != "failed" else _EXIT_STAGE_FAILED

    except _PreflightError as exc:
        _emit(_failure(exc.kind, str(exc)))
        return _EXIT_STAGE_FAILED
    except NotImplementedError as exc:
        _emit(_failure("TOOL_ERROR", f"not implemented: {exc}"))
        return _EXIT_TOOL_ERROR
    except Exception as exc:  # imported lazily so module always loads
        from pilot.tools._schema import SchemaValidationError
        if isinstance(exc, SchemaValidationError):
            _emit(_failure("TOOL_ERROR", f"schema validation failed: {exc}"))
            return _EXIT_TOOL_ERROR
        raise
    except FileNotFoundError as exc:
        _emit(_failure("TOOL_ERROR", f"file not found: {exc}"))
        return _EXIT_STAGE_FAILED
    except Exception as exc:  # noqa: BLE001
        _emit(_failure("UNKNOWN", f"{type(exc).__name__}: {exc}"))
        return _EXIT_STAGE_FAILED


if __name__ == "__main__":
    sys.exit(_cli())
