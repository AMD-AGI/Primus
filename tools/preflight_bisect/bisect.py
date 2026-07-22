#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# Minimal Slurm nodelist bisection for Primus preflight --perf-test.
# Run from repo root (or any cwd; script resolves repo root for runner/ path).
#
# Usage (typical):
#   export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate
#   cd /path/to/Primus
#   python tools/preflight_bisect/bisect.py --nodelist "node[01-32]" -p gpus ...
#
# Restricting the per-trial preflight test (all optional, default = stock full suite):
#   --all-reduce-inter          shorthand: inter-node all-reduce ONLY, a few sizes, no PDF
#                               (= --tests inter-allreduce --inter-group-sizes all
#                                  --inter-comm-sizes-mb <sizes> --disable-pdf)
#   --comm-sizes-mb 64,256,1024 sizes used by --all-reduce-inter
#   -- <preflight args...>      everything after a literal `--` is forwarded verbatim to
#                               preflight, e.g.  ... -- --tests inter-allreduce
#
# Post-bisection verification (both OFF by default; independent, composable):
#   --pin-suspects   rerun the failing subset for consistency, then pin each constituent
#                    against a known-good node -> a single culprit (or, if neither node
#                    fails alone, the pair flagged as an EDGE fault: bad link/switch/route).
#   --confirm-rest   coverage all-reduce over all nodes minus culprits, to rule out a
#                    second/confounding fault.
#
# Container mode (--container, OFF by default; production uses the stock direct/venv path):
#   routes each trial through runner/primus-cli-slurm-entry.sh `container` for an
#   in-container RoCE run. Off by default so nothing changes for the venv workflow.
#
# Caveats (see also --help):
#   - Scale-only hangs: subsets may all PASS while full N fails; suspects may be empty.
#   - Multiple bad nodes: union of singleton suspects from failing subtrees.
#   - Tune --trial-timeout-sec to ~2-3x healthy full-N runtime; too short causes false HANG.
#   - --scancel-user-on-hang kills ALL your Slurm jobs; do not use if you have other work.
###############################################################################
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Container-mode settings (only used when --container is passed). Site/version-
# specific, so they live in the run scripts and come in via the environment — no
# hard-coded values here. --image / --bnxt-tar override the env. main() requires
# these to be set when --container is passed.
_CONTAINER_NAME = os.environ.get("BISECT_CONTAINER_NAME", "")
_DEFAULT_IMAGE = os.environ.get("BISECT_IMAGE") or None
_DEFAULT_BNXT_TAR = os.environ.get("BISECT_BNXT_TAR") or None
# Sizes used by the --all-reduce-inter shorthand.
_DEFAULT_COMM_SIZES_MB = "64,256,1024"


def _repo_root() -> Path:
    # tools/preflight_bisect/bisect.py -> repo root is parent.parent.parent
    return Path(__file__).resolve().parent.parent.parent


def expand_nodelist(nodelist: str) -> list[str]:
    out = subprocess.check_output(
        ["scontrol", "show", "hostnames", nodelist],
        text=True,
        stderr=subprocess.PIPE,
    )
    hosts = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not hosts:
        raise SystemExit(f"scontrol produced no hostnames for nodelist={nodelist!r}")
    return hosts


def _fmt_dur(sec: float) -> str:
    return f"{sec:6.1f}s"


def _fmt_nodes(nodes: list[str]) -> str:
    """Compact slurm-style hostlist: [g32,g33,g54] -> 'g[32-33,54]'.

    Numeric suffixes are sorted and consecutive runs collapsed to a-b. Mixed
    prefixes fall back to a literal comma list. Leading zeros are not preserved
    (int() drops them); fine for un-padded names like g32."""
    if len(nodes) <= 1:
        return nodes[0] if nodes else ""
    m = [re.match(r"^(.*?)(\d+)$", n) for n in nodes]
    if not all(m) or len({x.group(1) for x in m}) != 1:
        return ",".join(nodes)
    prefix = m[0].group(1)
    nums = sorted(int(x.group(2)) for x in m)
    parts: list[str] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = n
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return f"{prefix}[{','.join(parts)}]"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _preflight_extra_args(args: argparse.Namespace) -> list[str]:
    """Flags appended after `preflight --perf-test ...`: the --all-reduce-inter
    shorthand (if set) followed by verbatim `-- ...` pass-through."""
    extra: list[str] = []
    if args.all_reduce_inter:
        extra += [
            "--tests", "inter-allreduce",
            "--inter-group-sizes", "all",
            "--inter-comm-sizes-mb", args.comm_sizes_mb,
            "--disable-pdf",
        ]
    extra += list(args.passthrough)
    return extra


def _cleanup_hung_trial(job_name: str, nodes: list[str], args: argparse.Namespace) -> None:
    """After a local SIGKILL, the Slurm job can keep holding the allocation and
    (in container mode) leave an orphaned docker container that name-conflicts the
    next trial. Free both, targeting only THIS trial's job (safe alongside other work)."""
    subprocess.run(
        ["scancel", "--name", job_name, "--user", os.environ.get("USER", "")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if args.container:
        # Prefix match so we also catch the per-job container name the runner
        # produces (primus-cli-container.sh appends -$SLURM_JOB_ID), not just the
        # bare _CONTAINER_NAME.
        for n in nodes:
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", n,
                 f"docker ps -aq --filter name={_CONTAINER_NAME} | xargs -r docker rm -f"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )


def _good_ref(subset: list[str], trials: list[dict[str, Any]]) -> str | None:
    """A known-good node to pin suspects against: any node from a PASSing n>=2 trial
    that is not itself in the failing subset."""
    subset_set = set(subset)
    for t in sorted(trials, key=lambda tr: tr["idx"]):
        if t["status"] == "pass" and t["n"] >= 2:
            for n in t["nodes"]:
                if n not in subset_set:
                    return n
    return None


@dataclass
class BisectState:
    max_concurrent_trials: int
    idx: int = 0
    trials: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _trial_slots: threading.BoundedSemaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._trial_slots = threading.BoundedSemaphore(self.max_concurrent_trials)

    def next_trial_idx(self) -> int:
        with self._lock:
            idx = self.idx
            self.idx += 1
            return idx

    def record_trial(
        self, idx: int, nodes: list[str], status: str, dur: float = 0.0,
        tag: str = "", phase: str = "bisect"
    ) -> None:
        record = {
            "idx": idx, "n": len(nodes), "status": status, "nodes": list(nodes),
            "dur": dur, "tag": tag, "phase": phase,
        }
        with self._lock:
            self.trials.append(record)

    def ordered_trials(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(self.trials, key=lambda trial: trial["idx"])

    def acquire_trial_slot(self) -> None:
        self._trial_slots.acquire()

    def release_trial_slot(self) -> None:
        self._trial_slots.release()


def run_trial(
    nodes: list[str],
    trial_idx: int,
    state: BisectState,
    args: argparse.Namespace,
    runner: Path,
    out_dir: Path,
    tag: str | None = None,
) -> tuple[str, float]:
    """Run one preflight perf trial. Returns (status, elapsed_sec) where status
    is 'pass', 'fail', or 'hang'.

    `tag` names the log file, the preflight --report-file-name, and the srun
    --job-name; it defaults to trial-NNN for the bisection phase, and is set
    explicitly (e.g. pin-<node>, confirm-rest) by the verification phases."""
    tag = tag or f"trial-{trial_idx:03d}"
    subset = ",".join(nodes)
    log_path = out_dir / f"{tag}.log"
    job_name = f"bisect-{tag}"
    cmd: list[str] = [
        "srun",
        f"--job-name={job_name}",
        f"-N{len(nodes)}",
        f"--nodelist={subset}",
        "-n",
        str(len(nodes)),
        "--ntasks-per-node=1",
        "-c",
        str(args.cpus_per_task),
        f"--gres=gpu:{args.gpus_per_node}",
        f"-t{args.slurm_time}",
    ]
    if args.partition:
        cmd.extend(["-p", args.partition])
    # Per-trial env overrides are propagated via srun --export so every rank on
    # every node sees them (the consolidated primus-cli launcher does accept
    # --env KEY=VALUE, but only rank 0 would see those; --export covers all
    # ranks). ALL keeps the caller's environment (notably VENV_ACTIVATE)
    # intact; the trailing K=V pairs override / add on top. SLURM tokenizes
    # --export on commas, so values must not contain ',' or whitespace
    # (NCCL flags never do).
    export_val = "ALL"
    if args.preflight_env:
        export_val += "," + ",".join(args.preflight_env)
    cmd.append(f"--export={export_val}")
    cmd.append(str(runner))
    # Launcher path: stock venv `direct` (default, production) or in-container.
    if args.container:
        cmd.extend([
            "container",
            "--image", args.image,
            "--volume", "/home:/home",
            "--env", "REBUILD_BNXT=1",
            "--env", f"PATH_TO_BNXT_TAR_PACKAGE={args.bnxt_tar}",
        ])
    else:
        cmd.append("direct")
    cmd.extend(["--", "preflight", "--perf-test", "--report-file-name", tag])
    cmd.extend(_preflight_extra_args(args))

    state.acquire_trial_slot()
    try:
        header = (
            f"CMD: {' '.join(cmd)}\n"
            f"NODES ({len(nodes)}): {subset}\n"
            f"START: {datetime.now(timezone.utc).isoformat()}\n\n"
        )
        print(
            f"[{tag}] N={len(nodes)} {_fmt_nodes(nodes)} -> {log_path.name}",
            flush=True,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        with log_path.open("wb", buffering=0) as logf:
            logf.write(header.encode())

            proc = subprocess.Popen(
                cmd,
                cwd=str(_repo_root()),
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                rc = proc.wait(timeout=args.trial_timeout_sec)
                return ("pass" if rc == 0 else "fail", time.monotonic() - start)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    pass  # process is stuck in D-state; SIGKILL is pending, move on
                # SIGKILL only kills the local srun client; free THIS trial's Slurm
                # allocation (and orphaned container) so the next trial isn't starved.
                _cleanup_hung_trial(job_name, nodes, args)
                if args.scancel_user_on_hang:
                    subprocess.run(
                        ["scancel", "--signal=KILL", "--user", os.environ.get("USER", "")],
                        stderr=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                    )
                return ("hang", time.monotonic() - start)
    finally:
        state.release_trial_slot()


def bisect(
    nodes: list[str],
    state: BisectState,
    args: argparse.Namespace,
    runner: Path,
    out_dir: Path,
) -> list[str]:
    idx = state.next_trial_idx()
    status, dur = run_trial(nodes, idx, state, args, runner, out_dir)
    state.record_trial(idx, nodes, status, dur)

    if status == "pass":
        return []
    if len(nodes) == 1:
        return list(nodes)

    mid = len(nodes) // 2
    if args.max_concurrent_trials == 1:
        left = bisect(nodes[:mid], state, args, runner, out_dir)
        right = bisect(nodes[mid:], state, args, runner, out_dir)
        combined = left + right
    else:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="preflight-bisect") as executor:
            right_future = executor.submit(bisect, nodes[mid:], state, args, runner, out_dir)
            left = bisect(nodes[:mid], state, args, runner, out_dir)
            right = right_future.result()
        combined = left + right

    # This subset failed but both halves passed on their own -> the fault is a
    # PROPERTY OF THE SUBSET, not of a single node (e.g. a fabric fault needs >=2
    # nodes to manifest; a lone node runs num_nodes=1 and skips inter-node comm).
    # Return the minimal failing subset so --pin-suspects can localize it.
    if combined:
        return combined
    return list(nodes)


def _run_and_record(
    nodes: list[str],
    state: BisectState,
    args: argparse.Namespace,
    runner: Path,
    out_dir: Path,
    tag: str,
    phase: str,
) -> str:
    """run_trial + record with an explicit tag/phase (used by the verification phases)."""
    idx = state.next_trial_idx()
    status, dur = run_trial(nodes, idx, state, args, runner, out_dir, tag=tag)
    state.record_trial(idx, nodes, status, dur, tag=tag, phase=phase)
    return status


def pin_suspects(
    suspects: list[str],
    state: BisectState,
    args: argparse.Namespace,
    runner: Path,
    out_dir: Path,
) -> tuple[list[str], list[str]]:
    """Localize a failing subset. Returns (culprits, edge_nodes):
      - culprits: node(s) that fail even when paired with a known-good node.
      - edge_nodes: non-empty only when NO constituent fails alone but the subset
        reproducibly fails -> a pairwise/edge fault (bad link/switch/route), not a
        node fault. Mutually exclusive with culprits.
    Empty (both) means the failure did not reproduce (transient) or could not be pinned."""
    if not suspects:
        return [], []

    # Step 1: rerun the failing subset to confirm the failure is consistent.
    if len(suspects) >= 2:
        st = _run_and_record(suspects, state, args, runner, out_dir, tag="pin-rerun", phase="pin")
        if st == "pass":
            print("[pin] failing subset PASSED on rerun -> transient, not convicting", flush=True)
            return [], []

    # Step 2: pin each constituent against a known-good reference node.
    ref = _good_ref(suspects, state.ordered_trials())
    if ref is None:
        print("[pin] no known-good reference available -> keeping raw suspects", flush=True)
        return list(suspects), []

    failed: list[str] = []
    for s in suspects:
        st = _run_and_record([s, ref], state, args, runner, out_dir, tag=f"pin-{s}", phase="pin")
        if st != "pass":
            failed.append(s)

    # Step 3: verdict.
    if failed:
        return failed, []
    # None failed alone, yet the subset fails -> the fault lives on the edge between them.
    return [], list(suspects)


def confirm_rest(
    all_nodes: list[str],
    excluded: list[str],
    state: BisectState,
    args: argparse.Namespace,
    runner: Path,
    out_dir: Path,
) -> tuple[str | None, list[str]]:
    """Coverage all-reduce over all nodes minus `excluded`, to rule out a second
    fault. Returns (status_or_None, rest_nodes); None if <2 nodes remain."""
    rest = [n for n in all_nodes if n not in set(excluded)]
    if len(rest) < 2:
        return None, rest
    status = _run_and_record(rest, state, args, runner, out_dir, tag="confirm-rest", phase="confirm-rest")
    return status, rest


def write_summary(out_dir: Path, nodes: list[str], result: dict[str, Any], trials: list[dict[str, Any]]) -> None:
    path = out_dir / "summary.txt"
    lines = [
        f"{datetime.now(timezone.utc).isoformat()} bisect nodes={len(nodes)}",
    ]
    for t in sorted(trials, key=lambda trial: trial["idx"]):
        r = _fmt_nodes(t["nodes"])
        phase = t.get("phase", "bisect")
        lines.append(
            f"[{t['idx']:03d}] N={t['n']:2d} {t['status'].upper():4s} {phase:12s} "
            f"{_fmt_dur(t.get('dur', 0.0))} nodes={r}"
        )
    lines.append("")

    suspects = result["suspects"]
    lines.append("SUSPECT_NODES: " + (" ".join(sorted(set(suspects))) if suspects else "(none)"))

    if result["pinned"]:
        culprits, edge = result["culprits"], result["edge"]
        if culprits:
            lines.append("PIN: CULPRIT_NODES: " + " ".join(sorted(set(culprits))))
        elif edge:
            lines.append(
                "PIN: EDGE_FAULT (pairwise, not node-localizable; suspect link/switch/route): "
                + " ".join(sorted(set(edge)))
            )
        else:
            lines.append("PIN: no culprit (failure did not reproduce or could not be pinned)")

    if result["confirmed"]:
        cov = result["coverage"]
        rest = _fmt_nodes(result["coverage_nodes"])
        if cov is None:
            lines.append("CONFIRM_REST: skipped (<2 nodes remaining)")
        else:
            lines.append(f"CONFIRM_REST: coverage over {rest} -> {cov.upper()}")

    fault = result["culprits"] or result["edge"] or result["suspects"]
    lines.append("FAULT: " + (" ".join(sorted(set(fault))) if fault else "(none)"))

    summary = "\n".join(lines) + "\n"
    path.write_text(summary, encoding="utf-8")
    print(summary, end="")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Recursively bisect a Slurm nodelist using Primus preflight --perf-test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment:
  Export VENV_ACTIVATE to your venv activate script before running (required by
  runner/primus-cli direct -> primus-cli-direct.sh). See docs/preflight-direct.md.

Caveats:
  - If the hang only reproduces at full scale, all subsets may PASS -> SUSPECT_NODES empty.
  - Multiple faulty nodes yield a union of suspects from failing singleton trials.
  - By default, failing sibling subsets launch in parallel (up to 2 concurrent trials).
  - --trial-timeout-sec too low marks healthy runs as HANG.
  - --scancel-user-on-hang cancels ALL jobs for $USER; only use it with --max-concurrent-trials=1.
""",
    )
    p.add_argument("--nodelist", required=True, help='Slurm nodelist expression, e.g. "node[01-32]"')
    p.add_argument("-p", "--partition", default="", help="Slurm partition (-p), optional")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bisect-out"),
        help="Directory for trial-*.log and summary.txt (default: ./bisect-out)",
    )
    p.add_argument(
        "--trial-timeout-sec",
        type=int,
        default=900,
        help="Wall-clock timeout per trial in seconds (default: 900)",
    )
    p.add_argument(
        "--slurm-time",
        default="00:45:00",
        help="srun -t limit per trial (default: 00:45:00)",
    )
    p.add_argument(
        "--max-concurrent-trials",
        type=_positive_int,
        default=2,
        help="Maximum concurrent subset trials (default: 2). Set to 1 to force sequential execution.",
    )
    p.add_argument("--cpus-per-task", type=int, default=128, help="srun -c (default: 128)")
    p.add_argument(
        "--gpus-per-node",
        type=int,
        default=8,
        help="GPUs per node; emitted as srun --gres=gpu:N (default: 8)",
    )
    p.add_argument(
        "--preflight-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Repeatable; propagated into each trial via "
            "'srun --export=ALL,KEY=VALUE,...'. Values must not contain ',' or whitespace."
        ),
    )
    p.add_argument(
        "--runner",
        type=Path,
        default=None,
        help=(
            "Override path to the launcher invoked per trial "
            "(default: repo runner/primus-cli). The bisector always appends "
            "'direct -- preflight --perf-test --report-file-name ...' after "
            "the runner path; custom runners that ignore positional args "
            "(e.g. fake_runner.sh) work transparently."
        ),
    )
    p.add_argument(
        "--scancel-user-on-hang",
        action="store_true",
        help="On timeout, also run: scancel --signal=KILL --user $USER (DANGEROUS)",
    )
    # --- per-trial preflight test selection (optional; default = stock full suite) ---
    p.add_argument(
        "--all-reduce-inter",
        action="store_true",
        help="Shorthand for inter-node all-reduce only: --tests inter-allreduce "
             "--inter-group-sizes all --inter-comm-sizes-mb <sizes> --disable-pdf",
    )
    p.add_argument(
        "--comm-sizes-mb",
        default=_DEFAULT_COMM_SIZES_MB,
        help=f"Sizes for --all-reduce-inter (default: {_DEFAULT_COMM_SIZES_MB})",
    )
    # --- post-bisection verification (both OFF by default; independent) ---
    p.add_argument(
        "--pin-suspects",
        action="store_true",
        help="Rerun the failing subset then pin each constituent vs a known-good node "
             "-> single culprit, or the pair flagged as an edge fault.",
    )
    p.add_argument(
        "--confirm-rest",
        action="store_true",
        help="Coverage all-reduce over all nodes minus culprits, to rule out a second fault.",
    )
    # --- container mode (OFF by default; production uses the stock direct/venv path) ---
    p.add_argument(
        "--container",
        action="store_true",
        help="Route each trial through runner/primus-cli-slurm-entry.sh `container` "
             "(in-container RoCE run) instead of the venv `direct` path.",
    )
    p.add_argument("--image", default=_DEFAULT_IMAGE, help="Container image (with --container)")
    p.add_argument("--bnxt-tar", default=_DEFAULT_BNXT_TAR, help="libbnxt_re tar for the in-container RoCE rebuild")
    p.add_argument(
        "passthrough",
        nargs="*",
        metavar="-- PREFLIGHT_ARGS",
        help="Everything after a literal `--` is forwarded verbatim to preflight.",
    )
    args = p.parse_args()
    if args.scancel_user_on_hang and args.max_concurrent_trials > 1:
        p.error("--scancel-user-on-hang is only supported with --max-concurrent-trials=1")
    if args.max_concurrent_trials > 1:
        print(
            f"WARNING: --max-concurrent-trials={args.max_concurrent_trials} runs trials "
            "concurrently. This can oversubscribe the reservation, muddle per-trial logs, "
            "and complicate hang cleanup. Use 1 (sequential) unless you know you want this.",
            file=sys.stderr,
        )

    if args.container:
        missing = []
        if not args.image:
            missing.append("BISECT_IMAGE (or --image)")
        if not args.bnxt_tar:
            missing.append("BISECT_BNXT_TAR (or --bnxt-tar)")
        if not _CONTAINER_NAME:
            missing.append("BISECT_CONTAINER_NAME")
        if missing:
            print(
                "ERROR: --container requires these to be set: " + ", ".join(missing) + "\n"
                "  Set them in your run script, e.g.\n"
                '    export BISECT_CONTAINER_NAME=primus-training\n'
                '    export BISECT_IMAGE=rocm/megatron-lm:v25.8_py310\n'
                '    export BISECT_BNXT_TAR=/path/to/libbnxt_re-<ver>.tar.gz',
                file=sys.stderr,
            )
            return 2

    # VENV_ACTIVATE is only needed by the venv `direct` path; container mode ignores it.
    if not args.container and not os.environ.get("VENV_ACTIVATE"):
        print(
            "ERROR: VENV_ACTIVATE is not set. Export the path to your venv's "
            "bin/activate before running, e.g.\n"
            "  export VENV_ACTIVATE=~/envs/preflight/.venv/bin/activate\n"
            "  (or pass --container to use the in-container path instead)",
            file=sys.stderr,
        )
        return 2

    default_runner = "primus-cli-slurm-entry.sh" if args.container else "primus-cli"
    runner = args.runner or (_repo_root() / "runner" / default_runner)
    if not runner.is_file():
        print(f"ERROR: runner not found: {runner}", file=sys.stderr)
        return 2

    try:
        hosts = expand_nodelist(args.nodelist)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: scontrol failed: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError:
        print("ERROR: scontrol not found; run this script on a Slurm login/head node.", file=sys.stderr)
        return 2

    out_dir = args.output_dir.resolve()
    state = BisectState(max_concurrent_trials=args.max_concurrent_trials)
    suspects = bisect(hosts, state, args, runner, out_dir)

    culprits, edge = [], []
    if args.pin_suspects and suspects:
        culprits, edge = pin_suspects(suspects, state, args, runner, out_dir)

    coverage, coverage_nodes = None, []
    if args.confirm_rest:
        excluded = culprits or edge or suspects
        coverage, coverage_nodes = confirm_rest(hosts, excluded, state, args, runner, out_dir)

    result = {
        "suspects": suspects,
        "pinned": args.pin_suspects,
        "culprits": culprits,
        "edge": edge,
        "confirmed": args.confirm_rest,
        "coverage": coverage,
        "coverage_nodes": coverage_nodes,
    }
    write_summary(out_dir, hosts, result, state.ordered_trials())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
