#!/usr/bin/env python3
###############################################################################
# Minimal Slurm nodelist bisection for Primus preflight --perf-test.
# Run from repo root (or any cwd; script resolves repo root for runner/ path).
#
# Usage (typical):
#   export VENV_PATH=~/envs/preflight/.venv/bin/activate
#   cd /path/to/Primus
#   python tools/preflight_bisect/bisect.py --nodelist "node[01-32]" -p gpus ...
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
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def _format_node_range(nodes: list[str]) -> str:
    if not nodes:
        return ""
    if len(nodes) == 1:
        return nodes[0]
    return f"{nodes[0]}..{nodes[-1]}"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


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

    def record_trial(self, idx: int, nodes: list[str], status: str) -> None:
        record = {"idx": idx, "n": len(nodes), "status": status, "nodes": list(nodes)}
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
) -> str:
    """Run one preflight perf trial. Returns 'pass', 'fail', or 'hang'."""
    subset = ",".join(nodes)
    log_path = out_dir / f"trial-{trial_idx:03d}.log"
    cmd: list[str] = [
        "srun",
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
    cmd.append(str(runner))
    # Use the space-separated form "--env KEY=VALUE" rather than the equals
    # form "--env=KEY=VALUE" so runner/primus-cli-direct-preflight.sh recognizes
    # --env as a runner-level option and consumes it. The equals form leaks past
    # the runner's arg parser into the primus CLI, which does not accept --env
    # and errors with "argument command: invalid choice: '--'".
    for kv in args.preflight_env:
        cmd.extend(["--env", kv])
    cmd.extend(
        [
            "--",
            "preflight",
            "--perf-test",
            "--report-file-name",
            f"trial-{trial_idx:03d}",
        ]
    )

    state.acquire_trial_slot()
    try:
        header = (
            f"CMD: {' '.join(cmd)}\n"
            f"NODES ({len(nodes)}): {subset}\n"
            f"START: {datetime.now(timezone.utc).isoformat()}\n\n"
        )
        print(
            f"[trial {trial_idx:03d}] N={len(nodes)} {_format_node_range(nodes)} -> {log_path.name}",
            flush=True,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
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
                return "pass" if rc == 0 else "fail"
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=60)
                if args.scancel_user_on_hang:
                    subprocess.run(
                        ["scancel", "--signal=KILL", "--user", os.environ.get("USER", "")],
                        stderr=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                    )
                return "hang"
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
    status = run_trial(nodes, idx, state, args, runner, out_dir)
    state.record_trial(idx, nodes, status)

    if status == "pass":
        return []
    if len(nodes) == 1:
        return list(nodes)

    mid = len(nodes) // 2
    if args.max_concurrent_trials == 1:
        left = bisect(nodes[:mid], state, args, runner, out_dir)
        right = bisect(nodes[mid:], state, args, runner, out_dir)
        return left + right

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="preflight-bisect") as executor:
        right_future = executor.submit(bisect, nodes[mid:], state, args, runner, out_dir)
        left = bisect(nodes[:mid], state, args, runner, out_dir)
        right = right_future.result()
    return left + right


def write_summary(out_dir: Path, nodes: list[str], suspects: list[str], trials: list[dict[str, Any]]) -> None:
    path = out_dir / "summary.txt"
    lines = [
        f"{datetime.now(timezone.utc).isoformat()} bisect nodes={len(nodes)}",
    ]
    for t in sorted(trials, key=lambda trial: trial["idx"]):
        nlist = t["nodes"]
        r = _format_node_range(nlist)
        lines.append(f"[{t['idx']:03d}] N={t['n']:2d} {t['status'].upper():4s}  nodes={r}")
    if suspects:
        lines.append("SUSPECT_NODES: " + " ".join(sorted(set(suspects))))
    else:
        lines.append("SUSPECT_NODES: (none)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path.read_text(), end="")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Recursively bisect a Slurm nodelist using Primus preflight --perf-test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment:
  Export VENV_PATH to your venv activate script before running (required by
  runner/primus-cli-direct-preflight.sh). See docs/run-preflight-without-container.md.

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
    p.add_argument("--gpus-per-node", type=int, default=8, help="GPUs per node; emitted as srun --gres=gpu:N (default: 8)")
    p.add_argument(
        "--preflight-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable; passed as runner/primus-cli-direct-preflight.sh --env KEY=VALUE",
    )
    p.add_argument(
        "--runner",
        type=Path,
        default=None,
        help="Override path to primus-cli-direct-preflight.sh (default: repo runner/)",
    )
    p.add_argument(
        "--scancel-user-on-hang",
        action="store_true",
        help="On timeout, also run: scancel --signal=KILL --user $USER (DANGEROUS)",
    )
    args = p.parse_args()
    if args.scancel_user_on_hang and args.max_concurrent_trials > 1:
        p.error("--scancel-user-on-hang is only supported with --max-concurrent-trials=1")

    runner = args.runner or (_repo_root() / "runner" / "primus-cli-direct-preflight.sh")
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
    write_summary(out_dir, hosts, suspects, state.ordered_trials())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
