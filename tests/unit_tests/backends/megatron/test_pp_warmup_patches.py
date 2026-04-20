###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
End-to-end validation of the pp_warmup patch
(``primus/backends/megatron/patches/pp_warmup_patches.py``).

This test is entirely self-contained in a single file so it can be run with:

    # From the Primus repo root:
    pytest -s tests/unit_tests/backends/megatron/test_pp_warmup_patches.py

What this test checks
---------------------
(1) **Deterministic loss parity** — with ``PRIMUS_DETERMINISTIC=1`` enabled,
    the ``lm loss`` values printed by the training loop MUST be
    byte-identical between the baseline run (``pp_warmup=False``) and the
    warmup run (``pp_warmup=True``). If they differ, pp_warmup is leaking
    state into real training.

(2) **Iter-time telemetry** — reports iter-1, iter-2 and iter3..last average
    elapsed times for both runs. When env var ``MIN_SPEEDUP`` is set, it
    also asserts that iter-1 with pp_warmup is at least that many times
    faster than iter-1 without pp_warmup. When unset, timings are printed
    but no assertion is made.

Environment variables
---------------------
    TRAIN_ITERS                number of training iters per run (default 20)
    MIN_SPEEDUP                optional iter-1 speedup assertion (e.g. "2.0")
    UT_LOG_PATH                where to write per-run logs (default: <repo>/ut_out)
    PRIMUS_PP_WARMUP_PLATFORM  override platform detection (MI300X / MI355X)

Auto-skip conditions
--------------------
The test is skipped automatically when:
    * the host is not POSIX (``primus-cli`` requires bash)
    * ``primus-cli`` launcher is missing
    * no CUDA/ROCm GPUs are visible
    * fewer GPUs are visible than ``PP_SIZE * EP_SIZE`` (= 4 * 2 = 8)
    * the required experiment config file is missing
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import List, Optional, Tuple

_MBS = 1
_GBS = 16
_PRIMUS_TOTAL_LAYERS = 4
_PRIMUS_MOE_LAYER_FREQ = 1
_PRIMUS_PP = 4
_PRIMUS_EP = 2
_MTP_NUM_LAYERS = 0

# Overridable via the TRAIN_ITERS env var.
_DEFAULT_TRAIN_ITERS = 20

# gfx-arch → config-directory mapping (same table used in
# tests/trainer/test_megatron_trainer.py).
_GFX_TO_PLATFORM = {
    "gfx942": "MI300X",
    "gfx950": "MI355X",
}

_MIN_SPEEDUP_ENV = "MIN_SPEEDUP"
_TRAIN_ITERS_ENV = "TRAIN_ITERS"

# Marker emitted by PrimusRuntime at the end of a successful training run.
# Used by run_training_script() in tests/utils.py as well — we keep the same
# convention here so silent failures are caught.
_TRAINING_COMPLETED_MARKER = "Training completed."


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
# tests/unit_tests/backends/megatron/test_pp_warmup_patches.py → repo root
_REPO_ROOT = _THIS_FILE.parents[4]
_PRIMUS_CLI = _REPO_ROOT / "primus-cli"


def _detect_gpu_platform() -> str:
    """Return the platform-config directory name (MI300X / MI355X).

    Priority:
        1. PRIMUS_PP_WARMUP_PLATFORM env var override
        2. Detected GFX arch via torch.cuda.get_device_properties
        3. Fallback: MI355X (the test will still skip via `_should_skip()` on non-GPU hosts).
    """
    override = os.environ.get("PRIMUS_PP_WARMUP_PLATFORM", "").strip()
    if override:
        return override
    try:
        import torch  # noqa: E402 — optional, guarded by try/except

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            props = torch.cuda.get_device_properties(0)
            arch_raw = getattr(props, "gcnArchName", "") or ""
            arch = arch_raw.split(":")[0].strip()
            if arch in _GFX_TO_PLATFORM:
                return _GFX_TO_PLATFORM[arch]
    except Exception:  # pragma: no cover - best-effort detection, skip logic enforces prereqs
        pass
    return "MI355X"


def _exp_config_path(platform: str) -> Path:
    return _REPO_ROOT / "examples" / "megatron" / "configs" / platform / "deepseek_v2_lite-BF16-pretrain.yaml"


# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------


def _should_skip() -> Optional[str]:
    """Return a reason string if the test should be skipped, else None."""
    if sys.platform == "win32":
        return "primus-cli requires bash; not supported on Windows"

    if not _PRIMUS_CLI.exists():
        return f"primus-cli launcher not found at {_PRIMUS_CLI}"

    platform = _detect_gpu_platform()
    cfg = _exp_config_path(platform)
    if not cfg.exists():
        return f"exp config file not found: {cfg}"

    try:
        import torch  # noqa: E402 — optional, guarded by try/except
    except Exception as exc:  # pragma: no cover - dependency-guarded
        return f"torch not importable in this environment: {exc!r}"

    if not torch.cuda.is_available():
        return "CUDA/ROCm not available"

    available = torch.cuda.device_count()
    needed = _PRIMUS_PP * _PRIMUS_EP
    if available < needed:
        return f"needs at least {needed} GPUs (PP={_PRIMUS_PP} * EP={_PRIMUS_EP}); " f"got {available}"
    return None


# ---------------------------------------------------------------------------
# Command builder + runner
# ---------------------------------------------------------------------------


def _build_training_cmd(
    train_iters: int,
    pp_warmup: bool,
    log_file: Path,
    exp_path: Path,
) -> Tuple[List[str], dict]:
    pp_warmup_flag = "True" if pp_warmup else "False"
    cmd = [
        "bash",
        str(_PRIMUS_CLI),
        "direct",
        "--log_file",
        str(log_file),
        "--",
        "train",
        "pretrain",
        "--config",
        str(exp_path),
        "--num_layers",
        str(_PRIMUS_TOTAL_LAYERS),
        "--train_iters",
        str(train_iters),
        "--micro_batch_size",
        str(_MBS),
        "--global_batch_size",
        str(_GBS),
        "--pipeline_model_parallel_size",
        str(_PRIMUS_PP),
        "--expert_model_parallel_size",
        str(_PRIMUS_EP),
        "--moe_layer_freq",
        str(_PRIMUS_MOE_LAYER_FREQ),
        "--mock_data",
        "True",
        "--pp_warmup",
        pp_warmup_flag,
        "--use_turbo_attention",
        "False",
        "--use_turbo_deepep",
        "False",
        "--mtp_num_layers",
        str(_MTP_NUM_LAYERS),
    ]

    env = os.environ.copy()
    env.update(
        {
            "MBS": str(_MBS),
            "GBS": str(_GBS),
            "TRAIN_ITERS": str(train_iters),
            "PRIMUS_TOTAL_LAYERS": str(_PRIMUS_TOTAL_LAYERS),
            "PRIMUS_MOE_LAYER_FREQ": str(_PRIMUS_MOE_LAYER_FREQ),
            "PRIMUS_PP": str(_PRIMUS_PP),
            "PRIMUS_EP": str(_PRIMUS_EP),
            "PP_WARMUP": pp_warmup_flag,
            "PRIMUS_DETERMINISTIC": "1",
            "EXP": str(exp_path),
            # Log file path is also consumed by tests/utils.py-style hooks.
            "TRAIN_LOG": str(log_file),
        }
    )
    return cmd, env


def _run_once(
    tag: str,
    pp_warmup: bool,
    train_iters: int,
    log_file: Path,
    exp_path: Path,
) -> str:
    """Run a single primus-cli training invocation, stream output, return log text."""
    cmd, env = _build_training_cmd(train_iters, pp_warmup, log_file, exp_path)
    pretty_cmd = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n=== [{tag}] pp_warmup={pp_warmup} ===", flush=True)
    print(f"[{tag}] cmd: {pretty_cmd}", flush=True)
    print(f"[{tag}] log: {log_file}", flush=True)

    # Make sure the log directory exists before primus-cli tries to tee to it.
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # Wipe any stale log from a previous run so our parsing can't see it.
    if log_file.exists():
        log_file.unlink()

    start = time.time()
    # check=False: we inspect the log / Training-completed marker regardless
    # of return code, matching run_training_script()'s resilient behaviour.
    proc = subprocess.run(
        cmd,
        check=False,
        cwd=str(_REPO_ROOT),
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )
    elapsed = time.time() - start
    print(f"[{tag}] exit={proc.returncode} elapsed={elapsed:.2f}s", flush=True)

    if not log_file.exists():
        raise AssertionError(
            f"[{tag}] log file was not produced: {log_file} " f"(primus-cli exit={proc.returncode})."
        )
    log_text = log_file.read_text(errors="replace")

    if _TRAINING_COMPLETED_MARKER not in log_text:
        tail = log_text[-4000:] if log_text else "<empty>"
        raise AssertionError(
            f"[{tag}] '{_TRAINING_COMPLETED_MARKER}' not found in log; training "
            f"failed. (exit={proc.returncode})\n--- log tail ---\n{tail}"
        )
    if proc.returncode != 0:
        print(
            f"[{tag}] WARN: non-zero exit {proc.returncode} but training " "completed; continuing.",
            flush=True,
        )
    return log_text


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

# Identify the training_log line: " iteration  3/  20 |".
_ITER_LINE_RE = re.compile(r"\biteration\s+\d+\s*/\s*\d+\s*\|")

# "lm loss: 9.710209E+00" — keep the value as a STRING (bash did the same
# via `diff`) so we can assert byte-for-byte equality.
_LM_LOSS_RE = re.compile(r"lm loss:\s*([-+0-9.eE]+)")

# "elapsed time per iteration (ms): 387.8" OR
# "elapsed time per iteration (ms): 387.8/397.1" (the primus training_log
# patches print <current>/<running-avg>; we capture the CURRENT value, which
# is what bash's `grep -oE '... [0-9]+(\.[0-9]+)?'` also captures).
_ITER_TIME_RE = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9]+(?:\.[0-9]+)?)")


def _extract_losses(log_text: str) -> List[str]:
    """Return ``lm loss`` values (as strings, one per iteration)."""
    losses: List[str] = []
    for line in log_text.splitlines():
        if not _ITER_LINE_RE.search(line):
            continue
        m = _LM_LOSS_RE.search(line)
        if m:
            losses.append(m.group(1))
    return losses


def _extract_iter_times(log_text: str) -> List[float]:
    """Return current-iter elapsed times in ms (one per iteration)."""
    times: List[float] = []
    for line in log_text.splitlines():
        if not _ITER_LINE_RE.search(line):
            continue
        m = _ITER_TIME_RE.search(line)
        if m:
            times.append(float(m.group(1)))
    return times


def _summarize_times(
    label: str, times: List[float]
) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """Return (iter1, iter2, iter3..last-avg, pretty_line)."""
    if not times:
        return None, None, None, f"{label}: <no iter times parsed>"
    t1 = times[0]
    t2 = times[1] if len(times) >= 2 else None
    rest = times[2:]
    avg_rest = (sum(rest) / len(rest)) if rest else None

    parts = [f"iter1 = {t1:9.1f} ms"]
    parts.append(f"iter2 = {t2:9.1f} ms" if t2 is not None else "iter2 =       N/A")
    if avg_rest is not None:
        parts.append(f"iter3..iter{len(times)} avg = {avg_rest:9.1f} ms  " f"(n_rest = {len(rest)})")
    else:
        parts.append("iter3..last avg =       N/A (< 3 iterations)")
    return t1, t2, avg_rest, f"{label}: " + "  |  ".join(parts)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestPPWarmupPatches(unittest.TestCase):
    """End-to-end validation of pp_warmup patches."""

    _platform: str
    _exp: Path
    _iters: int
    _out_dir: Path

    @classmethod
    def setUpClass(cls) -> None:
        reason = _should_skip()
        if reason:
            raise unittest.SkipTest(reason)

        cls._platform = _detect_gpu_platform()
        cls._exp = _exp_config_path(cls._platform)
        try:
            cls._iters = int(os.environ.get(_TRAIN_ITERS_ENV, _DEFAULT_TRAIN_ITERS))
        except ValueError:
            cls._iters = _DEFAULT_TRAIN_ITERS
        if cls._iters < 1:
            cls._iters = _DEFAULT_TRAIN_ITERS

        out_dir = Path(os.environ.get("UT_LOG_PATH", str(_REPO_ROOT / "ut_out"))).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        cls._out_dir = out_dir

        print(
            f"\n[pp_warmup_test] platform   : {cls._platform}\n"
            f"[pp_warmup_test] exp_config : {cls._exp}\n"
            f"[pp_warmup_test] train_iters: {cls._iters}\n"
            f"[pp_warmup_test] out_dir    : {cls._out_dir}",
            flush=True,
        )

    def test_pp_warmup_preserves_loss_and_reports_iter_times(self) -> None:
        baseline_log_file = self._out_dir / "log.test_pp_warmup_patches-baseline.txt"
        warmup_log_file = self._out_dir / "log.test_pp_warmup_patches-warmup.txt"

        # --- Run both experiments back-to-back on the same GPUs ---------------
        log_baseline = _run_once(
            tag="BASELINE",
            pp_warmup=False,
            train_iters=self._iters,
            log_file=baseline_log_file,
            exp_path=self._exp,
        )
        log_warmup = _run_once(
            tag="WARMUP",
            pp_warmup=True,
            train_iters=self._iters,
            log_file=warmup_log_file,
            exp_path=self._exp,
        )

        # --- (1) Deterministic loss parity ------------------------------------
        losses_baseline = _extract_losses(log_baseline)
        losses_warmup = _extract_losses(log_warmup)
        print("\n--- baseline (pp_warmup=False) losses ---")
        print("\n".join(losses_baseline) if losses_baseline else "<none>")
        print("--- warmup   (pp_warmup=True) losses ---")
        print("\n".join(losses_warmup) if losses_warmup else "<none>")

        self.assertGreater(
            len(losses_baseline),
            0,
            "Could not extract any 'lm loss' values from the baseline log; "
            "either the regex is wrong or the training log format has changed.",
        )
        self.assertEqual(
            len(losses_baseline),
            len(losses_warmup),
            f"Different iteration counts: baseline={len(losses_baseline)}, "
            f"warmup={len(losses_warmup)}. Both runs should produce the same "
            f"number of 'lm loss' lines.",
        )
        mismatches: List[str] = []
        for i, (a, b) in enumerate(zip(losses_baseline, losses_warmup), start=1):
            if a != b:
                mismatches.append(f"  iter {i}: baseline={a}  warmup={b}")
        self.assertFalse(
            mismatches,
            "Deterministic-mode lm loss differs between baseline and warmup "
            "runs. pp_warmup is leaking state into real training.\n" + "\n".join(mismatches),
        )
        print(
            f"\n[PASS] (1) deterministic loss matches bit-for-bit across "
            f"{len(losses_baseline)} iterations."
        )

        # --- (2) Iter-time summary + optional MIN_SPEEDUP assertion -----------
        times_baseline = _extract_iter_times(log_baseline)
        times_warmup = _extract_iter_times(log_warmup)

        b_t1, b_t2, b_avg, b_line = _summarize_times("baseline (no pp_warmup)", times_baseline)
        w_t1, w_t2, w_avg, w_line = _summarize_times("warmup   (pp_warmup=on)", times_warmup)

        print("\n================================================================")
        print("                     iter-time summary                          ")
        print("================================================================")
        print(b_line)
        print(w_line)

        speedup_t1: Optional[float] = None
        if b_t1 is not None and w_t1 is not None and w_t1 > 0:
            speedup_t1 = b_t1 / w_t1
            print()
            print(f"  iter-1 speedup        : {b_t1:9.1f} / {w_t1:9.1f} " f"= {speedup_t1:.3f}x")
            if b_t2 is not None and w_t2 is not None and w_t2 > 0:
                print(f"  iter-2 speedup        : {b_t2:9.1f} / {w_t2:9.1f} " f"= {b_t2 / w_t2:.3f}x")
            if b_avg is not None and w_avg is not None and w_avg > 0:
                ratio_avg = b_avg / w_avg
                print(
                    f"  iter3..last avg ratio : {b_avg:9.1f} / {w_avg:9.1f} "
                    f"= {ratio_avg:.3f}x  "
                    f"(sanity: should be ~1.0 — warmup must not slow down "
                    "steady state)"
                )
        print("================================================================")

        min_speedup_env = os.environ.get(_MIN_SPEEDUP_ENV, "").strip()
        if min_speedup_env:
            try:
                required = float(min_speedup_env)
            except ValueError:
                self.fail(
                    f"{_MIN_SPEEDUP_ENV}={min_speedup_env!r} is not a valid " "float; expected e.g. '2.0'."
                )
            self.assertIsNotNone(
                speedup_t1,
                f"{_MIN_SPEEDUP_ENV}={required} set, but iter-1 times could " "not be parsed from the logs.",
            )
            assert speedup_t1 is not None  # narrow for type-checkers
            self.assertGreaterEqual(
                speedup_t1,
                required,
                f"iter-1 speedup {speedup_t1:.3f}x < required "
                f"{_MIN_SPEEDUP_ENV}={required:.3f}x. If this fails "
                "sporadically due to GPU noise, bump MIN_SPEEDUP.",
            )
            print(
                f"[PASS] (2) iter-1 speedup {speedup_t1:.3f}x "
                f">= required {_MIN_SPEEDUP_ENV}={required:.3f}x."
            )
        else:
            print(f"[INFO] {_MIN_SPEEDUP_ENV} not set → iter-1 timings reported " "only; no assertion.")


if __name__ == "__main__":
    # Enables running as a plain script: `python test_pp_warmup_patches.py`.
    unittest.main(buffer=False, verbosity=2)
