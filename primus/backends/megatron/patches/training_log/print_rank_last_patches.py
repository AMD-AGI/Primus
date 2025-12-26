###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron training_log print_rank_last patch.

This module contains a focused patch for ``megatron.training.training.print_rank_last``
to inject additional information into Megatron training logs:

    - ROCm/HIP memory stats.
    - Running average throughput per GPU (TFLOP/s/GPU).
    - Running average token throughput per GPU (tokens/s/GPU).

Design:
    - We first parse Megatron's original ``log_string`` into a structured
      ``TrainingLogInfo`` so that all extensions share a single parse.
    - Extensions then inject additional information based on this parsed view
      and return updated log strings.
"""

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.rocm_mem_info import get_rocm_smi_mem_info
from primus.modules.module_utils import log_rank_0, log_rank_all, warning_rank_0


@dataclass
class TrainingLogInfo:
    """Structured view of Megatron's training_log output line."""

    iteration: Optional[int] = None
    train_iters: Optional[int] = None
    consumed_samples: Optional[int] = None
    elapsed_ms: Optional[float] = None
    throughput_tflops: Optional[float] = None
    # Index of the throughput segment within ``segments``, if present.
    throughput_index: Optional[int] = None
    global_batch_size: Optional[int] = None
    # Original segments split by '|' (trimmed), including unknown fields so that
    # formatting/extensions can preserve everything Megatron adds.
    segments: List[str] = field(default_factory=list)


def parse_training_log_line(log_string: str) -> TrainingLogInfo:
    """
    Best-effort parse of Megatron ``training_log`` output.

    The original line is split on '|' into segments, and we try to recognize a
    few well-known fields (iteration, elapsed time, throughput, batch size).
    All segments (including unknown ones) are preserved in ``info.segments`` to
    keep the representation extensible.
    """
    info = TrainingLogInfo()

    try:
        # Split by '|' and trim whitespace; keep all segments so we never drop
        # unknown fields.
        segments = [seg.strip() for seg in log_string.split("|")]
        info.segments = segments

        for idx, seg in enumerate(segments):
            if not seg:
                continue

            # iteration {iteration}/{train_iters}
            # Note: the first segment typically looks like
            #   "[2025-..] iteration   2/   50"
            # so we look for the specific "iteration <int>/<int>" pattern instead
            # of checking for the substring "iteration", to avoid intercepting
            # unrelated fields such as "elapsed time per iteration (ms)".
            iter_match = re.search(r"iteration\s+(\d+)\s*/\s*(\d+)", seg)
            if iter_match:
                info.iteration = int(iter_match.group(1))
                info.train_iters = int(iter_match.group(2))
                continue

            # consumed samples: {consumed}
            if seg.startswith("consumed samples:"):
                consumed_match = re.search(r"consumed samples:\s*([0-9]+)", seg)
                if consumed_match:
                    info.consumed_samples = int(consumed_match.group(1))
                continue

            # elapsed time per iteration (ms): {elapsed_ms}
            if seg.startswith("elapsed time per iteration (ms):"):
                elapsed_match = re.search(r"elapsed time per iteration \(ms\):\s*([0-9.+-eE]+)", seg)
                if elapsed_match:
                    info.elapsed_ms = float(elapsed_match.group(1))
                continue

            # throughput per GPU (TFLOP/s/GPU): {throughput}
            if seg.startswith("throughput per GPU (TFLOP/s/GPU):"):
                thr_match = re.search(r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.+-eE]+)", seg)
                if thr_match:
                    info.throughput_tflops = float(thr_match.group(1))
                    info.throughput_index = idx
                continue

            # global batch size: {batch_size}
            if seg.startswith("global batch size:"):
                batch_match = re.search(r"global batch size:\s*([0-9]+)", seg)
                if batch_match:
                    info.global_batch_size = int(batch_match.group(1))
                continue
    except Exception:
        # Parsing must never break logging.
        return info

    return info


def render_training_log_line(info: TrainingLogInfo) -> str:
    """
    Render a TrainingLogInfo structure back into a single log string.

    We keep the original segment order and simply join on ' | '. Any new
    segments appended by extensions are included at the end.
    """
    segments = [seg for seg in info.segments if seg]
    if not segments:
        return ""
    return " | ".join(segments)


def _touch_log_rank_all_for_tests() -> None:  # pragma: no cover
    """
    Helper to keep ``log_rank_all`` referenced so that it is available for
    unit tests that monkeypatch this symbol from this module.
    """
    if False:
        log_rank_all("")  # type: ignore[arg-type]


class MemoryStatsExtension:
    """
    Helper extension to collect and inject HIP and ROCm-SMI memory statistics
    into Megatron training logs.
    """

    def __init__(self, args: Any):
        self.args = args
        # Local cache of the last ROCm SMI stats string so we can reuse it on
        # iterations where we intentionally skip expensive SMI queries.
        self._last_rocm_mem_str: str = ""
        # Cache ROCm config flags to avoid repeated getattr lookups.
        self._use_rocm_mem: bool = bool(getattr(args, "use_rocm_mem_info", False))
        self._rocm_iters = getattr(args, "use_rocm_mem_info_iters", [])

    def inject(
        self,
        log_string: str,
        call_index: int,
        parsed: Optional[TrainingLogInfo] = None,
    ) -> str:
        hip_mem_str = ""
        rocm_mem_str = ""

        # 1. HIP Stats (Always available on ROCm)
        # We assume that if this extension is active, we want to see memory stats.
        # HIP stats are cheap and always available via PyTorch.
        try:
            hip_free, hip_total = torch.cuda.mem_get_info()
            hip_used = hip_total - hip_free
            hip_ratio = hip_used / hip_total
            hip_mem_str = (
                f" hip mem usage/free/total/usage_ratio: "
                f"{hip_used / 1024 ** 3:.2f}GB/"
                f"{hip_free / 1024 ** 3:.2f}GB/"
                f"{hip_total / 1024 ** 3:.2f}GB/"
                f"{hip_ratio * 100:.2f}%"
            )
        except Exception:
            # CUDA/ROCm may not be initialized (e.g., CPU-only UT).
            hip_mem_str = ""

        # 2. ROCm SMI Stats (Only if configured and iteration matches)
        # Only call expensive SMI if globally enabled OR current iteration is in list.
        # If we decide not to collect on this iteration but have a previously
        # collected value, reuse the last known ROCm SMI stats to keep the log
        # informative without incurring per-step overhead.
        should_collect_smi = self._use_rocm_mem or (call_index in self._rocm_iters)

        if should_collect_smi:
            try:
                local_rank = torch.cuda.current_device()
                r_total, r_used, r_free = get_rocm_smi_mem_info(local_rank)
                r_ratio = r_used / r_total
                rocm_mem_str = (
                    f" | rocm mem usage/free/total/usage_ratio: "
                    f"{r_used / 1024 ** 3:.2f}GB/"
                    f"{r_free / 1024 ** 3:.2f}GB/"
                    f"{r_total / 1024 ** 3:.2f}GB/"
                    f"{r_ratio * 100:.2f}%"
                )
                # Cache for reuse on non-sampled iterations
                self._last_rocm_mem_str = rocm_mem_str
            except Exception:
                # If SMI fails, fall back to last known value (if any)
                rocm_mem_str = self._last_rocm_mem_str
        else:
            # Not a sampling iteration; reuse last successful SMI stats if available.
            rocm_mem_str = self._last_rocm_mem_str

        combined = " ".join(s for s in [hip_mem_str, rocm_mem_str] if s)
        if not combined or parsed is None:
            # When no parsed structure is provided (e.g., in some tests), we keep
            # the original string unchanged and do not attempt to splice in
            # memory stats via string concatenation.
            return log_string

        # Append memory stats as a dedicated segment on the parsed structure so
        # that final rendering uses the canonical segment list.
        parsed.segments.append(combined.strip())
        # String result is ignored by the main patch when parsed is provided.
        return log_string


class ThroughputAverageExtension:
    """
    Helper extension to compute and inject running average throughput statistics
    (both TFLOPs and tokens) into Megatron training logs.

    Semantics mirror Primus MegatronTrainer:
        - Ignore the first `log_avg_skip_iterations` iterations for averaging.
        - Maintain a sliding window up to `log_avg_reset_interval` entries.
    """

    def __init__(self, args: Any):
        self._args = args
        # Cache seq_length and world_size once at construction time so we do not
        # repeatedly resolve them during throughput calculations.
        self._seq_len = getattr(args, "seq_length", None)
        self._world_size = getattr(args, "world_size", None)
        # Track throughput TFLOPs statistics across calls so we can log an average
        # throughput alongside Megatron's per-iteration throughput.
        self._recent_tflop_throughputs: list[float] = []
        # Track token throughput statistics across calls.
        self._recent_token_throughputs: list[float] = []
        # We follow the same warmup/reset semantics as Primus MegatronTrainer:
        #   - Ignore the first `log_avg_skip_iterations` iterations for averaging
        #   - Maintain a sliding window of size `log_avg_reset_interval`.
        self._log_avg_skip_iterations: int = int(getattr(args, "log_avg_skip_iterations", 0))
        self._log_avg_reset_interval: int = int(getattr(args, "log_avg_reset_interval", 1000))

        log_rank_0(
            f"[Patch:megatron.training_log] ThroughputAverageExtension initialized with "
            f"seq_len: {self._seq_len} "
            f"world_size: {self._world_size} "
            f"log_avg_skip_iterations: {self._log_avg_skip_iterations} "
            f"log_avg_reset_interval: {self._log_avg_reset_interval}"
        )

    def inject(self, log_string: str, parsed: Optional[TrainingLogInfo] = None) -> str:
        """
        Update ``parsed`` with running-average TFLOP and token throughput.

        - TFLOPs are rendered inline as:
              throughput per GPU (TFLOP/s/GPU): inst/avg
        - Tokens are appended as a new segment immediately after throughput:
              tokens per GPU (tokens/s/GPU): inst/avg
        """
        try:
            # If no parsed info is provided (e.g., unit tests calling this
            # extension directly), keep the original string unchanged.
            if parsed is None:
                return log_string

            iteration = parsed.iteration

            # ---------------- TFLOPs ----------------
            if parsed.throughput_tflops is not None:
                tflops_value = parsed.throughput_tflops

                # Handle warmup & sliding window logic for TFLOPs.
                if iteration is not None and (
                    iteration == self._log_avg_skip_iterations + 1
                    or len(self._recent_tflop_throughputs) >= self._log_avg_reset_interval
                ):
                    self._recent_tflop_throughputs.clear()

                # Only accumulate after skip window.
                if iteration is None or iteration > self._log_avg_skip_iterations:
                    self._recent_tflop_throughputs.append(tflops_value)

                if self._recent_tflop_throughputs:
                    avg_tflops = sum(self._recent_tflop_throughputs) / len(self._recent_tflop_throughputs)
                    idx = parsed.throughput_index
                    if idx is not None and 0 <= idx < len(parsed.segments):
                        parsed.segments[idx] = (
                            f"throughput per GPU (TFLOP/s/GPU): " f"{tflops_value:.1f}/{avg_tflops:.1f}"
                        )

            # ---------------- Tokens/s ----------------
            if parsed.elapsed_ms is None:
                warning_rank_0(f"[Patch:megatron.training_log] Elapsed time per iteration (ms) is missing")
                return log_string

            # Batch size must come from the parsed training_log line; if it is
            # missing we intentionally skip token throughput to avoid guessing.
            if parsed.global_batch_size is None:
                warning_rank_0(f"[Patch:megatron.training_log] Global batch size is missing")
                return log_string
            batch_size = int(parsed.global_batch_size)

            elapsed_ms = float(parsed.elapsed_ms)
            elapsed_s = elapsed_ms / 1000.0

            # Resolve seq_length/world_size strictly from args; if missing we
            # intentionally skip token throughput instead of guessing.
            if self._seq_len is None or self._world_size is None:
                warning_rank_0(f"[Patch:megatron.training_log] Seq length or world size is missing")
                return log_string

            tokens_per_iter = int(self._seq_len) * batch_size
            token_value = tokens_per_iter / max(elapsed_s, 1e-6) / int(self._world_size)

            # Handle warmup & sliding window logic for tokens.
            if iteration is not None and (
                iteration == self._log_avg_skip_iterations + 1
                or len(self._recent_token_throughputs) >= self._log_avg_reset_interval
            ):
                self._recent_token_throughputs.clear()

            if iteration is None or iteration > self._log_avg_skip_iterations:
                self._recent_token_throughputs.append(token_value)

            if not self._recent_token_throughputs:
                warning_rank_0(f"[Patch:megatron.training_log] No token throughput")
                return log_string

            avg_tokens = sum(self._recent_token_throughputs) / len(self._recent_token_throughputs)

            # Append token throughput directly after the TFLOP throughput within
            # the same segment. We do not create a new segment to keep the log
            # compact and closely aligned with Megatron's original formatting.
            idx = parsed.throughput_index
            if idx is not None and 0 <= idx < len(parsed.segments):
                parsed.segments[idx] = (
                    f"{parsed.segments[idx]} "
                    f" | tokens per GPU (tokens/s/GPU): {token_value:.1f}/{avg_tokens:.1f}"
                )

            # String result is ignored by the main patch when parsed is provided.
            return log_string
        except Exception:
            # Any parsing / numeric issues should not break logging.
            return log_string


@register_patch(
    "megatron.training_log.unified_patch",
    backend="megatron",
    phase="before_train",
    description="Patch training_log to use Primus print_rank_last (ROCm, throughput) only inside training_log.",
)
def patch_training_log_unified(ctx: PatchContext):
    """
    Patch Megatron's ``training_log`` so that ONLY calls to ``print_rank_last`` made
    *inside* ``training_log`` are intercepted by Primus.

    Implementation:
        - Wrap ``megatron.training.training.training_log`` with a small wrapper.
        - Inside the wrapper, temporarily override ``print_rank_last`` with
          ``PrintRankLastExtension(config)`` for the duration of the call.
        - Restore the original ``print_rank_last`` afterwards.
        - Other call sites that use ``print_rank_last`` outside of ``training_log``
          are not affected.
    """
    try:
        import megatron.training.training as megatron_training  # type: ignore

        # Get unified Megatron args (module_config.params) from context.
        config = get_args(ctx)

        # Check whether we should enable ROCm stats / throughput logging.
        use_rocm_mem = bool(getattr(config, "use_rocm_mem_info", False))
        rocm_iters = getattr(config, "use_rocm_mem_info_iters", [])

        enable_rocm_stats = bool(getattr(config, "log_throughput", False)) and (
            use_rocm_mem or (rocm_iters and len(rocm_iters) > 0)
        )

        if not enable_rocm_stats:
            # Nothing to do; leave Megatron's training_log and print_rank_last untouched.
            return

        original_training_log = megatron_training.training_log

        # Avoid double-wrapping training_log.
        if getattr(original_training_log, "_primus_training_log_print_rank_wrapper", False):
            return

        # Create helper extensions once so they keep state (ROCm cache, avg windows)
        # across all training_log invocations.
        mem_ext = MemoryStatsExtension(config)
        throughput_ext = ThroughputAverageExtension(config)
        call_count = 0
        # Capture the original ``print_rank_last`` so we can delegate actual
        # printing back to Megatron after mutating the log string.
        original_print_rank_last = megatron_training.print_rank_last

        def primus_print_rank_last(log_string: str) -> None:
            """
            Replacement for ``print_rank_last`` used only while inside training_log.

            Responsibilities:
                - Parse and enrich the log string with Primus metrics.
                - Delegate the final printing to Megatron's original
                  ``print_rank_last`` implementation.
            """
            nonlocal call_count
            try:
                # Track how many times we've seen print_rank_last in this run.
                call_count += 1
                # Parse the original log string once and share across extensions.
                parsed = parse_training_log_line(log_string)

                # Inject memory statistics, throughput, and token throughput by
                # mutating the parsed structure. These calls ignore their string
                # return value when `parsed` is provided.
                mem_ext.inject(log_string, call_count, parsed)
                throughput_ext.inject(log_string, parsed)

                # Render the final line from the parsed structure.
                updated = render_training_log_line(parsed)
            except Exception as e:
                # Logging must never break training; emit a warning and continue.
                warning_rank_0(f"[Patch:megatron.training_log] Failed to append training stats: {e}")
                updated = log_string

            # Delegate actual printing to Megatron's original implementation so
            # that rank filtering and any other side effects remain unchanged.
            original_print_rank_last(updated)

        def primus_training_log(*args, **kwargs):
            """
            Wrapper around Megatron's training_log that temporarily overrides
            ``print_rank_last`` for the duration of the call.
            """
            original_print = megatron_training.print_rank_last
            try:
                megatron_training.print_rank_last = primus_print_rank_last
                return original_training_log(*args, **kwargs)
            finally:
                megatron_training.print_rank_last = original_print

        setattr(primus_training_log, "_primus_training_log_print_rank_wrapper", True)
        megatron_training.training_log = primus_training_log

        log_rank_0("[Patch:megatron.training_log] Wrapped training_log with Primus print_rank_last hook")

    except ImportError as e:
        log_rank_0(f"[Patch:megatron.training_log][SKIP] Import failed: {e}")
    except Exception as e:
        # Catch-all to make sure patch does not crash training.
        log_rank_0(f"[Patch:megatron.training_log][ERROR] Unexpected error: {e}")
