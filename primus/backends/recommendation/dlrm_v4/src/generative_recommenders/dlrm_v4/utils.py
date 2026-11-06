# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe
"""
mlperf dlrm_v4 inference benchmarking tool.
"""

import contextlib
import gc
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import gin
import tensorboard  # @manual=//tensorboard:lib  # noqa: F401 - required implicit dep when using torch.utils.tensorboard
import torch
import torch.distributed as dist
from generative_recommenders.dlrm_v4 import host_sync_free as _hsf
from generative_recommenders.dlrm_v4.datasets.dataset import DLRMv4RandomDataset
from generative_recommenders.dlrm_v4.datasets.kuairand import DLRMv4KuaiRandDataset
from generative_recommenders.dlrm_v4.datasets.movie_lens import DLRMv4MovieLensDataset
from generative_recommenders.dlrm_v4.datasets.synthetic_movie_lens import (
    DLRMv4SyntheticMovieLensDataset,
)
from generative_recommenders.dlrm_v4.datasets.synthetic_streaming import (
    DLRMv4SyntheticStreamingDataset,
)
from generative_recommenders.dlrm_v4.datasets.yambda import DLRMv4YambdaDataset
from generative_recommenders.dlrm_v4.trace_postprocessing import (
    _rebuild_gpu_annotation_spans_from_file,
    _repair_gpu_nesting_from_file,
    _trim_warmup_from_trace,
)
from generative_recommenders.modules.multitask_module import (
    MultitaskTaskType,
    TaskConfig,
)
from torch.profiler import profile, profiler, ProfilerActivity  # pyre-ignore [21]
from torch.utils.tensorboard import SummaryWriter
from torchrec.metrics.accuracy import AccuracyMetricComputation
from torchrec.metrics.auc import AUCMetricComputation, compute_auc
from torchrec.metrics.gauc import GAUCMetricComputation
from torchrec.metrics.mae import MAEMetricComputation
from torchrec.metrics.metrics_namespace import MetricName, MetricPrefix
from torchrec.metrics.mse import MSEMetricComputation
from torchrec.metrics.ne import NEMetricComputation
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecMetricComputation,
)


class LifetimeAUCMetricComputation(AUCMetricComputation):
    """AUC over a 10M-sample (~5.5 eval-window) trailing buffer; emits with the
    LIFETIME prefix.

    NOTE: despite the name, this is NOT an uncapped since-step-0 AUC. The parent
    ``AUCMetricComputation`` evicts the prediction/label/weight buffers down to
    ``window_size`` in ``update()``; we instantiate it with
    ``window_size=10_000_000``, so "lifetime" is a ~10M-sample trailing window.
    Raise ``window_size`` (accepting unbounded buffer growth) if true cumulative
    AUC is ever required.

    Checkpoint correctness: torchrec registers the PREDICTIONS/LABELS/WEIGHTS
    buffers with ``persistent=False`` (so the default ``state_dict()`` drops
    them) and tracks a separate ``self._num_samples`` counter. Without the
    overrides below, every checkpoint resume would silently restart this metric
    from an empty buffer. We therefore serialize the buffers AND ``_num_samples``
    explicitly; restoring ``_num_samples`` is mandatory, since leaving it at 0
    makes the next ``update()`` take the init-sentinel branch and desync the
    windowed eviction. These buffers are per-rank-local (cross-rank gather only
    happens transiently at compute time), so the checkpoint layer MUST persist
    and restore them per-rank — see ``checkpoint.py``.
    """

    # Prefix used for the explicitly-serialized non-persistent buffers so the
    # keys can't collide with any persistent state the parent might register.
    _LIFETIME_KEY_PREFIX: str = "_lifetime_"

    def _compute(self) -> List[MetricComputationReport]:
        from typing import cast as _cast
        from torchrec.metrics.auc import LABELS, PREDICTIONS, WEIGHTS
        return [
            MetricComputationReport(
                name=MetricName.AUC,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_auc(
                    self._n_tasks,
                    _cast(List[torch.Tensor], getattr(self, PREDICTIONS)),
                    _cast(List[torch.Tensor], getattr(self, LABELS)),
                    _cast(List[torch.Tensor], getattr(self, WEIGHTS)),
                    self._apply_bin,
                ),
            )
        ]

    def lifetime_sample_count(self) -> int:
        """Current number of buffered samples (greppable for sanity logs)."""
        return int(getattr(self, "_num_samples", 0))

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        from torchrec.metrics.auc import LABELS, PREDICTIONS, WEIGHTS

        destination = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        # The parent registers these buffers persistent=False, so they are absent
        # from `destination`. Concatenate each buffer list to one (n_tasks, N)
        # tensor and serialize it alongside the sample counter.
        for attr in (PREDICTIONS, LABELS, WEIGHTS):
            buf = getattr(self, attr)
            if isinstance(buf, (list, tuple)) and len(buf) > 0:
                flat = torch.cat([t for t in buf], dim=-1)
            elif isinstance(buf, torch.Tensor):
                flat = buf
            else:
                flat = torch.empty(0)
            destination[prefix + self._LIFETIME_KEY_PREFIX + attr] = (
                flat.detach().cpu().clone()
            )
        destination[prefix + self._LIFETIME_KEY_PREFIX + "num_samples"] = (
            torch.tensor(int(getattr(self, "_num_samples", 0)), dtype=torch.long)
        )
        return destination

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        strict: bool = True,
    ) -> Any:
        from torchrec.metrics.auc import LABELS, PREDICTIONS, WEIGHTS

        # Copy so we can strip our custom keys before delegating to the parent
        # (whose strict load would otherwise reject them as unexpected).
        remaining = dict(state_dict)
        saved_bufs: Dict[str, torch.Tensor] = {}
        for attr in (PREDICTIONS, LABELS, WEIGHTS):
            key = self._LIFETIME_KEY_PREFIX + attr
            if key in remaining:
                saved_bufs[attr] = remaining.pop(key)
        num_key = self._LIFETIME_KEY_PREFIX + "num_samples"
        saved_num = remaining.pop(num_key, None)

        result = super().load_state_dict(remaining, strict=strict)

        if saved_bufs:
            # Device of the live (init-sentinel) buffers; keep restored buffers
            # co-located so subsequent update()/compute() stay on-device.
            existing = getattr(self, PREDICTIONS)
            dev = (
                existing[0].device
                if isinstance(existing, (list, tuple)) and len(existing) > 0
                else torch.device("cpu")
            )
            for attr, val in saved_bufs.items():
                setattr(self, attr, [val.to(dev)])
            if saved_num is not None:
                self._num_samples = int(saved_num.item())
        return result


# Sentinel "window size" used for the FRESH eval metrics so torchrec's windowed
# eviction never fires within a single eval pass (the per-pass reset bounds the
# buffer to exactly one full holdout pass). 1<<60 is far above any realistic
# per-rank sample count and avoids sys.maxsize overflow inside torchrec math.
UNBOUNDED_WINDOW: int = 1 << 60


class BinnedCumulativeAUC(RecMetricComputation):
    """Cumulative AUC via a fixed-resolution score histogram (LIFETIME prefix).

    Global AUC is a rank statistic, so it has no fixed-size additive sufficient
    statistic the way NE/Accuracy do - exact cumulative AUC otherwise needs every
    (score, label) pair retained and sorted (the buffer-based ``AUCMetricComputation``
    / ``LifetimeAUCMetricComputation``). Instead we keep two weighted histograms of
    positive/negative mass per score bin. This gives an AUC exact up to bin width
    with O(num_bins) memory that does NOT grow with sample count, and - because
    histograms are additive - cross-rank sync is a cheap all-reduce (dist_reduce_fx
    "sum") rather than all-gathering millions of predictions. The state is truly
    cumulative across all eval passes (never evicted, never reset on eval).

    Predictions MUST be probabilities in [0, 1] (the same tensor feeds NE, which
    requires probabilities; the model applies sigmoid in multitask_module). Values
    are clamped into [0, 1] defensively.
    """

    def __init__(self, *args, num_bins: int = 100_000, **kwargs) -> None:
        # window_size is irrelevant here (no windowed state); pass through.
        super().__init__(*args, **kwargs)
        self._num_bins: int = int(num_bins)
        self._add_state(
            "pos_hist",
            torch.zeros((self._n_tasks, self._num_bins), dtype=torch.float64),
            add_window_state=False,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            "neg_hist",
            torch.zeros((self._n_tasks, self._num_bins), dtype=torch.float64),
            add_window_state=False,
            dist_reduce_fx="sum",
            persistent=True,
        )

    def cumulative_sample_count(self) -> int:
        """Total weighted samples in the histograms (greppable for sanity logs)."""
        return int((self.pos_hist.sum() + self.neg_hist.sum()).item())

    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> None:
        if predictions is None or weights is None:
            raise ValueError(
                "BinnedCumulativeAUC.update requires predictions and weights"
            )
        preds = predictions.float().clamp_(0.0, 1.0)  # (n_tasks, n_examples)
        labels = labels.float()
        weights = weights.float()
        # Bin index per example; the top edge (p==1.0) folds into the last bin.
        idx = (preds * self._num_bins).long().clamp_(0, self._num_bins - 1)
        pos_w = (weights * labels).to(self.pos_hist.dtype)
        neg_w = (weights * (1.0 - labels)).to(self.neg_hist.dtype)
        self.pos_hist.scatter_add_(1, idx, pos_w)
        self.neg_hist.scatter_add_(1, idx, neg_w)

    def _compute(self) -> List[MetricComputationReport]:
        # By compute() time torchmetrics has all-reduced (summed) the histograms
        # across ranks, so these are the global per-bin masses.
        pos = self.pos_hist  # (n_tasks, num_bins)
        neg = self.neg_hist
        total_pos = pos.sum(dim=1)
        total_neg = neg.sum(dim=1)
        # Lower bin index == lower score. A positive in bin b outranks every
        # negative in bins < b (exclusive prefix sum), and ties in bin b score
        # 0.5. AUC = sum_b pos_b * (neg_below_b + 0.5*neg_b) / (P * N).
        neg_below = torch.cumsum(neg, dim=1) - neg
        numerator = (pos * (neg_below + 0.5 * neg)).sum(dim=1)
        denom = total_pos * total_neg
        auc = torch.where(
            denom > 0,
            numerator / denom,
            torch.full_like(numerator, 0.5),
        ).to(torch.float32)
        return [
            MetricComputationReport(
                name=MetricName.AUC,
                metric_prefix=MetricPrefix.LIFETIME,
                value=auc,
            )
        ]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("utils")


def _is_rocm() -> bool:
    """True on ROCm/AMD builds (``torch.version.hip`` set), False on CUDA/B200.

    The ProfilerStep-layout normalization and the sub-us kernel de-overlap are
    workarounds for how roctracer projects annotations/kernels onto HIP streams;
    CUDA/CUPTI traces don't have those artifacts, so these passes must be skipped
    on NVIDIA to avoid touching otherwise-correct traces.
    """
    return getattr(torch.version, "hip", None) is not None


def _on_trace_ready_fn(
    rank: Optional[int] = None,
    trace_dir: str = "/tmp/dlrm_v4_traces",
    keep_n_active: Optional[int] = None,
    trace_steps: Optional[List[int]] = None,
    keep_head_active: Optional[int] = None,
    skip_head: int = 0,
    capture_notes: Optional[List[dict]] = None,
    skip_boundary_steps: bool = False,
) -> Callable[[torch.profiler.profile], None]:
    """Create the on_trace_ready callback that exports a chrome trace to disk.

    Filename follows ``trace_step{step}_rank{rank}.json`` so multi-rank
    captures don't collide and ``tools/recommendation/dlrm_v4/stitch_traces.py`` can merge them
    by step number.

    The ``{step}`` label:

    * If ``trace_steps`` is provided (multi-window mode), the Nth callback
      invocation labels its file with ``trace_steps[N]`` -- i.e. the
      user-requested step that triggered the window. This is the most
      intuitive labelling.
    * Otherwise falls back to ``p.step_num`` (torch.profiler's internal
      counter at trigger time, off by ~warmup+active from the schedule
      arg).

    If ``keep_n_active`` is set, the exported file is post-processed to keep
    only the last N ProfilerStep-spans worth of events (i.e. drop WARMUP).
    ``keep_head_active`` narrows that further to the first N of those, which
    is how the window's skewed teardown steps are kept out of the artifact.
    ``skip_head`` drops that many recorded steps from the front of the export
    and is added to a ``trace_steps`` label, so a window that starts one step
    early to skip its head is still named by its first exported step.
    ``capture_notes`` (a list the ``Profiler`` appends to) is written into the
    trace's ``traceRendering`` and cleared after each export.
    ``skip_boundary_steps`` moves the exported steps past a recorded step
    that ran the streaming window-boundary barrier, when enough steps were
    recorded (see ``_trim_warmup_from_trace``); the file name is unchanged and
    ``traceCapture.exported_steps`` records the steps actually kept.
    """
    state = {"fire_count": 0}

    def handle_fn(p: torch.profiler.profile) -> None:
        if trace_steps:
            i = state["fire_count"]
            step_label = (
                trace_steps[i] + skip_head
                if i < len(trace_steps)
                else getattr(p, "step_num", 0)
            )
        else:
            step_label = getattr(p, "step_num", 0)
        state["fire_count"] += 1
        rank_str = f"_rank{rank}" if rank is not None else ""
        file_name = f"trace_step{step_label}{rank_str}.json"
        path = os.path.join(trace_dir, file_name)
        # Tracing is best-effort: no part of this callback may crash the
        # training run. It runs inside torch's profile.step(), which
        # Profiler.step() calls unguarded, so an exception here would end the
        # job. The op summary table is diagnostics only, and building it can
        # raise inside torch: key_averages() has raised UnicodeDecodeError on
        # with_stack=True captures whose window holds a FlyDSL JIT first step.
        # Log the failure and still export the trace.
        try:
            logger.warning(
                p.key_averages(group_by_input_shape=True).table(
                    sort_by="self_cuda_time_total"
                )
            )
        except Exception as exc:
            logger.warning(f"Trace op table failed for {path}: {exc!r} (skipping)")
        # A write/trim failure (permissions, disk full, malformed export) must
        # never crash the training run either. Degrade to a warning so the loop
        # continues — especially important since streaming enables output_trace
        # by default.
        try:
            os.makedirs(trace_dir, exist_ok=True)
            p.export_chrome_trace(path)
            logger.warning(f"Trace written to: {path}")
            if keep_n_active is not None and keep_n_active > 0:
                _trim_warmup_from_trace(
                    path,
                    keep_n_active,
                    keep_head_active,
                    skip_head=skip_head,
                    skip_boundary_steps=skip_boundary_steps,
                )
            # Rebuild GPU-side annotation spans (ProfilerStep#N included) from
            # the work each CPU span launched. Time-window spans are wrong once
            # the host runs ahead of the GPU (host-sync-free step), which is not
            # platform-specific, so this runs everywhere. Display-only.
            _rebuild_gpu_annotation_spans_from_file(
                path, capture_notes=capture_notes
            )
            # ROCm/AMD-only sub-us boundary repair. CUDA/CUPTI (e.g. B200)
            # traces don't exhibit the kernel-overlap artifacts, so skip on
            # NVIDIA. Best-effort like trim above.
            if _is_rocm():
                # Match Perfetto's integer-nanosecond boundaries. Repair only
                # small crossings; exact-touching siblings already nest cleanly.
                _repair_gpu_nesting_from_file(path)
        except Exception as exc:
            logger.warning(f"Trace export/trim failed for {path}: {exc!r} (skipping)")
        if capture_notes is not None:
            capture_notes.clear()

    return handle_fn


def profiler_or_nullcontext(
    enabled: bool, with_stack: bool, trace_dir: str = "/tmp/dlrm_v4_traces"
):
    """One-shot profile context for ad-hoc captures (no scheduling)."""
    return (
        profile(
            # pyre-fixme[16]: Module `profiler` has no attribute `ProfilerActivity`.
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=_on_trace_ready_fn(trace_dir=trace_dir),
            with_stack=with_stack,
        )
        if enabled
        else contextlib.nullcontext()
    )


def _multi_window_schedule(
    trace_steps,
    warmup: int,
    active: int,
):
    """Custom schedule that profiles around each step in ``trace_steps``.

    Step s gets:
        [s - warmup, s)        -> WARMUP
        [s, s + active - 1)    -> RECORD
        s + active - 1         -> RECORD_AND_SAVE
    """
    windows = [(s - warmup, s, s + active) for s in sorted(trace_steps)]

    def schedule_fn(step: int) -> torch.profiler.ProfilerAction:
        for warmup_start, active_start, active_end in windows:
            if warmup_start <= step < active_start:
                return torch.profiler.ProfilerAction.WARMUP
            if active_start <= step < active_end - 1:
                return torch.profiler.ProfilerAction.RECORD
            if step == active_end - 1:
                return torch.profiler.ProfilerAction.RECORD_AND_SAVE
        return torch.profiler.ProfilerAction.NONE

    return schedule_fn


def _post_export_gc_on() -> bool:
    """Profiler post-export full GC (traced runs only): on unless
    DLRM_HSTU_FP8_PROF_POST_GC=0 (the DLRM_HSTU_FP8_ prefix is forwarded into the
    container by scripts/launch_slurm.sh)."""
    return os.environ.get("DLRM_HSTU_FP8_PROF_POST_GC", "1").strip() != "0"


@gin.configurable
class Profiler:
    """Scheduled torch.profiler wrapper that writes Chrome traces to disk.

    Two modes (set via gin):

    * Single window (default): ``wait=10, warmup=20, active=50, repeat=1``.
      Captures one contiguous window starting after ``wait`` steps.
    * Multi-window: ``trace_steps=[500, 1000, 5000]`` (overrides wait+repeat).
      Captures a separate window around each listed step.

    All knobs are gin-tunable, e.g. in a gin file::

        Profiler.trace_dir = "/path/to/results/exp42/trace"
        Profiler.trace_steps = [500, 1000, 5000]
        Profiler.warmup = 5
        Profiler.active = 10

    Export selection (``trim_warmup``): ``skip_head`` recorded steps are
    dropped from the front, then ``keep_active`` steps are exported.
    ``skip_boundary_steps`` (default off) moves the exported steps past a
    recorded step that ran the streaming window-boundary barrier when the
    window recorded enough steps; with ``active=8, skip_head=1,
    keep_active=5`` it can never move (it needs ``active >= skip_head +
    2 * keep_active + 1`` to clear a boundary on the last exported step).
    Placing ``trace_steps`` inside one data window is the reliable fix.
    """

    def __init__(
        self,
        rank: int,
        active: int = 50,
        wait: int = 10,
        warmup: int = 20,
        repeat: int = 1,
        trace_steps: Optional[List[int]] = None,
        trace_dir: str = "/tmp/dlrm_v4_traces",
        trim_warmup: bool = True,
        keep_active: Optional[int] = None,
        record_shapes: bool = True,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        align_ranks: bool = True,
        skip_head: int = 0,
        skip_boundary_steps: bool = False,
    ) -> None:
        self.rank = rank
        self.trace_dir = trace_dir
        # Diagnostic override: $DLRM_TRACE_STEPS="a,b,..." replaces the gin
        # trace_steps so a run can be traced inside a chosen phase (e.g. the
        # cold-page-cache slow stretch) without editing the gin file. Unset =
        # gin value, unchanged behaviour.
        _env_steps = os.environ.get("DLRM_TRACE_STEPS", "").strip()
        if _env_steps:
            trace_steps = [int(s) for s in _env_steps.split(",") if s.strip()]
            logger.warning("Profiler: trace_steps overridden by $DLRM_TRACE_STEPS=%s", trace_steps)
        # $DLRM_TRACE_SKIP_HEAD: e.g. 0 to export the very first recorded step
        # (trace_steps=[0] -> the run's first training step, which the gin
        # skip_head=1 would otherwise drop). Unset = gin value.
        _env_skip = os.environ.get("DLRM_TRACE_SKIP_HEAD", "").strip()
        if _env_skip:
            skip_head = int(_env_skip)
            logger.warning("Profiler: skip_head overridden by $DLRM_TRACE_SKIP_HEAD=%d", skip_head)
        if trace_steps:
            sched = _multi_window_schedule(trace_steps, warmup, active)
        else:
            sched = torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=repeat
            )
        self._sched = sched
        self._align_ranks = align_ranks
        skip_head = max(0, int(skip_head or 0))
        if skip_head and not trim_warmup:
            logger.warning(
                "Profiler.skip_head=%d needs trim_warmup=True; ignored", skip_head
            )
            skip_head = 0
        if skip_head >= active:
            raise ValueError(
                f"Profiler.skip_head={skip_head} must be < active={active}"
            )
        self._skip_head = skip_head
        # Exported steps per window: recorded offsets
        # [skip_head, skip_head + keep).
        self._keep = (
            min(keep_active, active - skip_head) if keep_active else active - skip_head
        )
        self._active = active
        self._check_window(active, keep_active, skip_head, trim_warmup)
        self._install_gc_trace()
        # Loop events inside a recorded window (see note_event), written into
        # the exported trace and cleared after each export.
        self._notes: List[dict] = []
        self._noted: set = set()
        # A metrics drain at the end of step N leaves step N+1 starting from
        # an empty GPU queue under the host-sync-free step; noted on step().
        self._after_metric_log: Optional[dict] = None
        # Record `active` steps; drop the first `skip_head` of them, then
        # export only the first `keep_active` of the rest. Both default to
        # keeping everything, so this is inert unless gin sets them.
        keep_n = active if trim_warmup else None
        self._profiler: profiler.profile = torch.profiler.profile(
            schedule=sched,
            on_trace_ready=_on_trace_ready_fn(
                self.rank,
                trace_dir,
                keep_n,
                trace_steps,
                keep_active,
                skip_head=skip_head,
                capture_notes=self._notes,
                skip_boundary_steps=bool(skip_boundary_steps and trim_warmup),
            ),
            # pyre-fixme[16]: Module `profiler` has no attribute `ProfilerActivity`.
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
        )

    def _check_window(
        self,
        active: int,
        keep_active: Optional[int],
        skip_head: int,
        trim_warmup: bool,
    ) -> None:
        """Warn when the window cannot represent a host-sync-free step.

        Under the host-sync-free step (``host_sync_free.enabled()``) the host
        runs one step ahead of the GPU. Two recorded steps are then not
        training steps:

        * the FIRST recorded step. ``align_ranks`` drains the device and
          barriers before it, so it starts with an empty launch queue and no
          host lead; at 1n it ran ~5% faster than the steps after it.
          ``skip_head >= 1`` keeps it out of the export.
        * the LAST recorded step. Its GPU work finishes after the host has
          entered the next step, and the session teardown lands there. At
          least one recorded step past the export keeps both out
          (``active - keep_active - skip_head >= 1``).
        """
        if not _hsf.enabled() or not trim_warmup:
            return
        tail = active - skip_head - (keep_active or (active - skip_head))
        if skip_head < 1 or tail < 1:
            logger.warning(
                "Profiler window under the host-sync-free step: active=%d, "
                "keep_active=%s, skip_head=%d leaves %d unexported head and %d "
                "unexported tail step(s); need >= 1 of each (active - "
                "keep_active >= 1 + skip_head, skip_head >= 1). The first "
                "recorded step follows the align_ranks drain with an empty "
                "launch queue and the last one carries the next step's host "
                "lead and the session teardown, so neither is a training step.",
                active, keep_active, skip_head, skip_head, max(tail, 0),
            )

    def _window_position(self) -> Optional[tuple]:
        """(offset in the recorded window, exported?) for the current step."""
        record = (
            torch.profiler.ProfilerAction.RECORD,
            torch.profiler.ProfilerAction.RECORD_AND_SAVE,
        )
        n = getattr(self._profiler, "step_num", 0)
        try:
            if self._sched(n) not in record:
                return None
            start = n
            while start > 0 and n - start < self._active - 1 and (
                self._sched(start - 1) == torch.profiler.ProfilerAction.RECORD
            ):
                start -= 1
        except Exception:
            return None
        offset = n - start
        exported = self._skip_head <= offset < self._skip_head + self._keep
        return offset, exported

    def is_recording(self) -> bool:
        """True while the current step is inside a RECORD window."""
        return self._window_position() is not None

    def note_event(self, kind: str, **detail: Any) -> None:
        """Record that ``kind`` happened in the current profiler step.

        Called by the training loop (only when tracing, like every other
        Profiler call) at events that make a step unrepresentative: a
        streaming window boundary (``_window_boundary_barrier`` drains the
        run-ahead queue, and the first forward of the new window can stall),
        an eval pass, and a ``compute_and_log`` metrics drain at
        ``METRIC_LOG_FREQ`` multiples. Inside an exported step this warns and
        the event is written into the trace's ``traceRendering`` so a long
        step is read as the event, not as step time. Host-side bookkeeping
        only: no device work, no synchronization. A ``metric_log`` at step
        N also notes ``after_metric_log`` on step N+1 under the
        host-sync-free step, which starts from the drained queue.
        """
        if kind == "metric_log":
            # Also note the NEXT step (see step()), even when this one is not
            # recorded, so a recorded step that follows a drain is labelled.
            self._after_metric_log = {
                k: v for k, v in detail.items() if isinstance(v, (int, float, str))
            }
        pos = self._window_position()
        if pos is None:
            return
        offset, exported = pos
        step = getattr(self._profiler, "step_num", 0)
        key = (kind, step)
        if key in self._noted:
            return
        self._noted.add(key)
        note = {
            "kind": kind,
            "profiler_step": step,
            "window_offset": offset,
            "exported": exported,
            **{k: v for k, v in detail.items() if isinstance(v, (int, float, str))},
        }
        self._notes.append(note)
        if exported and self.rank == 0:
            logger.warning(
                "Profiler: %s at exported ProfilerStep#%d (window offset %d) %s; "
                "this step is not representative of steady-state step time. "
                "Move Profiler.trace_steps if the window should avoid it.",
                kind, step, offset, detail,
            )

    def _install_gc_trace(self) -> None:
        """Diagnostic, default off (DLRM_HSTU_FP8_GC_TRACE=1; traced runs only):
        log every full (generation-2) CPython collection of >= 50 ms on every
        rank with the profiler step, so a host stall can be matched to a GC
        pause. Never changes what is collected or when."""
        if os.environ.get("DLRM_HSTU_FP8_GC_TRACE", "0").strip() != "1":
            return
        state = {"t0": None}

        def _cb(phase: str, info: Dict[str, Any]) -> None:
            try:
                if info.get("generation") != 2:
                    return
                if phase == "start":
                    state["t0"] = time.perf_counter()
                elif state["t0"] is not None:
                    ms = (time.perf_counter() - state["t0"]) * 1e3
                    state["t0"] = None
                    if ms >= 50.0:
                        logger.warning(
                            "[gc2] rank=%d profiler_step=%d full collection %.1f ms "
                            "collected=%s uncollectable=%s",
                            self.rank, getattr(self._profiler, "step_num", -1), ms,
                            info.get("collected"), info.get("uncollectable"),
                        )
            except Exception:
                pass

        gc.callbacks.append(_cb)
        logger.warning("[gc2] rank=%d full-collection trace armed (>= 50 ms)", self.rank)

    def _window_follows(self, n: int, horizon: int = 64) -> bool:
        """True when the schedule enters WARMUP/RECORD again within ``horizon``
        steps after step ``n`` (then that window's pre-window GC collects)."""
        try:
            for k in range(n + 1, n + 1 + horizon):
                if self._sched(k) != torch.profiler.ProfilerAction.NONE:
                    return True
        except Exception:
            return True
        return False

    def step(self) -> None:
        """Advance the profiler to the next step.

        With ``align_ranks`` (default), the last WARMUP step before a window
        opens is followed by a device drain and a collective barrier, so every
        rank enters the first RECORDED step from the same instant.

        This exists for measurement fidelity only. It never runs on an untraced
        job -- ``Profiler`` is constructed only ``if output_trace`` -- and it
        cannot change training throughput.

        Why. Arming Kineto is per-process and unsynchronized, so the first
        RECORDED step otherwise absorbs the cost of turning the profiler on,
        which real training never pays: job 36093 measured ranks entering step
        52 up to 24.4 ms apart and the step running +66 ms (+21%), ~24 ms of it
        drawn as exposed collective that does not exist untraced.

        What it costs. The barrier does not delete the skew, it relocates it
        into the WARMUP step, which is never exported -- so the first RECORDED
        step comes out slightly optimistic. Comparing the same absolute step
        barriered vs not (jobs 36446/36454, ``tr_ctl``, same node): wall
        -2.94 ms (-0.9%) and NCCL -9.39 ms, against an unbarriered same-step
        control of -0.20 and -2.02 ms. Both settings are wrong; off is +21%
        pessimistic and on is ~-1% optimistic, so on is ~20x closer.

        Consequence. The first recorded step is a boundary step, not a training
        step. Steps 2..N are untouched. Arm-to-arm deltas are unaffected (every
        arm is barriered identically); only the absolute level carries the ~1%.

        This is deliberately NOT done per step. A barrier on every step would
        serialize the pipeline's run-ahead and change the thing being measured;
        one barrier during WARMUP is outside the recorded region and costs the
        window nothing.

        Placement. torch's ``profile.step()`` increments ``step_num`` and then
        transitions, so the action about to be entered is ``sched(step_num+1)``.
        Draining while the current action is still WARMUP keeps both the
        synchronize and the barrier's own collective out of the trace.

        Full GC before each window (independent of ``align_ranks``). The
        previous window's export (``key_averages``, ``export_chrome_trace``)
        leaves large cyclic garbage (FunctionEvent parent/child links) that
        only a full collection frees. Left to CPython's schedule, that
        collection lands in a RECORDED step of the next window on one rank at
        a time: the step-52-56 ship captures showed a ~1.0-1.1 s single-rank
        host stall in the second window only (best explanation, not proven).
        Collecting here moves it into the unexported WARMUP step, ahead of the
        drain/barrier that absorbs it. No ``gc.freeze()`` after it: startup
        already froze the setup-time objects (``_collect_and_freeze_gc``,
        DLRM_HSTU_FP8_GC_FREEZE), and re-freezing per window would move
        whatever is alive mid-training into the permanent generation, where
        cycles among those objects are never collected again (the leak seen
        when re-freezing per data window). Traced runs only.

        Full GC after the last export (independent of ``align_ranks``; kill
        switch DLRM_HSTU_FP8_PROF_POST_GC=0). The pre-window collection above
        has no counterpart after the LAST window: its export garbage
        (many cyclic objects per rank) stays until each rank's own
        CPython full collection, which lands at a different later step
        on every rank and pauses that rank; under the per-step
        collectives every such pause stalls all ranks, so a traced run
        shows a cluster of up to one stall per rank some steps after the
        last export, at a step count that shifts with the code and the node
        count (no untraced run has it). Collecting on every rank in the same
        step right after the export turns that cluster into one pause at an
        unexported step, as the pre-window collection does for the earlier
        windows. Traced runs only; training math and data are unchanged.
        """
        n = getattr(self._profiler, "step_num", 0)
        try:
            cur, nxt = self._sched(n), self._sched(n + 1)
        except Exception:
            cur = nxt = None
        _record = (
            torch.profiler.ProfilerAction.RECORD,
            torch.profiler.ProfilerAction.RECORD_AND_SAVE,
        )
        if cur == torch.profiler.ProfilerAction.WARMUP and nxt in _record:
            # Best-effort: a capture must never be able to kill a run.
            try:
                t0 = time.perf_counter()
                found = gc.collect()
                if self.rank == 0:
                    logger.warning(
                        "Profiler: gc.collect() before trace window at profiler "
                        "step %d: %d unreachable object(s), %.1f ms",
                        n, found, (time.perf_counter() - t0) * 1e3,
                    )
            except Exception as exc:
                logger.warning(f"gc before trace window failed: {exc!r}")
            if self._align_ranks:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                except Exception as exc:
                    logger.warning(f"rank-align before trace window failed: {exc!r}")
        self._profiler.step()
        if (
            cur == torch.profiler.ProfilerAction.RECORD_AND_SAVE
            and _post_export_gc_on()
            and not self._window_follows(n)
        ):
            # Full GC right after the export of a window that no other window
            # follows (traced runs only; see the docstring above): pay the
            # export's full collection on every rank at this same, unexported
            # step instead of at each rank's own later CPython collection. A
            # window that another window follows is covered by that window's
            # pre-window collection above. Best-effort: a capture must never be
            # able to kill a run.
            try:
                t0 = time.perf_counter()
                found = gc.collect()
                if self.rank == 0:
                    logger.warning(
                        "Profiler: gc.collect() after trace export at profiler "
                        "step %d: %d unreachable object(s), %.1f ms",
                        n, found, (time.perf_counter() - t0) * 1e3,
                    )
            except Exception as exc:
                logger.warning(f"gc after trace export failed: {exc!r}")
        if self._after_metric_log is not None:
            detail, self._after_metric_log = self._after_metric_log, None
            if _hsf.enabled():
                self.note_event("after_metric_log", **detail)


class _NoOpSummaryWriter:
    """Drop-in stand-in for SummaryWriter used when TensorBoard is disabled
    (empty ``tensorboard_log_path``). All scalar writes become no-ops so the
    metrics path (compute + text-log + ``.metrics.jsonl`` sinks) runs unchanged
    and never crashes on TB file I/O. The shared ``/apps`` tfevents writer was a
    crash source under transient filer ``Errno 121`` (Remote I/O) errors, and
    nothing we consume reads TensorBoard — eval-window AUCs come from the JSONL
    sink and the text log."""

    def add_scalar(self, *args, **kwargs) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


@gin.configurable
class MetricsLogger:
    """
    Logger for tracking and computing recommendation metrics.

    Supports both classification metrics (NE, Accuracy, GAUC) and regression
    metrics (MSE, MAE) based on multitask configuration.

    Args:
        multitask_configs: List of task configurations defining metric types.
        batch_size: Batch size for metric computation.
        window_size: Window size for running metric aggregation.
        device: Device to place metric tensors on.
        rank: Process rank for distributed training.
        tensorboard_log_path: Optional path for TensorBoard logging.
    """

    def __init__(
        self,
        multitask_configs: List[TaskConfig],
        batch_size: int,
        window_size: int,
        device: torch.device,
        rank: int,
        tensorboard_log_path: str = "",
        world_size: int = 1,
        auc_threshold: Optional[float] = None,
        num_flops_per_sample: float = 0.0,
        gpu_peak_flops: float = 0.0,
        model: Optional[torch.nn.Module] = None,
        eval_cumulative: bool = False,
        cumulative_auc_bins: int = 100_000,
        train_lifetime_auc_mode: str = "binned",
        eval_lifetime_auc_mode: str = "binned",
        lifetime_auc_window: int = 10_000_000,
    ) -> None:
        # tflops/mfu reporting state (optional — when both num_flops_per_sample
        # and gpu_peak_flops are set, the train perf line gains tflops_algo/gpu,
        # mfu, tflops_real/gpu, hfu, fill. The jagged ("real") numbers come
        # from `model._last_jagged_flops_per_sample` stashed by DlrmHSTU.main_forward.
        self._num_flops_per_sample: float = max(0.0, float(num_flops_per_sample))
        self._gpu_peak_flops: float = max(0.0, float(gpu_peak_flops))
        self._model_ref: Optional[torch.nn.Module] = model
        if rank == 0 and self._num_flops_per_sample > 0 and self._gpu_peak_flops > 0:
            logger.info(
                f"FLOPS reporting enabled: {self._num_flops_per_sample / 1e9:.1f} "
                f"GFLOP/sample (dense fwd+bwd), GPU peak {self._gpu_peak_flops / 1e12:.0f} TFLOPS"
            )
        self.multitask_configs: List[TaskConfig] = multitask_configs
        all_classification_tasks: List[str] = [
            task.task_name
            for task in self.multitask_configs
            if task.task_type != MultitaskTaskType.REGRESSION
        ]
        all_regression_tasks: List[str] = [
            task.task_name
            for task in self.multitask_configs
            if task.task_type == MultitaskTaskType.REGRESSION
        ]
        assert all_classification_tasks + all_regression_tasks == [
            task.task_name for task in multitask_configs
        ]
        self.task_names: List[str] = all_classification_tasks + all_regression_tasks

        # Eval metric semantics:
        #   eval_cumulative=False (default, legacy / static / non-streaming eval):
        #     a single eval set with the configured window_size, including a
        #     lifetime AUC. Unchanged behavior.
        #   eval_cumulative=True (streaming fixed-holdout eval): a FRESH eval set
        #     (window_size=UNBOUNDED, reset each pass -> per-pass full-holdout
        #     "window_*") PLUS a CUMULATIVE set ("eval_cum", never reset ->
        #     "lifetime_*"). NE/Accuracy/GAUC are cumulative for free via their
        #     persistent scalar sums; AUC cumulative uses the selected backend.
        #
        # Lifetime-AUC backend is configurable independently for train and eval:
        #   "binned" (default): BinnedCumulativeAUC - exact-cumulative AUC via an
        #     O(num_bins) score histogram (additive all-reduce, no unbounded
        #     buffer, memory independent of #samples/#windows).
        #   "capped": LifetimeAUCMetricComputation - AUC over a trailing buffer of
        #     `lifetime_auc_window` samples/rank (the legacy approach; per-rank
        #     buffer all-gathered at compute).
        self._eval_cumulative: bool = eval_cumulative
        self._cumulative_auc_bins: int = int(cumulative_auc_bins)
        self._train_lifetime_auc_mode: str = str(train_lifetime_auc_mode)
        self._eval_lifetime_auc_mode: str = str(eval_lifetime_auc_mode)
        self._lifetime_auc_window: int = int(lifetime_auc_window)
        n_cls = len(all_classification_tasks)
        n_reg = len(all_regression_tasks)

        def _make_lifetime_auc(mode: str) -> RecMetricComputation:
            if mode == "binned":
                # window_size=0: no torchrec windowed state; histograms only.
                return BinnedCumulativeAUC(
                    my_rank=rank, batch_size=batch_size, n_tasks=n_cls,
                    window_size=0, num_bins=self._cumulative_auc_bins,
                ).to(device)
            if mode == "capped":
                return LifetimeAUCMetricComputation(
                    my_rank=rank, batch_size=batch_size, n_tasks=n_cls,
                    window_size=self._lifetime_auc_window,
                ).to(device)
            raise ValueError(
                f"lifetime_auc_mode must be 'binned' or 'capped', got {mode!r}"
            )

        def _make_class(ws: int, lifetime_mode: Optional[str]) -> List[RecMetricComputation]:
            mets: List[RecMetricComputation] = [
                NEMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=ws).to(device),
                AccuracyMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=ws).to(device),
                GAUCMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=ws).to(device),
                AUCMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=ws).to(device),
            ]
            if lifetime_mode is not None:
                mets.append(_make_lifetime_auc(lifetime_mode))
            return mets

        def _make_class_cumulative() -> List[RecMetricComputation]:
            # NE/Accuracy/GAUC: cumulative via persistent lifetime sums (window
            # value ignored at compute). AUC: selected lifetime backend.
            return [
                NEMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=window_size).to(device),
                AccuracyMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=window_size).to(device),
                GAUCMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=window_size).to(device),
                _make_lifetime_auc(self._eval_lifetime_auc_mode),
            ]

        def _make_reg(ws: int) -> List[RecMetricComputation]:
            return [
                MSEMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_reg, window_size=ws).to(device),
                MAEMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_reg, window_size=ws).to(device),
            ]

        self.class_metrics: Dict[str, List[RecMetricComputation]] = {"train": [], "eval": []}
        self.regression_metrics: Dict[str, List[RecMetricComputation]] = {"train": [], "eval": []}
        if eval_cumulative:
            self.class_metrics["eval_cum"] = []
            self.regression_metrics["eval_cum"] = []

        if all_classification_tasks:
            self.class_metrics["train"] = _make_class(window_size, lifetime_mode=self._train_lifetime_auc_mode)
            if eval_cumulative:
                self.class_metrics["eval"] = _make_class(UNBOUNDED_WINDOW, lifetime_mode=None)
                self.class_metrics["eval_cum"] = _make_class_cumulative()
            else:
                self.class_metrics["eval"] = _make_class(window_size, lifetime_mode=self._eval_lifetime_auc_mode)

        if all_regression_tasks:
            self.regression_metrics["train"] = _make_reg(window_size)
            if eval_cumulative:
                self.regression_metrics["eval"] = _make_reg(UNBOUNDED_WINDOW)
                self.regression_metrics["eval_cum"] = _make_reg(window_size)
            else:
                self.regression_metrics["eval"] = _make_reg(window_size)

        self.global_step: Dict[str, int] = {"train": 0, "eval": 0}
        # GAUC max_length from the host length tag when the host-sync-free step
        # is on (checked per call; falls back to the device .item() otherwise).
        _hsf.install_gauc_patch()
        _hsf.install_kjt_lpk_patch()
        # Sequence-embedding comm permutes (yambda_5b.gin item (22b));
        # each install() is a no-op unless its switch is on (absent = off).
        # N8: row-wise unbucketize backward as an inverse gather (row-wise
        # sequence sharding, i.e. NODES>=2). Switch DLRM_HSTU_FP8_UNBUCKETIZE_GATHER.
        from generative_recommenders.dlrm_v4 import unbucketize_gather as _ubg

        _ubg.install()
        # N3: sequence a2a fwd/bwd permutes as a row-index gather in the wire
        # dtype (bitwise identical; switch DLRM_HSTU_FP8_SEQ_A2A_ROWPERM).
        # Keeps the '## All2All_Seq_{fwd,bwd}_permute ##' trace labels.
        from generative_recommenders.dlrm_v4 import seq_a2a_rowperm as _sarp

        _sarp.install()
        # B1'' (comm iter 46): Rkeep rebag and Triton bounds check for the
        # sequence-embedding (pooling NONE) TBE, patched into torchrec
        # BatchedFusedEmbedding.forward (bitwise; switches
        # DLRM_HSTU_FP8_TBE_REBAG / DLRM_HSTU_FP8_TBE_BOUNDS_TRITON). A no-op
        # unless one of them is on; the KJT recat part needs no patch here.
        from generative_recommenders.dlrm_v4 import b1pp as _b1pp

        _b1pp.install()
        self._pending_train_loss: Optional[Any] = None
        # MLPerf `samples_count` progress unit: global trained samples, persisted
        # alongside global_step so a resumed run continues the count.
        self.cumulative_train_samples: int = 0
        # Whether the MLPerf run markers (RUN_START etc.) were already emitted for
        # this logical run. Checkpointed so a resume relaunch knows the run is
        # already open and does NOT re-emit INIT_START/RUN_START (the compliance
        # checker requires EXACTLY_ONE); the resumed process continues the same
        # event stream and emits the single RUN_STOP at convergence/end.
        self.mlperf_run_started: bool = False
        self._rank: int = int(rank)
        # Optional MLPerf logger + LR accessor wired by the streaming loop (duck-
        # typed to avoid a train-module import cycle); drives the train_loss event.
        self.mlperf_logger: Optional[Any] = None
        self.lr_getter: Optional[Callable[[], float]] = None
        self.tb_logger: Optional[SummaryWriter] = None
        if tensorboard_log_path != "":
            self.tb_logger = SummaryWriter(log_dir=tensorboard_log_path, purge_step=0)
            self.tb_logger.flush()
        else:
            # TB disabled: use a no-op writer so the existing call sites (and the
            # `assert self.tb_logger is not None` in compute_and_log) keep working
            # while no tfevents are written to the fragile shared filer.
            self.tb_logger = _NoOpSummaryWriter()

        # Throughput / time-to-target tracking. Counters are train-only; eval
        # samples are not relevant for headline samples/sec numbers.
        self._world_size: int = max(1, int(world_size))
        self._auc_threshold: Optional[float] = auc_threshold
        self._time_to_target_logged: bool = False
        self._perf_t_start: float = time.perf_counter()
        self._perf_t_window: float = self._perf_t_start
        self._perf_steps_in_window: int = 0
        self._perf_total_samples: int = 0
        self._perf_samples_counter: torch.Tensor = torch.zeros(
            1, dtype=torch.long, device=device
        )
        # Non-train wall-time to exclude from the train step-time window so
        # `step_ms` reports the canonical per-step compute latency even when an
        # interval coincides with eval or checkpointing. Categorized so the
        # excluded time is also reportable (eval_ms / ckpt_ms) rather than just
        # discarded. The trainer brackets eval/ckpt regions via
        # pause_perf(cat)/resume_perf(cat); accumulators reset each train-perf log.
        self._perf_excluded: Dict[str, float] = {"eval": 0.0, "ckpt": 0.0}
        self._perf_pause: Dict[str, Optional[float]] = {}

    def pause_perf(self, category: str) -> None:
        """Start excluding wall-time under `category` (e.g. "eval"/"ckpt") from
        the train step-time window. Idempotent: a second pause without an
        intervening resume is a no-op (keeps the earliest start)."""
        if self._perf_pause.get(category) is None:
            self._perf_pause[category] = time.perf_counter()

    def resume_perf(self, category: str) -> None:
        """Stop excluding `category` and fold the elapsed interval into the
        per-category accumulator. No-op if not currently paused."""
        t0 = self._perf_pause.get(category)
        if t0 is not None:
            self._perf_excluded[category] = (
                self._perf_excluded.get(category, 0.0)
                + (time.perf_counter() - t0)
            )
            self._perf_pause[category] = None

    @property
    def auc_threshold(self) -> Optional[float]:
        """Configured time-to-target AUC threshold (None if unset)."""
        return self._auc_threshold

    @property
    def all_metrics(self) -> Dict[str, List[RecMetricComputation]]:
        """
        Get all metrics for train and eval modes.

        Returns:
            Dictionary mapping mode ('train'/'eval') to list of metric computations.
        """
        out = {
            "train": self.class_metrics["train"] + self.regression_metrics["train"],
            "eval": self.class_metrics["eval"] + self.regression_metrics["eval"],
        }
        if "eval_cum" in self.class_metrics or "eval_cum" in self.regression_metrics:
            out["eval_cum"] = self.class_metrics.get(
                "eval_cum", []
            ) + self.regression_metrics.get("eval_cum", [])
        return out

    def update(
        self,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        labels: torch.Tensor,
        num_candidates: torch.Tensor,
        mode: str = "train",
    ) -> None:
        """
        Update metrics with new batch of predictions and labels.

        Args:
            predictions: Model prediction tensor.
            weights: Sample weight tensor.
            labels: Ground truth label tensor.
            num_candidates: Number of candidates per sample (for GAUC).
            mode: Either 'train' or 'eval'.
        """
        # On eval, update BOTH the fresh set and the never-reset cumulative set
        # (if enabled) from the same batch.
        update_targets = list(self.all_metrics[mode])
        if mode == "eval" and "eval_cum" in self.all_metrics:
            update_targets = update_targets + self.all_metrics["eval_cum"]
        for metric in update_targets:
            if isinstance(metric, GAUCMetricComputation):
                metric.update(
                    predictions=predictions,
                    labels=labels,
                    weights=weights,
                    num_candidates=num_candidates,
                )
            else:
                metric.update(
                    predictions=predictions,
                    labels=labels,
                    weights=weights,
                )
        self.advance_step(num_candidates, mode=mode, perf_counters=True)

    def advance_step(
        self,
        num_candidates: torch.Tensor,
        mode: str = "train",
        perf_counters: bool = False,
    ) -> None:
        """Step bookkeeping WITHOUT any metric accumulation.

        Everything ``update()`` does that training depends on lives here, so the
        METRIC_LOG_FREQ=0 path (training metrics fully off) calls this instead of
        ``update()`` and keeps identical counters:

          * ``global_step[mode]`` -- LR warmup, checkpoint/eval cadences, the
            MLPerf train_loss cadence and die_at_step all read the train step;
          * ``cumulative_train_samples`` -- the MLPerf ``samples_count``
            (host-side ``numel()``, no device work, no sync).

        ``perf_counters`` additionally feeds the throughput window read only by
        ``compute_and_log`` (a per-step on-device add); ``update()`` passes True,
        the metrics-off path False since nothing ever reads it.
        """
        self.global_step[mode] += 1
        if mode == "train":
            if perf_counters:
                # Accumulate on-device to avoid a per-step GPU->CPU sync; we read
                # the counter only at compute_and_log boundaries.
                self._perf_samples_counter += num_candidates.sum().to(
                    self._perf_samples_counter.dtype
                )
                self._perf_steps_in_window += 1
            # MLPerf progress counter: per-rank sample count scaled by world size
            # approximates global trained samples without an extra collective.
            self.cumulative_train_samples += int(num_candidates.numel()) * self._world_size

    def compute(self, mode: str = "train") -> Dict[str, float]:
        """
        Compute and return all metrics for the current window.

        Args:
            mode: Either 'train' or 'eval'.

        Returns:
            Dictionary mapping metric names to their computed values.
        """
        all_computed_metrics = {}

        if mode == "eval" and "eval_cum" in self.all_metrics:
            # Dual-set eval: `window_*` (fresh per-pass) from the reset-each-pass
            # set; `lifetime_*` (cumulative across passes) from the never-reset
            # set. Filter each set to the matching prefix, and drop GAUC's
            # auxiliary `*_num_samples` reports. Key names are unchanged
            # (`window_auc`, `lifetime_ne`, ...) so dashboards keep working.
            def _emit(
                metrics: List[RecMetricComputation], keep_prefix: str
            ) -> None:
                for metric in metrics:
                    for computed in metric.compute():
                        pfx = str(computed.metric_prefix)
                        name = str(computed.name)
                        if pfx != keep_prefix or name.endswith("num_samples"):
                            continue
                        all_values = computed.value.cpu()
                        for i, task_name in enumerate(self.task_names):
                            if i >= len(all_values):
                                break
                            all_computed_metrics[f"metric/{pfx}{name}/{task_name}"] = (
                                all_values[i]
                            )

            _emit(self.all_metrics["eval"], "window_")
            _emit(self.all_metrics["eval_cum"], "lifetime_")
        else:
            for metric in self.all_metrics[mode]:
                computed_metrics = metric.compute()
                for computed in computed_metrics:
                    all_values = computed.value.cpu()
                    for i, task_name in enumerate(self.task_names):
                        if i >= len(all_values):
                            break
                        key = f"metric/{str(computed.metric_prefix) + str(computed.name)}/{task_name}"
                        all_computed_metrics[key] = all_values[i]

        logger.info(
            f"{mode} - Step {self.global_step[mode]} metrics: {all_computed_metrics}"
        )
        return all_computed_metrics

    def _global_mean_loss(self, loss_terms: Dict[str, torch.Tensor]) -> float:
        """Cross-rank mean of the summed per-task losses.

        The 1-element all-reduce MUST run on every rank in lockstep; callers gate
        it on a rank-identical counter (global_step / a deterministic frequency)
        so it cannot desync.
        """
        return float(self._global_mean_loss_tensor(loss_terms))

    def maybe_log_mlperf_train_loss(
        self, aux_losses: Dict[str, torch.Tensor], every: int
    ) -> None:
        """Emit the MLPerf ``train_loss`` event on its OWN cadence.

        Decoupled from the console/TB metric cadence (``compute_and_log``): call
        this every train step with the just-computed ``aux_losses`` and the
        desired interval ``every`` (in global train steps). The cross-rank loss
        all-reduce only fires on the cadence, gated by ``global_step["train"]``
        which is incremented identically on all ranks in ``update()`` — so the
        collective stays in lockstep. No-op when MLPerf logging is disabled
        (``mlperf_logger is None`` on every rank) or ``every <= 0``. The
        streaming loop passes ``every = 0`` by default (run_knobs.
        resolve_mlperf_loss_every: 0 = OFF, the submission logging profile); the
        early return below then skips the event and the all_reduce on every rank
        alike, so collectives stay in lockstep and no train_loss is ever left
        pending for the pre-emit hook.
        """
        if self.mlperf_logger is None or every <= 0 or not aux_losses:
            return
        if self.global_step["train"] % every != 0:
            return
        c = self.mlperf_logger.constants
        md: Dict[str, Any] = {c.SAMPLES_COUNT: self.cumulative_train_samples}
        if self.lr_getter is not None:
            try:
                md["lr"] = float(self.lr_getter())
            except Exception:
                pass
        key = getattr(c, "TRAIN_LOSS", "train_loss")
        if _hsf.enabled():
            # Host-sync-free step: the same all-reduce, issued now in lockstep,
            # but the value is read one step late from a pinned non-blocking
            # copy. The previous step's event is emitted first (its copy is long
            # done by now), and any other MLLOG event flushes it beforehand via
            # the logger's pre-emit hook, so event order and metadata match the
            # synchronous path; only the wall time of the emit moves.
            self.flush_mlperf_train_loss()
            loss_t = self._global_mean_loss_tensor(aux_losses)
            self._pending_train_loss = (key, _hsf.PendingScalar(loss_t), md)
            if getattr(self.mlperf_logger, "pre_emit_hook", None) is None:
                self.mlperf_logger.pre_emit_hook = self.flush_mlperf_train_loss
            return
        train_loss = self._global_mean_loss(aux_losses)
        self.mlperf_logger.event(key=key, value=train_loss, metadata=md)

    def _global_mean_loss_tensor(
        self, loss_terms: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        loss_t = torch.stack(
            [v.detach().float().sum() for v in loss_terms.values()]
        ).sum()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss_t, op=torch.distributed.ReduceOp.SUM)
            loss_t = loss_t / self._world_size
        return loss_t

    def flush_mlperf_train_loss(self) -> None:
        """Emit the deferred train_loss event, if any (no collective inside)."""
        pending = getattr(self, "_pending_train_loss", None)
        if pending is None:
            return
        self._pending_train_loss = None
        key, value, md = pending
        if self.mlperf_logger is not None:
            self.mlperf_logger.event(key=key, value=value.get(), metadata=md)

    def compute_and_log(
        self,
        mode: str = "train",
        additional_logs: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, float]:
        """
        Compute metrics and log to TensorBoard.

        Args:
            mode: Either 'train' or 'eval'.
            additional_logs: Optional additional data to log.

        Returns:
            Dictionary mapping metric names to their computed values.

        Raises:
            AssertionError: If TensorBoard logger is not configured.
        """
        assert self.tb_logger is not None
        all_computed_metrics = self.compute(mode)
        for k, v in all_computed_metrics.items():
            self.tb_logger.add_scalar(  # pyre-ignore [16]
                f"{mode}_{k}",
                v,
                global_step=self.global_step[mode],
            )

        if additional_logs is not None:
            for tag, data in additional_logs.items():
                for data_name, data_value in data.items():
                    self.tb_logger.add_scalar(
                        f"{tag}/{mode}_{data_name}",
                        data_value.detach().clone().cpu(),
                        global_step=self.global_step[mode],
                    )

        # Global (cross-rank mean) train loss to console/TB at the metric-logging
        # cadence. The 1-element all-reduce runs on every rank in lockstep, so it
        # cannot desync. The MLPerf `train_loss` EVENT is emitted separately via
        # ``maybe_log_mlperf_train_loss`` so its cadence can be tuned independently
        # of this console/TB cadence (see that method).
        if mode == "train" and additional_logs is not None and "losses" in additional_logs:
            loss_terms = additional_logs["losses"]
            if loss_terms:
                train_loss = self._global_mean_loss(loss_terms)
                self.tb_logger.add_scalar(
                    "train_loss", train_loss, global_step=self.global_step["train"]
                )
                if self._rank == 0:
                    logger.info(
                        f"train - Step {self.global_step['train']} "
                        f"train_loss={train_loss:.5f}"
                    )

        # Throughput metrics (train only). One GPU->CPU sync per call.
        if mode == "train" and self._perf_steps_in_window > 0:
            now = time.perf_counter()
            wall_dt = max(now - self._perf_t_window, 1e-6)
            # Subtract bracketed eval/checkpoint wall-time so step_ms / sps /
            # MFU reflect canonical train-step compute, not eval+ckpt stalls
            # that happened to land in this window. The excluded time is also
            # surfaced separately below (eval_ms / ckpt_ms) rather than discarded.
            eval_s = self._perf_excluded.get("eval", 0.0)
            ckpt_s = self._perf_excluded.get("ckpt", 0.0)
            dt = max(wall_dt - eval_s - ckpt_s, 1e-6)
            n_samples = int(self._perf_samples_counter.item())
            self._perf_total_samples += n_samples
            local_sps = n_samples / dt
            global_sps = local_sps * self._world_size
            step_ms = dt * 1000.0 / self._perf_steps_in_window
            wall_step_ms = wall_dt * 1000.0 / self._perf_steps_in_window
            eval_ms = eval_s * 1000.0
            ckpt_ms = ckpt_s * 1000.0
            elapsed = now - self._perf_t_start
            step = self.global_step["train"]
            self.tb_logger.add_scalar(
                "perf/train_samples_per_sec_local", local_sps, global_step=step
            )
            self.tb_logger.add_scalar(
                "perf/train_samples_per_sec_global", global_sps, global_step=step
            )
            self.tb_logger.add_scalar(
                "perf/train_step_time_ms", step_ms, global_step=step
            )
            # Inclusive (old-semantics) per-step wall time and the eval/ckpt
            # breakdown that was excluded from step_ms above.
            self.tb_logger.add_scalar(
                "perf/train_wall_step_time_ms", wall_step_ms, global_step=step
            )
            self.tb_logger.add_scalar(
                "perf/window_eval_time_ms", eval_ms, global_step=step
            )
            self.tb_logger.add_scalar(
                "perf/window_ckpt_time_ms", ckpt_ms, global_step=step
            )
            self.tb_logger.add_scalar(
                "perf/train_total_samples", self._perf_total_samples, global_step=step
            )
            self.tb_logger.add_scalar(
                "perf/train_elapsed_sec", elapsed, global_step=step
            )
            # TFLOPS / MFU reporting (algo = dense yardstick, real = jagged).
            #   tflops_algo/gpu, mfu  — uses max_seq_len^2 attention work (the
            #     MFU yardstick: the FLOPs the workload would do if every
            #     user's UIH filled the padded seq length).
            #   tflops_real/gpu, hfu — uses this batch's mean(s_i^2) and
            #     discounts a targets-only last layer (actual GPU work;
            #     hardware utilization).
            #   fill                  — ragged / algo as a percent; how much of
            #     the padded sequence budget this batch's sequences fill.
            #   exec                  — real / algo as a percent; how much of
            #     the algo budget the model actually executed, so it carries
            #     both the ragged fill and any work a knob removed outright
            #     (exec / fill isolates the latter).
            # The jagged stash is read from the inner model; the model ref may
            # be a DMP wrapper, so unwrap via .module if present.
            tflops_str = ""
            if self._num_flops_per_sample > 0 and self._gpu_peak_flops > 0:
                local_flops = self._num_flops_per_sample * local_sps
                tflops_algo = local_flops / 1e12
                mfu = 100.0 * local_flops / self._gpu_peak_flops
                self.tb_logger.add_scalar("perf/train_tflops_algo_gpu", tflops_algo, global_step=step)
                self.tb_logger.add_scalar("perf/train_mfu_pct", mfu, global_step=step)
                tflops_str = f" tflops_algo/gpu={tflops_algo:.1f} mfu={mfu:.1f}%"
                jagged_t = None
                executed_t = None
                m = self._model_ref
                if m is not None:
                    inner = m.module if hasattr(m, "module") else m
                    _getter = getattr(inner, "jagged_flops_per_sample", None)
                    if callable(_getter):
                        # Lazy stash (DLRM_HSTU_FP8_LAZY_FLOPS_STASH).
                        jagged_t, executed_t = _getter()
                    else:
                        jagged_t = getattr(
                            inner, "_last_jagged_flops_per_sample", None
                        )
                        executed_t = getattr(
                            inner, "_last_executed_flops_per_sample", None
                        )
                if jagged_t is not None:
                    if executed_t is None:
                        executed_t = jagged_t
                    # One D->H sync for both, not two.
                    jagged, executed = torch.stack([jagged_t, executed_t]).tolist()
                    if 0 < executed <= jagged < self._num_flops_per_sample:
                        tflops_real = executed * local_sps / 1e12
                        hfu = 100.0 * executed * local_sps / self._gpu_peak_flops
                        fill = 100.0 * jagged / self._num_flops_per_sample
                        executed_pct = 100.0 * executed / self._num_flops_per_sample
                        self.tb_logger.add_scalar("perf/train_tflops_real_gpu", tflops_real, global_step=step)
                        self.tb_logger.add_scalar("perf/train_hfu_pct", hfu, global_step=step)
                        self.tb_logger.add_scalar("perf/train_fill_pct", fill, global_step=step)
                        self.tb_logger.add_scalar("perf/train_exec_pct", executed_pct, global_step=step)
                        tflops_str += f" tflops_real/gpu={tflops_real:.1f} hfu={hfu:.1f}% fill={fill:.1f}% exec={executed_pct:.1f}%"
            logger.info(
                f"train - Step {step} perf: local_sps={local_sps:.1f} "
                f"global_sps={global_sps:.1f} step_ms={step_ms:.2f} "
                f"elapsed_sec={elapsed:.1f} total_samples={self._perf_total_samples} "
                f"wall_step_ms={wall_step_ms:.2f} eval_ms={eval_ms:.1f} "
                f"ckpt_ms={ckpt_ms:.1f}"
                + tflops_str
            )
            self._perf_t_window = now
            self._perf_steps_in_window = 0
            self._perf_samples_counter.zero_()
            # Reset the excluded-time accumulators for the next window. Any
            # still-open pause (eval/ckpt straddling this log) is cleared so its
            # remaining time is not double-counted; in practice perf logs fire
            # only after a train step, never mid eval/ckpt.
            self._perf_excluded = {"eval": 0.0, "ckpt": 0.0}
            self._perf_pause = {}

        # Time-to-target: latch wall-clock once any task's AUC crosses threshold.
        # Matches MLPerf DLRM-DCNv2 reporting style (default upstream target 0.80275).
        if (
            self._auc_threshold is not None
            and not self._time_to_target_logged
        ):
            for key, val in all_computed_metrics.items():
                metric_short = key.split("/")[-2] if "/" in key else key
                if metric_short.endswith("auc") and not metric_short.endswith("gauc"):
                    if float(val) >= self._auc_threshold:
                        ttt = time.perf_counter() - self._perf_t_start
                        self.tb_logger.add_scalar(
                            f"perf/time_to_auc_{self._auc_threshold:.5f}_sec",
                            ttt,
                            global_step=self.global_step[mode],
                        )
                        logger.info(
                            f"REACHED AUC>={self._auc_threshold} on {key}="
                            f"{float(val):.6f} at elapsed_sec={ttt:.2f} "
                            f"step={self.global_step[mode]}"
                        )
                        self._time_to_target_logged = True
                        break

        return all_computed_metrics

    def reset(self, mode: str = "train"):
        """
        Reset all metrics for a given mode.

        Args:
            mode: Either 'train' or 'eval'.
        """
        for metric in self.all_metrics[mode]:
            metric.reset()


# the datasets we support
SUPPORTED_DATASETS = [
    "debug",
    "movielens-1m",
    "movielens-20m",
    "movielens-13b",
    "movielens-18b",
    "kuairand-1k",
    "streaming-400m",
    "streaming-200b",
    "streaming-100b",
    "sampled-streaming-100b",
    "yambda-5b",
]


@gin.configurable
def env_path(key: str = "", default: str = "") -> str:
    """Resolve a path from os.environ[key], falling back to `default`.

    Intended as a gin macro so paths can be overridden via env vars without
    editing the gin file. Example gin usage:

        DATA_PATH = @env_path()
        env_path.key = "DLRM_DATA_PATH"
        env_path.default = "/some/default/path"
        make_train_test_dataloaders.new_path_prefix = %DATA_PATH
    """
    return os.environ.get(key, default) if key else default


@gin.configurable
def env_str(key: str = "", default: str = "") -> str:
    """Resolve a string from os.environ[key], falling back to `default`.

    Companion to `env_int`/`env_float` for categorical/string overrides (e.g. a
    metric backend selector). Example gin usage:

        MetricsLogger.train_lifetime_auc_mode = @tlam/env_str()
        tlam/env_str.key     = "TRAIN_LIFETIME_AUC_MODE"
        tlam/env_str.default = "binned"
    """
    raw = os.environ.get(key) if key else None
    return raw if raw else default


@gin.configurable
def env_str_map(
    key: str = "",
    default: Optional[Dict[str, str]] = None,
    merge: bool = False,
) -> Dict[str, str]:
    """Parse os.environ[key] as 'name=value,name=value' into a dict.

    Falls back to `default` (gin) when the env var is unset/empty. Companion to
    `env_str` for map-valued overrides (e.g. per-table embedding placement).
    Example gin usage:

        make_optimizer_and_shard.embedding_placement_overrides = @env_str_map()
        env_str_map.key     = "EMB_PLACEMENT_OVERRIDES"
        env_str_map.default = {}

    Example env override: EMB_PLACEMENT_OVERRIDES="uid=uvm_caching,item_id=hbm".

    `merge` controls how the parsed env entries combine with `default`:
      * False (default): the env var REPLACES `default` wholesale (whole-dict
        override) — set the env var and you fully define the map.
      * True: the parsed env entries are LAYERED ON TOP of `default` (per-key
        override) — gin `default` defines the base map and the env var tweaks
        only the named keys, leaving the rest of `default` intact.
    """
    base = dict(default or {})
    raw = os.environ.get(key) if key else None
    if not raw:
        return base
    parsed: Dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, _, v = pair.partition("=")
        parsed[k.strip()] = v.strip()
    return {**base, **parsed} if merge else parsed


@gin.configurable
def env_int(key: str = "", default: int = 0) -> int:
    """Resolve an int from os.environ[key], falling back to `default`.

    Companion to `env_path` for numeric overrides. Example gin usage:

        make_optimizer_and_shard.hbm_cap_gb = @env_int()
        env_int.key = "HBM_CAP_GB"
        env_int.default = 260
    """
    raw = os.environ.get(key) if key else None
    return int(raw) if raw else default


@gin.configurable
def env_float(key: str = "", default: float = 0.0) -> float:
    """Resolve a float from os.environ[key], falling back to `default`.

    Companion to `env_int` for fractional/duration overrides (e.g. a
    checkpoint time interval in seconds). Example gin usage:

        streaming_train_eval_loop.checkpoint_time_interval_s = @env_float()
        env_float.key     = "CKPT_TIME_INTERVAL_S"
        env_float.default = 3600.0
    """
    raw = os.environ.get(key) if key else None
    return float(raw) if raw else default


_GPU_PEAK_FLOPS_TABLE: Dict[str, Dict[str, float]] = {
    # Per-GPU peak TFLOPS by dtype. Values from vendor datasheets / Primus-DLRM
    # peak_table. Used as the denominator in MFU/HFU. Keyed by case-insensitive
    # substring of torch.cuda.get_device_name(0).
    # MI355X and MI350X are the same CDNA4 silicon (256 CU); the MI355X's 1.4 kW
    # liquid-cooled envelope clocks it to 2.4 GHz vs 2.2 GHz, hence 2.5 vs 2.3
    # PFLOPS dense BF16. Values here are dense, never the 2:4-sparsity numbers.
    "MI355X": {"bf16": 2500e12, "fp32": 575e12},
    "MI350X": {"bf16": 2300e12, "fp32": 575e12},
    "MI300X": {"bf16": 1300e12, "fp32": 653e12},
    "MI325X": {"bf16": 1300e12, "fp32": 653e12},
    "B200":   {"bf16": 2250e12, "fp32": 1125e12},
    "H100":   {"bf16": 990e12,  "fp32": 67e12},
    "A100":   {"bf16": 312e12,  "fp32": 19.5e12},
}


def get_gpu_peak_flops(dtype: str = "bf16") -> float:
    """Peak FLOPS for the current GPU at the given dtype.

    Falls back to MI350X's number with a warning when the device name doesn't
    match any table entry — better to over-report MFU than to silently skip.
    """
    if not torch.cuda.is_available():
        return 0.0
    name = torch.cuda.get_device_name(0)
    for gpu_key, peaks in _GPU_PEAK_FLOPS_TABLE.items():
        if gpu_key in name:
            return peaks.get(dtype, peaks["bf16"])
    logger.warning(
        f"Unknown GPU for peak FLOPS: {name}; defaulting to MI350X bf16 (2300 TF)"
    )
    return _GPU_PEAK_FLOPS_TABLE["MI350X"]["bf16"]


@gin.configurable
def run_results_dir(run_name: str = "default", subdir: str = "results") -> str:
    """Resolve ``<recommendation>/<subdir>/<run_name>`` from this file's location.

    Used as a gin macro to give per-run output directories that persist on the
    host (recommendation is bind-mounted into the training container).

    Example gin usage::

        RUN_NAME = @env_path()
        env_path.key     = "RUN_NAME"
        env_path.default = "default"
        run_results_dir.run_name = %RUN_NAME
        Profiler.trace_dir = @run_results_dir()
    """
    # utils.py lives at <recommendation>/generative_recommenders/dlrm_v4/utils.py;
    # parents[2] climbs to <recommendation>/.
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / subdir / run_name)


@gin.configurable
def get_dataset(
    name: str,
    new_path_prefix: str = "",
    history_length: Optional[int] = None,
    min_history: Optional[int] = None,
    history_strategy: str = "interleaved",
    streaming_window_seconds: int = 86400,
    streaming_sort_within_window: bool = False,
    streaming_shuffle_fraction: float = 0.0,
    streaming_shuffle_seed: int = 0,
    train_split_percentage: float = 1.0,
    split_salt: int = 0,
    use_offline_ts_indices: bool = False,
):
    """
    Get dataset class and configuration by name.

    Args:
        name: Dataset identifier (must be in SUPPORTED_DATASETS).
        new_path_prefix: Optional prefix to prepend to data paths.

    Returns:
        Tuple of (dataset_class, kwargs_dict) for dataset instantiation.

    Raises:
        AssertionError: If dataset name is not supported.
    """
    assert name in SUPPORTED_DATASETS, f"dataset {name} not supported"
    if name == "debug":
        return DLRMv4RandomDataset, {}
    if name == "movielens-1m":
        return (
            DLRMv4MovieLensDataset,
            {
                "ratings_file": os.path.join(
                    new_path_prefix, "data/ml-1m/sasrec_format.csv"
                ),
            },
        )
    if name == "movielens-20m":
        return (
            DLRMv4MovieLensDataset,
            {
                "ratings_file": os.path.join(
                    new_path_prefix, "data/ml-20m/sasrec_format.csv"
                ),
            },
        )
    if name == "movielens-13b":
        return (
            DLRMv4SyntheticMovieLensDataset,
            {
                "ratings_file_prefix": os.path.join(
                    new_path_prefix, "data/ml-13b/16x16384"
                ),
            },
        )
    if name == "movielens-18b":
        return (
            DLRMv4SyntheticMovieLensDataset,
            {
                "ratings_file_prefix": os.path.join(
                    new_path_prefix, "data/ml-18b/20x36864"
                ),
            },
        )
    if name == "kuairand-1k":
        return (
            DLRMv4KuaiRandDataset,
            {
                "seq_logs_file": os.path.join(
                    new_path_prefix, "data/KuaiRand-1K/data/processed_seqs.csv"
                ),
            },
        )
    if name == "streaming-400m":
        return (
            DLRMv4SyntheticStreamingDataset,
            {
                "ratings_file_prefix": os.path.join(
                    new_path_prefix, "data/streaming-400m/"
                ),
                "train_ts": 8,
                "total_ts": 10,
                "num_files": 3,
                "num_users": 150_000,
                "num_items": 1_500_000,
                "num_categories": 128,
            },
        )
    if name == "streaming-200b":
        return (
            DLRMv4SyntheticStreamingDataset,
            {
                "ratings_file_prefix": os.path.join(
                    new_path_prefix, "data/streaming-200b/"
                ),
                "train_ts": 90,
                "total_ts": 100,
                "num_files": 100,
                "num_users": 10_000_000,
                "num_items": 1_000_000_000,
                "num_categories": 128,
            },
        )
    if name == "streaming-100b":
        return (
            DLRMv4SyntheticStreamingDataset,
            {
                "ratings_file_prefix": os.path.join(
                    new_path_prefix, "data/streaming-100b/"
                ),
                "train_ts": 90,
                "total_ts": 100,
                "num_files": 100,
                "num_users": 5_000_000,
                "num_items": 1_000_000_000,
                "num_categories": 128,
            },
        )
    if name == "yambda-5b":
        from generative_recommenders.dlrm_v4.configs import YAMBDA_5B_CROSS_SPECS

        return (
            DLRMv4YambdaDataset,
            {
                # Layout: <new_path_prefix>/processed_5b/{train_sessions.parquet,...}
                # and <new_path_prefix>/shared_metadata/{artist,album}_item_mapping.parquet.
                # The dataset auto-builds a MAP_SHARED-mmap'd cache of the
                # flat columns + LISTEN-anchor positions under
                # <processed_dir>/hstu_cache_L<history_length>/ on first use;
                # all ranks on a node share the same physical pages.
                "processed_dir": os.path.join(new_path_prefix, "processed_5b"),
                "metadata_dir": os.path.join(new_path_prefix, "shared_metadata"),
                # Per-pool truncation cap; total interleaved UIH ~ 3*L/3 = L.
                # Override via `get_dataset.history_length = N` in gin.
                "history_length": history_length if history_length is not None else 4096,
                "scan_window": 20000,
                # Anchor-eligibility floor: a LISTEN event qualifies once the
                # user has >= min_history prior events. Decoupled from
                # history_length (gather cap) since jagged attention handles
                # short UIH. None = legacy (require a full history_length).
                # Override via `get_dataset.min_history = N` / $MIN_HISTORY.
                "min_history": min_history,
                # UIH construction: "interleaved" (per-pool L//3 cap) or
                # "last_n" (last history_length pooled events, no per-pool
                # split). Strategy-independent on disk — both reuse the same
                # hstu_cache_L<history_length>/ and positions file (the gather
                # runs at sample-construction time), so switching needs no
                # rebuild. Override via $HISTORY_STRATEGY.
                "history_strategy": history_strategy,
                "cross_specs": YAMBDA_5B_CROSS_SPECS,
                # Temporal-streaming knobs (only used under --mode
                # streaming-train-eval; ignored by the default train-eval path).
                "streaming_window_seconds": streaming_window_seconds,
                "streaming_sort_within_window": streaming_sort_within_window,
                "use_offline_ts_indices": use_offline_ts_indices,
                # In-window shuffle diversity dial in [0,1]: K=round(frac*N) within-
                # segment shuffle. 0=off/user-major, 1=full. Config-invariant and
                # deterministic by (seed, ts).
                "streaming_shuffle_fraction": streaming_shuffle_fraction,
                "streaming_shuffle_seed": streaming_shuffle_seed,
                # User-level train:eval holdout for the streaming path. 1.0 =
                # no holdout (legacy). <1.0 holds out (1 - tsp) of users as a
                # fixed eval set; those users are never trained.
                "train_split_percentage": train_split_percentage,
                "split_salt": split_salt,
            },
        )
    if name == "sampled-streaming-100b":
        return (
            DLRMv4SyntheticStreamingDataset,
            {
                "ratings_file_prefix": os.path.join(
                    new_path_prefix, "data/streaming-100b/sampled_data/"
                ),
                "train_ts": 90,
                "total_ts": 100,
                "num_files": 1,
                "num_users": 50_000,
                "num_items": 1_000_000_000,
                "num_categories": 128,
            },
        )
