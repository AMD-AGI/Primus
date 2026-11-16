# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# pyre-strict
"""Pure (torch-free) resolution of the streaming loop's logging/eval-pool knobs.

Kept out of train/utils.py so the decisions, and the exact "armed OK" strings a
run log is checked for, are unit-testable on a CPU host without torch.

  * metric cadence (ship-4 item 1): METRIC_LOG_FREQ is the console/TB metric
    cadence; MLPERF_TRAIN_LOSS_LOG_FREQ the MLLOG train_loss cadence. They are
    independent.
  * training metrics off: METRIC_LOG_FREQ <= 0 (0 is the documented value and
    the gin default) disables training-side metric logging completely -- no
    per-step torchmetrics update, no train compute_and_log, no metric-cadence
    diagnostics. Step bookkeeping (global_step, samples_count) and EVAL are
    unaffected.
  * MLPerf train_loss off: MLPERF_TRAIN_LOSS_LOG_FREQ <= 0 (gin default 0)
    emits no train_loss event and runs no loss all_reduce. MLPerf 6.1 does not
    require train_loss; every required MLLOG event is unaffected.
  * eval pool workers (ship-4 item 2): EVAL_NUM_WORKERS sets num_workers of the
    persistent eval DataLoader pool only; < 0 inherits the train pools'
    NUM_WORKERS (the pre-ship-4 behaviour).
"""

from typing import Dict, Optional

# Grep-able arming literals. A run log containing these proves the item is on.
METRIC_CADENCE_ARMED = "armed OK: metric_cadence"
EVAL_POOL_ARMED = "armed OK: eval_pool"
METRICS_OFF_ARMED = (
    "armed OK: training metrics OFF (METRIC_LOG_FREQ=0): "
    "no train metric update/compute_and_log"
)
# Submission logging profile: MLPerf train_loss off (MLPERF_TRAIN_LOSS_LOG_FREQ <= 0).
TRAIN_LOSS_OFF_ARMED = (
    "armed OK: MLPerf train_loss OFF (MLPERF_TRAIN_LOSS_LOG_FREQ=0): "
    "no train_loss event or loss all_reduce"
)


def metric_logging_enabled(metric_log_frequency: int) -> bool:
    """Training-side metric logging is on iff the cadence is positive."""
    return metric_log_frequency is not None and int(metric_log_frequency) > 0


def metric_log_due(step: int, metric_log_frequency: int) -> bool:
    """`step % metric_log_frequency == 0`, with <= 0 meaning OFF (never due,
    never a ZeroDivisionError)."""
    return metric_logging_enabled(metric_log_frequency) and (
        int(step) % int(metric_log_frequency) == 0
    )


def resolve_mlperf_loss_every(
    metric_log_frequency: int, mlperf_train_loss_log_frequency: int
) -> int:
    """MLPerf train_loss cadence. <= 0 (the gin default 0) means OFF: no
    train_loss event and no cross-rank loss all_reduce. MLPerf Training 6.1 does
    not require train_loss (no compliance-checker rule), so the submission
    logging profile turns it off; throughput/ladder runs set an explicit N > 0
    (e.g. 1 or 20) to get per-step MLLOG timestamps. Independent of the metric
    cadence."""
    if mlperf_train_loss_log_frequency and mlperf_train_loss_log_frequency > 0:
        return int(mlperf_train_loss_log_frequency)
    return 0


def mlperf_train_loss_log_line(mlperf_loss_every: int) -> str:
    """Arming line for the MLPerf train_loss cadence (rank 0)."""
    if mlperf_loss_every <= 0:
        return TRAIN_LOSS_OFF_ARMED
    return f"MLPerf train_loss ON: every {mlperf_loss_every} steps (MLPERF_TRAIN_LOSS_LOG_FREQ)"


def metric_cadence_log_line(metric_log_frequency: int, mlperf_loss_every: int) -> str:
    """One line describing the resolved cadences.

    ARMED when the console/TB metric cadence is decoupled from, and coarser
    than, the MLPerf train_loss cadence (e.g. an opt-in 50 vs 5). Otherwise the
    line says OFF so it can never be mistaken for the armed literal.
    Metric cadence <= 0: the single METRICS_OFF_ARMED line (plus the resolved
    train_loss cadence) instead.
    """
    if not metric_logging_enabled(metric_log_frequency):
        return f"{METRICS_OFF_ARMED} mlperf_train_loss_every={mlperf_loss_every}"
    if metric_log_frequency > mlperf_loss_every:
        return (
            f"{METRIC_CADENCE_ARMED} metric_log_frequency={metric_log_frequency} "
            f"mlperf_train_loss_every={mlperf_loss_every}"
        )
    return (
        f"metric_cadence OFF (coupled) metric_log_frequency={metric_log_frequency} "
        f"mlperf_train_loss_every={mlperf_loss_every}"
    )


def resolve_eval_dataloader_kwargs(eval_num_workers: Optional[int]) -> Dict[str, int]:
    """Extra kwargs for the persistent eval pool's make_persistent_streaming_dataloader.

    None or < 0 -> {} (the pool inherits the gin num_workers binding, i.e. the
    train pools' NUM_WORKERS, byte-identical to before). >= 0 -> num_workers.
    """
    if eval_num_workers is None:
        return {}
    n = int(eval_num_workers)
    if n < 0:
        return {}
    return {"num_workers": n}


def eval_pool_log_line(eval_dl_kwargs: Dict[str, int]) -> str:
    if "num_workers" in eval_dl_kwargs:
        return (
            f"{EVAL_POOL_ARMED} num_workers={eval_dl_kwargs['num_workers']} "
            "from EVAL_NUM_WORKERS (train pools keep NUM_WORKERS)"
        )
    return "eval_pool OFF: inherits NUM_WORKERS (EVAL_NUM_WORKERS<0)"
