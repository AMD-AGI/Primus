###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Measure the bench GPU's effective compute clock and HBM bandwidth.

Benchmark-mode projection anchors layer times to the bench host, so the clock
and bandwidth that host actually sustains are part of the result.  This
package measures both on every bench GPU:

* the **compute clock**, back-calculated from a few of the workload's own
  GEMMs (see :mod:`.compute_clock`);
* the **memory bandwidth**, from STREAM copy/scale/add/triad, derated to what
  a layer's memory-bound kernels sustain and fed to the projection's
  memory-bound terms (see :mod:`.stream_bandwidth`, which also explains why
  bandwidth -- not a memory clock -- is what can be measured).
"""

from primus.core.projection.hardware_calibration.calibration import (
    PAYLOAD_KEY,
    SKIP_ENV,
    TAG,
    calibrated_model_hbm_bandwidth_gbps,
    calibration_skip_reason,
    format_calibration_report,
    gather_rank_summaries,
    run_hardware_calibration,
)
from primus.core.projection.hardware_calibration.stream_bandwidth import (
    STREAM_DERATE_ENV,
    STREAM_WORKLOAD_DERATE,
    derive_model_hbm_bandwidth,
)

__all__ = [
    "PAYLOAD_KEY",
    "SKIP_ENV",
    "STREAM_DERATE_ENV",
    "STREAM_WORKLOAD_DERATE",
    "TAG",
    "calibrated_model_hbm_bandwidth_gbps",
    "calibration_skip_reason",
    "derive_model_hbm_bandwidth",
    "format_calibration_report",
    "gather_rank_summaries",
    "run_hardware_calibration",
]
