###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Power projection: per-GPU power under a cap, node/cluster power, capacity planning.

Entry points:

* :func:`project_power` -- power on top of a performance projection result.
* :func:`fit_power_profile` / :func:`load_telemetry` -- calibrate the model to
  measured telemetry.
* :func:`capacity_plan` -- cluster throughput under a facility power budget.
"""

from primus.core.projection.power.footprint import (
    IterationBreakdown,
    OpRecorder,
    WorkloadFootprint,
    build_iteration_phases,
)
from primus.core.projection.power.infrastructure import (
    InfraConfig,
    capacity_plan,
    cluster_power,
)
from primus.core.projection.power.model import (
    ParametricPowerModel,
    Phase,
    PowerSolution,
)
from primus.core.projection.power.projector import (
    PowerConfig,
    PowerReport,
    project_power,
)
from primus.core.projection.power.registry import (
    available_power_models,
    get_power_model,
    register_power_model,
)
from primus.core.projection.power.specs import (
    GPUPowerProfile,
    public_archs,
    resolve_power_profile,
)
from primus.core.projection.power.telemetry import (
    TelemetrySample,
    fit_power_profile,
    load_telemetry,
    make_telemetry_sampler,
)

__all__ = [
    "GPUPowerProfile",
    "InfraConfig",
    "IterationBreakdown",
    "OpRecorder",
    "ParametricPowerModel",
    "Phase",
    "PowerConfig",
    "PowerReport",
    "PowerSolution",
    "TelemetrySample",
    "WorkloadFootprint",
    "available_power_models",
    "build_iteration_phases",
    "capacity_plan",
    "cluster_power",
    "fit_power_profile",
    "get_power_model",
    "load_telemetry",
    "make_telemetry_sampler",
    "project_power",
    "public_archs",
    "register_power_model",
    "resolve_power_profile",
]
