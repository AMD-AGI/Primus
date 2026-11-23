###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MLPerf Training 6.1 disclosure keys for Flux on the diffusion backend.

Megatron's ``extract_mlperf_configs`` needs a Megatron args object. Flux logs
from the FSDP2 trainer, so this helper only reads environment variables and
the parallelism values the caller already knows.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

# mlperf_logging 6.1.0 ruleset. Kept here so a missing or older logging package
# still fails at startup instead of after the run.
ALLOWED_NUMERICAL_PRECISIONS = frozenset(
    {
        "fp64",
        "fp32",
        "tf32",
        "fp16",
        "fp8",
        "mxfp6",
        "nvfp4",
        "mxfp4",
        "bfloat16",
        "Graphcore FLOAT 16.16",
        "int8",
        "uint8",
        "int4",
        "uint4",
    }
)

_PRECISION_PARTS = ("linear", "attn", "comm")
_LINEAR_ALIAS = "MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR"


def _environ(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _precision_value(part: str, environ: Mapping[str, str]) -> str:
    primary = f"MLLOG_LOWEST_NUMERICAL_PRECISION_IN_{part.upper()}"
    if primary in environ:
        value = environ.get(primary, "").strip()
        source = primary
    elif part == "linear":
        # Pre-6.1 launchers set this name. IN_LINEAR wins when it is set,
        # including when it is set to an empty string.
        value = environ.get(_LINEAR_ALIAS, "").strip()
        source = f"{primary} (alias {_LINEAR_ALIAS})"
    else:
        value = ""
        source = primary
    if value not in ALLOWED_NUMERICAL_PRECISIONS:
        allowed = ", ".join(sorted(ALLOWED_NUMERICAL_PRECISIONS))
        raise ValueError(f"{source}={value!r}: MLPerf 6.1 requires one of {allowed}")
    return value


def mlperf_v61_disclosure(
    *,
    tensor_parallelism: int,
    pipeline_parallelism: int,
    context_parallelism: int,
    expert_parallelism: int,
    micro_batch_size: int,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Return the 6.1 disclosure events, in compliance-log order.

    Raises ``ValueError`` for a missing or non-enum precision, and for a
    missing ``MLLOG_CONFIG_FILENAME``. Callers must invoke this on every rank
    before training; the exception is not rank-gated.
    """
    env = _environ(environ)
    config_filename = env.get("MLLOG_CONFIG_FILENAME", "").strip()
    if not config_filename:
        raise ValueError("MLLOG_CONFIG_FILENAME must name the submission config_*.sh")
    disclosure: dict[str, object] = {
        f"lowest_numerical_precision_in_{part}": _precision_value(part, env) for part in _PRECISION_PARTS
    }
    disclosure["tensor_parallelism"] = int(tensor_parallelism)
    disclosure["pipeline_parallelism"] = int(pipeline_parallelism)
    disclosure["context_parallelism"] = int(context_parallelism)
    disclosure["expert_parallelism"] = int(expert_parallelism)
    disclosure["micro_batch_size"] = int(micro_batch_size)
    disclosure["config_filename"] = config_filename
    return disclosure
