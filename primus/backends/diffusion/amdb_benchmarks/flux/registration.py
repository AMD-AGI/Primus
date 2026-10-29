###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Build the FLUX.1-dev benchmark on the native FLUX modules."""

from __future__ import annotations

import os
from typing import Any

from primus.backends.diffusion.amdb_benchmarks.download import materialize_pretrained
from primus.backends.diffusion.models.registrations.flux import (
    build_flux_model as build_native_flux_model,
)
from primus.backends.diffusion.utils.log import logger


def build_flux_model(model_config: dict[str, Any]):
    """Native FLUX.1-dev with the AMDiffusionBenchmark load and noise schedule."""
    config = dict(model_config or {})
    config["model_preset"] = config.get("model_preset") or "flux.1-dev"
    cfg = dict(config.get("config") or {})
    cfg.setdefault("timestep_sampling", "amdb_logit_normal")
    cfg.setdefault("shift", 3.0)
    cfg["strict_pretrained_load"] = True
    params = dict(cfg.get("params") or {})
    # BFL FLUX.1-dev uses FP32 RMSNorm statistics with eps 1e-6. The native
    # flux.1-dev trainer leaves this unset and keeps the TorchTitan default.
    params.setdefault("qk_norm_eps", 1e-6)
    cfg["params"] = params

    pretrained = config.get("load_from_pretrained_path") or config.get("pretrained_path") or ""
    pretrained = materialize_pretrained("flux-dev", str(pretrained))
    if pretrained:
        config["load_from_pretrained_path"] = pretrained

    encoder = dict(config.get("encoder") or cfg.get("encoder") or {})
    if not encoder.get("autoencoder") and pretrained:
        checkpoint = os.path.expanduser(str(pretrained))
        root = checkpoint if os.path.isdir(checkpoint) else os.path.dirname(checkpoint)
        ae_path = os.path.join(root, "ae.safetensors")
        if os.path.isfile(ae_path):
            encoder["autoencoder"] = ae_path
        else:
            logger.warning(f"FLUX autoencoder was not found next to {checkpoint}")
    if encoder:
        config["encoder"] = encoder
    config["config"] = cfg
    return build_native_flux_model(config)
