###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Primus-side FLUX data loaders for the nemo_automodel backend.

``synthetic.py`` provides an in-memory generator matching the shapes the upstream
FLUX cache produces, for throughput / memory / MFU measurement without an input
pipeline. Real training reads the upstream cache through
``nemo_automodel.components.datasets.diffusion.build_text_to_image_multiresolution_dataloader``.

Neither is registered anywhere: the diffusion recipe accepts any ``_target_`` that
resolves to a ``@dataclass`` exposing ``build()``, so YAML points at the class directly.
"""
