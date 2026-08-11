###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Per-model hooks for the AutoModel diffusion backend.

One subpackage per model. A model subpackage must not import another's: anything
two models share belongs in ``quantization/`` (or a future ``common/``). That
isolation is what lets a single model be reviewed, or upstreamed, on its own.

Deliberately no re-exports, so that importing one model never drags in another's
optional dependencies. See ``quantization/__init__.py``.
"""
