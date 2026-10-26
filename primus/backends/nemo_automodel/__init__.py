###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Deprecated alias of :mod:`primus.backends.automodel`, kept so imports of the
original module paths keep working. Configs with ``framework: nemo_automodel``
are mapped to ``automodel`` by the config parser, not here.
"""

import warnings

from primus.backends.automodel import AutoModelAdapter, AutoModelPretrainTrainer

warnings.warn(
    "primus.backends.nemo_automodel is deprecated; import primus.backends.automodel instead.",
    DeprecationWarning,
    stacklevel=2,
)

NemoAutomodelAdapter = AutoModelAdapter
NemoAutomodelPretrainTrainer = AutoModelPretrainTrainer
