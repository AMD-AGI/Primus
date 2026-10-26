###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Deprecated alias of :mod:`primus.backends.automodel`, kept so imports of the
original module paths, and lookups of ``nemo_automodel`` in the ``BackendRegistry``,
keep working. Configs and ``--backend`` values that say ``nemo_automodel`` are
mapped to ``automodel`` through :mod:`primus.backends.backend_aliases` before any
backend is loaded, so a normal launch never imports this package.

Importing any of its modules imports this package first, so the warning below is
issued once per process, for whichever old path is imported first.
"""

import warnings

from primus.backends.automodel import AutoModelAdapter, AutoModelPretrainTrainer
from primus.core.backend.backend_registry import BackendRegistry

warnings.warn(
    "primus.backends.nemo_automodel is deprecated; import primus.backends.automodel instead.",
    FutureWarning,
    stacklevel=2,
)

NemoAutomodelAdapter = AutoModelAdapter
NemoAutomodelPretrainTrainer = AutoModelPretrainTrainer

BackendRegistry.register_adapter("nemo_automodel", NemoAutomodelAdapter)
BackendRegistry.register_trainer_class(NemoAutomodelPretrainTrainer, "nemo_automodel")
