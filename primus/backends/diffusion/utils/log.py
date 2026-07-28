###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Logger proxy for the diffusion backend.

Primus configures loguru sinks whose format references bound ``extra`` fields
(``rank``, ``world_size``, ``user``, ``team``, ``module_name``, ``node_ip``).
Those fields are only present on the Primus-bound logger created in
``primus.core.utils.logger.setup_logger``. Emitting from the raw, unbound
global loguru logger (``from loguru import logger``) produces records without
those extras, which makes every sink raise "Logging error in Handler".

This module exposes a ``logger`` proxy that forwards attribute access to the
live Primus-bound logger, so diffusion code keeps the familiar
``logger.info(...)`` style while inheriting the bound extras. Before the Primus
logger is initialized (e.g. standalone tooling), it falls back to the raw
loguru logger, which at that point still uses loguru's default sink.
"""

from __future__ import annotations

from primus.core.utils import logger as _primus_logger_module


class _LoggerProxy:
    @staticmethod
    def _resolve():
        bound = getattr(_primus_logger_module, "_logger", None)
        if bound is not None:
            return bound
        from loguru import logger as _raw_logger

        return _raw_logger

    def __getattr__(self, name):
        return getattr(self._resolve(), name)


logger = _LoggerProxy()
