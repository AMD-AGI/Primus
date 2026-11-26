###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Renamed backends: the one place a rename is recorded.

The config parser, the launcher's ``--backend`` handling (``prepare_experiment.sh``)
and the CLI's framework lookups resolve names through :func:`resolve_backend_alias`.
A renamed backend's deprecated package registers its old name with the
``BackendRegistry`` itself, for code that looks a backend up directly.

Dependency-free, so the launcher scripts can import it before any backend is set up.
"""

import warnings

# Old name -> current name.
RENAMED_BACKENDS = {"nemo_automodel": "automodel"}


def resolve_backend_alias(name: str) -> str:
    """The current name of a renamed backend, with a FutureWarning; any other name unchanged."""
    current = RENAMED_BACKENDS.get(name)
    if current is None:
        return name
    warnings.warn(
        f"framework '{name}' is deprecated; use 'framework: {current}'.",
        FutureWarning,
        stacklevel=2,
    )
    return current
