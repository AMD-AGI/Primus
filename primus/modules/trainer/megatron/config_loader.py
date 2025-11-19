###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Helpers for injecting Megatron runtime defaults into Primus configs."""

from __future__ import annotations

import argparse
from functools import lru_cache
from types import SimpleNamespace
from typing import Mapping


def _build_megatron_parser() -> argparse.ArgumentParser:
    """Construct a Megatron argument parser without touching sys.argv."""
    from megatron.training.arguments import add_megatron_arguments

    parser = argparse.ArgumentParser(
        description="Primus Megatron default argument loader", allow_abbrev=False
    )
    return add_megatron_arguments(parser)


@lru_cache(maxsize=1)
def _load_megatron_defaults() -> Mapping[str, object]:
    """Return a cached dictionary of Megatron's CLI default values."""
    parser = _build_megatron_parser()
    args = parser.parse_args([])  # Only capture defaults.
    return vars(args).copy()


def apply_megatron_defaults(namespace: SimpleNamespace) -> SimpleNamespace:
    """Populate missing fields on a namespace using Megatron defaults."""
    defaults = _load_megatron_defaults()
    for key, value in defaults.items():
        if hasattr(namespace, key):
            continue
        setattr(namespace, key, value)
    return namespace
