# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Primus dataset agent: plug a real dataset into one backend and model."""

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    from tools.dataset_agent.orchestrator import main as orchestrator_main

    return orchestrator_main(argv)
