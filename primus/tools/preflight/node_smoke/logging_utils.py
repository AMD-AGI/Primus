###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Hostname normalization + log/warn helpers used everywhere in node-smoke.

The log prefix ``[HH:MM:SS][node-smoke][<short-host>]`` is part of the
operator-facing contract -- existing log-scraping tooling assumes it,
so it is preserved verbatim across the refactor.
"""

from __future__ import annotations

import socket
import sys
import time


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _short_name(h: str) -> str:
    """Return the leading short-hostname segment.

    SLURM tools (`scontrol show hostnames`, `srun --nodelist=`,
    `srun --exclude=`) all operate on short hostnames, so we normalize
    everywhere so the produced ``passing_nodes.txt`` / ``failing_nodes.txt``
    can be piped straight into them. ``socket.gethostname()`` returns the
    FQDN on some clusters, hence this helper.
    """
    if not h:
        return h
    return h.split(".", 1)[0]


def _this_host_short() -> str:
    """This node's short hostname (first segment of socket.gethostname())."""
    return _short_name(socket.gethostname())


def _log(msg: str) -> None:
    print(f"[{_ts()}][node-smoke][{_this_host_short()}] {msg}", flush=True)


def _warn(msg: str) -> None:
    print(
        f"[{_ts()}][node-smoke][{_this_host_short()}] WARN: {msg}",
        file=sys.stderr,
        flush=True,
    )
