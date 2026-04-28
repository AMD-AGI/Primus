###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Verbs ``ib_write_bw`` helpers (subprocess only; no PyTorch)."""

from __future__ import annotations

import os
import re
import subprocess
from typing import List, Optional

DEFAULT_PORT_ENV = "PRIMUS_IB_WRITE_BW_PORT"


def default_port() -> int:
    return int(os.environ.get(DEFAULT_PORT_ENV, "2000"))


def first_ib_device_name() -> Optional[str]:
    ib_path = "/sys/class/infiniband"
    try:
        names = sorted(os.listdir(ib_path))
        return names[0] if names else None
    except OSError:
        return None


def parse_peak_gbps(text: str) -> Optional[float]:
    best: Optional[float] = None
    for line in text.splitlines():
        for m in re.finditer(r"(\d+\.\d+)\s*Gb/sec", line, re.I):
            v = float(m.group(1))
            best = v if best is None else max(best, v)
        for m in re.finditer(r"(\d+\.\d+)\s*GB/sec", line):
            v = float(m.group(1))
            best = v if best is None else max(best, v)
    return best


def ib_write_bw_server_cmd(ib_dev: str, port: int) -> List[str]:
    return ["ib_write_bw", "-d", ib_dev, "-q", "4", "-a", "--report_gbits", "-F", "-p", str(port)]


def ib_write_bw_client_cmd(ib_dev: str, server_host: str, port: int) -> List[str]:
    return [
        "ib_write_bw",
        "-d",
        ib_dev,
        "-q",
        "4",
        "-a",
        "--report_gbits",
        "-F",
        server_host,
        "-p",
        str(port),
    ]


def run_ib_write_bw_server(ib_dev: str, port: int, *, timeout_sec: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        ib_write_bw_server_cmd(ib_dev, port),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )


def run_ib_write_bw_client(ib_dev: str, server_host: str, port: int, *, timeout_sec: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ib_write_bw_client_cmd(ib_dev, server_host, port),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
