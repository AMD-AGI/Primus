###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

import torch.distributed as dist

from primus.tools.preflight.global_vars import RANK, WORLD_SIZE


def _first_ib_device_name() -> Optional[str]:
    ib_path = "/sys/class/infiniband"
    try:
        names = sorted(os.listdir(ib_path))
        return names[0] if names else None
    except OSError:
        return None


def _resolve_server_host() -> str:
    return os.environ.get("MASTER_ADDR", "localhost").strip() or "localhost"


def _parse_peak_gbps(text: str) -> Optional[float]:
    """Best-effort peak Gb/sec from ib_write_bw stdout/stderr."""
    best: Optional[float] = None
    for line in text.splitlines():
        for m in re.finditer(r"(\d+\.\d+)\s*Gb/sec", line, re.I):
            v = float(m.group(1))
            best = v if best is None else max(best, v)
        for m in re.finditer(r"(\d+\.\d+)\s*GB/sec", line):
            v = float(m.group(1))
            best = v if best is None else max(best, v)
    return best


def append_cluster_sphere_verbs_rdma_section(args: Any, markdown_file: str) -> None:
    """
    Run ib_write_bw between two ranks (WORLD_SIZE == 2) and append results to the perf markdown.
    Server is rank 0; client is rank 1 connecting to MASTER_ADDR.
    """
    from primus.tools.preflight.cluster_sphere.report import wants_cluster_sphere_rdma_bw

    if not wants_cluster_sphere_rdma_bw(args):
        return

    lines: List[str] = ["\n---\n\n## Cluster Sphere — Verbs RDMA (ib_write_bw)\n\n"]

    if WORLD_SIZE != 2:
        lines.append(
            f"This check requires exactly **two** distributed processes (`WORLD_SIZE=2`). "
            f"Current world size: **{WORLD_SIZE}**. Skipping `ib_write_bw`.\n\n"
        )
        if RANK == 0:
            _append_lines(markdown_file, lines)
        return

    if shutil.which("ib_write_bw") is None:
        lines.append(
            "`ib_write_bw` was not found in PATH (install "
            "[linux-rdma/perftest](https://github.com/linux-rdma/perftest)).\n\n"
        )
        if RANK == 0:
            _append_lines(markdown_file, lines)
        return

    ib_dev = _first_ib_device_name()
    if not ib_dev:
        lines.append("No InfiniBand / RDMA device under `/sys/class/infiniband`; skipping.\n\n")
        if RANK == 0:
            _append_lines(markdown_file, lines)
        return

    port = int(os.environ.get("PRIMUS_IB_WRITE_BW_PORT", "2000"))
    server_host = _resolve_server_host()

    server_holder: Dict[str, Any] = {}

    def server_main() -> None:
        cmd = [
            "ib_write_bw",
            "-d",
            ib_dev,
            "-q",
            "4",
            "-a",
            "--report_gbits",
            "-F",
            "-p",
            str(port),
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
            )
            server_holder["out"] = (proc.stdout or "") + "\n" + (proc.stderr or "")
            server_holder["rc"] = proc.returncode
        except subprocess.TimeoutExpired as e:
            server_holder["out"] = str(e)
            server_holder["rc"] = -1

    th: Optional[threading.Thread] = None
    if RANK == 0:
        th = threading.Thread(target=server_main, daemon=True)
        th.start()
        time.sleep(2)

    dist.barrier()

    client_payload: Optional[Dict[str, Any]] = None
    if RANK == 1:
        time.sleep(5)
        cmd = [
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
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            peak = _parse_peak_gbps(out)
            client_payload = {"stdout": out, "rc": proc.returncode, "peak_gbps": peak}
        except subprocess.TimeoutExpired as e:
            client_payload = {"stdout": str(e), "rc": -1, "peak_gbps": None}

    dist.barrier()

    if RANK == 0 and th is not None:
        th.join(timeout=200)

    local_obj = client_payload if RANK == 1 else None
    gathered_objs: Optional[List[Any]] = None
    if RANK == 0:
        gathered_objs = [None] * WORLD_SIZE
        dist.gather_object(local_obj, object_gather_list=gathered_objs, dst=0)
    else:
        dist.gather_object(local_obj, dst=0)

    if RANK == 0:
        client = None
        if gathered_objs is not None and len(gathered_objs) > 1:
            client = gathered_objs[1]

        peak = None
        client_txt = ""
        if isinstance(client, dict):
            peak = client.get("peak_gbps")
            client_txt = str(client.get("stdout") or "")

        if peak is None:
            peak = _parse_peak_gbps(client_txt)

        lines.append(f"- **Device**: `{ib_dev}`\n")
        lines.append(f"- **Server**: `{server_host}:{port}` (rank 0)\n")
        lines.append("- **Client**: rank 1\n")
        if peak is not None:
            lines.append(f"- **Peak bandwidth (parsed)**: **{peak:.2f} Gb/sec**\n")
        else:
            lines.append("- **Peak bandwidth**: could not parse (see raw output below)\n")
        lines.append("\n")

        if client_txt:
            lines.append("<details><summary>Client output</summary>\n\n```\n")
            lines.append(client_txt[:12000])
            lines.append("\n```\n\n</details>\n\n")

        so = server_holder.get("out")
        if so:
            lines.append("<details><summary>Server output</summary>\n\n```\n")
            lines.append(str(so)[:12000])
            lines.append("\n```\n\n</details>\n\n")

        _append_lines(markdown_file, lines)


def _append_lines(path: str, lines: List[str]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(lines))
