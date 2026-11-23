# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""Which ROCr IPC mode the ranks actually run (init-time log line, no effect on training).

RCCL maps intra-node peer buffers with hipIpcGetMemHandle / hipIpcOpenMemHandle while
the communicators connect (all of it inside comm init on ROCm). ROCr implements that
IPC in one of two ways:

* default (DMA-BUF): the exporting process serves its buffers' dmabuf fds over an
  abstract UNIX socket named ``hsa<pid>`` (padded with NULs). It is created at the
  process's first IPC export and is listed in ``/proc/net/unix`` as ``@hsa<pid>@@@...``.
  The container runs with ``--network=host``, so a node's sockets share one list.
* legacy (``HSA_ENABLE_IPC_MODE_LEGACY=1``): KFD export/import ioctls; no socket.

``log_rocr_ipc_mode`` reads that list once, after the process group is initialised
(before RUN_START, outside the timed region), and logs one line per node (local rank 0)
with the env value, whether this process owns such a socket, and how many the node has.
It proves which mode a run used (the env alone does not: the launcher must forward it
into the container and ROCr must accept it). Never raises.
"""

import logging
import os
import re
import socket
from typing import Dict, Iterable, Optional

logger: logging.Logger = logging.getLogger(__name__)

_SOCK_RE = re.compile(r"^@hsa(\d+)(?:@|$)")


def _socket_pids(lines: Iterable[str]) -> set:
    """PIDs that own a ROCr DMA-BUF IPC socket (@hsa<pid>) in a /proc/net/unix dump."""
    pids = set()
    for line in lines:
        parts = line.split()
        if len(parts) < 8:  # header or an unnamed socket (no Path column)
            continue
        m = _SOCK_RE.match(parts[-1])
        if m:
            pids.add(int(m.group(1)))
    return pids


def detect_rocr_ipc_mode(
    pid: Optional[int] = None,
    net_unix: str = "/proc/net/unix",
    environ: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Returns {env, own_socket, node_sockets, mode, why}.

    mode: "legacy" (env "1" and no own socket), "dmabuf" (own socket present),
    "unknown" (no own socket and env not "1": this process exported nothing yet),
    "mismatch" (env "1" but an own socket exists: the switch did not take effect),
    "unreadable" (the list could not be read).
    """
    env_map = os.environ if environ is None else environ
    pid = os.getpid() if pid is None else pid
    env = env_map.get("HSA_ENABLE_IPC_MODE_LEGACY")
    out: Dict[str, object] = {"env": env, "own_socket": None, "node_sockets": None}
    try:
        with open(net_unix, "r", errors="replace") as f:
            pids = _socket_pids(f)
    except OSError as e:
        out.update(mode="unreadable", why=f"{net_unix}: {e.__class__.__name__}")
        return out
    own = pid in pids
    out.update(own_socket=own, node_sockets=len(pids))
    legacy_env = (env or "").strip() == "1"
    if own and legacy_env:
        out.update(mode="mismatch", why="HSA_ENABLE_IPC_MODE_LEGACY=1 but this process serves a DMA-BUF IPC socket")
    elif own:
        out.update(mode="dmabuf", why="this process serves a DMA-BUF IPC socket @hsa<pid>")
    elif legacy_env:
        out.update(mode="legacy", why="HSA_ENABLE_IPC_MODE_LEGACY=1 and no DMA-BUF IPC socket after comm init")
    else:
        out.update(mode="unknown", why="no IPC export by this process yet (no intra-node peer?)")
    return out


def log_rocr_ipc_mode(rank: int, local_rank: int, world_size: int) -> Optional[Dict[str, object]]:
    """One line per node (local rank 0); a mismatch is logged by every rank. Never raises."""
    try:
        info = detect_rocr_ipc_mode()
        msg = (
            f"[ipc] ROCr IPC mode={info['mode']} env HSA_ENABLE_IPC_MODE_LEGACY="
            f"{info['env'] if info['env'] is not None else '<unset>'} own_socket={info['own_socket']} "
            f"node_sockets={info['node_sockets']} ({info.get('why', '')}) host={socket.gethostname()} "
            f"rank={rank} local_rank={local_rank} world_size={world_size}"
        )
        if info["mode"] == "mismatch":
            logger.warning("NOT armed: " + msg)
        elif local_rank == 0:
            if info["mode"] == "legacy":
                logger.warning("armed OK: " + msg)
            else:
                logger.warning(msg)
        return info
    except Exception as e:  # pragma: no cover - diagnostics must never break init
        logger.warning(f"[ipc] ROCr IPC mode check failed: {e.__class__.__name__}: {e}")
        return None
