# Copyright (c) Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe
"""RCCL settings that corrupt data on this fabric, corrected before the process group exists.

The 4-node MI355X crash (runner plans/dlrmv4 09; 2026-10-11/12, cluster B, 8 AMD
Pensando Pollara 400 "ionic" NICs per node, RCCL's built-in NET/IB transport):
- About one 4-node run in three died inside an eval pass. Before
  TORCHREC_VALIDATE_COLLECTIVES=1 it showed as a garbage stride_per_rank and
  a ``_get_recat`` int32 overflow; with it, as a collective-tag mismatch. The
  other nodes then hung in a collective.
- The collective diagnostics (host_sync_free.py on dlrmv4_opt-mi355x,
  2e9ca72 and c810aba) caught one: a single
  144-byte peer-to-peer chunk of the splits all_to_all, from an inter-node
  peer, arrived holding fp16 values within +-3.2e-4. That is the previous
  transfer on the same communicator, the sequence-embedding all_to_all.
- Every rank had issued the identical sequence of collectives: corrupted
  bytes, not a desync.
- The cluster recipe set ``NCCL_GDR_FLUSH_DISABLE=1`` (RCCL's default is 0).
  The GDR flush makes the NIC's writes into GPU memory visible before the
  receiving kernel reads them; without it the receiver can read a buffer
  that still holds the previous transfer.
- With the flush on, 2 x 45 minutes of evals on the same four nodes, seed
  and code had no mismatch (the off arm failed at about the 40th eval), and
  steps and evals were no slower (run_start to first eval 425.4-425.9 s
  against 424.5-426.6 s).

That recipe (ROCm/cvs ainic_env_script.sh) is written for the AMD ANP net
plugin (librccl-anp.so), not for RCCL's built-in IB transport, which is what
these runs use. So: ionic devices present, no ANP plugin, and
``NCCL_GDR_FLUSH_DISABLE=1`` -> set it to 0 and say so.
``DLRM_ALLOW_GDR_FLUSH_DISABLE=1`` keeps the caller's value (A/B arms only).

Also warned (no value is changed, so what runs is what was measured): names
RCCL does not read. RCCL reads ``NCCL_NET_OPTIONAL_RECV_COMPLETION``
(``NET_OPTIONAL_RECV_COMPLETION`` has no effect) and
``RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING`` (the ``NCCL_`` spelling has
none). Names checked against ROCm/rccl src/transport/net_ib*.cc and
NCCL_PARAM's ``NCCL_`` prefix.

No torch import, so it is cheap to call and to test.
"""

import logging
import os
from typing import Dict, List, MutableMapping

logger: logging.Logger = logging.getLogger(__name__)

IB_ROOT = "/sys/class/infiniband"
ALLOW_ENV = "DLRM_ALLOW_GDR_FLUSH_DISABLE"

# (name with no effect, name RCCL reads)
MISNAMED = (
    ("NET_OPTIONAL_RECV_COMPLETION", "NCCL_NET_OPTIONAL_RECV_COMPLETION"),
    ("NCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING", "RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING"),
)


def ionic_devices(ib_root: str = IB_ROOT) -> List[str]:
    try:
        return sorted(d for d in os.listdir(ib_root) if d.startswith("ionic"))
    except OSError:
        return []


def uses_anp_plugin(environ: Dict[str, str]) -> bool:
    plugin = os.path.basename(environ.get("NCCL_NET_PLUGIN", "") or "").lower()
    return "anp" in plugin


def check_rccl_env(
    environ: MutableMapping[str, str] = os.environ,  # pyre-ignore [9]
    ib_root: str = IB_ROOT,
) -> List[str]:
    """Correct and report the settings above. Returns one line per finding."""
    notes: List[str] = []
    ionic = ionic_devices(ib_root)
    if ionic and not uses_anp_plugin(environ) and environ.get("NCCL_GDR_FLUSH_DISABLE") == "1":
        if environ.get(ALLOW_ENV) == "1":
            notes.append(
                f"NCCL_GDR_FLUSH_DISABLE=1 kept ({ALLOW_ENV}=1) on {len(ionic)} ionic NICs with "
                "RCCL's built-in IB transport: inter-node data can arrive stale (plans/dlrmv4 09)"
            )
        else:
            environ["NCCL_GDR_FLUSH_DISABLE"] = "0"
            notes.append(
                f"NCCL_GDR_FLUSH_DISABLE=1 -> 0: {len(ionic)} ionic NICs with RCCL's built-in IB "
                "transport (no ANP plugin); without the GDR flush a receiver can read the previous "
                f"transfer's bytes (the 4-node eval crash, plans/dlrmv4 09). {ALLOW_ENV}=1 keeps 1."
            )
    for wrong, right in MISNAMED:
        if wrong in environ and right not in environ:
            notes.append(f"{wrong}={environ[wrong]} has no effect: RCCL reads {right}")
    return notes


def apply(rank: int = 0) -> List[str]:
    """Run the check on this process's environment; local rank 0 logs it."""
    notes = check_rccl_env()
    if notes and rank % 8 == 0:
        for n in notes:
            logger.warning("fabric_guard: %s", n)
    return notes
