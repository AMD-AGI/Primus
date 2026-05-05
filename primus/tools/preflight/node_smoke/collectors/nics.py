###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tier 1 -- B. NIC / RDMA roll-call (per-port state + GIDs from sysfs)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..shell_utils import _read_text


def _collect_nic_status(expected_count: Optional[int]) -> Dict[str, Any]:
    """Inventory every RDMA port on this node and flag the ones that would
    silently break inter-node training.

    Reads everything from ``/sys/class/infiniband`` (kernel ``ib_core``
    ABI) so the check is **vendor- and fabric-agnostic** -- works on
    Mellanox/NVIDIA (``mlx5_ib``), Broadcom (``bnxt_re``), Intel
    (``irdma``), Marvell (``qedr``), AWS EFA (``efa``), Huawei
    (``hns_roce``), etc., and on either RoCE-over-Ethernet or true
    InfiniBand fabrics. We do not depend on ``ibv_devinfo`` /
    ``ibstat`` / vendor SDKs being present in the container.

    Per port we capture:

    * ``link_layer`` (``Ethernet`` for RoCE, ``InfiniBand`` for IB) --
      determines which GID rule applies below;
    * link state (``state``: ``ACTIVE``/``DOWN``/``INIT``) and physical
      state (``phys_state``: ``LinkUp``/``Polling``/...);
    * link rate (Gb/s);
    * netdev + MTU (so the aggregator can detect MTU drift, which silently
      tanks RoCE all-reduce throughput);
    * GID counts -- total non-zero GIDs and the subset configured as
      ``RoCE v2`` (an empty RoCE v2 set is a frequent cause of training
      jobs hanging at the first inter-node collective on RoCE clusters).

    Issues are pushed into ``out["issues"]`` (each a short string). Hard
    issues (port not Active / missing GIDs / wrong NIC count) are
    treated as node FAIL by ``_node_status_from``. The GID check is
    fabric-aware:

    * RoCE/Ethernet port must have at least one ``RoCE v2`` GID
      configured (RoCEv1 is essentially unused for AI training).
    * InfiniBand port must have at least one valid (non-zero) GID --
      it is normally auto-populated by the SM; an empty GID table on
      an ACTIVE IB port indicates a subnet-manager problem.
    * Unknown ``link_layer`` (very old kernels) falls back to "must
      have any valid GID" so we don't false-FAIL exotic configurations.
    """
    out: Dict[str, Any] = {
        "expected_count": expected_count,
        "ports": [],
        "issues": [],
    }
    base = "/sys/class/infiniband"
    if not os.path.isdir(base):
        # Container may not expose the IB stack; report and let the operator
        # decide. We only mark this as a hard issue when the user explicitly
        # asked for a positive expected_count.
        msg = f"{base} missing -- no RDMA stack visible"
        if expected_count and expected_count > 0:
            out["issues"].append(msg)
        else:
            out["info"] = msg
        return out

    try:
        devs = sorted(os.listdir(base))
    except Exception as e:
        out["issues"].append(f"failed to list {base}: {e}")
        return out

    for dev in devs:
        port_dir = os.path.join(base, dev, "ports")
        if not os.path.isdir(port_dir):
            continue
        try:
            ports = sorted(os.listdir(port_dir))
        except Exception:
            continue
        for port_str in ports:
            try:
                port = int(port_str)
            except ValueError:
                continue
            p = os.path.join(port_dir, port_str)

            # Sysfs values look like "4: ACTIVE" / "5: LinkUp" / "400 Gb/sec (4X NDR)"
            state_raw = _read_text(os.path.join(p, "state"))
            phys_raw = _read_text(os.path.join(p, "phys_state"))
            rate_raw = _read_text(os.path.join(p, "rate"))
            state = state_raw.split(":", 1)[-1].strip() if state_raw else ""
            phys = phys_raw.split(":", 1)[-1].strip() if phys_raw else ""
            rate_gbps: Optional[int] = None
            try:
                rate_gbps = int(rate_raw.split()[0])
            except Exception:
                pass

            # Fabric type: "Ethernet" -> RoCE, "InfiniBand" -> IB.
            # Determines which GID rule to apply below. Provided by
            # ib_core for every RDMA driver since Linux 3.x; missing
            # only on extremely old kernels.
            link_layer = (_read_text(os.path.join(p, "link_layer")) or "").strip() or None

            # GID inventory. A GID is "all-zero" until configured.
            gid_count = 0
            rocev2_count = 0
            gids_dir = os.path.join(p, "gids")
            types_dir = os.path.join(p, "gid_attrs", "types")
            valid_gid_indices: List[int] = []
            if os.path.isdir(gids_dir):
                try:
                    for gn in sorted(os.listdir(gids_dir), key=lambda s: int(s) if s.isdigit() else 0):
                        if not gn.isdigit():
                            continue
                        g = _read_text(os.path.join(gids_dir, gn))
                        if g and g != "0000:0000:0000:0000:0000:0000:0000:0000":
                            gid_count += 1
                            valid_gid_indices.append(int(gn))
                except Exception:
                    pass
            if os.path.isdir(types_dir):
                for idx in valid_gid_indices:
                    t = _read_text(os.path.join(types_dir, str(idx)))
                    if "RoCE v2" in t or "RoCEv2" in t:
                        rocev2_count += 1

            # Linked netdev + MTU.
            ifname: Optional[str] = None
            mtu: Optional[int] = None
            net_dir = os.path.join(base, dev, "device", "net")
            if os.path.isdir(net_dir):
                try:
                    nets = sorted(os.listdir(net_dir))
                    if nets:
                        ifname = nets[0]
                        mtu_raw = _read_text(f"/sys/class/net/{ifname}/mtu")
                        try:
                            mtu = int(mtu_raw)
                        except Exception:
                            mtu = None
                except Exception:
                    pass

            out["ports"].append(
                {
                    "device": dev,
                    "port": port,
                    "link_layer": link_layer,
                    "state": state or None,
                    "phys_state": phys or None,
                    "rate_gbps": rate_gbps,
                    "ifname": ifname,
                    "mtu": mtu,
                    "gid_count": gid_count,
                    "rocev2_gid_count": rocev2_count,
                }
            )

            # Per-port hard issues -> node FAIL.
            if state and state.upper() != "ACTIVE":
                out["issues"].append(f"{dev}:{port} state={state} (expected ACTIVE)")
            if phys and phys.upper() != "LINKUP":
                out["issues"].append(f"{dev}:{port} phys_state={phys} (expected LinkUp)")
            # Fabric-aware GID requirement on ACTIVE ports.
            if state.upper() == "ACTIVE":
                ll = (link_layer or "").lower()
                if ll == "ethernet":
                    # RoCE: needs at least one RoCE v2 GID; RoCEv1 is
                    # essentially unused for AI training and we treat
                    # its absence as a hard fail.
                    if rocev2_count == 0:
                        out["issues"].append(
                            f"{dev}:{port} no RoCE v2 GIDs configured " f"(RoCE/Ethernet fabric)"
                        )
                elif ll == "infiniband":
                    # True IB: subnet manager normally populates GIDs.
                    # Empty GID table on an ACTIVE IB port = SM problem.
                    if gid_count == 0:
                        out["issues"].append(
                            f"{dev}:{port} no GIDs populated " f"(InfiniBand fabric -- check subnet manager)"
                        )
                else:
                    # Unknown / missing link_layer (very old kernel):
                    # require at least one valid GID rather than a
                    # specific type, to avoid false FAIL.
                    if gid_count == 0:
                        out["issues"].append(
                            f"{dev}:{port} no valid GIDs configured " f"(link_layer unknown)"
                        )

    if expected_count is not None and len(out["ports"]) != expected_count:
        out["issues"].append(f"RDMA NIC port count {len(out['ports'])} != expected {expected_count}")

    return out
