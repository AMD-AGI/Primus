###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tier 1 -- B. NIC / RDMA roll-call (per-port state + GIDs from sysfs).

Selector chain
==============

Many clusters expose more RDMA-capable ports in ``/sys/class/infiniband/``
than the training job will actually use. Common reasons:

* a separate front-end / management RoCE NIC that is physically present
  but admin-disabled (no SFP, BIOS port-disable, netdev down) -- typical
  ``state=DOWN phys_state=Disabled`` ports;
* a storage / control-plane RoCE NIC that is fully up but is reserved
  for sockets (it shows up in ``NCCL_SOCKET_IFNAME`` rather than
  ``NCCL_IB_HCA``).

Failing the node on those ports is wrong: the operator already told
NCCL/RCCL not to use them, and the training job will run fine. So the
collector resolves a *training-NIC selector* and only enforces the hard
rules (state must be ACTIVE, phys_state must be LinkUp, RoCE v2 GID
present, ...) on the included subset. Excluded ports stay in
``ports`` for diagnostics and are summarised in ``info_issues``.

Selector precedence (highest first):

1. ``allowlist`` argument -- typically wired to ``--rdma-nic-allowlist``
   on the CLI. NCCL syntax (see :func:`_parse_nic_selector` below).
2. ``NCCL_IB_HCA`` env -- mirrors what NCCL/RCCL itself will use, so the
   smoke test and the training launch agree by construction.
3. ``phys_state in {Disabled, Sleep}`` heuristic -- ports in those
   administratively-down phys_states are intentionally not in use
   (admin-disabled at firmware/driver level, no SFP, BIOS-disabled, ...)
   and are auto-excluded. Crucially, real failure modes on a port that
   *is* intended to be used produce a different phys_state
   (``Polling``, ``LinkErrorRecovery``, ``LinkUp`` with ``state!=ACTIVE``,
   ...), so this heuristic does not mask cable / driver problems.
4. Fallback: every IB port must be ACTIVE / LinkUp (the strict rule).

Empty-set guard: if the resolved included set is empty (every port was
excluded by some combination of the above) but the host has at least
one IB port at all, that's still a node FAIL -- a node with zero
training NICs cannot participate in inter-node training.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

from ..shell_utils import _read_text

# Phys-states we treat as "administratively off, not a failure".
# See module docstring for why this set is conservative.
_ADMIN_DOWN_PHYS_STATES: Set[str] = {"disabled", "sleep"}


def _parse_nic_selector(raw: str) -> Dict[str, Any]:
    """Parse an ``NCCL_IB_HCA``-style selector string.

    Accepts the full NCCL syntax:

    * Comma-separated entries of the form ``device[:port]``. ``port``
      is an integer; if omitted, the entry matches any port on the
      device.
    * Optional ``^`` prefix on the *whole* string makes it a denylist
      (every device matches except the listed ones).
    * Optional ``=`` prefix on an individual entry forces an exact
      device-name match (e.g. ``=mlx5`` matches device ``mlx5`` only,
      not ``mlx5_0``). Without ``=``, an entry is a *prefix* match,
      matching ``mlx5`` against ``mlx5_0`` / ``mlx5_1`` / ...

    Returns a dict with:

    * ``mode``: ``"allowlist"`` or ``"denylist"``.
    * ``entries``: list of ``(device_pattern, exact_match, port_or_None)``.
    * ``raw``: the original input, after stripping the global ``^``.

    Returns ``{"mode": "passthrough", ...}`` for empty / whitespace-only
    input, signalling the caller to fall through to the next selector.
    """
    s = (raw or "").strip()
    if not s:
        return {"mode": "passthrough", "entries": [], "raw": raw or ""}

    mode = "allowlist"
    if s.startswith("^"):
        mode = "denylist"
        s = s[1:].strip()

    entries: List[Tuple[str, bool, Optional[int]]] = []
    for token in s.split(","):
        t = token.strip()
        if not t:
            continue
        exact = False
        if t.startswith("="):
            exact = True
            t = t[1:].strip()
            if not t:
                continue
        # Split off an optional :port suffix. We split from the right so
        # device names containing ':' (none in the wild today, but we
        # don't want to be the thing that breaks if they appear) are
        # preserved.
        port: Optional[int] = None
        if ":" in t:
            dev_part, _, port_str = t.rpartition(":")
            try:
                port = int(port_str)
                t = dev_part
            except ValueError:
                # Not actually a port suffix -- treat the whole thing
                # as the device name.
                port = None
        entries.append((t, exact, port))

    if not entries:
        return {"mode": "passthrough", "entries": [], "raw": raw or ""}
    return {"mode": mode, "entries": entries, "raw": raw or ""}


def _selector_matches(
    sel: Dict[str, Any],
    device: str,
    port: int,
) -> bool:
    """Return True iff ``(device, port)`` matches an allow/denylist selector
    parsed by :func:`_parse_nic_selector`.

    ``passthrough`` selectors trivially match everything (the caller
    should not normally pass them here; instead it should treat them as
    "no selector at all" and fall through to the next layer).
    """
    mode = sel.get("mode", "passthrough")
    if mode == "passthrough":
        return True
    entries = sel.get("entries") or []
    listed = False
    for dev_pat, exact, want_port in entries:
        # Device name match: exact iff the entry started with '='; else
        # prefix match (NCCL semantics).
        if exact:
            dev_ok = device == dev_pat
        else:
            dev_ok = device.startswith(dev_pat)
        if not dev_ok:
            continue
        # Port filter: an entry with no :port matches any port on that
        # device.
        if want_port is not None and want_port != port:
            continue
        listed = True
        break
    if mode == "allowlist":
        return listed
    # Denylist: ports NOT in the list are accepted.
    return not listed


def _resolve_selector(
    allowlist_arg: Optional[str],
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Resolve the training-NIC selector for this node.

    Precedence:

    1. ``allowlist_arg`` (typically the value of ``--rdma-nic-allowlist``).
    2. ``NCCL_IB_HCA`` env var.
    3. Heuristic fallback (``"heuristic"``) -- the caller handles this
       by auto-excluding ports whose ``phys_state`` is in
       :data:`_ADMIN_DOWN_PHYS_STATES`.

    Returns a dict with ``source`` (``"cli"`` / ``"env"`` / ``"heuristic"``)
    plus the parsed-selector fields when applicable.
    """
    if env is None:
        env = dict(os.environ)
    # 1. CLI / explicit arg.
    if allowlist_arg:
        parsed = _parse_nic_selector(allowlist_arg)
        if parsed.get("mode") != "passthrough":
            parsed["source"] = "cli"
            return parsed
    # 2. NCCL_IB_HCA env.
    env_raw = env.get("NCCL_IB_HCA") or ""
    if env_raw.strip():
        parsed = _parse_nic_selector(env_raw)
        if parsed.get("mode") != "passthrough":
            parsed["source"] = "env"
            return parsed
    # 3. Heuristic fallback.
    return {
        "source": "heuristic",
        "mode": "heuristic",
        "entries": [],
        "raw": "",
        "admin_down_phys_states": sorted(_ADMIN_DOWN_PHYS_STATES),
    }


def _collect_nic_status(
    expected_count: Optional[int],
    *,
    allowlist: Optional[str] = None,
) -> Dict[str, Any]:
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

    Per-port hard issues are only emitted for ports in the resolved
    *training-NIC selector* (see module docstring). Excluded ports stay
    in ``ports`` for diagnostics and are summarised in ``info_issues``.

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
    selector = _resolve_selector(allowlist)
    out: Dict[str, Any] = {
        "expected_count": expected_count,
        "selector": selector,
        "ports": [],
        "included_ports": [],
        "excluded_ports": [],
        "issues": [],
        "info_issues": [],
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

            port_rec = {
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
            out["ports"].append(port_rec)

            # ------------------------------------------------------------
            # Selector: is this port part of the training-NIC set?
            # ------------------------------------------------------------
            label = f"{dev}:{port}"
            include = True
            exclude_reason = ""
            if selector.get("source") == "heuristic":
                # Auto-exclude admin-down phys_states; keep everything
                # else and let the strict per-port rules run.
                if (phys or "").lower() in _ADMIN_DOWN_PHYS_STATES:
                    include = False
                    exclude_reason = (
                        f"phys_state={phys} (admin-disabled, not used for training)"
                    )
            else:
                # Explicit allow/denylist (CLI or env).
                if not _selector_matches(selector, dev, port):
                    include = False
                    src = selector.get("source") or "selector"
                    src_label = (
                        "NCCL_IB_HCA" if src == "env" else "--rdma-nic-allowlist" if src == "cli" else src
                    )
                    extra = ""
                    if (state or "").upper() != "ACTIVE" or (phys or "").lower() != "linkup":
                        extra = f" (state={state} phys_state={phys})"
                    exclude_reason = f"excluded by {src_label}{extra}"

            if not include:
                out["excluded_ports"].append(label)
                out["info_issues"].append(f"{label} {exclude_reason}")
                continue
            out["included_ports"].append(label)

            # ------------------------------------------------------------
            # Per-port hard issues -> node FAIL. Only for included ports.
            # ------------------------------------------------------------
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

    # The expected-count check operates on the *included* set: operators
    # almost always set --expected-rdma-nics to mean "training NIC count"
    # (e.g. 8 = 1 NIC per GPU), not "everything visible under
    # /sys/class/infiniband/" -- which on multi-role nodes includes
    # frontend / management / storage RoCE NICs.
    if expected_count is not None and len(out["included_ports"]) != expected_count:
        out["issues"].append(
            f"RDMA NIC port count {len(out['included_ports'])} != expected {expected_count}"
        )

    # Empty-set guard: if we saw IB ports on this host but every single
    # one got excluded, the node cannot participate in inter-node
    # training. This catches "the whole RoCE card disabled itself" and
    # similar disasters that would otherwise PASS silently because every
    # port met the exclusion criteria.
    if out["ports"] and not out["included_ports"]:
        out["issues"].append(
            "no included RDMA NIC ports remain after selector "
            f"({selector.get('source')}); node cannot participate in inter-node training"
        )

    return out
