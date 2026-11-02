###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Print the MASTER_PORT a torchrun launch should use (stdlib only).

``python3 local_master_port.py <nnodes> <master_addr> <master_port>``

A single-node run (``1`` or ``1:1``) with a loopback rendezvous (an IP in
127.0.0.0/8, ``::1``, ``localhost``; any case, ``[::1]``) needs no other node to agree on the port. When that port is in use on
this host (EADDRINUSE, e.g. an ephemeral port of another process on a shared node),
torchrun would fail at once, so a port the kernel reports free is printed instead.
"In use" is checked on IPv4 and, when the host has it, IPv6 (a v6-only listener also
holds the port). The new port is released before torchrun binds it, so another process
can still take it in between (rare: the kernel hands out ephemeral ports in turn); the
check removes the common failure, it does not guarantee the bind.

The given port is printed unchanged for: several nodes (``2``, ``1:2``), a MASTER_ADDR
that is not loopback, a value that is not a port number (not an integer, or outside
1-65535: torchrun reports it), a bind error other than EADDRINUSE (e.g. EACCES: the
port is not known to be in use), an ``<nnodes>`` that is not a node count, no unused
port found (after ``_FREE_TRIES`` kernel picks, or a bind error), and
``PRIMUS_KEEP_MASTER_PORT`` set to 1 / true / yes / on (keep the port and let torchrun
fail). ``primus-cli direct`` also keeps a MASTER_PORT given with ``--env`` /
``direct.env`` and logs a warning when this helper fails.
"""

import errno
import ipaddress
import os
import socket
import sys

LOCAL_ADDRS = ("localhost", "::1")
KEEP_ENV = "PRIMUS_KEEP_MASTER_PORT"
_TRUE = ("1", "true", "yes", "on")
_FREE_TRIES = 20  # a kernel-picked IPv4 port may be in use on IPv6: try again


def is_local(addr: str) -> bool:
    """A loopback address: ``localhost`` or a loopback IP (127.0.0.0/8, ``::1``), any case,
    ``[::1]`` as written in URLs. The value is parsed as an IP, so a hostname such as
    ``127.example.com`` is not loopback."""
    a = addr.strip().lower()
    if a.startswith("[") and a.endswith("]"):
        a = a[1:-1]
    if a in LOCAL_ADDRS:
        return True
    try:
        return ipaddress.ip_address(a).is_loopback
    except ValueError:
        return False


def is_single_node(nnodes: str) -> bool:
    """``1`` or an elastic ``1:1`` (torchrun's ``MIN:MAX``, both parts 1); ValueError when
    not a node count. A range such as ``2:1`` or ``0:1`` is not single-node."""
    parts = str(nnodes).split(":")
    if len(parts) > 2:
        raise ValueError(nnodes)
    return all(int(p) == 1 for p in parts)


def keep_requested(environ=os.environ) -> bool:
    return environ.get(KEEP_ENV, "").strip().lower() in _TRUE


def _listen(family: int, port: int) -> int:
    """Bind a listener on ``port`` (0: the kernel picks) with SO_REUSEADDR, as torch's
    TCPStore does, close it and return the bound port; OSError when it cannot
    (``errno.EADDRINUSE``: in use). IPv6 is probed v6-only so each family is checked on
    its own; together the two probes cover a dual-stack bind (TCPStore's ``::``), which
    fails when either family holds the port."""
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if family == socket.AF_INET6:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        sock.bind(("::" if family == socket.AF_INET6 else "", port))
        sock.listen(1)
        return sock.getsockname()[1]
    finally:
        sock.close()


def has_ipv6() -> bool:
    """This host can listen on IPv6 (else only IPv4 can hold a port)."""
    try:
        _listen(socket.AF_INET6, 0)
        return True
    except OSError:
        return False


def is_port_in_use(port: int, ipv6=None) -> bool:
    """A bind of ``port`` fails with EADDRINUSE on IPv4 or (when the host has it) IPv6.
    Other bind errors (EACCES, EADDRNOTAVAIL, EMFILE, ...) do not say the port is in use:
    False, so the caller keeps the port and torchrun reports the real error."""
    families = [socket.AF_INET]
    if has_ipv6() if ipv6 is None else ipv6:
        families.append(socket.AF_INET6)
    for family in families:
        try:
            _listen(family, port)
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                return True
    return False


def find_unused_port(ipv6=None):
    """A port not in use on any family the host has (picked by the kernel on IPv4), else None."""
    ipv6 = has_ipv6() if ipv6 is None else ipv6
    for _ in range(_FREE_TRIES):
        try:
            port = _listen(socket.AF_INET, 0)
        except OSError:
            return None
        if not is_port_in_use(port, ipv6):
            return port
    return None


def choose_port(nnodes: str, addr: str, port: str, environ=os.environ) -> str:
    try:
        single = is_single_node(nnodes)
        wanted = int(port)
    except ValueError:
        return port
    if not 1 <= wanted <= 65535 or not single or not is_local(addr):
        return port
    if keep_requested(environ):
        return port
    ipv6 = has_ipv6()  # probed once for both checks
    if not is_port_in_use(wanted, ipv6):
        return port
    unused = find_unused_port(ipv6)
    return str(unused) if unused else port


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: local_master_port.py <nnodes> <master_addr> <master_port>")
    print(choose_port(*sys.argv[1:]))
