#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Make the RDMA userspace provider (libibverbs driver) match the host NIC driver.

Training images ship a fixed libibverbs provider per NIC family: libbnxt_re for
Broadcom and libionic for AMD Pensando AINIC. The kernel half of the driver
lives on the host, so when the host runs a different release the provider in
the image can reject every device ("does not support the kernel ABI") and RCCL
quietly falls back to TCP sockets.

runner/helpers/hooks/02_setup_nic_userspace_driver.sh runs this before every
launch. For each NIC family found under /sys/class/infiniband it:

  1. probes libibverbs: if the current provider enumerates and opens every
     device, auto mode stops here;
  2. reads the host driver version from sysfs (bnxt_re module version, ionic
     fw_ver, which is the AINIC bundle version);
  3. looks for a provider of that version: PATH_TO_BNXT_TAR_PACKAGE, vendor
     bundles on PRIMUS_NIC_DRIVER_SEARCH_PATH (plain or inside archives), the
     host's own libbnxt_re (mounted by primus-cli-container.sh), and for AINIC
     the public APT repository;
  4. builds it (libbnxt_re source tarball) or unpacks it (.deb), installs it,
     re-probes, and rolls back anything that does not work.

System paths are only modified as root inside a container (where they are
ephemeral). Everywhere else, including bare-metal hosts, the provider goes into
a private prefix and is selected with RDMAV_DRIVERS / LD_LIBRARY_PATH, handed
back to the launcher as env.* lines (hook protocol).

Usage:
  nic_userspace_driver.py setup     # what the hook runs
  nic_userspace_driver.py status    # report only, change nothing
"""

from __future__ import annotations

import argparse
import errno
import gzip
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

SYSFS = Path("/sys")
DEV_INFINIBAND = Path("/dev/infiniband")
OS_RELEASE = Path("/etc/os-release")
# Keep in sync with the default in runner/primus-cli-container.sh.
DEFAULT_SEARCH_PATH = "/opt/broadcom:/opt/bnxt-bundles:/opt/amd/ainic"
# primus-cli-container.sh mounts the host's libbnxt_re here as bnxt_re/<version>/<file>.
HOST_PROVIDER_ROOT = Path("/run/primus/host-rdma-providers")
DEFAULT_AINIC_REPO_URL = "https://repo.radeon.com/amdainic/pensando/ubuntu"
SYSTEM_BACKUP_ROOT = Path("/var/tmp/primus-rdma-providers")

DOWNLOAD_TIMEOUT_S = 30
MAX_ATTEMPTS = 4
MAX_SCAN_FILES = 50000
MAX_SCAN_DEPTH = 8
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2", ".tbz2", ".tar", ".zip")
KIND_ORDER = {"source": 0, "deb": 1, "prebuilt": 2, "download": 3}


class SetupError(Exception):
    """A provider could not be found, staged, installed or verified."""


def _log(level: str, msg: str) -> None:
    stamp = (
        time.strftime("[%Y-%m-%d %H:%M:%S] ") if os.environ.get("PRIMUS_LOG_TIMESTAMP", "1") == "1" else ""
    )
    node = f"[NODE-{os.environ.get('NODE_RANK', '?')}({socket.gethostname()})]"
    stream = sys.stderr if level in ("WARN", "ERROR") else sys.stdout
    print(f"{stamp}{node} [{level}] [nic-driver] {msg}", file=stream, flush=True)


def info(msg: str) -> None:
    _log("INFO", msg)


def warn(msg: str) -> None:
    _log("WARN", msg)


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    mode: str = "auto"  # auto | force | off
    strict: bool = False
    search_path: List[Path] = field(default_factory=list)
    bnxt_package: Optional[Path] = None
    forced_providers: frozenset = frozenset()
    ainic_repo_url: str = DEFAULT_AINIC_REPO_URL
    version_overrides: Dict[str, str] = field(default_factory=dict)  # provider -> version to install

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "Settings":
        env = os.environ if env is None else env
        raw = env.get("PRIMUS_NIC_USERSPACE_DRIVER", "auto").strip().lower()
        if raw in ("off", "0", "false", "no"):
            mode = "off"
        elif raw == "force":
            mode = "force"
        else:
            if raw not in ("", "auto", "1", "on", "true", "yes"):
                warn(f"unknown PRIMUS_NIC_USERSPACE_DRIVER={raw!r}; using auto (choices: auto, force, off)")
            mode = "auto"
        search = env.get("PRIMUS_NIC_DRIVER_SEARCH_PATH", DEFAULT_SEARCH_PATH)
        package = env.get("PATH_TO_BNXT_TAR_PACKAGE", "").strip()
        repo = env.get("PRIMUS_AINIC_REPO_URL", DEFAULT_AINIC_REPO_URL).strip().rstrip("/")
        # For hosts whose ionic fw_ver is not the AINIC bundle version.
        ainic_bundle = env.get("PRIMUS_AINIC_BUNDLE_VERSION", "").strip()
        return cls(
            mode=mode,
            strict=env.get("PRIMUS_NIC_DRIVER_STRICT", "0").strip().lower() in ("1", "true", "yes", "on"),
            search_path=[Path(p) for p in search.split(":") if p.strip()],
            bnxt_package=Path(package) if package else None,
            # Legacy switch from the old 02_rebuild_bnxt.sh hook.
            forced_providers=(
                frozenset({"bnxt_re"}) if env.get("REBUILD_BNXT", "0").strip() == "1" else frozenset()
            ),
            ainic_repo_url="" if repo.lower() in ("none", "off", "0") else repo,
            version_overrides={"ionic": ainic_bundle} if ainic_bundle else {},
        )


# ---------------------------------------------------------------------------
# Host / environment inspection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RdmaDevice:
    name: str
    vendor: str
    fw_ver: str
    uverbs: str = ""  # character device name under /dev/infiniband, e.g. "uverbs3"


def list_rdma_devices(sysfs: Path = SYSFS) -> List[RdmaDevice]:
    try:
        entries = sorted((sysfs / "class" / "infiniband").iterdir())
    except OSError:
        return []
    uverbs = {}
    try:
        for entry in (sysfs / "class" / "infiniband_verbs").iterdir():
            if entry.name.startswith("uverbs"):
                uverbs[_read(entry / "ibdev")] = entry.name
    except OSError:
        pass
    return [
        RdmaDevice(
            e.name, _read(e / "device" / "vendor").lower(), _read(e / "fw_ver"), uverbs.get(e.name, "")
        )
        for e in entries
    ]


def device_accessible(device: RdmaDevice) -> bool:
    # libibverbs skips devices whose character device is absent, so only these can be judged.
    return bool(device.uverbs) and os.access(DEV_INFINIBAND / device.uverbs, os.R_OK | os.W_OK)


def running_in_container() -> bool:
    # Only positive markers: a false positive would let the system installer modify a bare-metal host.
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return True
    if any(
        os.environ.get(k)
        for k in ("container", "KUBERNETES_SERVICE_HOST", "APPTAINER_CONTAINER", "SINGULARITY_CONTAINER")
    ):
        return True
    return bool(re.search(r"docker|kubepods|containerd|libpod|lxc", _read(Path("/proc/1/cgroup"))))


def os_codename(os_release: Path = OS_RELEASE) -> str:
    values = {}
    for line in _read(os_release).splitlines():
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"')
    return values.get("VERSION_CODENAME") or values.get("UBUNTU_CODENAME", "")


def deb_architecture() -> str:
    if shutil.which("dpkg"):
        arch = subprocess.run(["dpkg", "--print-architecture"], capture_output=True, text=True).stdout.strip()
        if arch:
            return arch
    machine = platform.machine()
    return {"x86_64": "amd64", "aarch64": "arm64"}.get(machine, machine)


# ---------------------------------------------------------------------------
# libibverbs probe
# ---------------------------------------------------------------------------

# Runs in a fresh interpreter so each probe sees a clean libibverbs init and the
# environment under test (RDMAV_DRIVERS, LD_LIBRARY_PATH, LD_DEBUG). Creating a
# PD, CQ and RC QP exercises the provider-specific kernel ABI beyond the version
# check done when listing devices.
_PROBE_SCRIPT = r"""
import ctypes, json, os, sys
from ctypes import POINTER, byref, c_char_p, c_int, c_uint32, c_void_p
out = {"enumerated": [], "opened": [], "qp_failures": {}, "sonames": {}, "libibverbs": None, "error": None}

def mapped(stem):
    with open("/proc/self/maps") as maps:
        for line in maps:
            path = line.split()[-1]
            if os.path.basename(path).startswith(stem):
                return path
    return None

# Libraries applications link directly (libionic.so.1) are resolved before any provider loads.
for soname in filter(None, os.environ.get("PRIMUS_NIC_PROBE_SONAMES", "").split(":")):
    try:
        ctypes.CDLL(soname)
        out["sonames"][soname] = mapped(soname.split(".so")[0] + ".so")
    except OSError:
        out["sonames"][soname] = None
try:
    lib = ctypes.CDLL("libibverbs.so.1", use_errno=True)
except OSError as exc:
    out["error"] = "cannot load libibverbs.so.1: %s" % exc
    print(json.dumps(out))
    sys.exit(0)
out["libibverbs"] = mapped("libibverbs.so")

class QpCap(ctypes.Structure):
    _fields_ = [(n, c_uint32) for n in ("max_send_wr", "max_recv_wr", "max_send_sge", "max_recv_sge", "max_inline_data")]

class QpInitAttr(ctypes.Structure):
    _fields_ = [("qp_context", c_void_p), ("send_cq", c_void_p), ("recv_cq", c_void_p), ("srq", c_void_p),
                ("cap", QpCap), ("qp_type", c_int), ("sq_sig_all", c_int)]

for name, restype, argtypes in (
    ("ibv_get_device_list", POINTER(c_void_p), [POINTER(c_int)]),
    ("ibv_free_device_list", None, [POINTER(c_void_p)]),
    ("ibv_get_device_name", c_char_p, [c_void_p]),
    ("ibv_open_device", c_void_p, [c_void_p]),
    ("ibv_close_device", c_int, [c_void_p]),
    ("ibv_alloc_pd", c_void_p, [c_void_p]),
    ("ibv_dealloc_pd", c_int, [c_void_p]),
    ("ibv_create_cq", c_void_p, [c_void_p, c_int, c_void_p, c_void_p, c_int]),
    ("ibv_destroy_cq", c_int, [c_void_p]),
    ("ibv_create_qp", c_void_p, [c_void_p, POINTER(QpInitAttr)]),
    ("ibv_destroy_qp", c_int, [c_void_p]),
):
    getattr(lib, name).restype = restype
    getattr(lib, name).argtypes = argtypes

def call(name, *args):
    ctypes.set_errno(0)
    return getattr(lib, name)(*args), ctypes.get_errno()

def exercise(ctx):
    pd, err = call("ibv_alloc_pd", ctx)
    if not pd:
        return ["ibv_alloc_pd", err]
    cq, err = call("ibv_create_cq", ctx, 16, None, None, 0)
    failure = None if cq else ["ibv_create_cq", err]
    if cq:
        attr = QpInitAttr(send_cq=cq, recv_cq=cq, cap=QpCap(16, 16, 1, 1, 0), qp_type=2)  # IBV_QPT_RC
        qp, err = call("ibv_create_qp", pd, byref(attr))
        failure = None if qp else ["ibv_create_qp", err]
        if qp:
            lib.ibv_destroy_qp(qp)
        lib.ibv_destroy_cq(cq)
    lib.ibv_dealloc_pd(pd)
    return failure

num = c_int(0)
devs = lib.ibv_get_device_list(byref(num))
if devs:
    for i in range(num.value):
        name = lib.ibv_get_device_name(devs[i]).decode()
        out["enumerated"].append(name)
        ctx = lib.ibv_open_device(devs[i])
        if ctx:
            out["opened"].append(name)
            failure = exercise(ctx)
            if failure:
                out["qp_failures"][name] = failure
            lib.ibv_close_device(ctx)
    lib.ibv_free_device_list(devs)
print(json.dumps(out))
"""
# Failures caused by limits (RLIMIT_MEMLOCK, permissions), not by the provider.
RESOURCE_ERRNOS = {errno.EPERM, errno.EACCES, errno.EAGAIN, errno.ENOMEM, errno.EMFILE}

_ABI_RE = re.compile(
    r"Driver (\S+) does not support the kernel ABI of \d+ \(supports \d+ to \d+\) for device (\S+)"
)
_INIT_RE = re.compile(r"calling init: (\S+)")


@dataclass
class ProbeResult:
    enumerated: List[str] = field(default_factory=list)
    opened: List[str] = field(default_factory=list)
    qp_failures: Dict[str, List] = field(default_factory=dict)  # device -> [verbs call, errno]
    sonames: Dict[str, Optional[str]] = field(default_factory=dict)  # soname -> file the linker chose
    abi_mismatch: Dict[str, str] = field(default_factory=dict)  # device -> libibverbs warning
    loaded: List[str] = field(default_factory=list)  # objects whose initializer ran (LD_DEBUG=libs)
    libibverbs: Optional[str] = None
    error: Optional[str] = None

    def provider_failures(self, names: Sequence[str]) -> Dict[str, str]:
        return {
            n: f"{call} fails on {n}: {os.strerror(err) if err else 'no error code'}"
            for n, (call, err) in self.qp_failures.items()
            if n in names and err not in RESOURCE_ERRNOS
        }

    def limit_failures(self, names: Sequence[str]) -> Dict[str, str]:
        return {
            n: f"{call} fails on {n}: {os.strerror(err)}"
            for n, (call, err) in self.qp_failures.items()
            if n in names and err in RESOURCE_ERRNOS
        }


def run_probe(extra_env: Optional[Dict[str, str]] = None, trace: bool = False) -> ProbeResult:
    env = dict(os.environ)
    env.update(extra_env or {})
    env.pop("LD_DEBUG", None)
    if trace:
        env["LD_DEBUG"] = "libs"
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _PROBE_SCRIPT], env=env, capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ProbeResult(error=f"libibverbs probe did not run: {exc}")
    data = None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            try:
                data = json.loads(line)
                break
            except ValueError:
                pass
    if data is None:
        return ProbeResult(
            error=f"libibverbs probe exited with {proc.returncode}: {proc.stderr.strip()[-400:]}"
        )
    return ProbeResult(
        enumerated=data["enumerated"],
        opened=data["opened"],
        qp_failures=data.get("qp_failures", {}),
        sonames=data.get("sonames", {}),
        abi_mismatch={os.path.basename(m.group(2)): m.group(0) for m in _ABI_RE.finditer(proc.stderr)},
        loaded=_INIT_RE.findall(proc.stderr) if trace else [],
        libibverbs=data.get("libibverbs"),
        error=data.get("error"),
    )


@dataclass
class ProviderLayout:
    """How this environment's libibverbs names and finds providers."""

    suffix: str  # e.g. "-rdmav34.so"
    provider_dir: Optional[Path]  # compiled-in provider directory (tried before the linker path)
    lib_dir: Path  # directory holding libibverbs.so.1
    config_dir: Path  # driver config directory (libibverbs.d)


def read_provider_layout(libibverbs: str) -> ProviderLayout:
    data = Path(libibverbs).read_bytes()
    match = re.search(rb"\x00(/[\x21-\x7e]*)/lib%s(-rdmav\d+\.so)\x00", data)
    provider_dir = Path(match.group(1).decode()) if match else None
    if not match:
        match = re.search(rb"\x00lib%s(-rdmav\d+\.so)\x00", data)
    if not match:
        raise SetupError(f"cannot tell how {libibverbs} names its providers")
    config = re.search(rb"\x00(/[\x21-\x7e]*libibverbs\.d)\x00", data)
    return ProviderLayout(
        suffix=match.groups()[-1].decode(),
        provider_dir=provider_dir,
        lib_dir=Path(os.path.realpath(libibverbs)).parent,
        config_dir=Path(config.group(1).decode()) if config else Path("/etc/libibverbs.d"),
    )


def assess(devices: Sequence[RdmaDevice], probe: ProbeResult) -> Optional[str]:
    """Why the current provider cannot drive these (accessible) devices, or None when it can."""
    names = [d.name for d in devices]
    missing = [n for n in names if n not in probe.enumerated]
    if missing:
        abi = [probe.abi_mismatch[n] for n in names if n in probe.abi_mismatch]
        detail = f' ("{abi[0]}")' if abi else ""
        return f"libibverbs lists {len(names) - len(missing)} of {len(names)} devices{detail}"
    unopened = [n for n in names if n not in probe.opened]
    if unopened:
        return f"libibverbs cannot open {', '.join(unopened)}"
    failures = probe.provider_failures(names)
    if failures:
        return next(iter(failures.values()))
    return None


def _same_file(a: Optional[str], b: Path) -> bool:
    try:
        return a is not None and os.path.samefile(a, b)
    except OSError:
        return False


def verify(devices: Sequence[RdmaDevice], installed: "Installed") -> Optional[str]:
    env = dict(installed.env, PRIMUS_NIC_PROBE_SONAMES=":".join(installed.sonames))
    probe = run_probe(env, trace=True)
    if probe.error:
        return probe.error
    problem = assess(devices, probe)
    if problem:
        return problem
    # In user mode the image's provider is still loaded first and warns before ours claims the device.
    names = [d.name for d in devices]
    abi = [n for n in names if n in probe.abi_mismatch]
    if abi and not installed.env:
        return f'kernel ABI warning for {", ".join(abi)}: "{probe.abi_mismatch[abi[0]]}"'
    if not any(_same_file(p, installed.provider_path) for p in probe.loaded):
        return f"libibverbs did not load {installed.provider_path}"
    for soname, expected in installed.sonames.items():
        got = probe.sonames.get(soname)
        if not _same_file(got, expected):
            return f"{soname} resolves to {got or 'nothing'} instead of {expected}, so a process would load two copies"
    for failure in probe.limit_failures(names).values():
        warn(f"{failure}; RCCL needs locked memory (--ulimit memlock=-1 or --privileged)")
        break
    return None


# ---------------------------------------------------------------------------
# Archives and packages
# ---------------------------------------------------------------------------


def is_archive_name(name: str) -> bool:
    return name.lower().endswith(ARCHIVE_SUFFIXES)


def archive_members(archive: Path) -> List[str]:
    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                return [i.filename for i in zf.infolist() if not i.is_dir()]
        # "r:*" sniffs the compression, so a gzip stream named .tar.xz still opens.
        with tarfile.open(archive, "r:*") as tf:
            return [m.name for m in tf.getmembers() if m.isfile()]
    except (OSError, EOFError, tarfile.TarError, zipfile.BadZipFile) as exc:
        warn(f"skipping unreadable archive {archive}: {exc}")
        return []


def extract_member(archive: Path, member: str, scratch: Path) -> Path:
    target = Path(tempfile.mkdtemp(dir=scratch, prefix="member-")) / Path(member).name
    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf, zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
        else:
            with tarfile.open(archive, "r:*") as tf:
                src = tf.extractfile(member)
                if src is None:
                    raise SetupError(f"{member} in {archive} is not a regular file")
                with src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
    except (OSError, EOFError, KeyError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise SetupError(f"cannot extract {member} from {archive}: {exc}") from exc
    return target


def iter_archive_files(
    archive: Path, scratch: Path, descend: Callable[[str], bool], depth: int, origin: str = ""
) -> Iterator[Tuple[Path, str, str]]:
    """Yield (archive, member, origin) for files in archive, recursing into nested archives that descend() accepts."""
    origin = origin or str(archive)
    nested = []
    for member in archive_members(archive):
        yield archive, member, f"{origin} -> {member}"
        if depth > 1 and is_archive_name(member) and descend(os.path.basename(member)):
            nested.append(member)
    for member in nested:
        try:
            inner = extract_member(archive, member, scratch)
        except SetupError as exc:
            warn(str(exc))
            continue
        yield from iter_archive_files(inner, scratch, descend, depth - 1, f"{origin} -> {member}")


def scan_search_path(roots: Sequence[Path]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
            continue
        if not root.is_dir():
            continue
        base = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root):
            deep = len(Path(dirpath).parts) - base >= MAX_SCAN_DEPTH
            dirnames[:] = [] if deep else sorted(d for d in dirnames if not d.startswith("."))
            for name in sorted(filenames):
                files.append(Path(dirpath) / name)
                if len(files) >= MAX_SCAN_FILES:
                    warn(
                        f"stopped scanning after {MAX_SCAN_FILES} files; narrow PRIMUS_NIC_DRIVER_SEARCH_PATH"
                    )
                    return files
    return files


def extract_deb(deb: Path, scratch: Path) -> Path:
    if not shutil.which("dpkg-deb"):
        raise SetupError("dpkg-deb is required to unpack .deb packages")
    root = Path(tempfile.mkdtemp(dir=scratch, prefix="deb-"))
    proc = subprocess.run(["dpkg-deb", "-x", str(deb), str(root)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SetupError(f"dpkg-deb -x {deb} failed: {proc.stderr.strip()}")
    return root


def safe_extractall(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:*") as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(dest, filter="data")
            return
        root = os.path.realpath(dest)
        for member in tf.getmembers():
            target = os.path.realpath(os.path.join(dest, member.name))
            escapes = target != root and not target.startswith(root + os.sep)
            if (
                escapes
                or member.isdev()
                or ((member.issym() or member.islnk()) and os.path.isabs(member.linkname))
            ):
                raise SetupError(f"refusing to extract {member.name} from {archive}")
        tf.extractall(dest)


def elf_soname(path: Path) -> Optional[str]:
    """DT_SONAME of a 64-bit little-endian ELF shared object."""
    data = path.read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        return None
    (phoff,) = struct.unpack_from("<Q", data, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
    dynamic, loads = None, []
    for i in range(phnum):
        p_type, _, p_offset, p_vaddr, _, p_filesz, _, _ = struct.unpack_from(
            "<IIQQQQQQ", data, phoff + i * phentsize
        )
        if p_type == 2:
            dynamic = (p_offset, p_filesz)
        elif p_type == 1:
            loads.append((p_vaddr, p_offset, p_filesz))
    if dynamic is None:
        return None
    strtab = soname = None
    for off in range(dynamic[0], dynamic[0] + dynamic[1], 16):
        tag, val = struct.unpack_from("<qQ", data, off)
        if tag == 0:
            break
        if tag == 5:
            strtab = val
        elif tag == 14:
            soname = val
    if strtab is None or soname is None:
        return None
    for vaddr, offset, size in loads:
        if vaddr <= strtab < vaddr + size:
            start = strtab - vaddr + offset + soname
            return data[start : data.index(b"\x00", start)].decode()
    return None


def provider_in_tree(root: Path, provider: str, suffix: str) -> Path:
    """The real shared object behind lib<provider>-rdmavN.so in an unpacked package."""
    links = sorted(root.rglob(f"lib{provider}-rdmav*.so"))
    if not links:
        raise SetupError(f"package has no lib{provider}-rdmav*.so")
    if links[0].name != f"lib{provider}{suffix}":
        raise SetupError(f"package provides {links[0].name} but this libibverbs loads lib{provider}{suffix}")
    real = Path(os.path.realpath(links[0]))
    if not str(real).startswith(os.path.realpath(root) + os.sep) or not real.is_file():
        raise SetupError(f"{links[0].name} in the package does not resolve to a file inside the package")
    return real


# ---------------------------------------------------------------------------
# Provider families
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    version: str  # provider release it provides; "" when unknown
    kind: str  # source | deb | prebuilt | download
    path: str  # file on disk, archive holding `member`, or URL
    origin: str
    member: Optional[str] = None
    explicit: bool = False


@dataclass
class Staged:
    """A provider ready to install."""

    version: str
    real_file: Path
    aliases: List[str] = field(default_factory=list)  # SONAMEs applications link against


@dataclass
class Context:
    settings: Settings
    layout: ProviderLayout
    scratch: Path
    host_version: str
    devices: List[RdmaDevice]


def materialize(cand: Candidate, scratch: Path) -> Path:
    return extract_member(Path(cand.path), cand.member, scratch) if cand.member else Path(cand.path)


def _version_tuple(version: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", version))


def _closeness(version: str, want: str) -> Tuple:
    have, target = _version_tuple(version), _version_tuple(want)
    same_major = bool(have) and bool(target) and have[0] == target[0]
    return (0 if same_major else 1, tuple(abs(a - b) for a, b in zip_longest(have, target, fillvalue=0)))


class Family:
    provider = ""  # libibverbs provider name: lib<provider>-rdmavN.so
    label = ""
    vendor_ids: Tuple[str, ...] = ()
    links_shared_lib = False  # applications link the provider library directly (libionic.so.1)
    stale_glob: Optional[str] = None  # older copies to retire so ldconfig cannot pick them
    no_version = "the host driver does not report its version"

    def matches(self, device: RdmaDevice) -> bool:
        return device.vendor in self.vendor_ids

    def host_version(self, devices: Sequence[RdmaDevice], sysfs: Path) -> str:
        raise NotImplementedError

    def candidates(self, ctx: Context) -> Iterator[Candidate]:
        raise NotImplementedError

    def stage(self, cand: Candidate, ctx: Context) -> Staged:
        raise NotImplementedError

    def not_found(self, ctx: Context) -> str:
        raise NotImplementedError


_BNXT_SOURCE_RE = re.compile(r"^libbnxt_re-(\d+(?:\.\d+)+)\.(?:tar\.gz|tgz)$")
_BNXT_DEB_RE = re.compile(r"^bnxt-rocelib[_-](\d+(?:\.\d+)+)[-_].*\.deb$")
_BNXT_BUILD_TOOLS = ("make", "aclocal", "libtoolize", "autoheader", "automake", "autoconf")


def _classify_bnxt(name: str) -> Optional[Tuple[str, str]]:
    base = os.path.basename(name)
    for kind, pattern in (("source", _BNXT_SOURCE_RE), ("deb", _BNXT_DEB_RE)):
        match = pattern.match(base)
        if match:
            return kind, match.group(1)
    return None


def _bnxt_descend(name: str) -> bool:
    return bool(re.match(r"(?i)(bcm|nxe)", name)) or name.lower().endswith(".zip")


class BnxtFamily(Family):
    provider = "bnxt_re"
    label = "Broadcom bnxt_re"
    vendor_ids = ("0x14e4",)

    def host_version(self, devices: Sequence[RdmaDevice], sysfs: Path) -> str:
        return _read(sysfs / "module" / "bnxt_re" / "version")

    def _explicit(self, ctx: Context, roots: List[Path]) -> Iterator[Candidate]:
        package = ctx.settings.bnxt_package
        if package is None:
            return
        if package.is_dir():
            roots.insert(0, package)
        elif not package.is_file():
            warn(f"PATH_TO_BNXT_TAR_PACKAGE={package} is not readable here; searching elsewhere")
        elif package.name.startswith("libbnxt_re") and package.name.endswith((".tar.gz", ".tgz")):
            match = re.search(r"\d+(?:\.\d+)+", package.name)
            yield Candidate(
                match.group(0) if match else "", "source", str(package), str(package), explicit=True
            )
        elif package.name.endswith(".deb"):
            classified = _classify_bnxt(package.name)
            yield Candidate(
                classified[1] if classified else "", "deb", str(package), str(package), explicit=True
            )
        else:
            roots.insert(0, package)  # a vendor bundle archive

    def _host_provider(self, version: str) -> Optional[Candidate]:
        if not version:
            return None
        found = sorted((HOST_PROVIDER_ROOT / "bnxt_re" / version).glob("libbnxt_re-rdmav*.so"))
        if not found:
            return None
        return Candidate(
            version, "prebuilt", str(found[0]), f"host-installed libbnxt_re {version} ({found[0]})"
        )

    def candidates(self, ctx: Context) -> Iterator[Candidate]:
        want = ctx.host_version
        roots = list(ctx.settings.search_path)
        yield from self._explicit(ctx, roots)

        # Bundle names carry the release (bcm_<fw_ver>.tar.gz); the driver version names its components.
        hints = [h for h in [want] + sorted({d.fw_ver for d in ctx.devices}) if h]
        plain, archives = [], []
        for path in scan_search_path(roots):
            classified = _classify_bnxt(path.name)
            if classified:
                plain.append(Candidate(classified[1], classified[0], str(path), str(path)))
            elif is_archive_name(path.name) and (
                path in roots or _bnxt_descend(path.name) or any(h in path.name for h in hints)
            ):
                archives.append(path)
        yield from sorted((c for c in plain if c.version == want), key=lambda c: KIND_ORDER[c.kind])

        archives.sort(key=lambda a: (not any(h in a.name for h in hints), a.name))
        nested = []
        for archive in archives:
            exact = []
            for holder, member, origin in iter_archive_files(archive, ctx.scratch, _bnxt_descend, depth=2):
                classified = _classify_bnxt(member)
                if classified:
                    cand = Candidate(classified[1], classified[0], str(holder), origin, member=member)
                    (exact if cand.version == want else nested).append(cand)
            yield from sorted(exact, key=lambda c: KIND_ORDER[c.kind])

        host = self._host_provider(want)
        if host:
            yield host

        others = [c for c in plain + nested if c.version != want]
        yield from sorted(others, key=lambda c: (_closeness(c.version, want), KIND_ORDER[c.kind]))

    def stage(self, cand: Candidate, ctx: Context) -> Staged:
        path = materialize(cand, ctx.scratch)
        if cand.kind == "source":
            real = self._build(path, cand.version, ctx)
        elif cand.kind == "deb":
            real = provider_in_tree(extract_deb(path, ctx.scratch), self.provider, ctx.layout.suffix)
        else:
            real = path
        if real.name != f"libbnxt_re{ctx.layout.suffix}":
            raise SetupError(
                f"{real.name} does not match this libibverbs (expects libbnxt_re{ctx.layout.suffix})"
            )
        return Staged(version=cand.version, real_file=real)

    def _build(self, tarball: Path, version: str, ctx: Context) -> Path:
        missing = [tool for tool in _BNXT_BUILD_TOOLS if not shutil.which(tool)]
        if not (shutil.which("cc") or shutil.which("gcc")):
            missing.append("a C compiler")
        if not any(Path(d, "infiniband", "verbs.h").exists() for d in ("/usr/include", "/usr/local/include")):
            missing.append("libibverbs headers (infiniband/verbs.h)")
        if missing:
            raise SetupError(f"cannot build libbnxt_re here, missing {', '.join(missing)}")
        work = Path(tempfile.mkdtemp(dir=ctx.scratch, prefix="build-"))
        safe_extractall(tarball, work / "src")
        sources = sorted((work / "src").glob("*/configure.ac"))
        if not sources:
            raise SetupError(f"{tarball} does not look like a libbnxt_re source tarball")
        src = sources[0].parent
        prefix = work / "install"
        # configure.ac locates verbs.h under $PREFIX when that variable is set.
        env = {k: v for k, v in os.environ.items() if k != "PREFIX"}
        steps = [
            ["./configure", f"--prefix={prefix}"],
            ["make", f"-j{min(os.cpu_count() or 1, 16)}"],
            ["make", "install"],
        ]
        if (src / "autogen.sh").exists():
            steps.insert(0, ["sh", "autogen.sh"])
        log_path = work / "build.log"
        started = time.time()
        info(f"building libbnxt_re {version or ''} from {tarball.name}")
        with open(log_path, "w") as log:
            for cmd in steps:
                log.write(f"$ {' '.join(cmd)}\n")
                log.flush()
                rc = subprocess.run(cmd, cwd=src, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                if rc != 0:
                    fd, kept_name = tempfile.mkstemp(
                        prefix=f"primus-libbnxt_re-{version or 'unknown'}-", suffix=".log"
                    )
                    os.close(fd)
                    kept = Path(kept_name)
                    shutil.copyfile(log_path, kept)
                    tail = "\n".join(kept.read_text(errors="replace").splitlines()[-25:])
                    raise SetupError(
                        f"`{' '.join(cmd)}` failed with exit code {rc} (full log: {kept}):\n{tail}"
                    )
        built = [p for p in sorted((prefix / "lib").glob("libbnxt_re-rdmav*.so")) if not p.is_symlink()]
        if not built:
            raise SetupError(f"the build did not produce libbnxt_re-rdmav*.so under {prefix / 'lib'}")
        info(f"built {built[0].name} in {time.time() - started:.0f}s")
        return built[0]

    def not_found(self, ctx: Context) -> str:
        roots = ":".join(str(p) for p in ctx.settings.search_path) or "<empty>"
        return (
            f"no libbnxt_re {ctx.host_version} found (searched PATH_TO_BNXT_TAR_PACKAGE, "
            f"PRIMUS_NIC_DRIVER_SEARCH_PATH={roots} and the host's installed libbnxt_re). Put the Broadcom "
            f"NetXtreme-E Linux bundle that matches the host driver (it contains "
            f"drivers_linux/bnxt_rocelib/libbnxt_re-{ctx.host_version}.tar.gz) in one of those directories "
            f"on every node, or point PATH_TO_BNXT_TAR_PACKAGE at the tarball"
        )


_AINIC_BUNDLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _ionic_descend(name: str) -> bool:
    lowered = name.lower()
    return any(key in lowered for key in ("host_sw_pkg", "rdma-core-debs", "libionic-debs", "ainic"))


def _ionic_rank(path: str) -> int:
    # Bundles can carry both archives; rdma-core-debs holds what the published repo installs.
    return 0 if "rdma-core-debs" in path else 1


class IonicFamily(Family):
    provider = "ionic"
    label = "AMD Pensando AINIC (ionic)"
    vendor_ids = ("0x1dd8",)
    links_shared_lib = True
    stale_glob = "libionic.so.1.*"
    no_version = (
        "the ionic devices do not report fw_ver, so the AINIC bundle version is unknown; "
        "set PRIMUS_AINIC_BUNDLE_VERSION to the bundle installed on the hosts"
    )

    def host_version(self, devices: Sequence[RdmaDevice], sysfs: Path) -> str:
        versions = [d.fw_ver for d in devices if d.fw_ver]
        if not versions:
            return ""
        common = max(sorted(set(versions)), key=versions.count)
        if len(set(versions)) > 1:
            warn(f"ionic devices report different fw_ver values {sorted(set(versions))}; using {common}")
        return common

    def candidates(self, ctx: Context) -> Iterator[Candidate]:
        want, codename = ctx.host_version, os_codename()

        def in_codename_dir(path: str) -> bool:
            # Bundles ship one directory of .debs per distribution codename.
            return f"/{codename}/" in f"/{path}"

        if codename:
            files = scan_search_path(ctx.settings.search_path)
            local = [
                Candidate(want, "deb", str(f), str(f))
                for f in files
                if f.name.startswith("libionic1_")
                and f.name.endswith(".deb")
                and any(want in part for part in f.parts)
                and in_codename_dir(str(f))
            ]
            yield from sorted(local, key=lambda c: _ionic_rank(c.path))
            for archive in (f for f in files if is_archive_name(f.name) and want in f.name):
                found = [
                    Candidate(want, "deb", str(holder), origin, member=member)
                    for holder, member, origin in iter_archive_files(
                        archive, ctx.scratch, _ionic_descend, depth=3
                    )
                    if os.path.basename(member).startswith("libionic1_")
                    and member.endswith(".deb")
                    and in_codename_dir(member)
                ]
                yield from sorted(found, key=lambda c: _ionic_rank(c.origin))
        if ctx.settings.ainic_repo_url and _AINIC_BUNDLE_RE.match(want):
            url = f"{ctx.settings.ainic_repo_url}/{want}"
            yield Candidate(want, "download", url, f"{url} ({codename or 'unknown codename'})")

    def stage(self, cand: Candidate, ctx: Context) -> Staged:
        deb = self._download(cand, ctx) if cand.kind == "download" else materialize(cand, ctx.scratch)
        real = provider_in_tree(extract_deb(deb, ctx.scratch), self.provider, ctx.layout.suffix)
        soname = elf_soname(real)
        if not soname:
            raise SetupError(f"cannot read the SONAME of {real.name}")
        return Staged(version=cand.version, real_file=real, aliases=[soname] if soname != real.name else [])

    def _download(self, cand: Candidate, ctx: Context) -> Path:
        codename = os_codename()
        if not codename:
            raise SetupError(f"cannot determine the OS codename from {OS_RELEASE}")
        index_base = f"{cand.path}/dists/{codename}/main/binary-{deb_architecture()}"
        packages, errors = None, []
        for name in ("Packages", "Packages.gz"):
            try:
                with urllib.request.urlopen(f"{index_base}/{name}", timeout=DOWNLOAD_TIMEOUT_S) as resp:
                    data = resp.read()
                packages = (gzip.decompress(data) if name.endswith(".gz") else data).decode()
                break
            except (OSError, ValueError, urllib.error.URLError) as exc:
                errors.append(f"{index_base}/{name}: {exc}")
                if not isinstance(exc, urllib.error.HTTPError):
                    break  # unreachable (air-gapped, DNS, timeout): do not wait out a second timeout
        if packages is None:
            raise SetupError("cannot fetch the package index: " + "; ".join(errors))
        stanza = next((s for s in parse_packages_index(packages) if s.get("Package") == "libionic1"), None)
        if stanza is None or "Filename" not in stanza:
            raise SetupError(f"{index_base}/Packages has no libionic1 package")
        url = f"{cand.path}/{stanza['Filename']}"
        dest = ctx.scratch / os.path.basename(stanza["Filename"])
        info(f"downloading libionic1 {stanza.get('Version', '')} from {url}")
        try:
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as resp, open(dest, "wb") as out:
                shutil.copyfileobj(resp, out)
        except (OSError, urllib.error.URLError) as exc:
            raise SetupError(f"cannot download {url}: {exc}") from exc
        expected = stanza.get("SHA256")
        if expected and hashlib.sha256(dest.read_bytes()).hexdigest() != expected:
            raise SetupError(f"{url} does not match the SHA256 in the package index")
        return dest

    def not_found(self, ctx: Context) -> str:
        roots = ":".join(str(p) for p in ctx.settings.search_path) or "<empty>"
        return (
            f"no libionic1 for AINIC bundle {ctx.host_version} ({os_codename() or 'unknown codename'}) found under "
            f"PRIMUS_NIC_DRIVER_SEARCH_PATH={roots}"
            + ("" if ctx.settings.ainic_repo_url else " and the package repository is disabled")
        )


def parse_packages_index(text: str) -> List[Dict[str, str]]:
    stanzas, current = [], {}
    for line in text.splitlines():
        if not line.strip():
            if current:
                stanzas.append(current)
            current = {}
        elif not line[0].isspace() and ":" in line:
            key, _, value = line.partition(":")
            current[key.strip()] = value.strip()
    if current:
        stanzas.append(current)
    return stanzas


FAMILIES: Tuple[Family, ...] = (BnxtFamily(), IonicFamily())


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


@dataclass
class Installed:
    provider_path: Path  # what libibverbs should load
    env: Dict[str, str] = field(default_factory=dict)  # empty for system installs
    sonames: Dict[str, Path] = field(default_factory=dict)  # soname -> file the linker must resolve it to
    undo: List[Tuple[Path, Optional[Path]]] = field(default_factory=list)  # (target, backup)
    backup_dir: Optional[Path] = None
    ldconfig: bool = False


def _ldconfig() -> None:
    if shutil.which("ldconfig"):
        subprocess.run(["ldconfig"], capture_output=True)


class SystemInstaller:
    """Overlay the provider onto the paths libibverbs already uses (root, inside a container only)."""

    mode = "system"

    def __init__(self, layout: ProviderLayout, backup_root: Path = SYSTEM_BACKUP_ROOT):
        self.layout = layout
        self.backup_root = backup_root

    def install(self, family: Family, staged: Staged) -> Installed:
        lay = self.layout
        assert lay.provider_dir is not None
        self.backup_root.mkdir(parents=True, exist_ok=True)
        backups = Path(tempfile.mkdtemp(dir=self.backup_root, prefix=f"{family.provider}-"))
        inst = Installed(
            provider_path=lay.provider_dir / f"lib{family.provider}{lay.suffix}", backup_dir=backups
        )
        try:
            if family.links_shared_lib:
                real = lay.lib_dir / staged.real_file.name
                self._put(inst, backups, real, source=staged.real_file)
                for alias in staged.aliases:
                    self._put(inst, backups, lay.lib_dir / alias, link=real.name)
                    inst.sonames[alias] = real
                self._put(inst, backups, inst.provider_path, link=os.path.relpath(real, lay.provider_dir))
                for old in sorted(lay.lib_dir.glob(family.stale_glob or "")):
                    if old != real and old.is_file() and not old.is_symlink():
                        self._put(inst, backups, old)
                inst.ldconfig = True
            else:
                self._put(inst, backups, inst.provider_path, source=staged.real_file)
            config = lay.config_dir / f"{family.provider}.driver"
            if _read(config).split() != ["driver", family.provider]:
                self._put(inst, backups, config, text=f"driver {family.provider}\n")
            if inst.ldconfig:
                _ldconfig()
        except Exception as exc:
            self.rollback(inst)
            raise SetupError(f"cannot install into {lay.provider_dir}: {exc}") from exc
        return inst

    @staticmethod
    def _put(
        inst: Installed,
        backups: Path,
        target: Path,
        source: Optional[Path] = None,
        link: Optional[str] = None,
        text: Optional[str] = None,
    ) -> None:
        """Replace (or with no content given, remove) target, keeping a backup for rollback."""
        backup = None
        if os.path.lexists(target):
            backup = backups / f"{len(inst.undo)}-{target.name}"
            if target.is_symlink():
                os.symlink(os.readlink(target), backup)
            else:
                shutil.copy2(target, backup)
        inst.undo.append((target, backup))
        if source is None and link is None and text is None:
            os.remove(target)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.primus-tmp")
        if os.path.lexists(tmp):
            os.remove(tmp)
        if link is not None:
            os.symlink(link, tmp)
        elif source is not None:
            shutil.copy2(source, tmp)
        else:
            tmp.write_text(text or "")
        os.replace(tmp, target)

    def rollback(self, inst: Installed) -> None:
        restored = True
        for target, backup in reversed(inst.undo):
            try:
                if os.path.lexists(target):
                    os.remove(target)
                if backup is not None and backup.is_symlink():
                    os.symlink(os.readlink(backup), target)
                elif backup is not None:
                    shutil.copy2(backup, target)
            except OSError as exc:
                warn(f"could not restore {target} (backup kept in {inst.backup_dir}): {exc}")
                restored = False
        if inst.ldconfig:
            _ldconfig()
        inst.undo.clear()
        if restored:
            self.commit(inst)

    def commit(self, inst: Installed) -> None:
        if inst.backup_dir is not None:
            shutil.rmtree(inst.backup_dir, ignore_errors=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _private_dir(path: Path) -> bool:
    """True if path is a real directory owned by us that nobody else can write to."""
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISDIR(st.st_mode) and st.st_uid == os.getuid() and not st.st_mode & 0o022


class UserInstaller:
    """Install into a private prefix and select it per process via RDMAV_DRIVERS / LD_LIBRARY_PATH."""

    mode = "user"

    def __init__(self, layout: ProviderLayout, base_dir: Optional[Path] = None):
        self.layout = layout
        self.base_dir = base_dir or Path(tempfile.gettempdir()) / f"primus-rdma-providers-{os.getuid()}"

    def install(self, family: Family, staged: Staged) -> Installed:
        # Processes load code from here, so never reuse a directory someone else could have prepared.
        if not _private_dir(self.base_dir):
            warn(f"{self.base_dir} is not a private directory; using a fresh one")
            self.base_dir = Path(tempfile.mkdtemp(prefix=f"primus-rdma-providers-{os.getuid()}-"))
        digest = _sha256(staged.real_file)
        prefix = self.base_dir / f"{family.provider}-{staged.version or 'unknown'}-{digest[:12]}"
        provider = prefix / f"lib{family.provider}{self.layout.suffix}"
        real = prefix / staged.real_file.name
        if prefix.exists() and not (real.is_file() and not real.is_symlink() and _sha256(real) == digest):
            shutil.rmtree(prefix, ignore_errors=True)
        if not prefix.exists():
            tmp = None
            try:
                tmp = Path(tempfile.mkdtemp(dir=self.base_dir, prefix=".staging-"))
                shutil.copy2(staged.real_file, tmp / staged.real_file.name)
                for name in [provider.name] + staged.aliases:
                    if name != staged.real_file.name:
                        os.symlink(staged.real_file.name, tmp / name)
                os.replace(tmp, prefix)
            except OSError as exc:
                if tmp is not None:
                    shutil.rmtree(tmp, ignore_errors=True)
                if not provider.exists():  # another launch on this node may have won the rename
                    raise SetupError(f"cannot install into {prefix}: {exc}") from exc
        env = {"RDMAV_DRIVERS": str(prefix / f"lib{family.provider}")}
        sonames = {}
        if family.links_shared_lib:
            env["LD_LIBRARY_PATH"] = str(prefix)
            sonames = {alias: real for alias in staged.aliases}
        return Installed(provider_path=provider, env=env, sonames=sonames)

    def rollback(self, inst: Installed) -> None:
        pass  # nothing outside the private prefix was touched

    def commit(self, inst: Installed) -> None:
        pass


def choose_installer(layout: ProviderLayout):
    reasons = []
    if os.geteuid() != 0:
        reasons.append("not running as root")
    if not running_in_container():
        reasons.append("not inside a container")
    if layout.provider_dir is None:
        reasons.append("libibverbs has no provider directory")
    else:
        dirs = (layout.provider_dir, layout.lib_dir, layout.config_dir)
        reasons += [f"{d} is not writable" for d in dirs if not os.access(d, os.W_OK)]
    if not reasons:
        return SystemInstaller(layout)
    info(f"using a private provider prefix ({'; '.join(reasons)}); system libraries stay untouched")
    return UserInstaller(layout)


def merge_env(into: Dict[str, str], env: Dict[str, str], prepend: bool = True) -> None:
    """Join colon-separated path lists (RDMAV_DRIVERS, LD_LIBRARY_PATH) instead of overwriting them."""
    for key, value in env.items():
        if not into.get(key):
            into[key] = value
        else:
            into[key] = f"{value}:{into[key]}" if prepend else f"{into[key]}:{value}"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def install_matching_provider(
    family: Family,
    devices: List[RdmaDevice],
    layout: ProviderLayout,
    settings: Settings,
    installer,
    sysfs: Path,
) -> Installed:
    """Install and verify a provider matching the host driver for `devices` (the accessible ones)."""
    version = settings.version_overrides.get(family.provider) or family.host_version(devices, sysfs)
    source = " (PRIMUS_AINIC_BUNDLE_VERSION)" if family.provider in settings.version_overrides else ""
    info(f"{family.label}: host driver version {version or 'unknown'}{source}")
    if not version and not (family.provider == "bnxt_re" and settings.bnxt_package):
        raise SetupError(family.no_version)
    tried: List[str] = []
    seen = set()
    with tempfile.TemporaryDirectory(prefix=f"primus-{family.provider}-") as scratch:
        ctx = Context(settings, layout, Path(scratch), version, devices)
        for cand in family.candidates(ctx):
            if len(tried) >= MAX_ATTEMPTS:
                break
            key = (os.path.realpath(cand.path) if cand.kind != "download" else cand.path, cand.member)
            if key in seen:
                continue
            seen.add(key)
            if version and cand.version and cand.version != version:
                warn(
                    f"{family.label}: trying {family.provider} {cand.version} (host runs {version}): {cand.origin}"
                )
            else:
                info(f"{family.label}: trying {cand.origin}")
            try:
                inst = installer.install(family, family.stage(cand, ctx))
            except Exception as exc:  # a corrupt archive or failed build only rules out this candidate
                tried.append(f"{cand.origin}: {exc}")
                warn(f"{family.label}: {exc}")
                continue
            try:
                problem = verify(devices, inst)
            except Exception as exc:  # never leave an unverified provider installed
                problem = f"verification failed: {exc!r}"
            if problem is None:
                installer.commit(inst)
                info(
                    f"{family.label}: installed {family.provider} {cand.version or ''} from {cand.origin} "
                    f"({installer.mode} install, {len(devices)} devices verified)"
                )
                return inst
            installer.rollback(inst)
            tried.append(f"{cand.origin}: {problem}")
            warn(f"{family.label}: {cand.origin} did not work ({problem}); rolled back")
        if not tried:
            raise SetupError(family.not_found(ctx))
    raise SetupError("no usable provider: " + " | ".join(tried))


def setup(settings: Settings, sysfs: Path = SYSFS) -> int:
    if settings.mode == "off":
        return 0
    devices = list_rdma_devices(sysfs)
    groups = [(fam, [d for d in devices if fam.matches(d)]) for fam in FAMILIES]
    groups = [(fam, devs) for fam, devs in groups if devs]
    if not groups:
        if settings.forced_providers:
            info("REBUILD_BNXT=1 but no Broadcom RDMA devices are present; nothing to do")
        return 0

    probe = run_probe()
    try:
        if probe.error or not probe.libibverbs:
            raise SetupError(probe.error or "libibverbs not found")
        layout = read_provider_layout(probe.libibverbs)
    except (OSError, SetupError) as exc:
        warn(f"cannot check the RDMA userspace driver: {exc}")
        return 1 if settings.strict else 0

    multi_node = os.environ.get("NNODES", "1").strip() not in ("", "0", "1")
    env_out: Dict[str, str] = {}
    broken: List[str] = []
    installer = None
    for fam, all_devs in groups:
        devs = [d for d in all_devs if device_accessible(d)]
        if not devs:
            warn(
                f"{fam.label}: {len(all_devs)} devices are in /sys but none of their /dev/infiniband/uverbs* "
                f"character devices is accessible here (container started without --device /dev/infiniband?)"
            )
            broken.append(fam.label)
            continue
        if len(devs) < len(all_devs):
            skipped = ", ".join(d.name for d in all_devs if d not in devs)
            info(f"{fam.label}: {skipped} not passed into this environment; checking the other {len(devs)}")
        for failure in probe.limit_failures([d.name for d in devs]).values():
            warn(f"{fam.label}: {failure}; RCCL needs locked memory (--ulimit memlock=-1 or --privileged)")
            break
        problem = assess(devs, probe)
        forced = settings.mode == "force" or fam.provider in settings.forced_providers
        if problem is None and not forced:
            info(
                f"{fam.label}: the {fam.provider} provider in this environment works with all {len(devs)} devices"
            )
            continue
        info(
            f"{fam.label}: {problem or 'reinstall requested'}; installing a provider that matches the host driver"
        )
        installer = installer or choose_installer(layout)
        try:
            inst = install_matching_provider(fam, devs, layout, settings, installer, sysfs)
        except SetupError as exc:
            warn(f"{fam.label}: {exc}")
            if problem is None:
                warn(f"{fam.label}: keeping the existing provider, which already works")
                continue
            warn(
                f"{fam.label}: RDMA is unusable on this node, so RCCL falls back to TCP sockets here. "
                f"What to do: docs/04-technical-guides/multi-node-networking.md, 'When RDMA still does not work'"
            )
            if multi_node and not settings.strict:
                warn(
                    f"{fam.label}: nodes that did get a matching provider will use RDMA, and RCCL can hang on "
                    "mixed transports; set PRIMUS_NIC_DRIVER_STRICT=1 to stop the job instead"
                )
            broken.append(fam.label)
            continue
        if inst.env and fam.provider == "bnxt_re":
            info(
                f"{fam.label}: the image's own provider still prints 'does not support the kernel ABI'; harmless"
            )
        merge_env(env_out, inst.env)

    merge_env(env_out, {k: os.environ[k] for k in env_out if os.environ.get(k)}, prepend=False)
    for key, value in env_out.items():
        print(f"env.{key}={value}", flush=True)
    if broken and settings.strict:
        warn(f"PRIMUS_NIC_DRIVER_STRICT=1: aborting because RDMA is unusable for {', '.join(broken)}")
        return 1
    return 0


def status(settings: Settings, sysfs: Path = SYSFS) -> int:
    devices = list_rdma_devices(sysfs)
    print(f"mode: {settings.mode}; search path: {':'.join(map(str, settings.search_path)) or '<empty>'}")
    if not devices:
        print("no RDMA devices under /sys/class/infiniband")
        return 0
    probe = run_probe(trace=True)
    print(f"libibverbs: {probe.libibverbs or probe.error}")
    if probe.libibverbs:
        try:
            layout = read_provider_layout(probe.libibverbs)
            print(f"provider naming: {layout.provider_dir or '<linker path>'}/lib<name>{layout.suffix}")
        except (OSError, SetupError) as exc:
            print(f"provider naming: unknown ({exc})")
    print(f"running in a container: {running_in_container()}; root: {os.geteuid() == 0}")
    for fam in FAMILIES:
        devs = [d for d in devices if fam.matches(d)]
        if not devs:
            continue
        usable = [d for d in devs if device_accessible(d)]
        loaded = sorted({p for p in probe.loaded if f"lib{fam.provider}-rdmav" in os.path.basename(p)})
        override = settings.version_overrides.get(fam.provider)
        host = fam.host_version(devs, sysfs) or "unknown"
        print(
            f"{fam.label}: {len(devs)} devices, host driver {host}"
            + (f" (override: {override})" if override else "")
        )
        for d in devs:
            access = f"/dev/infiniband/{d.uverbs}" if d in usable else "no accessible character device"
            print(f"  {d.name}: fw {d.fw_ver or '?'}, {access}")
        print(f"  provider loaded from: {', '.join(loaded) or 'none'}")
        limits = "; ".join(probe.limit_failures([d.name for d in usable]).values())
        state = assess(usable, probe) if usable else "no device is accessible"
        print(f"  status: {state or 'OK'}" + (f" ({limits})" if limits else ""))
    others = sorted({d.vendor for d in devices if not any(f.matches(d) for f in FAMILIES)})
    if others:
        print(f"other RDMA vendors (not managed here): {', '.join(others)}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("setup", "status"))
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    if args.command == "status":
        return status(settings)
    try:
        return setup(settings)
    except Exception as exc:  # an unexpected failure here must not block a launch that may not need RDMA
        warn(f"NIC userspace driver setup crashed: {exc!r}")
        return 1 if settings.strict else 0


if __name__ == "__main__":
    sys.exit(main())
