###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# RDMA device scan and NCCL environment guidance (ROCm-oriented; in-tree).
###############################################################################

from __future__ import annotations

import glob
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from primus.tools.preflight.network.info import Finding

LIB_SEARCH_PATHS = [
    "/usr/lib",
    "/usr/lib64",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/local/lib",
    "/etc/libibverbs.d",
]


@dataclass
class _DeviceInfo:
    rdma: str
    pci: str
    netdev: str
    firmware: str
    gid_index: str
    gid_value: str
    vendor: str


@dataclass
class ClusterSphereEnvResult:
    """Structured output for preflight report and Findings."""

    devices: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    firmware_by_version: Dict[str, List[str]] = field(default_factory=dict)
    nccl_exports: List[str] = field(default_factory=list)
    rocshmem_exports: List[str] = field(default_factory=list)
    docker_cmd: str = ""
    socket_ifname: str = ""


class EnvRecommenderEngine:
    """
    RDMA device scan and environment recommendations (no stdout; results in
    ClusterSphereEnvResult + Findings).
    """

    def __init__(self) -> None:
        self._devices: List[_DeviceInfo] = []

    @property
    def devices(self) -> List[Dict[str, Any]]:
        return [d.__dict__ for d in self._devices]

    def scan_rdma_devices(self) -> bool:
        self._devices = []
        rdma_paths = sorted(glob.glob("/sys/class/infiniband/*"))
        if not rdma_paths:
            return False

        for path in rdma_paths:
            rdma = os.path.basename(path)
            pci = self._get_pci_device(path)
            netdev = self._find_netdev_for_pci(pci)
            ibv_out = self._ibv_devinfo(rdma)
            firmware = self._get_firmware_version(ibv_out)
            gid_index, gid_value = self._get_gid_info(ibv_out)
            vendor = self._rdma_vendor_from_pci(pci)
            self._devices.append(
                _DeviceInfo(
                    rdma=rdma,
                    pci=pci,
                    netdev=netdev,
                    firmware=firmware,
                    gid_index=gid_index,
                    gid_value=gid_value,
                    vendor=vendor,
                )
            )
        return bool(self._devices)

    def _get_pci_device(self, device_path: str) -> str:
        try:
            link = os.path.join(device_path, "device")
            if os.path.islink(link):
                return os.path.basename(os.readlink(link))
        except OSError:
            pass
        return "UNKNOWN_PCI"

    def _find_netdev_for_pci(self, target_pci: str) -> str:
        for netdev in glob.glob("/sys/class/net/*"):
            try:
                link = os.path.join(netdev, "device")
                if os.path.islink(link) and os.path.basename(os.readlink(link)) == target_pci:
                    return os.path.basename(netdev)
            except OSError:
                pass
        return "NO_NETDEV"

    def get_socket_ifname_value(self) -> str:
        try:
            out = subprocess.check_output("ip route show default | awk '{print $5}'", shell=True, text=True).strip()
            if not out:
                return "NA"
            ifnames = list(dict.fromkeys(out.splitlines()))
            return ifnames[0] if ifnames else "NA"
        except Exception:
            return "NA"

    def _rdma_vendor_from_pci(self, pci: str) -> str:
        try:
            out = subprocess.check_output(["lspci", "-s", pci, "-nn"], text=True).lower()
            if "pensando" in out:
                return "AINIC"
            if "broadcom" in out:
                return "BNXT"
            if "mellanox" in out:
                return "MLNX"
        except Exception:
            pass
        return "UNKNOWN"

    def _ibv_devinfo(self, rdma: str) -> str:
        try:
            result = subprocess.run(
                ["ibv_devinfo", "-d", rdma, "-v"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass
        return ""

    def _get_firmware_version(self, output: str) -> str:
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("fw_ver:"):
                return line.split("fw_ver:", 1)[1].strip()
        return "UNKNOWN"

    def _get_gid_info(self, output: str) -> Tuple[str, str]:
        for line in output.splitlines():
            if "::ffff:" in line and "GID[" in line:
                idx = re.search(r"GID\[\s*(\d+)\]", line)
                ip = re.search(r"(::ffff:[0-9.]+)", line)
                if idx and ip:
                    return idx.group(1), ip.group(1)
        return "-", "N/A"

    def _find_lib(self, patterns: List[str]) -> Optional[str]:
        for base in LIB_SEARCH_PATHS:
            for pat in patterns:
                matches = glob.glob(os.path.join(base, "**", pat), recursive=True)
                if matches:
                    return matches[0]
        return None

    def _find_all_libs(self, patterns: List[str]) -> List[str]:
        found: Set[str] = set()
        for base in LIB_SEARCH_PATHS:
            for pat in patterns:
                found.update(glob.glob(os.path.join(base, "**", pat), recursive=True))
        return sorted(found)

    def _docker_cmd_bnxt(self, warnings: List[str]) -> str:
        bnxt_rdma = self._find_lib(["libbnxt_re-rdmav*.so"])
        rdmacm = self._find_lib(["librdmacm.so.1"])
        ibverbs = self._find_lib(["libibverbs.so.1"])
        libnl3 = self._find_lib(["libnl-3.so.200"])
        libnl3_router = self._find_lib(["libnl-route-3.so.200"])

        if not bnxt_rdma:
            warnings.append("Docker (BNXT): Missing libbnxt_re-rdma*.so")
        if not rdmacm:
            warnings.append("Docker (BNXT): Missing librdmacm.so")
        if not libnl3:
            warnings.append("Docker (BNXT): Missing libnl-3.so")

        lines = [
            "docker run --rm -it \\",
            "    --device /dev/dri \\",
            "    --device /dev/infiniband \\",
            "    --device /dev/kfd \\",
            "    --network host \\",
            "    --ipc host \\",
            "    --privileged \\",
            "    --ulimit memlock=-1:-1 \\",
            "    --group-add video \\",
            "    --cap-add SYS_PTRACE \\",
            "    --security-opt seccomp=unconfined \\",
            "    --shm-size 64G \\",
            "    -v /sys:/sys \\",
            "    -v $HOME/.ssh:/root/.ssh \\",
            "    -v $HOME:$HOME \\",
            "    -v /dev/infiniband:/dev/infiniband \\",
            "    -v /sys/class/infiniband:/sys/class/infiniband:ro \\",
            "    -v /sys/class/net:/sys/class/net:ro \\",
            "    -v /sys/bus/pci:/sys/bus/pci:ro \\",
            "    -v /etc/libibverbs.d:/etc/libibverbs.d:ro \\",
            "    -v /etc/rdma:/etc/rdma:ro \\",
        ]
        if bnxt_rdma:
            lines.append(f"        -v {bnxt_rdma}:{bnxt_rdma}:ro \\")
        if rdmacm:
            lines.append(f"                -v {rdmacm}:{rdmacm}:ro \\")
        if ibverbs:
            lines.append(f"                -v {ibverbs}:{ibverbs}:ro \\")
        if libnl3:
            lines.append(f"                -v {libnl3}:{libnl3}:ro \\")
        if libnl3_router:
            lines.append(f"                -v {libnl3_router}:{libnl3_router}:ro \\")
        lines.append("                <image>")
        return "\n".join(lines)

    def _docker_cmd_mlnx(self) -> str:
        return (
            "docker run --rm -it \\\n"
            "    --device /dev/dri \\\n"
            "    --device /dev/infiniband \\\n"
            "    --device /dev/kfd \\\n"
            "    --network host \\\n"
            "    --ipc host \\\n"
            "    --privileged \\\n"
            "    --ulimit memlock=-1:-1 \\\n"
            "    --group-add video \\\n"
            "    --cap-add SYS_PTRACE \\\n"
            "    --security-opt seccomp=unconfined \\\n"
            "    --shm-size 64G \\\n"
            "    -v /sys:/sys \\\n"
            "    -v $HOME/.ssh:/root/.ssh \\\n"
            "    -v $HOME:$HOME \\\n"
            "    -v /dev/infiniband:/dev/infiniband \\\n"
            "    -v /sys/class/infiniband:/sys/class/infiniband:ro \\\n"
            "    -v /sys/class/net:/sys/class/net:ro \\\n"
            "    -v /sys/bus/pci:/sys/bus/pci:ro \\\n"
            "    <image>"
        )

    def _docker_cmd_ionic(self, warnings: List[str]) -> str:
        ionic_rdma = self._find_lib(["libionic-rdmav*.so"])
        ionic_so = self._find_all_libs(["libionic.so*"])
        ionic_driver = self._find_lib(["ionic.driver"])

        if not ionic_rdma:
            warnings.append("Docker (AINIC): Missing libionic-rdma*.so")
        if not ionic_so:
            warnings.append("Docker (AINIC): Missing libionic.so")
        if not ionic_driver:
            warnings.append("Docker (AINIC): Missing ionic.driver")

        lines = [
            "docker run --rm -it \\",
            "    --device /dev/dri \\",
            "    --device /dev/infiniband \\",
            "    --device /dev/kfd \\",
            "    --network host \\",
            "    --ipc host \\",
            "    --privileged \\",
            "    --ulimit memlock=-1:-1 \\",
            "    --group-add video \\",
            "    --cap-add SYS_PTRACE \\",
            "    --security-opt seccomp=unconfined \\",
            "    --shm-size 64G \\",
            "    -v /sys:/sys \\",
            "    -v $HOME/.ssh:/root/.ssh \\",
            "    -v $HOME:$HOME \\",
            "    -v /dev/infiniband:/dev/infiniband \\",
            "    -v /sys/class/infiniband:/sys/class/infiniband:ro \\",
            "    -v /sys/class/net:/sys/class/net:ro \\",
            "    -v /sys/bus/pci:/sys/bus/pci:ro \\",
        ]
        if ionic_rdma:
            lines.append(f"        -v {ionic_rdma}:{ionic_rdma}:ro \\")
        for so_file in ionic_so:
            lines.append(f"                -v {so_file}:{so_file}:ro \\")
        if ionic_driver:
            lines.append(f"                -v {ionic_driver}:{ionic_driver}:ro \\")
        lines.append("                <image>")
        return "\n".join(lines)

    def generate_docker_launch_command(self, warnings: List[str]) -> str:
        vendors = {d.vendor for d in self._devices}
        if len(vendors) > 1:
            warnings.append("Multiple RDMA vendors detected; verify Docker snippet manually.")

        if "AINIC" in vendors:
            return self._docker_cmd_ionic(warnings)
        if "BNXT" in vendors:
            return self._docker_cmd_bnxt(warnings)
        if "MLNX" in vendors:
            return self._docker_cmd_mlnx()

        warnings.append("Vendor UNKNOWN or unsupported for auto-generated Docker command.")
        return ""

    def build_nccl_exports(self, warnings: List[str]) -> List[str]:
        exports = ["export NCCL_IGNORE_CPU_AFFINITY=1"]

        gid_numeric: List[int] = []
        for d in self._devices:
            if str(d.gid_index).isdigit():
                gid_numeric.append(int(d.gid_index))

        unique_gid = sorted(set(gid_numeric))
        if len(unique_gid) > 1:
            warnings.append("Multiple GID indices detected; verify NCCL_IB_GID_INDEX against detailed device table.")

        if unique_gid:
            exports.append(f"export NCCL_IB_GID_INDEX={max(unique_gid)}")
        else:
            exports.append("export NCCL_IB_GID_INDEX=0")

        firmware_version = max((d.firmware for d in self._devices), default="UNKNOWN")
        parts: List[str] = []
        for d in self._devices:
            part = d.rdma if str(d.gid_index).isdigit() and d.firmware == firmware_version else ""
            if part:
                parts.append(part)
        nccl_hca = ",".join(parts) if parts else ",".join(d.rdma for d in self._devices)
        exports.append(f"export NCCL_IB_HCA={nccl_hca}")

        socket_ifname = self.get_socket_ifname_value()
        exports.append(f"export NCCL_SOCKET_IFNAME={socket_ifname}")
        exports.append(f"export GLOO_SOCKET_IFNAME={socket_ifname}")
        return exports

    def build_rocshmem_exports(self) -> List[str]:
        return [
            "export ROCSHMEM_HEAP_SIZE=7524589824",
            "export ROCSHMEM_MAX_NUM_CONTEXTS=256",
        ]

    def build_result(self) -> ClusterSphereEnvResult:
        warnings: List[str] = []
        result = ClusterSphereEnvResult(warnings=list(warnings))

        if not self.scan_rdma_devices():
            warnings.append("No RDMA devices found under /sys/class/infiniband.")
            result.warnings = warnings
            return result

        fw_map: Dict[str, List[str]] = {}
        for d in self._devices:
            fw_map.setdefault(d.firmware, []).append(d.rdma)
        result.devices = self.devices
        result.firmware_by_version = fw_map
        if len(fw_map) > 1:
            warnings.append("Multiple firmware versions detected — standardization recommended.")

        result.nccl_exports = self.build_nccl_exports(warnings)
        result.rocshmem_exports = self.build_rocshmem_exports()
        result.socket_ifname = self.get_socket_ifname_value()
        result.docker_cmd = self.generate_docker_launch_command(warnings)
        result.warnings = warnings
        return result


def collect_cluster_sphere_env_findings() -> List[Finding]:
    """Run env recommender on the local host; returns Findings for preflight aggregation."""
    engine = EnvRecommenderEngine()
    result = engine.build_result()

    findings: List[Finding] = []

    if not result.devices:
        findings.append(
            Finding(
                "warn",
                "Cluster Sphere env recommender: no RDMA devices found",
                {"warnings": result.warnings},
            )
        )
        return findings

    findings.append(
        Finding(
            "info",
            "Cluster Sphere RDMA environment recommendations",
            {
                "devices": result.devices,
                "nccl_exports": result.nccl_exports,
                "rocshmem_exports": result.rocshmem_exports,
                "docker_cmd": result.docker_cmd,
                "socket_ifname": result.socket_ifname,
                "firmware_by_version": result.firmware_by_version,
                "warnings": result.warnings,
            },
        )
    )

    return findings
