###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Host system probes for CPU, memory, PCIe, etc.
"""

from __future__ import annotations

import os
import platform
import re
import socket
import subprocess
from typing import Any, Dict, List, Optional


def get_hostname() -> str:
    """Get hostname."""
    return os.environ.get("HOSTNAME") or socket.gethostname()


def get_kernel_version() -> str:
    """Get kernel version."""
    return platform.release()


def get_os_info() -> Dict[str, str]:
    """Get OS information."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }


def get_cpu_info() -> Dict[str, Any]:
    """Get CPU information from /proc/cpuinfo and lscpu."""
    info: Dict[str, Any] = {
        "physical_cores": 0,
        "logical_cores": 0,
        "model_name": "",
        "vendor": "",
        "sockets": 0,
        "cores_per_socket": 0,
        "threads_per_core": 0,
        "numa_nodes": 0,
    }

    # Try lscpu first (more reliable)
    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    if key == "CPU(s)":
                        info["logical_cores"] = int(val)
                    elif key == "Model name":
                        info["model_name"] = val
                    elif key == "Vendor ID":
                        info["vendor"] = val
                    elif key == "Socket(s)":
                        info["sockets"] = int(val)
                    elif key == "Core(s) per socket":
                        info["cores_per_socket"] = int(val)
                    elif key == "Thread(s) per core":
                        info["threads_per_core"] = int(val)
                    elif key == "NUMA node(s)":
                        info["numa_nodes"] = int(val)
            # Calculate physical cores
            if info["sockets"] > 0 and info["cores_per_socket"] > 0:
                info["physical_cores"] = info["sockets"] * info["cores_per_socket"]
    except Exception:
        pass

    # Fallback to /proc/cpuinfo
    if info["logical_cores"] == 0:
        try:
            with open("/proc/cpuinfo", "r") as f:
                content = f.read()
                processors = content.count("processor")
                info["logical_cores"] = processors
                # Try to get model name
                match = re.search(r"model name\s*:\s*(.+)", content)
                if match:
                    info["model_name"] = match.group(1).strip()
        except Exception:
            pass

    return info


def get_memory_info() -> Dict[str, Any]:
    """Get memory information from /proc/meminfo."""
    info: Dict[str, Any] = {
        "total_gb": 0,
        "available_gb": 0,
        "free_gb": 0,
        "buffers_gb": 0,
        "cached_gb": 0,
        "swap_total_gb": 0,
        "swap_free_gb": 0,
    }

    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    # Parse value (usually in kB)
                    match = re.match(r"(\d+)\s*(\w+)?", val)
                    if match:
                        num = int(match.group(1))
                        unit = match.group(2) or "kB"
                        # Convert to GB
                        if unit.lower() == "kb":
                            gb = num / (1024 * 1024)
                        elif unit.lower() == "mb":
                            gb = num / 1024
                        elif unit.lower() == "gb":
                            gb = num
                        else:
                            gb = num / (1024 * 1024)  # assume kB

                        if key == "MemTotal":
                            info["total_gb"] = round(gb, 2)
                        elif key == "MemAvailable":
                            info["available_gb"] = round(gb, 2)
                        elif key == "MemFree":
                            info["free_gb"] = round(gb, 2)
                        elif key == "Buffers":
                            info["buffers_gb"] = round(gb, 2)
                        elif key == "Cached":
                            info["cached_gb"] = round(gb, 2)
                        elif key == "SwapTotal":
                            info["swap_total_gb"] = round(gb, 2)
                        elif key == "SwapFree":
                            info["swap_free_gb"] = round(gb, 2)
    except Exception:
        pass

    return info


def get_numa_info() -> Dict[str, Any]:
    """Get NUMA node information."""
    info: Dict[str, Any] = {
        "nodes": 0,
        "node_cpus": {},
        "node_memory_gb": {},
    }

    try:
        result = subprocess.run(["numactl", "--hardware"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("available:"):
                    match = re.search(r"(\d+)\s+nodes", line)
                    if match:
                        info["nodes"] = int(match.group(1))
                elif line.startswith("node ") and "cpus:" in line:
                    match = re.match(r"node\s+(\d+)\s+cpus:\s*(.*)", line)
                    if match:
                        node = int(match.group(1))
                        cpus = match.group(2).strip()
                        info["node_cpus"][node] = cpus
                elif line.startswith("node ") and "size:" in line:
                    match = re.match(r"node\s+(\d+)\s+size:\s*(\d+)\s*MB", line)
                    if match:
                        node = int(match.group(1))
                        size_mb = int(match.group(2))
                        info["node_memory_gb"][node] = round(size_mb / 1024, 2)
    except FileNotFoundError:
        # numactl not installed
        pass
    except Exception:
        pass

    return info


def get_pcie_info() -> List[Dict[str, Any]]:
    """Get PCIe device information using lspci."""
    devices: List[Dict[str, Any]] = []

    try:
        # Get GPU devices - check multiple PCIe class codes:
        # 0300: VGA compatible controller (some GPUs)
        # 0302: 3D controller (AMD Instinct GPUs like MI300X, MI355X)
        # 0380: Display controller (other accelerators)
        gpu_class_codes = ["0300", "0302", "0380"]
        for class_code in gpu_class_codes:
            result = subprocess.run(
                ["lspci", "-nn", "-d", f"::{class_code}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if line.strip():
                        devices.append({"type": "GPU", "info": line.strip()})

        # Get Infiniband/Network devices
        result = subprocess.run(
            ["lspci", "-nn", "-d", "::0207"],  # Infiniband
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.strip():
                    devices.append({"type": "Infiniband", "info": line.strip()})

        # Get Ethernet controllers
        result = subprocess.run(
            ["lspci", "-nn", "-d", "::0200"],  # Ethernet
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.strip():
                    devices.append({"type": "Ethernet", "info": line.strip()})

    except FileNotFoundError:
        pass
    except Exception:
        pass

    return devices


def get_pcie_link_info() -> Dict[str, Any]:
    """
    Get PCIe link speed and width for GPUs.

    Returns dict with:
    - gpu_links: list of {bdf, speed, width, max_speed, max_width, bandwidth_gbps}
    - summary: {min_speed, max_speed, min_width, max_width, total_bandwidth_gbps}
    """
    info: Dict[str, Any] = {
        "gpu_links": [],
        "summary": {},
    }

    # PCIe speed to GT/s mapping
    speed_map = {
        "2.5GT/s": 2.5,
        "5GT/s": 5.0,
        "8GT/s": 8.0,
        "16GT/s": 16.0,
        "32GT/s": 32.0,
        "64GT/s": 64.0,
    }

    # PCIe generation names
    gen_map = {
        2.5: "Gen1",
        5.0: "Gen2",
        8.0: "Gen3",
        16.0: "Gen4",
        32.0: "Gen5",
        64.0: "Gen6",
    }

    try:
        # Get GPU BDFs (Bus:Device.Function)
        gpu_bdfs = []
        gpu_class_codes = ["0300", "0302", "0380"]
        for class_code in gpu_class_codes:
            result = subprocess.run(
                ["lspci", "-n", "-d", f"::{class_code}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if line.strip():
                        bdf = line.split()[0]
                        gpu_bdfs.append(bdf)

        # Get link status for each GPU
        for bdf in gpu_bdfs:
            result = subprocess.run(
                ["lspci", "-vvv", "-s", bdf],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                output = result.stdout
                link_info: Dict[str, Any] = {"bdf": bdf}

                # Parse LnkCap (max capability)
                cap_match = re.search(r"LnkCap:.*Speed\s+(\d+(?:\.\d+)?GT/s).*Width\s+x(\d+)", output)
                if cap_match:
                    max_speed_str = cap_match.group(1)
                    max_width = int(cap_match.group(2))
                    max_speed_gts = speed_map.get(max_speed_str, 0)
                    link_info["max_speed"] = gen_map.get(max_speed_gts, max_speed_str)
                    link_info["max_speed_gts"] = max_speed_gts
                    link_info["max_width"] = max_width

                # Parse LnkSta (current status)
                sta_match = re.search(r"LnkSta:.*Speed\s+(\d+(?:\.\d+)?GT/s).*Width\s+x(\d+)", output)
                if sta_match:
                    speed_str = sta_match.group(1)
                    width = int(sta_match.group(2))
                    speed_gts = speed_map.get(speed_str, 0)
                    link_info["speed"] = gen_map.get(speed_gts, speed_str)
                    link_info["speed_gts"] = speed_gts
                    link_info["width"] = width

                    # Calculate bandwidth: GT/s * width * (encoding efficiency)
                    # Gen1/2: 8b/10b encoding (80% efficiency)
                    # Gen3+: 128b/130b encoding (~98.5% efficiency)
                    if speed_gts <= 5.0:
                        efficiency = 0.8
                    else:
                        efficiency = 0.985
                    bandwidth_gbps = speed_gts * width * efficiency / 8  # Convert to GB/s
                    link_info["bandwidth_gbps"] = round(bandwidth_gbps, 2)

                if link_info.get("speed"):
                    info["gpu_links"].append(link_info)

        # Generate summary
        if info["gpu_links"]:
            speeds = [l.get("speed_gts", 0) for l in info["gpu_links"] if l.get("speed_gts")]
            widths = [l.get("width", 0) for l in info["gpu_links"] if l.get("width")]
            bandwidths = [l.get("bandwidth_gbps", 0) for l in info["gpu_links"] if l.get("bandwidth_gbps")]

            if speeds:
                min_speed = min(speeds)
                max_speed = max(speeds)
                info["summary"]["min_speed"] = gen_map.get(min_speed, f"{min_speed}GT/s")
                info["summary"]["max_speed"] = gen_map.get(max_speed, f"{max_speed}GT/s")
            if widths:
                info["summary"]["min_width"] = f"x{min(widths)}"
                info["summary"]["max_width"] = f"x{max(widths)}"
            if bandwidths:
                info["summary"]["total_bandwidth_gbps"] = round(sum(bandwidths), 2)
                info["summary"]["per_gpu_bandwidth_gbps"] = round(sum(bandwidths) / len(bandwidths), 2)

    except FileNotFoundError:
        pass
    except Exception:
        pass

    # Fallback: use sysfs for containers where lspci -vvv may not work
    if not info["gpu_links"]:
        info = _get_pcie_link_info_sysfs()

    return info


def _get_pcie_link_info_sysfs() -> Dict[str, Any]:
    """
    Get PCIe link info from sysfs (works better in containers).

    Reads /sys/class/drm/card*/device/current_link_speed and current_link_width
    for physical GPU devices (not XCP partitions).
    """
    import glob

    info: Dict[str, Any] = {
        "gpu_links": [],
        "summary": {},
    }

    # PCIe speed to GT/s mapping
    speed_map = {
        "2.5 GT/s": 2.5,
        "5 GT/s": 5.0,
        "8 GT/s": 8.0,
        "16 GT/s": 16.0,
        "32 GT/s": 32.0,
        "64 GT/s": 64.0,
    }

    gen_map = {
        2.5: "Gen1",
        5.0: "Gen2",
        8.0: "Gen3",
        16.0: "Gen4",
        32.0: "Gen5",
        64.0: "Gen6",
    }

    try:
        # Find card devices that are real PCI devices (not XCP partitions)
        for card_path in sorted(glob.glob("/sys/class/drm/card*")):
            device_path = os.path.join(card_path, "device")
            if not os.path.isdir(device_path):
                continue

            # Check if this is a real PCI device (has current_link_speed)
            speed_file = os.path.join(device_path, "current_link_speed")
            width_file = os.path.join(device_path, "current_link_width")
            max_speed_file = os.path.join(device_path, "max_link_speed")
            max_width_file = os.path.join(device_path, "max_link_width")

            if not os.path.exists(speed_file):
                continue

            link_info: Dict[str, Any] = {
                "card": os.path.basename(card_path),
            }

            try:
                # Read current link speed
                with open(speed_file, "r") as f:
                    speed_str = f.read().strip()
                    # Parse "32.0 GT/s PCIe" -> 32.0
                    speed_match = re.match(r"(\d+(?:\.\d+)?)\s*GT/s", speed_str)
                    if speed_match:
                        speed_gts = float(speed_match.group(1))
                        link_info["speed_gts"] = speed_gts
                        link_info["speed"] = gen_map.get(speed_gts, f"{speed_gts}GT/s")

                # Read current link width
                with open(width_file, "r") as f:
                    width = int(f.read().strip())
                    link_info["width"] = width

                # Read max link speed
                if os.path.exists(max_speed_file):
                    with open(max_speed_file, "r") as f:
                        max_speed_str = f.read().strip()
                        max_speed_match = re.match(r"(\d+(?:\.\d+)?)\s*GT/s", max_speed_str)
                        if max_speed_match:
                            max_speed_gts = float(max_speed_match.group(1))
                            link_info["max_speed_gts"] = max_speed_gts
                            link_info["max_speed"] = gen_map.get(max_speed_gts, f"{max_speed_gts}GT/s")

                # Read max link width
                if os.path.exists(max_width_file):
                    with open(max_width_file, "r") as f:
                        link_info["max_width"] = int(f.read().strip())

                # Calculate bandwidth
                if "speed_gts" in link_info and "width" in link_info:
                    speed_gts = link_info["speed_gts"]
                    width = link_info["width"]
                    efficiency = 0.8 if speed_gts <= 5.0 else 0.985
                    bandwidth_gbps = speed_gts * width * efficiency / 8
                    link_info["bandwidth_gbps"] = round(bandwidth_gbps, 2)

                info["gpu_links"].append(link_info)

            except Exception:
                continue

        # Generate summary
        if info["gpu_links"]:
            speeds = [l.get("speed_gts", 0) for l in info["gpu_links"] if l.get("speed_gts")]
            widths = [l.get("width", 0) for l in info["gpu_links"] if l.get("width")]
            bandwidths = [l.get("bandwidth_gbps", 0) for l in info["gpu_links"] if l.get("bandwidth_gbps")]

            if speeds:
                min_speed = min(speeds)
                max_speed = max(speeds)
                info["summary"]["min_speed"] = gen_map.get(min_speed, f"{min_speed}GT/s")
                info["summary"]["max_speed"] = gen_map.get(max_speed, f"{max_speed}GT/s")
            if widths:
                info["summary"]["min_width"] = f"x{min(widths)}"
                info["summary"]["max_width"] = f"x{max(widths)}"
            if bandwidths:
                info["summary"]["total_bandwidth_gbps"] = round(sum(bandwidths), 2)
                info["summary"]["per_gpu_bandwidth_gbps"] = round(sum(bandwidths) / len(bandwidths), 2)

    except Exception:
        pass

    return info


def get_gpu_count_rocm() -> int:
    """Get GPU count using rocm-smi (more reliable in containers)."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Count lines that look like GPU entries (contain "GPU[")
            count = sum(1 for line in result.stdout.splitlines() if "GPU[" in line)
            if count > 0:
                return count
    except FileNotFoundError:
        pass
    except Exception:
        pass

    # Fallback: try HIP_VISIBLE_DEVICES or count /dev/dri/renderD* devices
    hip_devices = os.environ.get("HIP_VISIBLE_DEVICES", "")
    if hip_devices:
        return len([x for x in hip_devices.split(",") if x.strip()])

    # Count render devices
    try:
        import glob

        render_devices = glob.glob("/dev/dri/renderD*")
        if render_devices:
            return len(render_devices)
    except Exception:
        pass

    return 0


def get_pcie_topology() -> Optional[str]:
    """Get PCIe topology using lstopo or lspci -t."""
    try:
        # Try lstopo first (more detailed)
        result = subprocess.run(
            ["lstopo-no-graphics", "--of", "txt"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    except Exception:
        pass

    try:
        # Fallback to lspci -t
        result = subprocess.run(["lspci", "-t"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    except Exception:
        pass

    return None


def is_container() -> bool:
    """Check if running in a container."""
    # Check for /.dockerenv
    if os.path.exists("/.dockerenv"):
        return True
    # Check cgroup
    try:
        with open("/proc/1/cgroup", "r") as f:
            content = f.read()
            if "docker" in content or "kubepods" in content or "containerd" in content:
                return True
    except Exception:
        pass
    return False


def is_slurm_job() -> bool:
    """Check if running in a Slurm job."""
    return "SLURM_JOB_ID" in os.environ
