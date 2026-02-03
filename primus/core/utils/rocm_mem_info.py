###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import re
import subprocess
from typing import Optional


def get_rocm_smi_mem_info(device_id: int):
    try:
        out = subprocess.check_output(["rocm-smi", "--showmeminfo", "vram", f"-d={device_id}"], text=True)
    except FileNotFoundError:
        raise RuntimeError("rocm-smi not found, please ensure ROCm is installed and in PATH")

    # mem in Bytes
    total_mem, used_mem = None, None
    for line in out.splitlines():
        if "Total Memory" in line:
            total_mem = int(line.split(":")[-1].strip())
        elif "Total Used Memory" in line:
            used_mem = int(line.split(":")[-1].strip())

    assert total_mem is not None
    assert used_mem is not None
    free_mem = total_mem - used_mem

    return total_mem, used_mem, free_mem


def get_rocm_smi_gpu_util(device_id: int) -> Optional[float]:
    """
    Get GPU utilization percentage from rocm-smi.

    Args:
        device_id: The GPU device ID to query

    Returns:
        GPU utilization as a float (0-100), or None if unable to parse
    """
    try:
        out = subprocess.check_output(["rocm-smi", "--showuse", f"-d={device_id}"], text=True)
    except FileNotFoundError:
        raise RuntimeError("rocm-smi not found, please ensure ROCm is installed and in PATH")
    except subprocess.CalledProcessError:
        return None

    match = re.search(r"GPU\s*use.*?:\s*([0-9]+)", out, re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))
