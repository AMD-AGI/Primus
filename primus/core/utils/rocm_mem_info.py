###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import re
import subprocess


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


def get_rocm_smi_gpu_util(device_id: int) -> float:
    try:
        out = subprocess.check_output(["rocm-smi", "--showuse", f"-d={device_id}"], text=True)
    except FileNotFoundError:
        raise RuntimeError("rocm-smi not found, please ensure ROCm is installed and in PATH")

    match = re.search(r"GPU\s*use.*?:\s*([0-9]+)", out, re.IGNORECASE)
    if not match:
        raise RuntimeError("Unable to parse GPU utilization from rocm-smi output")
    return float(match.group(1))
