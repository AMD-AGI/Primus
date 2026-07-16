###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import re
import subprocess


def get_rocm_smi_gpu_util(device_id: int):
    """
    Return current GPU utilization (0-100) for the given device via rocm-smi --showuse.

    Returns:
        float: GPU use percentage (0-100), or raises on failure (caller should catch and use fallback).
    """
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showuse", f"-d={device_id}"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except FileNotFoundError:
        raise RuntimeError("rocm-smi not found, please ensure ROCm is installed and in PATH")
    except subprocess.TimeoutExpired:
        raise RuntimeError("rocm-smi --showuse timed out")
    except subprocess.CalledProcessError as e:
        output = e.output.strip() if isinstance(e.output, str) and e.output else "No output captured."
        raise RuntimeError(f"rocm-smi --showuse failed with exit code {e.returncode}. Output: {output}")

    # Parse output: look for GPU use (%) or similar (e.g. "GPU use (%): 42" or "GPU Use: 42%")
    for line in out.splitlines():
        line_lower = line.lower()
        if "use" not in line_lower and "busy" not in line_lower:
            continue
        # Prefer a number that follows a use/busy label.
        labeled_match = re.search(
            r"\b(?:use|busy)\b[^0-9%]*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
            line_lower,
        )
        if labeled_match:
            val = float(labeled_match.group(1))
            if 0 <= val <= 100:
                return val

        # Otherwise, take the last percentage on the line to avoid grabbing GPU index.
        percent_numbers = re.findall(r"(\d+(?:\.\d+)?)\s*%", line)
        for n in reversed(percent_numbers):
            val = float(n)
            if 0 <= val <= 100:
                return val

    raise RuntimeError(f"rocm-smi --showuse did not report a GPU use percentage for device {device_id}")


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
