###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the platform capture in tools/perf/lib/common.sh.

The perf_host_* functions run on every node of a batch and feed the system
description that results are quoted with, so these pin down its wording, the
host GPU driver decoding and the multi-node comparison. The library is
sourced under the runners' shell options (set -uo pipefail, nullglob) with
stdin held open, so an unset variable or a command left reading stdin fails
here instead of hanging a batch.

Pure CPU; needs bash, no GPU and no root.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LIB = REPO_ROOT / "tools" / "perf" / "lib"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def bash(script: str, timeout: int = 120) -> str:
    """Run ``script`` after sourcing lib/common.sh and lib/control.sh."""
    read_end, write_end = os.pipe()
    try:
        out = subprocess.run(
            [
                "bash",
                "-c",
                f"set -uo pipefail\nshopt -s nullglob\nsource {LIB}/common.sh\n"
                f"PERF_MULTINODE=${{PERF_MULTINODE:-0}}\nsource {LIB}/control.sh\n{script}",
            ],
            stdin=read_end,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PRIMUS_PATH": str(REPO_ROOT)},
            check=False,
        )
    finally:
        os.close(read_end)
        os.close(write_end)
    assert out.returncode == 0, out.stderr
    assert "unbound variable" not in out.stderr, out.stderr
    return out.stdout


# Every key the banner reads.
SETTINGS_KEYS = (
    "cpu_governor numa_balancing transparent_hugepages thp_defrag iommu amdgpu_driver kernel cpu_model "
    "cpus memory os kernel_cmdline server board bios cpu_sockets cpu_cores_per_socket cpu_threads_per_core "
    "numa_nodes cpu_frequency cpu_idle memory_installed memory_dimms rdma_nics gpu_count gpu_product "
    "gpu_vram gpu_power_cap gpu_partition gpu_clocks gpu_vbios gpu_pcie gpu_driver gpu_driver_package"
).split()

MI355X_SETTINGS = """\
cpu_governor: schedutil x256
amdgpu_driver: 6.14.14
cpu_model: AMD EPYC 9575F 64-Core Processor
cpus: 256
memory: 3023 GiB
os: Ubuntu 22.04.5 LTS
server: Supermicro AS -4126GS-NMR-LCC
bios: 1.7 (09/12/2025, American Megatrends International, LLC.)
cpu_sockets: 2
cpu_cores_per_socket: 64
cpu_threads_per_core: 2
numa_nodes: 2
memory_installed: 3072 GiB Registered-DDR5, 24 channels
memory_dimms: 128 GiB Samsung M321RAJA0MB2-CCPWF, 6400 MT/s, configured 6000 MT/s
gpu_count: 8
gpu_product: AMD Instinct MI355 OAM
gpu_vram: 288 GiB
gpu_power_cap: 1400 W
gpu_driver: amdgpu 30.10.1 (ROCm 7.0.1)
"""


def describe(settings: str, gpu_model: str = "AMD Instinct MI355X", vram_type: str = "HBM3E") -> str:
    """perf_system_description for canned settings, with amd-smi's answers stubbed."""
    return bash(
        f"PERF_HOST_SETTINGS=$(cat <<'EOF'\n{settings}EOF\n)\n"
        f"perf_gpu_model() {{ echo '{gpu_model}'; }}\n"
        f"perf_gpu_vram_type() {{ printf '%s' '{vram_type}'; }}\n"
        "perf_system_description\n"
    ).strip()


def test_system_description_reads_like_a_results_footnote():
    assert describe(MI355X_SETTINGS) == (
        "Dual AMD EPYC 9575F 64-Core Processor-based Supermicro AS -4126GS-NMR-LCC server with "
        "8x AMD Instinct MI355X (288GB HBM3E 1400W) GPUs, 1 NUMA node per socket, System BIOS 1.7, "
        "Ubuntu 22.04.5 LTS, Host GPU driver amdgpu 30.10.1 (ROCm 7.0.1)"
    )


def test_system_description_leaves_out_what_is_unknown():
    settings = (
        MI355X_SETTINGS.replace("cpu_sockets: 2", "cpu_sockets: 1")
        .replace("numa_nodes: 2", "numa_nodes: 4")
        .replace("bios: 1.7 (09/12/2025, American Megatrends International, LLC.)", "bios: unknown (Vultr)")
        .replace("server: Supermicro AS -4126GS-NMR-LCC", "server: unknown")
        .replace("gpu_power_cap: 1400 W", "gpu_power_cap: 1400 W x7, 1000 W x1")
    )
    # amd-smi sees no GPU (a Slurm step without them): sysfs' product name,
    # and no memory type.
    assert describe(settings, gpu_model="unknown", vram_type="") == (
        "Single AMD EPYC 9575F 64-Core Processor-based server with 8x AMD Instinct MI355 OAM "
        "(288GB 1400 W x7, 1000 W x1) GPUs, 4 NUMA nodes per socket, Ubuntu 22.04.5 LTS, "
        "Host GPU driver amdgpu 30.10.1 (ROCm 7.0.1)"
    )


def test_system_description_without_gpus():
    settings = re.sub(r"(?m)^gpu_.*\n", "", MI355X_SETTINGS) + "gpu_count: 0\ngpu_driver: not loaded\n"
    assert describe(settings) == (
        "Dual AMD EPYC 9575F 64-Core Processor-based Supermicro AS -4126GS-NMR-LCC server, "
        "1 NUMA node per socket, System BIOS 1.7, Ubuntu 22.04.5 LTS"
    )


@pytest.mark.parametrize(
    "package,rocm,expected",
    [
        ("1:6.14.14.30100100-2212064.22.04", "7.0.1", "amdgpu 30.10.1 (ROCm 7.0.1)"),
        ("1:6.14.14.30100101-2212064.22.04", "", "amdgpu 30.10.1.1"),
        # rpm: no epoch.
        ("6.16.6.30200000-2238411.el9", "7.1.0", "amdgpu 30.20.0 (ROCm 7.1.0)"),
        # Before the 30.x numbering the release was ROCm's.
        ("1:6.10.5.60401-2164967.22.04", "6.4.1", "ROCm 6.4.1"),
        ("1:6.10.5-custom", "", "amdgpu-dkms 1:6.10.5-custom"),
    ],
)
def test_host_gpu_driver_from_its_package(package, rocm, expected):
    assert bash(f"perf_host_gpu_driver '{package}' '{rocm}'").strip() == expected


def test_tally_names_the_most_common_value_first():
    assert bash("printf 'a\\nb\\na\\n\\n' | perf_host_tally").strip() == "a x2, b x1"
    assert bash("printf 'a\\na\\n' | perf_host_tally").strip() == "a"
    assert bash("printf '' | perf_host_tally") == ""
    assert bash("perf_host_field gpu_vram ''").strip() == "gpu_vram: unknown"


def test_this_hosts_settings_are_one_key_value_per_line():
    lines = bash("perf_host_settings").splitlines()
    keys = []
    for line in lines:
        m = re.match(r"^([a-z_]+): (.*)$", line)
        assert m, f"not a 'key: value' line: {line!r}"
        keys.append(m.group(1))
    assert len(keys) == len(set(keys)), "duplicate keys"
    assert set(SETTINGS_KEYS) <= set(keys), set(SETTINGS_KEYS) - set(keys)


def test_snapshot_section_is_the_settings_and_details_follow(tmp_path):
    snapshot, settings = tmp_path / "batch_host.txt", tmp_path / "settings.txt"
    bash(f"perf_host_snapshot > {snapshot}\nperf_host_settings > {settings}\n")
    section = bash(f'perf_host_section {snapshot} "$(hostname)"')
    # Values are live readings (clocks, NIC state) that can change between the two calls.
    assert [line.split(":", 1)[0] for line in section.splitlines()] == [
        line.split(":", 1)[0] for line in settings.read_text().splitlines()
    ]
    text = snapshot.read_text()
    assert text.startswith(f"== {socket.gethostname()} ==\n")
    for name in (
        "lscpu",
        "numa",
        "memory",
        "gpus (sysfs)",
        "rdma",
        "driver packages and repositories",
        "interfaces",
    ):
        assert f"\n-- {name} --\n" in text, name


SNAPSHOT = """\
== node-a ==
bios: 1.7
gpu_power_cap: 1400 W
-- lscpu --
CPU MHz: 1500
== node-b ==
bios: 1.8
gpu_power_cap: 1400 W
-- lscpu --
CPU MHz: 3300
== node-c ==
bios: 1.7
gpu_power_cap: 1400 W
-- lscpu --
CPU MHz: 2000

== other output ==
srun: warning: something
"""


def test_nodes_that_differ_are_named(tmp_path):
    path = tmp_path / "batch_host_x.txt"
    path.write_text(SNAPSHOT)
    # The raw sections differ too, but only settings are compared.
    assert bash(f"perf_host_differences {path}").strip() == "bios: 1.7 (2 nodes); 1.8 (1 node)"
    assert bash(f"perf_host_section {path} node-b") == "bios: 1.8\ngpu_power_cap: 1400 W\n"
    # A runner outside the allocation gets the first node's.
    assert bash(f"perf_host_section {path} login-1").startswith("bios: 1.7\n")


def test_multinode_host_check(tmp_path):
    path = tmp_path / "batch_host_x.txt"
    path.write_text(SNAPSHOT)
    out = bash(
        "PERF_MULTINODE=1 NNODES=4\n"
        "perf_gpu_model() { echo unknown; }\nperf_gpu_vram_type() { :; }\n"
        f"perf_load_host_settings {path} >/dev/null\n"
        'echo "$PERF_HOST_CHECK"\n'
    )
    assert out.strip() == "differs between nodes: bios (see batch_host_x.txt); only 3 of 4 nodes reported"
    assert path.read_text().endswith("== differences between nodes ==\nbios: 1.7 (2 nodes); 1.8 (1 node)\n")

    path.write_text(SNAPSHOT.replace("bios: 1.8", "bios: 1.7"))
    out = bash(
        "PERF_MULTINODE=1 NNODES=3\n"
        "perf_gpu_model() { echo unknown; }\nperf_gpu_vram_type() { :; }\n"
        f'perf_load_host_settings {path}\necho "$PERF_HOST_CHECK"\n'
    )
    assert out.strip() == "same on all 3 nodes"
