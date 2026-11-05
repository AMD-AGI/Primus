# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Training test launches must not reuse a parent process's rendezvous port."""

import os
import re
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from tests.utils import isolated_torchrun_env


def test_concurrent_launches_with_occupied_inherited_port(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import os, torch\n"
        "import torch.distributed as dist\n"
        "dist.init_process_group('gloo')\n"
        "value = torch.tensor([dist.get_rank() + int(os.environ['TEST_JOB']) * 10])\n"
        "dist.all_reduce(value)\n"
        "assert value.item() == 1 + int(os.environ['TEST_JOB']) * 20\n"
        "print('DIST_PORT=' + os.environ['MASTER_PORT'], flush=True)\n"
        "dist.destroy_process_group()\n"
    )
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        parent_env = dict(os.environ, NNODES="1", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        original_env = parent_env.copy()

        def launch(job):
            env = isolated_torchrun_env(parent_env)
            env["TEST_JOB"] = str(job)
            return subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--nnodes=1",
                    "--nproc_per_node=2",
                    "--master_addr=127.0.0.1",
                    f"--master_port={port}",
                    str(worker),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(launch, (0, 1)))

        for result in results:
            assert result.returncode == 0, result.stdout + result.stderr
            ports = re.findall(r"DIST_PORT=(\d+)", result.stdout)
            assert len(ports) == 2, result.stdout
            assert len(set(ports)) == 1
            assert ports[0] != str(port)
        assert parent_env == original_env


def test_multinode_rendezvous_is_preserved():
    env = {"NNODES": "2", "MASTER_ADDR": "node0", "MASTER_PORT": "29500"}
    assert isolated_torchrun_env(env) == env


def test_slurm_multinode_rendezvous_is_preserved():
    env = {"SLURM_NNODES": "2", "MASTER_ADDR": "node0", "MASTER_PORT": "29500"}
    assert isolated_torchrun_env(env) == env
