###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LIFECYCLE = ROOT / "runner" / "lib" / "container_lifecycle.sh"


@pytest.fixture
def runtime(tmp_path):
    fake = tmp_path / "runtime"
    fake.write_text(
        """#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
root = Path(os.environ['FAKE_RUNTIME_ROOT'])
args = sys.argv[1:]
cid = 'a' * 64
with (root / 'calls').open('a') as log:
    log.write(json.dumps(args) + '\\n')
if args[0] == 'run':
    (root / 'container').write_text(cid)
    (root / 'started').touch()
    mode = os.environ['FAKE_RUNTIME_MODE']
    if mode == 'wait':
        while True:
            time.sleep(0.1)
    # Detached mode leaves a running container after a successful client exit.
    if mode != 'detach':
        (root / 'container').unlink()
    sys.exit(17 if mode == 'fail' else 0)
elif args[0] == 'ps':
    assert args == ['ps', '-aq', '--filter', 'label=io.primus.launch=test-launch']
    if (root / 'container').exists():
        print(cid)
elif args[0] == 'rm':
    assert args == ['rm', '-f', cid], 'Must never remove an unrelated container'
    (root / 'container').unlink()
else:
    raise AssertionError(args)
""",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return fake


def launch(runtime, tmp_path, mode):
    env = os.environ.copy()
    env.update(FAKE_RUNTIME_ROOT=str(tmp_path), FAKE_RUNTIME_MODE=mode)
    return subprocess.Popen(
        [
            "/bin/bash",
            "-c",
            'set -euo pipefail; source "$1"; '
            'run_container_with_cleanup test-launch "$2" run '
            "--label io.primus.launch=test-launch example-image",
            "bash",
            str(LIFECYCLE),
            str(runtime),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


@pytest.mark.parametrize("mode,expected", [("success", 0), ("fail", 17), ("detach", 0)])
def test_container_client_status_and_detached_launch(runtime, tmp_path, mode, expected):
    process = launch(runtime, tmp_path, mode)
    _, stderr = process.communicate(timeout=10)
    assert process.returncode == expected, stderr
    assert (tmp_path / "container").exists() == (mode == "detach")


@pytest.mark.parametrize(
    "signum,expected", [(signal.SIGTERM, 143), (signal.SIGINT, 130), (signal.SIGHUP, 129)]
)
def test_signal_removes_only_this_launch_container(runtime, tmp_path, signum, expected):
    process = launch(runtime, tmp_path, "wait")
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "started").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (tmp_path / "started").exists(), "Fake container did not start"
        process.send_signal(signum)
        _, stderr = process.communicate(timeout=10)
        assert process.returncode == expected, stderr
        assert not (tmp_path / "container").exists()
        assert '["rm", "-f",' in (tmp_path / "calls").read_text()
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            process.communicate(timeout=5)
