#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# System hook: enable build uccl settings.
#
# Trigger:
#   export REBUILD_UEP=1
#
###############################################################################

set -euo pipefail

if [[ "${REBUILD_UEP:-0}" != "1" ]]; then
    exit 0
fi

UCCL_DIR="/tmp/uccl"
UCCL_BUILD_DIR="${UCCL_BUILD_DIR:-/tmp/uccl_${HOSTNAME:-$(hostname)}}"
UCCL_REF="${UCCL_REF:-b99aefd263bf89ed79dbb3207dde5a5b979ea530}" # [EP] Replace all __shfl_sync with shared memory broadcasts

LOG_INFO_RANK0 "[hook system] REBUILD_UEP=1 → Building uccl in /tmp "
LOG_INFO_RANK0 "  Build directory : ${UCCL_BUILD_DIR}"

if [ -d "$UCCL_DIR" ]; then
	LOG_INFO_RANK0 "[hook system] Found existed uccl in /tmp, remove it"
	rm -rf $UCCL_DIR
fi

cd /tmp && git clone https://github.com/uccl-project/uccl.git


pushd $UCCL_DIR

# install dependencies
apt update && apt install -y rdma-core libibverbs-dev libnuma-dev libgoogle-glog-dev
python3 -m pip install --no-cache-dir nanobind

if [[ -n "$UCCL_REF" ]]; then
	LOG_INFO_RANK0 "Checking out UCCL ref: ${UCCL_REF}"
    git fetch --all --tags
    git checkout "${UCCL_REF}"
fi

python3 - <<'PY'
import re
from pathlib import Path

pattern = re.compile(
    r"((?:send_head|send_nvl_head)\.data_ptr\(\),\n)"
    r"(\s*)None,\n"
    r"(\s*)async_finish,\n"
    r"(\s*)allocate_on_comm_stream,"
)
replacement = (
    r'\1\2getattr(previous_event, "event", None),\n'
    r"\3async_finish,\n"
    r"\4allocate_on_comm_stream,"
)

candidates = [
    Path("ep/deep_ep_wrapper/deep_ep/buffer.py"),
    Path("ep/bench/buffer.py"),
]

patched_total = 0
incomplete = []

for path in candidates:
    if not path.exists():
        continue

    text = path.read_text()
    updated, count = pattern.subn(replacement, text)
    if count:
        path.write_text(updated)
        print(f"[hook system] Patched {count} DeepEP previous_event call site(s) in {path}")
        patched_total += count

    current = updated if count else text
    if pattern.search(current):
        incomplete.append(str(path))

if incomplete:
    raise SystemExit(
        "[hook system] DeepEP previous_event compatibility patch is incomplete for: "
        + ", ".join(incomplete)
    )

if patched_total == 0:
    print("[hook system] No DeepEP previous_event compatibility patch needed")
else:
    print(f"[hook system] Patched {patched_total} DeepEP previous_event call site(s) in total")
PY

LOG_INFO_RANK0 "[hook system] Building uccl ep"
cd ep && python3 setup.py build && cd ..

LOG_INFO_RANK0 "[hook system] Building uccl ep done"

cp ep/build/**/*.so uccl

pip3 install --no-build-isolation .
LOG_INFO_RANK0 "[hook system] Install uccl done"
# install deep_ep_wrapper
cd $UCCL_DIR/ep/deep_ep_wrapper
pip3 install --no-build-isolation .
LOG_INFO_RANK0 "[hook system] Install deep_ep done"

LOG_INFO_RANK0 "[hook system] Building uccl done."

popd
