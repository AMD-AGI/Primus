#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
set -euxo pipefail

start_time=$(date +%s)

# Get directory of this script
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_PATH="$SCRIPT_PATH/../../"
TORCHTITAN_PATH="$PRIMUS_PATH/third_party/torchtitan/"

pip install --upgrade pip
pip install --no-cache-dir -r "$PRIMUS_PATH/requirements.txt"
pip install tomli_w

PYTEST_COV_ARGS=(
  --cov-branch
  --cov torchtitan
  --cov-append
  --no-cov-on-fail
  --cov-report term
  --cov-report xml
  --cov-report html
)

clear_previous_runs() {
  pgrep -f python | xargs -r -n 1 kill -9 2>/dev/null || true
  sleep 10
}

cd "$TORCHTITAN_PATH"
find . -name '.coverage*' -delete
if git apply --reverse --check "$SCRIPT_PATH/torchtitan_ut.patch"; then
  echo "Patch already applied, skipping."
else
  git apply "$SCRIPT_PATH/torchtitan_ut.patch"
fi

INIT_FILES=(
  torchtitan/experiments/deepseek_v3/unit_testing/__init__.py
  torchtitan/experiments/multimodal/tokenizer/__init__.py
  torchtitan/experiments/kernels/__init__.py
  torchtitan/experiments/kernels/moe/__init__.py
  torchtitan/experiments/kernels/moe/unit_tests/__init__.py
  torchtitan/experiments/kernels/triton_contiguous_group_gemm/__init__.py
  torchtitan/experiments/kernels/triton_mg_group_gemm/__init__.py
)

create_init_files() {
  cd "$TORCHTITAN_PATH"
  touch "${INIT_FILES[@]}"
}

remove_init_files() {
  cd "$TORCHTITAN_PATH"
  rm -f "${INIT_FILES[@]}"
}

trap remove_init_files EXIT

TEST_PATHS=(
  tests/unit_tests
  torchtitan/experiments/flux/tests/unit_tests
  torchtitan/experiments/simple_fsdp/tests
  torchtitan/experiments/multimodal/tests
  torchtitan/experiments/deepseek_v3/unit_testing/test_create_m_indices.py
  torchtitan/experiments/deepseek_v3/unit_testing/permute_indices_testing.py
  torchtitan/experiments/kernels/moe/unit_tests
  torchtitan/experiments/kernels/triton_contiguous_group_gemm/unit_test_cg.py
  torchtitan/experiments/kernels/triton_mg_group_gemm/torchao_pr/unit_test_backwards.py
  torchtitan/experiments/kernels/triton_mg_group_gemm/torchao_pr/unit_test_forwards.py
)

clear_previous_runs
create_init_files
pytest -vxrs "${PYTEST_COV_ARGS[@]}" \
  --deselect torchtitan/experiments/kernels/triton_mg_group_gemm/torchao_pr/unit_test_backwards.py::TestMG_GroupedGEMM_Backward \
  --deselect torchtitan/experiments/kernels/triton_mg_group_gemm/torchao_pr/unit_test_forwards.py::TestMG_GroupedGEMM \
  "${TEST_PATHS[@]}"

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total execution time: ${elapsed} seconds"
