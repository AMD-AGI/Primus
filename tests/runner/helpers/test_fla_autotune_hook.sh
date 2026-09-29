#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Skip-path tests for the FLA autotune pretrain hook. Matching GDN/Hylo/KDA
# configs are not executed here because the patch mutates installed FLA.
###############################################################################
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HOOK="$PROJECT_ROOT/runner/helpers/hooks/train/pretrain/z_patch_fla_triton_autotune.sh"

TESTS_RUN=0
TESTS_PASSED=0

assert_eq() {
    TESTS_RUN=$((TESTS_RUN + 1))
    if [[ "$1" == "$2" ]]; then
        echo "  ✓ PASS: $3"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo "  ✗ FAIL: $3 (expected $2, got $1)"
    fi
}

assert_not_contains() {
    TESTS_RUN=$((TESTS_RUN + 1))
    if ! grep -q "$2" <<<"$1"; then
        echo "  ✓ PASS: $3"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo "  ✗ FAIL: $3"
    fi
}

echo "FLA autotune hook skip-path tests"

out="$(bash "$HOOK" train pretrain --config examples/megatron/configs/MI355X/llama3.1_8B-MXFP4-pretrain.yaml 2>&1)"
assert_eq "$?" "0" "non-hybrid config exits 0"
assert_not_contains "$out" "applying FLA" "non-hybrid config does not apply the patch"

out="$(bash "$HOOK" train pretrain 2>&1)"
assert_eq "$?" "0" "missing --config exits 0"

echo ""
echo "Result: $TESTS_PASSED / $TESTS_RUN passed"
[[ "$TESTS_PASSED" -eq "$TESTS_RUN" ]]
