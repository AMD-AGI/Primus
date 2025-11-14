#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Run all Primus CLI tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source common library for logging
# shellcheck disable=SC1091
source "$PROJECT_ROOT/runner/lib/common.sh"

export NODE_RANK=0

echo "========================================="
echo "  Primus CLI Test Suite Runner"
echo "========================================="
echo ""

# Test results
TOTAL_SUITES=0
PASSED_SUITES=0
FAILED_SUITES=0

# Find all test scripts
TEST_SCRIPTS=(
    "$SCRIPT_DIR/lib/test_common.sh"
    "$SCRIPT_DIR/lib/test_validation.sh"
    "$SCRIPT_DIR/lib/test_config.sh"
    "$SCRIPT_DIR/helpers/test_execute_hooks.sh"
    "$SCRIPT_DIR/helpers/test_execute_patches.sh"
    "$SCRIPT_DIR/helpers/test_primus_env.sh"
    "$SCRIPT_DIR/test_primus_cli.sh"
    "$SCRIPT_DIR/test_primus_cli_slurm.sh"
    "$SCRIPT_DIR/test_primus_cli_container.sh"
    "$SCRIPT_DIR/test_primus_cli_direct.sh"
)

# Run each test suite
for test_script in "${TEST_SCRIPTS[@]}"; do
    if [[ ! -f "$test_script" ]]; then
        LOG_WARN "Test script not found: $test_script (skipping)"
        continue
    fi

    ((TOTAL_SUITES++)) || true

    test_name=$(basename "$test_script")
    LOG_INFO "========================================="
    LOG_INFO "Running: $test_name"
    LOG_INFO "========================================="
    echo ""

    if bash "$test_script"; then
        LOG_SUCCESS "✓ $test_name PASSED"
        ((PASSED_SUITES++)) || true
    else
        LOG_ERROR "✗ $test_name FAILED"
        ((FAILED_SUITES++)) || true
    fi
    echo ""
    echo ""
done

# Final summary
echo "========================================="
echo "  Final Test Results"
echo "========================================="
LOG_INFO "Total test suites: $TOTAL_SUITES"
LOG_SUCCESS "Passed: $PASSED_SUITES"
if [[ "$FAILED_SUITES" -gt 0 ]]; then
    LOG_ERROR "Failed: $FAILED_SUITES"
else
    LOG_INFO "Failed: $FAILED_SUITES"
fi
echo "========================================="

if [[ "$FAILED_SUITES" -eq 0 ]]; then
    LOG_SUCCESS "🎉 All test suites passed! ✓"
    exit 0
else
    LOG_ERROR "❌ Some test suites failed! ✗"
    exit 1
fi
