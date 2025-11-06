#!/bin/bash
###############################################################################
# Test script for primus-cli main entry
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source common library
# shellcheck disable=SC1091
source "$PROJECT_ROOT/runner/lib/common.sh"

export NODE_RANK=0

echo "========================================="
echo "  Primus CLI Main Entry Tests"
echo "========================================="
echo ""

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

test_pass() {
    LOG_SUCCESS "✓ $1"
    ((TESTS_PASSED++)) || true
}

test_fail() {
    LOG_ERROR "✗ $1"
    ((TESTS_FAILED++)) || true
}

# Test 1: Help option
LOG_INFO "Test 1: Testing --help option..."
if bash "$PROJECT_ROOT/runner/primus-cli" --help 2>&1 | grep -q "Primus Unified Launcher CLI"; then
    test_pass "--help shows usage"
else
    test_fail "--help shows usage"
fi
echo ""

# Test 2: Version option
LOG_INFO "Test 2: Testing --version option..."
if bash "$PROJECT_ROOT/runner/primus-cli" --version 2>&1 | grep -q "Primus CLI version"; then
    test_pass "--version shows version"
else
    test_fail "--version shows version"
fi
echo ""

# Test 3: No arguments
LOG_INFO "Test 3: Testing no arguments (should show help)..."
if bash "$PROJECT_ROOT/runner/primus-cli" 2>&1 | grep -q "Usage:"; then
    test_pass "No arguments shows help"
else
    test_fail "No arguments shows help"
fi
echo ""

# Test 4: Unknown mode
LOG_INFO "Test 4: Testing unknown mode..."
if bash "$PROJECT_ROOT/runner/primus-cli" unknown-mode 2>&1 | grep -q "Unknown mode"; then
    test_pass "Unknown mode shows error"
else
    test_fail "Unknown mode shows error"
fi
echo ""

# Test 5: Dry-run mode
LOG_INFO "Test 5: Testing --dry-run option..."
if bash "$PROJECT_ROOT/runner/primus-cli" --dry-run direct --help 2>&1 | grep -q "DRY-RUN"; then
    test_pass "--dry-run shows dry-run message"
else
    test_fail "--dry-run shows dry-run message"
fi
echo ""

# Test 6: Debug mode
LOG_INFO "Test 6: Testing --debug option..."
export NODE_RANK=0
if bash "$PROJECT_ROOT/runner/primus-cli" --debug direct --help 2>&1 | grep -q "+"; then
    test_pass "--debug enables trace mode"
else
    test_fail "--debug enables trace mode"
fi
echo ""

# Test 7: Log level
LOG_INFO "Test 7: Testing --log-level option..."
if bash "$PROJECT_ROOT/runner/primus-cli" --log-level DEBUG direct --help 2>&1 | grep -q "Primus Direct"; then
    test_pass "--log-level DEBUG works"
else
    test_fail "--log-level DEBUG works"
fi
echo ""

# Test 8: Direct mode help
LOG_INFO "Test 8: Testing direct mode help..."
if bash "$PROJECT_ROOT/runner/primus-cli" direct --help 2>&1 | grep -q "Primus Direct Launcher"; then
    test_pass "direct --help works"
else
    test_fail "direct --help works"
fi
echo ""

# Test 9: Container mode help
LOG_INFO "Test 9: Testing container mode help..."
if bash "$PROJECT_ROOT/runner/primus-cli" container --help 2>&1 | grep -q "Docker/Podman container"; then
    test_pass "container --help works"
else
    test_fail "container --help works"
fi
echo ""

# Test 10: Slurm mode help
LOG_INFO "Test 10: Testing slurm mode help..."
if bash "$PROJECT_ROOT/runner/primus-cli" slurm --help 2>&1 | grep -q "Primus Slurm Launcher"; then
    test_pass "slurm --help works"
else
    test_fail "slurm --help works"
fi
echo ""

# Summary
echo "========================================="
echo "  Test Summary"
echo "========================================="
LOG_SUCCESS "Passed: $TESTS_PASSED"
if [[ "$TESTS_FAILED" -gt 0 ]]; then
    LOG_ERROR "Failed: $TESTS_FAILED"
else
    LOG_INFO "Failed: $TESTS_FAILED"
fi
echo "Total: $((TESTS_PASSED + TESTS_FAILED))"
echo "========================================="

if [[ "$TESTS_FAILED" -eq 0 ]]; then
    LOG_SUCCESS "All primus-cli tests passed! ✓"
    exit 0
else
    LOG_ERROR "Some primus-cli tests failed! ✗"
    exit 1
fi
