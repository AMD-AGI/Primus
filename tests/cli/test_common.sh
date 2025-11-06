#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Test script for Primus CLI common library

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source common library
export PRIMUS_LOG_COLOR=0  # Disable colors in tests
# shellcheck disable=SC1091
source "$PROJECT_ROOT/runner/lib/common.sh"

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Test helper function
test_function() {
    local name="$1"
    local expected="$2"
    local actual="$3"

    if [[ "$expected" == "$actual" ]]; then
        LOG_SUCCESS "✓ $name: PASSED"
        ((TESTS_PASSED++)) || true
    else
        LOG_ERROR "✗ $name: FAILED (expected: $expected, got: $actual)"
        ((TESTS_FAILED++)) || true
    fi
}

echo "========================================="
echo "  Primus CLI Common Library Test Suite"
echo "========================================="
echo ""

# Test 1: Logging functions
LOG_INFO "Test 1: Testing logging functions..."
LOG_DEBUG "This is a debug message"
LOG_INFO "This is an info message"
LOG_WARN "This is a warning message"
LOG_ERROR "This is an error message (expected)"
LOG_SUCCESS "This is a success message"
test_function "Logging functions" "pass" "pass"
echo ""

# Test 2: Path utilities
LOG_INFO "Test 2: Testing path utilities..."
TEST_DIR="/tmp/primus-test-$$"
mkdir -p "$TEST_DIR"
ensure_dir "$TEST_DIR/subdir"
if [[ -d "$TEST_DIR/subdir" ]]; then
    test_function "ensure_dir" "pass" "pass"
else
    test_function "ensure_dir" "pass" "fail"
fi
rm -rf "$TEST_DIR"
echo ""

# Test 3: String utilities
LOG_INFO "Test 3: Testing string utilities..."
result=$(trim "  hello world  ")
test_function "trim" "hello world" "$result"

if contains "hello world" "world"; then
    test_function "contains" "pass" "pass"
else
    test_function "contains" "pass" "fail"
fi
echo ""

# Test 4: System utilities
LOG_INFO "Test 4: Testing system utilities..."
cpu_count=$(get_cpu_count)
if [[ "$cpu_count" -gt 0 ]]; then
    test_function "get_cpu_count" "pass" "pass"
    LOG_INFO "  CPU count: $cpu_count"
else
    test_function "get_cpu_count" "pass" "fail"
fi

mem_gb=$(get_memory_gb)
if [[ "$mem_gb" -gt 0 ]]; then
    test_function "get_memory_gb" "pass" "pass"
    LOG_INFO "  Memory: ${mem_gb}GB"
else
    test_function "get_memory_gb" "pass" "fail"
fi
echo ""

# Test 5: Environment utilities
LOG_INFO "Test 5: Testing environment utilities..."
export TEST_VAR="test_value"
set_default "TEST_VAR" "default_value"
test_function "set_default (existing)" "test_value" "$TEST_VAR"

set_default "NEW_VAR" "new_value"
test_function "set_default (new)" "new_value" "$NEW_VAR"
echo ""

# Test 6: Command validation
LOG_INFO "Test 6: Testing command validation..."
if require_command "bash" 2>/dev/null; then
    test_function "require_command (exists)" "pass" "pass"
else
    test_function "require_command (exists)" "pass" "fail"
fi

# Test require_command with missing command in subshell to avoid script exit
if (require_command "nonexistent_command_12345" 2>/dev/null); then
    test_function "require_command (missing)" "pass" "fail"
else
    test_function "require_command (missing)" "pass" "pass"
fi
echo ""

# Test 7: Environment file loading
LOG_INFO "Test 7: Testing environment file loading..."
TEST_ENV_FILE="/tmp/test-env-$$"
cat > "$TEST_ENV_FILE" << EOF
# Test environment file
TEST_KEY1=value1
TEST_KEY2="value with spaces"
# Comment line
TEST_KEY3=value3
EOF

load_env_file "$TEST_ENV_FILE" >/dev/null 2>&1
if [[ "${TEST_KEY1:-}" == "value1" ]] && [[ "${TEST_KEY2:-}" == "value with spaces" ]] && [[ "${TEST_KEY3:-}" == "value3" ]]; then
    test_function "load_env_file" "pass" "pass"
else
    test_function "load_env_file" "pass" "fail"
fi
rm -f "$TEST_ENV_FILE"
echo ""

# Test 8: Log formatting
LOG_INFO "Test 8: Testing log formatting..."
export NODE_RANK=0
log_exported_vars "Test Variables" TEST_KEY1 TEST_KEY2 TEST_KEY3
test_function "log_exported_vars" "pass" "pass"
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
    LOG_SUCCESS "All tests passed! ✓"
    exit 0
else
    LOG_ERROR "Some tests failed! ✗"
    exit 1
fi
