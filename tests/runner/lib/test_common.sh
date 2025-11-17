#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Unit tests for Primus CLI common library

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Source common library (needed for testing)
export PRIMUS_LOG_COLOR=0  # Disable colors
# shellcheck disable=SC1091
source "$PROJECT_ROOT/runner/lib/common.sh"

# Test helper functions
assert_equals() {
    local expected="$1"
    local actual="$2"
    local test_name="$3"

    ((TESTS_RUN++))

    if [[ "$expected" == "$actual" ]]; then
        echo -e "${GREEN}✓${NC} $test_name"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $test_name"
        echo "  Expected: $expected"
        echo "  Actual:   $actual"
        ((TESTS_FAILED++))
        return 1
    fi
}

assert_pass() {
    local test_name="$1"
    ((TESTS_RUN++))
    echo -e "${GREEN}✓${NC} $test_name"
    ((TESTS_PASSED++))
}

assert_fail() {
    local test_name="$1"
    local reason="${2:-}"
    ((TESTS_RUN++))
    echo -e "${RED}✗${NC} $test_name"
    if [[ -n "$reason" ]]; then
        echo "  Reason: $reason"
    fi
    ((TESTS_FAILED++))
}

assert_contains() {
    local haystack="$1"
    local needle="$2"
    local test_name="$3"

    ((TESTS_RUN++))

    if echo "$haystack" | grep -qF -- "$needle"; then
        echo -e "${GREEN}✓${NC} $test_name"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $test_name"
        echo "  Expected to contain: $needle"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Print test section header
print_section() {
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# ============================================================================
# Test 1: Logging functions
# ============================================================================
test_logging_functions() {
    print_section "Test 1: Logging Functions"

    # Just verify functions exist and don't crash
    if LOG_INFO "Test message" >/dev/null 2>&1; then
        assert_pass "LOG_INFO exists and works"
    else
        assert_fail "LOG_INFO exists and works"
    fi

    if LOG_SUCCESS "Success message" >/dev/null 2>&1; then
        assert_pass "LOG_SUCCESS exists and works"
    else
        assert_fail "LOG_SUCCESS exists and works"
    fi
}

# ============================================================================
# Test 2: Path utilities
# ============================================================================
test_path_utilities() {
    print_section "Test 2: Path Utilities"

    local test_dir="/tmp/primus-test-$$"
    mkdir -p "$test_dir"

    if ensure_dir "$test_dir/subdir" 2>/dev/null && [[ -d "$test_dir/subdir" ]]; then
        assert_pass "ensure_dir creates directories"
    else
        assert_fail "ensure_dir creates directories"
    fi

    rm -rf "$test_dir"
}

# ============================================================================
# Test 3: String utilities
# ============================================================================
test_string_utilities() {
    print_section "Test 3: String Utilities"

    local result
    result=$(trim "  hello world  " 2>/dev/null)
    assert_equals "hello world" "$result" "trim removes whitespace"

    if contains "hello world" "world" 2>/dev/null; then
        assert_pass "contains finds substring"
    else
        assert_fail "contains finds substring"
    fi

    if ! contains "hello world" "xyz" 2>/dev/null; then
        assert_pass "contains rejects non-substring"
    else
        assert_fail "contains rejects non-substring"
    fi
}

# ============================================================================
# Test 4: System utilities
# ============================================================================
test_system_utilities() {
    print_section "Test 4: System Utilities"

    local cpu_count
    cpu_count=$(get_cpu_count 2>/dev/null)
    if [[ "$cpu_count" -gt 0 ]]; then
        assert_pass "get_cpu_count returns positive value ($cpu_count CPUs)"
    else
        assert_fail "get_cpu_count returns positive value"
    fi

    local mem_gb
    mem_gb=$(get_memory_gb 2>/dev/null)
    if [[ "$mem_gb" -gt 0 ]]; then
        assert_pass "get_memory_gb returns positive value (${mem_gb}GB)"
    else
        assert_fail "get_memory_gb returns positive value"
    fi
}

# ============================================================================
# Test 5: Environment utilities
# ============================================================================
test_environment_utilities() {
    print_section "Test 5: Environment Utilities"

    export TEST_VAR="test_value"
    set_default "TEST_VAR" "default_value" 2>/dev/null
    assert_equals "test_value" "$TEST_VAR" "set_default preserves existing value"

    set_default "NEW_VAR" "new_value" 2>/dev/null
    assert_equals "new_value" "$NEW_VAR" "set_default sets new value"
}

# ============================================================================
# Test 6: Command validation
# ============================================================================
test_command_validation() {
    print_section "Test 6: Command Validation"

    if require_command "bash" 2>/dev/null; then
        assert_pass "require_command accepts existing command"
    else
        assert_fail "require_command accepts existing command"
    fi

    # Test in subshell to avoid script exit
    if (require_command "nonexistent_command_12345" 2>/dev/null); then
        assert_fail "require_command rejects non-existent command"
    else
        assert_pass "require_command rejects non-existent command"
    fi
}

# ============================================================================
# Test 7: Environment file loading
# ============================================================================
test_env_file_loading() {
    print_section "Test 7: Environment File Loading"

    local test_env="/tmp/test-env-$$"
    cat > "$test_env" << 'EOF'
# Test environment file
TEST_KEY1=value1
TEST_KEY2="value with spaces"
# Comment line
TEST_KEY3=value3
EOF

    load_env_file "$test_env" >/dev/null 2>&1

    if [[ "${TEST_KEY1:-}" == "value1" ]] && \
       [[ "${TEST_KEY2:-}" == "value with spaces" ]] && \
       [[ "${TEST_KEY3:-}" == "value3" ]]; then
        assert_pass "load_env_file loads environment variables"
    else
        assert_fail "load_env_file loads environment variables"
    fi

    rm -f "$test_env"
}

# ============================================================================
# Test 8: Exported variables logging
# ============================================================================
test_exported_vars_logging() {
    print_section "Test 8: Exported Variables Logging"

    export NODE_RANK=0
    local output
    output=$(log_exported_vars "Test Variables" TEST_KEY1 TEST_KEY2 TEST_KEY3 2>&1)

    assert_contains "$output" "Test Variables" "log_exported_vars shows title"
    assert_contains "$output" "TEST_KEY1" "log_exported_vars shows variable names"
}

# ============================================================================
# Run all tests
# ============================================================================
main() {
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Unit Tests for Primus CLI Common Library                   ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    test_logging_functions
    test_path_utilities
    test_string_utilities
    test_system_utilities
    test_environment_utilities
    test_command_validation
    test_env_file_loading
    test_exported_vars_logging

    # Print summary
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Test Summary:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Total:  $TESTS_RUN"
    echo -e "  Passed: ${GREEN}$TESTS_PASSED${NC}"
    if [[ $TESTS_FAILED -gt 0 ]]; then
        echo -e "  Failed: ${RED}$TESTS_FAILED${NC}"
    else
        echo "  Failed: 0"
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}✓ All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}✗ Some tests failed${NC}"
        return 1
    fi
}

# Run tests
main
