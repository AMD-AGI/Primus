#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Unit tests for Primus CLI validation library

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Source libraries (needed for testing)
export PRIMUS_LOG_COLOR=0  # Disable colors
# shellcheck disable=SC1091
source "$PROJECT_ROOT/runner/lib/common.sh"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/runner/lib/validation.sh"

# Test helper functions
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
# Test 1: Distributed parameters validation
# ============================================================================
test_distributed_params() {
    print_section "Test 1: Distributed Parameters Validation"

    export NNODES=2
    export NODE_RANK=0
    export GPUS_PER_NODE=8
    export MASTER_ADDR="localhost"
    export MASTER_PORT=1234

    if validate_distributed_params 2>/dev/null; then
        assert_pass "Valid distributed parameters accepted"
    else
        assert_fail "Valid distributed parameters accepted"
    fi
}

# ============================================================================
# Test 2: GPUS_PER_NODE validation
# ============================================================================
test_gpus_per_node_validation() {
    print_section "Test 2: GPUS_PER_NODE Validation"

    export GPUS_PER_NODE=8
    if validate_gpus_per_node 2>/dev/null; then
        assert_pass "Valid GPUS_PER_NODE=8 accepted"
    else
        assert_fail "Valid GPUS_PER_NODE=8 accepted"
    fi

    export GPUS_PER_NODE=10
    local output
    output=$(validate_gpus_per_node 2>&1)
    assert_contains "$output" "must be between" "Invalid GPUS_PER_NODE rejected"

    export GPUS_PER_NODE=8  # Reset
}

# ============================================================================
# Test 3: Integer validation
# ============================================================================
test_integer_validation() {
    print_section "Test 3: Integer Validation"

    if validate_integer "123" "test_value" 2>/dev/null; then
        assert_pass "Valid integer accepted"
    else
        assert_fail "Valid integer accepted"
    fi

    if ! (validate_integer "abc" "test_value" 2>/dev/null); then
        assert_pass "Non-integer correctly rejected"
    else
        assert_fail "Non-integer correctly rejected"
    fi

    if ! (validate_integer "12.5" "test_value" 2>/dev/null); then
        assert_pass "Float correctly rejected"
    else
        assert_fail "Float correctly rejected"
    fi
}

# ============================================================================
# Test 4: Integer range validation
# ============================================================================
test_integer_range_validation() {
    print_section "Test 4: Integer Range Validation"

    if validate_integer_range "5" 1 10 "test_value" 2>/dev/null; then
        assert_pass "Value in range accepted"
    else
        assert_fail "Value in range accepted"
    fi

    if ! (validate_integer_range "15" 1 10 "test_value" 2>/dev/null); then
        assert_pass "Value above range rejected"
    else
        assert_fail "Value above range rejected"
    fi

    if ! (validate_integer_range "0" 1 10 "test_value" 2>/dev/null); then
        assert_pass "Value below range rejected"
    else
        assert_fail "Value below range rejected"
    fi
}

# ============================================================================
# Test 5: Container runtime detection
# ============================================================================
test_container_runtime() {
    print_section "Test 5: Container Runtime Detection"

    if (validate_container_runtime 2>/dev/null); then
        assert_pass "Container runtime detected: ${CONTAINER_RUNTIME:-none}"
    else
        echo -e "${YELLOW}  ℹ No container runtime found (docker/podman) - this is OK${NC}"
        assert_pass "Container runtime check completed (no runtime found)"
    fi
}

# ============================================================================
# Test 6: NNODES validation
# ============================================================================
test_nnodes_validation() {
    print_section "Test 6: NNODES Validation"

    export NNODES=4
    if validate_nnodes 2>/dev/null; then
        assert_pass "Valid NNODES=4 accepted"
    else
        assert_fail "Valid NNODES=4 accepted"
    fi

    export NNODES=0
    if ! (validate_nnodes 2>/dev/null); then
        assert_pass "NNODES=0 correctly rejected"
    else
        assert_fail "NNODES=0 correctly rejected"
    fi

    export NNODES=-1
    if ! (validate_nnodes 2>/dev/null); then
        assert_pass "Negative NNODES rejected"
    else
        assert_fail "Negative NNODES rejected"
    fi

    export NNODES=2  # Reset
}

# ============================================================================
# Test 7: NODE_RANK validation
# ============================================================================
test_node_rank_validation() {
    print_section "Test 7: NODE_RANK Validation"

    export NNODES=4
    export NODE_RANK=2
    if validate_node_rank 2>/dev/null; then
        assert_pass "Valid NODE_RANK=2 (NNODES=4) accepted"
    else
        assert_fail "Valid NODE_RANK=2 (NNODES=4) accepted"
    fi

    export NODE_RANK=5
    if ! (validate_node_rank 2>/dev/null); then
        assert_pass "NODE_RANK >= NNODES correctly rejected"
    else
        assert_fail "NODE_RANK >= NNODES correctly rejected"
    fi

    export NODE_RANK=-1
    if ! (validate_node_rank 2>/dev/null); then
        assert_pass "Negative NODE_RANK rejected"
    else
        assert_fail "Negative NODE_RANK rejected"
    fi

    export NODE_RANK=0  # Reset
}

# ============================================================================
# Test 8: MASTER_PORT validation
# ============================================================================
test_master_port_validation() {
    print_section "Test 8: MASTER_PORT Validation"

    export MASTER_PORT=8888
    if validate_master_port 2>/dev/null; then
        assert_pass "Valid MASTER_PORT=8888 accepted"
    else
        assert_fail "Valid MASTER_PORT=8888 accepted"
    fi

    export MASTER_PORT=100
    if ! (validate_master_port 2>/dev/null); then
        assert_pass "MASTER_PORT < 1024 correctly rejected"
    else
        assert_fail "MASTER_PORT < 1024 correctly rejected"
    fi

    export MASTER_PORT=70000
    if ! (validate_master_port 2>/dev/null); then
        assert_pass "MASTER_PORT > 65535 correctly rejected"
    else
        assert_fail "MASTER_PORT > 65535 correctly rejected"
    fi

    export MASTER_PORT=1234  # Reset
}

# ============================================================================
# Test 9: Volume format validation
# ============================================================================
test_volume_format_validation() {
    print_section "Test 9: Volume Format Validation"

    # Valid formats
    if validate_volume_format "/host:/container" "[test]" 2>/dev/null; then
        assert_pass "Valid volume format /host:/container accepted"
    else
        assert_fail "Valid volume format /host:/container accepted"
    fi

    if validate_volume_format "/host:/container:ro" "[test]" 2>/dev/null; then
        assert_pass "Valid volume format with options accepted"
    else
        assert_fail "Valid volume format with options accepted"
    fi

    if validate_volume_format "/workspace" "[test]" 2>/dev/null; then
        assert_pass "Single path format accepted"
    else
        assert_fail "Single path format accepted"
    fi

    # Invalid formats
    if ! (validate_volume_format ":/container" "[test]" 2>/dev/null); then
        assert_pass "Empty source path rejected"
    else
        assert_fail "Empty source path rejected"
    fi

    if ! (validate_volume_format "/host:" "[test]" 2>/dev/null); then
        assert_pass "Empty destination path rejected"
    else
        assert_fail "Empty destination path rejected"
    fi

    if ! (validate_volume_format "/host:/container:invalid" "[test]" 2>/dev/null); then
        assert_pass "Invalid volume option rejected"
    else
        assert_fail "Invalid volume option rejected"
    fi
}

# ============================================================================
# Test 10: Memory format validation
# ============================================================================
test_memory_format_validation() {
    print_section "Test 10: Memory Format Validation"

    # Valid formats
    if validate_memory_format "256G" "memory" 2>/dev/null; then
        assert_pass "Valid memory format 256G accepted"
    else
        assert_fail "Valid memory format 256G accepted"
    fi

    if validate_memory_format "1024M" "memory" 2>/dev/null; then
        assert_pass "Valid memory format 1024M accepted"
    else
        assert_fail "Valid memory format 1024M accepted"
    fi

    if validate_memory_format "512" "memory" 2>/dev/null; then
        assert_pass "Memory format without unit accepted"
    else
        assert_fail "Memory format without unit accepted"
    fi

    # Invalid formats
    if ! (validate_memory_format "256GB" "memory" 2>/dev/null); then
        assert_pass "Invalid memory format 256GB rejected"
    else
        assert_fail "Invalid memory format 256GB rejected"
    fi

    if ! (validate_memory_format "abc" "memory" 2>/dev/null); then
        assert_pass "Non-numeric memory format rejected"
    else
        assert_fail "Non-numeric memory format rejected"
    fi
}

# ============================================================================
# Test 11: CPUs format validation
# ============================================================================
test_cpus_format_validation() {
    print_section "Test 11: CPUs Format Validation"

    # Valid formats
    if validate_cpus_format "32" "cpus" 2>/dev/null; then
        assert_pass "Valid cpus format 32 accepted"
    else
        assert_fail "Valid cpus format 32 accepted"
    fi

    if validate_cpus_format "16.5" "cpus" 2>/dev/null; then
        assert_pass "Valid cpus format 16.5 accepted"
    else
        assert_fail "Valid cpus format 16.5 accepted"
    fi

    # Invalid formats
    if ! (validate_cpus_format "abc" "cpus" 2>/dev/null); then
        assert_pass "Non-numeric cpus format rejected"
    else
        assert_fail "Non-numeric cpus format rejected"
    fi

    if ! (validate_cpus_format "16.5.2" "cpus" 2>/dev/null); then
        assert_pass "Invalid decimal cpus format rejected"
    else
        assert_fail "Invalid decimal cpus format rejected"
    fi
}

# ============================================================================
# Test 12: Environment variable format validation
# ============================================================================
test_env_format_validation() {
    print_section "Test 12: Environment Variable Format Validation"

    # Valid formats
    if validate_env_format "KEY=VALUE" "[test]" 2>/dev/null; then
        assert_pass "Valid env format KEY=VALUE accepted"
    else
        assert_fail "Valid env format KEY=VALUE accepted"
    fi

    local multi_env="KEY1=VALUE1
KEY2=VALUE2"
    if validate_env_format "$multi_env" "[test]" 2>/dev/null; then
        assert_pass "Multiple env variables accepted"
    else
        assert_fail "Multiple env variables accepted"
    fi

    # Test pass-through format (KEY without =)
    if validate_env_format "HF_TOKEN" "[test]" 2>/dev/null; then
        assert_pass "Pass-through env format KEY accepted"
    else
        assert_fail "Pass-through env format KEY accepted"
    fi

    local mixed_env="KEY1=VALUE1
HF_TOKEN
KEY2=VALUE2"
    if validate_env_format "$mixed_env" "[test]" 2>/dev/null; then
        assert_pass "Mixed KEY=VALUE and KEY formats accepted"
    else
        assert_fail "Mixed KEY=VALUE and KEY formats accepted"
    fi

    # Invalid formats
    if ! (validate_env_format "123INVALID" "[test]" 2>/dev/null); then
        assert_pass "Env starting with number rejected"
    else
        assert_fail "Env starting with number rejected"
    fi

    if ! (validate_env_format "INVALID-KEY" "[test]" 2>/dev/null); then
        assert_pass "Env with invalid character (dash) rejected"
    else
        assert_fail "Env with invalid character (dash) rejected"
    fi
}

# ============================================================================
# Test 13: Config parameter validation
# ============================================================================
test_config_param_validation() {
    print_section "Test 13: Config Parameter Validation"

    # Valid parameter
    if validate_config_param "value" "param_name" 2>/dev/null; then
        assert_pass "Non-empty parameter accepted"
    else
        assert_fail "Non-empty parameter accepted"
    fi

    # Empty parameter
    if ! (validate_config_param "" "param_name" 2>/dev/null); then
        assert_pass "Empty parameter rejected"
    else
        assert_fail "Empty parameter rejected"
    fi
}

# ============================================================================
# Test 14: Config array validation
# ============================================================================
test_config_array_validation() {
    print_section "Test 14: Config Array Validation"

    # Valid array
    if validate_config_array "item1
item2" "array_name" 2>/dev/null; then
        assert_pass "Non-empty array accepted"
    else
        assert_fail "Non-empty array accepted"
    fi

    # Empty array marker
    if ! (validate_config_array "[]" "array_name" 2>/dev/null); then
        assert_pass "Empty array marker [] rejected"
    else
        assert_fail "Empty array marker [] rejected"
    fi

    # Empty string
    if ! (validate_config_array "" "array_name" 2>/dev/null); then
        assert_pass "Empty array rejected"
    else
        assert_fail "Empty array rejected"
    fi
}

# ============================================================================
# Test 15: Positional arguments validation
# ============================================================================
test_positional_args_validation() {
    print_section "Test 15: Positional Arguments Validation"

    # Valid arguments
    # shellcheck disable=SC2034
    local valid_args=("arg1" "arg2" "arg3")
    if validate_positional_args valid_args 2>/dev/null; then
        assert_pass "Non-empty arguments array accepted"
    else
        assert_fail "Non-empty arguments array accepted"
    fi

    # Empty arguments
    # shellcheck disable=SC2034
    local empty_args=()
    if ! (validate_positional_args empty_args 2>/dev/null); then
        assert_pass "Empty arguments array rejected"
    else
        assert_fail "Empty arguments array rejected"
    fi
}

# ============================================================================
# Run all tests
# ============================================================================
main() {
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Unit Tests for Primus CLI Validation Library               ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    test_distributed_params
    test_gpus_per_node_validation
    test_integer_validation
    test_integer_range_validation
    test_container_runtime
    test_nnodes_validation
    test_node_rank_validation
    test_master_port_validation
    test_volume_format_validation
    test_memory_format_validation
    test_cpus_format_validation
    test_env_format_validation
    test_config_param_validation
    test_config_array_validation
    test_positional_args_validation

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
