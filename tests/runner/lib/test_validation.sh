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
