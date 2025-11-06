#!/bin/bash
###############################################################################
# Test script for validation library
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source libraries
# shellcheck disable=SC1091
source "$PROJECT_ROOT/runner/lib/common.sh"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/runner/lib/validation.sh"

export NODE_RANK=0

echo "========================================="
echo "  Primus CLI Validation Library Tests"
echo "========================================="
echo ""

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

test_pass() {
    LOG_SUCCESS "✓ $1: PASSED"
    ((TESTS_PASSED++)) || true
}

test_fail() {
    LOG_ERROR "✗ $1: FAILED"
    ((TESTS_FAILED++)) || true
}

# Test 1: Distributed parameters validation
LOG_INFO "Test 1: Validating distributed parameters..."
export NNODES=2
export NODE_RANK=0
export GPUS_PER_NODE=8
export MASTER_ADDR="localhost"
export MASTER_PORT=1234

if validate_distributed_params 2>/dev/null; then
    test_pass "Distributed parameters validation"
else
    test_fail "Distributed parameters validation"
fi
echo ""

# Test 2: Invalid GPUS_PER_NODE
LOG_INFO "Test 2: Testing invalid GPUS_PER_NODE (should fail gracefully)..."
export GPUS_PER_NODE=10
if validate_gpus_per_node 2>&1 | grep -q "must be between"; then
    test_pass "Invalid GPUS_PER_NODE correctly rejected"
else
    test_fail "Invalid GPUS_PER_NODE not caught"
fi
export GPUS_PER_NODE=8
echo ""

# Test 3: Integer validation
LOG_INFO "Test 3: Testing integer validation..."
if validate_integer "123" "test_value" 2>/dev/null; then
    test_pass "Integer validation"
else
    test_fail "Integer validation"
fi

if ! validate_integer "abc" "test_value" 2>/dev/null; then
    test_pass "Non-integer correctly rejected"
else
    test_fail "Non-integer not caught"
fi
echo ""

# Test 4: Integer range validation
LOG_INFO "Test 4: Testing integer range validation..."
if validate_integer_range "5" 1 10 "test_value" 2>/dev/null; then
    test_pass "Integer range validation (valid)"
else
    test_fail "Integer range validation (valid)"
fi

if ! validate_integer_range "15" 1 10 "test_value" 2>/dev/null; then
    test_pass "Out of range correctly rejected"
else
    test_fail "Out of range not caught"
fi
echo ""

# Test 5: Container runtime detection
LOG_INFO "Test 5: Testing container runtime detection..."
if validate_container_runtime 2>/dev/null; then
    test_pass "Container runtime detected: ${CONTAINER_RUNTIME}"
else
    LOG_WARN "⚠ No container runtime found (docker/podman) - this is expected on some systems"
    test_pass "Container runtime check (no runtime found)"
fi
echo ""

# Test 6: NNODES validation
LOG_INFO "Test 6: Testing NNODES validation..."
export NNODES=4
if validate_nnodes 2>/dev/null; then
    test_pass "NNODES validation (valid)"
else
    test_fail "NNODES validation (valid)"
fi

export NNODES=0
if ! validate_nnodes 2>/dev/null; then
    test_pass "NNODES=0 correctly rejected"
else
    test_fail "NNODES=0 not caught"
fi
export NNODES=2
echo ""

# Test 7: NODE_RANK validation
LOG_INFO "Test 7: Testing NODE_RANK validation..."
export NNODES=4
export NODE_RANK=2
if validate_node_rank 2>/dev/null; then
    test_pass "NODE_RANK validation (valid)"
else
    test_fail "NODE_RANK validation (valid)"
fi

export NODE_RANK=5
if ! validate_node_rank 2>/dev/null; then
    test_pass "NODE_RANK >= NNODES correctly rejected"
else
    test_fail "NODE_RANK >= NNODES not caught"
fi
export NODE_RANK=0
echo ""

# Test 8: MASTER_PORT validation
LOG_INFO "Test 8: Testing MASTER_PORT validation..."
export MASTER_PORT=8888
if validate_master_port 2>/dev/null; then
    test_pass "MASTER_PORT validation (valid)"
else
    test_fail "MASTER_PORT validation (valid)"
fi

export MASTER_PORT=100
if ! validate_master_port 2>/dev/null; then
    test_pass "MASTER_PORT < 1024 correctly rejected"
else
    test_fail "MASTER_PORT < 1024 not caught"
fi
export MASTER_PORT=1234
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
    LOG_SUCCESS "All validation tests passed! ✓"
    exit 0
else
    LOG_ERROR "Some validation tests failed! ✗"
    exit 1
fi
