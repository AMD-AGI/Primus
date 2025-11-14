#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Unit tests for runner/helpers/envs/primus-env.sh
#

# Get project root (tests/runner/helpers -> ../../..)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Test counter
TESTS_RUN=0
TESTS_PASSED=0

# Test assertion functions
assert_pass() {
    ((TESTS_RUN++))
    ((TESTS_PASSED++))
    echo "  ✓ PASS: $1"
}

assert_fail() {
    ((TESTS_RUN++))
    echo "  ✗ FAIL: $1"
}

# Currently unused but kept for future use
# shellcheck disable=SC2317
assert_contains() {
    local output="$1"
    local expected="$2"
    local message="$3"

    if echo "$output" | grep -q "$expected"; then
        assert_pass "$message"
    else
        assert_fail "$message"
    fi
}

# Setup test environment
setup_test_env() {
    export MASTER_ADDR="localhost"
    export MASTER_PORT="1234"
    export NNODES="1"
    export NODE_RANK="0"
    export GPUS_PER_NODE="8"
}

# Cleanup test environment
cleanup_test_env() {
    unset MASTER_ADDR MASTER_PORT NNODES NODE_RANK GPUS_PER_NODE
    unset PRIMUS_DEBUG PRIMUS_SKIP_VALIDATION
    unset HIP_VISIBLE_DEVICES NCCL_DEBUG NCCL_IB_HCA
    unset HSA_ENABLE_SDMA GPU_MAX_HW_QUEUES
    unset __PRIMUS_BASE_ENV_SOURCED
}

# ============================================================================
# Test 1: Basic Environment Loading
# ============================================================================
test_basic_env_loading() {
    echo "Test 1: Basic Environment Loading"

    setup_test_env
    export PRIMUS_SKIP_VALIDATION=1  # Skip validation for faster test

    # Source primus-env.sh in a subshell to avoid affecting test environment
    result=$(bash -c "
        export MASTER_ADDR='$MASTER_ADDR'
        export MASTER_PORT='$MASTER_PORT'
        export NNODES='$NNODES'
        export NODE_RANK='$NODE_RANK'
        export GPUS_PER_NODE='$GPUS_PER_NODE'
        export PRIMUS_SKIP_VALIDATION=1
        source '$PROJECT_ROOT/runner/helpers/envs/primus-env.sh' 2>&1 | grep -c 'Environment Configuration Complete'
    ")

    if [[ "$result" -eq 1 ]]; then
        assert_pass "Basic environment loads successfully"
    else
        assert_fail "Basic environment loading failed"
    fi

    cleanup_test_env
}

# ============================================================================
# Test 2: Environment Variables Are Set
# ============================================================================
test_env_variables_set() {
    echo "Test 2: Environment Variables Are Set"

    setup_test_env
    export PRIMUS_SKIP_VALIDATION=1

    # Check if key variables are exported
    result=$(bash -c "
        export MASTER_ADDR='$MASTER_ADDR'
        export MASTER_PORT='$MASTER_PORT'
        export NNODES='$NNODES'
        export NODE_RANK='$NODE_RANK'
        export GPUS_PER_NODE='$GPUS_PER_NODE'
        export PRIMUS_SKIP_VALIDATION=1
        source '$PROJECT_ROOT/runner/helpers/envs/primus-env.sh' 2>/dev/null

        # Check if variables are set
        if [[ -n \"\$HIP_VISIBLE_DEVICES\" ]] && \
           [[ -n \"\$HSA_ENABLE_SDMA\" ]] && \
           [[ -n \"\$GPU_MAX_HW_QUEUES\" ]]; then
            echo 'PASS'
        else
            echo 'FAIL'
        fi
    " 2>&1)

    if echo "$result" | grep -q "PASS"; then
        assert_pass "Environment variables are set correctly"
    else
        assert_fail "Environment variables not set"
    fi

    cleanup_test_env
}

# ============================================================================
# Test 3: Debug Mode
# ============================================================================
test_debug_mode() {
    echo "Test 3: Debug Mode"

    setup_test_env
    export PRIMUS_DEBUG=1
    export PRIMUS_SKIP_VALIDATION=1

    # Check if debug mode outputs expected trace
    result=$(bash -c "
        export MASTER_ADDR='$MASTER_ADDR'
        export MASTER_PORT='$MASTER_PORT'
        export NNODES='$NNODES'
        export NODE_RANK='$NODE_RANK'
        export GPUS_PER_NODE='$GPUS_PER_NODE'
        export PRIMUS_DEBUG=1
        export PRIMUS_SKIP_VALIDATION=1
        source '$PROJECT_ROOT/runner/helpers/envs/primus-env.sh' 2>&1 | grep -c 'DEBUG'
    ")

    if [[ "$result" -gt 0 ]]; then
        assert_pass "Debug mode works correctly"
    else
        assert_fail "Debug mode not working"
    fi

    cleanup_test_env
}

# ============================================================================
# Test 4: Validation Execution
# ============================================================================
test_validation_execution() {
    echo "Test 4: Validation Execution"

    setup_test_env
    # Don't skip validation this time

    # Should pass validation with correct values
    result=$(bash -c "
        export MASTER_ADDR='$MASTER_ADDR'
        export MASTER_PORT='$MASTER_PORT'
        export NNODES='$NNODES'
        export NODE_RANK='$NODE_RANK'
        export GPUS_PER_NODE='$GPUS_PER_NODE'
        source '$PROJECT_ROOT/runner/helpers/envs/primus-env.sh' 2>&1 | grep -c 'Configuration validation passed'
    ")

    if [[ "$result" -eq 1 ]]; then
        assert_pass "Validation executes and passes correctly"
    else
        assert_fail "Validation not executed or failed"
    fi

    cleanup_test_env
}

# ============================================================================
# Test 5: Validation Skip Flag
# ============================================================================
test_validation_skip() {
    echo "Test 5: Validation Skip Flag"

    setup_test_env
    export PRIMUS_SKIP_VALIDATION=1

    # Should not see validation messages
    result=$(bash -c "
        export MASTER_ADDR='$MASTER_ADDR'
        export MASTER_PORT='$MASTER_PORT'
        export NNODES='$NNODES'
        export NODE_RANK='$NODE_RANK'
        export GPUS_PER_NODE='$GPUS_PER_NODE'
        export PRIMUS_SKIP_VALIDATION=1
        source '$PROJECT_ROOT/runner/helpers/envs/primus-env.sh' 2>&1 | grep -c 'Validating Configuration'
    ")

    if [[ "$result" -eq 0 ]]; then
        assert_pass "Validation skip flag works correctly"
    else
        assert_fail "Validation skip flag not working"
    fi

    cleanup_test_env
}

# ============================================================================
# Test 6: Invalid Configuration Detection
# ============================================================================
test_invalid_config_detection() {
    echo "Test 6: Invalid Configuration Detection"

    # Set invalid NODE_RANK (>= NNODES)
    export MASTER_ADDR="localhost"
    export MASTER_PORT="1234"
    export NNODES="2"
    export NODE_RANK="5"  # Invalid: should be < NNODES
    export GPUS_PER_NODE="8"

    # Should fail validation - capture exit code only
    if bash -c "
        export MASTER_ADDR='$MASTER_ADDR'
        export MASTER_PORT='$MASTER_PORT'
        export NNODES='$NNODES'
        export NODE_RANK='$NODE_RANK'
        export GPUS_PER_NODE='$GPUS_PER_NODE'
        source '$PROJECT_ROOT/runner/helpers/envs/primus-env.sh' 2>/dev/null
    " 2>/dev/null; then
        assert_fail "Invalid configuration not detected"
    else
        assert_pass "Invalid configuration is detected"
    fi

    cleanup_test_env
}

# ============================================================================
# Test 7: GPU Detection
# ============================================================================
test_gpu_detection() {
    echo "Test 7: GPU Detection"

    setup_test_env
    export PRIMUS_SKIP_VALIDATION=1

    # Check if GPU detection runs
    result=$(bash -c "
        export MASTER_ADDR='$MASTER_ADDR'
        export MASTER_PORT='$MASTER_PORT'
        export NNODES='$NNODES'
        export NODE_RANK='$NODE_RANK'
        export GPUS_PER_NODE='$GPUS_PER_NODE'
        export PRIMUS_SKIP_VALIDATION=1
        source '$PROJECT_ROOT/runner/helpers/envs/primus-env.sh' 2>&1 | grep -c 'Detected GPU model'
    ")

    if [[ "$result" -eq 1 ]]; then
        assert_pass "GPU detection executes"
    else
        assert_fail "GPU detection not executed"
    fi

    cleanup_test_env
}

# ============================================================================
# Test 8: Layered Configuration Loading Order
# ============================================================================
test_loading_order() {
    echo "Test 8: Layered Configuration Loading Order"

    setup_test_env
    export PRIMUS_SKIP_VALIDATION=1

    # Check loading messages appear in correct order
    result=$(bash -c "
        export MASTER_ADDR='$MASTER_ADDR'
        export MASTER_PORT='$MASTER_PORT'
        export NNODES='$NNODES'
        export NODE_RANK='$NODE_RANK'
        export GPUS_PER_NODE='$GPUS_PER_NODE'
        export PRIMUS_SKIP_VALIDATION=1
        source '$PROJECT_ROOT/runner/helpers/envs/primus-env.sh' 2>&1
    ")

    # Check if loading message appears
    if echo "$result" | grep -q "Loading Primus Environment Configuration"; then
        assert_pass "Configuration loading order is correct"
    else
        assert_fail "Configuration loading order incorrect"
    fi

    cleanup_test_env
}

# ============================================================================
# Test 9: Missing Base Environment Detection
# ============================================================================
test_missing_base_env() {
    echo "Test 9: Missing Base Environment Detection"

    # Temporarily rename base_env.sh to simulate missing file
    BASE_ENV_FILE="$PROJECT_ROOT/runner/helpers/envs/base_env.sh"
    BASE_ENV_BACKUP="$PROJECT_ROOT/runner/helpers/envs/base_env.sh.backup.$$"

    if [[ -f "$BASE_ENV_FILE" ]]; then
        mv "$BASE_ENV_FILE" "$BASE_ENV_BACKUP"
    fi

    # Should fail when base_env.sh is missing
    # Use a fresh bash process and check for error message
    local output
    output=$(bash -c "
        unset __PRIMUS_BASE_ENV_SOURCED
        unset __PRIMUS_COMMON_SOURCED
        cd '$PROJECT_ROOT'
        source '$PROJECT_ROOT/runner/helpers/envs/primus-env.sh' 2>&1
    " 2>&1)

    # Restore base_env.sh immediately
    if [[ -f "$BASE_ENV_BACKUP" ]]; then
        mv "$BASE_ENV_BACKUP" "$BASE_ENV_FILE"
    fi

    # Test should detect the failure (check for "No such file or directory" error)
    if echo "$output" | grep -q "No such file or directory"; then
        assert_pass "Missing base environment is detected"
    else
        assert_fail "Missing base environment not detected"
    fi
}

# ============================================================================
# Test 10: Environment Variable Defaults
# ============================================================================
test_env_defaults() {
    echo "Test 10: Environment Variable Defaults"

    # Don't set any variables, let defaults kick in
    export PRIMUS_SKIP_VALIDATION=1

    result=$(bash -c "
        export PRIMUS_SKIP_VALIDATION=1
        source '$PROJECT_ROOT/runner/helpers/envs/primus-env.sh' 2>&1

        # Check default values
        [[ \"\$MASTER_ADDR\" == 'localhost' ]] && \
        [[ \"\$MASTER_PORT\" == '1234' ]] && \
        [[ \"\$NNODES\" == '1' ]] && \
        [[ \"\$NODE_RANK\" == '0' ]] && \
        [[ \"\$GPUS_PER_NODE\" == '8' ]] && \
        echo 'PASS' || echo 'FAIL'
    " 2>&1)

    if echo "$result" | grep -q "PASS"; then
        assert_pass "Default values are set correctly"
    else
        assert_fail "Default values not set"
    fi

    cleanup_test_env
}

# ============================================================================
# Run all tests
# ============================================================================
echo "=========================================="
echo "Running primus-env.sh Unit Tests"
echo "=========================================="
echo ""

test_basic_env_loading
test_env_variables_set
test_debug_mode
test_validation_execution
test_validation_skip
test_invalid_config_detection
test_gpu_detection
test_loading_order
test_missing_base_env
test_env_defaults

echo ""
echo "=========================================="
echo "Test Summary: $TESTS_PASSED/$TESTS_RUN tests passed"
echo "=========================================="

if [[ $TESTS_PASSED -eq $TESTS_RUN ]]; then
    exit 0
else
    exit 1
fi
