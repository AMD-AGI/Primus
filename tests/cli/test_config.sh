#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Test script for configuration file support

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source libraries
export PRIMUS_LOG_COLOR=0  # Disable colors in tests
# shellcheck disable=SC1091
source "$PROJECT_ROOT/runner/lib/common.sh"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/runner/lib/config.sh"

export NODE_RANK=0

echo "========================================="
echo "  Primus CLI Configuration Tests"
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

# Test 1: Config functions exist
LOG_INFO "Test 1: Testing config functions exist..."
if type load_config &>/dev/null; then
    test_pass "load_config function exists"
else
    test_fail "load_config function exists"
fi

if type get_config &>/dev/null; then
    test_pass "get_config function exists"
else
    test_fail "get_config function exists"
fi

if type set_config &>/dev/null; then
    test_pass "set_config function exists"
else
    test_fail "set_config function exists"
fi
echo ""

# Test 2: Default configuration values
LOG_INFO "Test 2: Testing default configuration values..."
default_log_level=$(get_config "global.log_level")
if [[ "$default_log_level" == "INFO" ]]; then
    test_pass "Default log_level is INFO"
else
    test_fail "Default log_level is INFO (got: $default_log_level)"
fi

default_gpus=$(get_config "distributed.gpus_per_node")
if [[ "$default_gpus" == "8" ]]; then
    test_pass "Default gpus_per_node is 8"
else
    test_fail "Default gpus_per_node is 8 (got: $default_gpus)"
fi
echo ""

# Test 3: Set and get config
LOG_INFO "Test 3: Testing set_config and get_config..."
set_config "test.key" "test_value"
value=$(get_config "test.key")
if [[ "$value" == "test_value" ]]; then
    test_pass "set_config and get_config work"
else
    test_fail "set_config and get_config work (got: $value)"
fi
echo ""

# Test 4: YAML config loading
LOG_INFO "Test 4: Testing YAML config loading..."
TEST_YAML="/tmp/test-primus-$$-yaml"
cat > "$TEST_YAML" << 'EOF'
global:
  log_level: DEBUG

distributed:
  gpus_per_node: 4
  master_port: 5678

container:
  image: "rocm/primus:test"
  cpus: 32
EOF

if load_yaml_config "$TEST_YAML" 2>/dev/null; then
    test_pass "YAML config loads without error"

    # Check if values were loaded
    loaded_gpus=$(get_config "distributed.gpus_per_node")
    if [[ "$loaded_gpus" == "4" ]]; then
        test_pass "YAML values loaded correctly (gpus_per_node: 4)"
    else
        test_fail "YAML values loaded correctly (got: $loaded_gpus)"
    fi

    loaded_cpus=$(get_config "container.cpus")
    if [[ "$loaded_cpus" == "32" ]]; then
        test_pass "YAML container values loaded (cpus: 32)"
    else
        test_fail "YAML container values loaded (got: $loaded_cpus)"
    fi
else
    test_fail "YAML config loads without error"
fi
rm -f "$TEST_YAML"
echo ""

# Test 5: Shell config loading
LOG_INFO "Test 5: Testing shell config loading..."
TEST_SHELL="/tmp/test-primusrc-$$"
cat > "$TEST_SHELL" << 'EOF'
# Test shell config
PRIMUS_GLOBAL_LOG_LEVEL="WARN"
PRIMUS_DISTRIBUTED_GPUS_PER_NODE="16"
PRIMUS_CONTAINER_MEMORY="512G"
EOF

if load_shell_config "$TEST_SHELL" 2>/dev/null; then
    test_pass "Shell config loads without error"

    # Check if values were loaded
    loaded_memory=$(get_config "container.memory")
    if [[ "$loaded_memory" == "512G" ]]; then
        test_pass "Shell values loaded correctly (memory: 512G)"
    else
        test_fail "Shell values loaded correctly (got: $loaded_memory)"
    fi
else
    test_fail "Shell config loads without error"
fi
rm -f "$TEST_SHELL"
echo ""

# Test 6: --show-config option
LOG_INFO "Test 6: Testing --show-config option..."
if bash "$PROJECT_ROOT/runner/primus-cli" --show-config 2>&1 | grep -q "Current Configuration"; then
    test_pass "--show-config works"
else
    test_fail "--show-config works"
fi
echo ""

# Test 7: --config option with file
LOG_INFO "Test 7: Testing --config FILE option..."
TEST_CONFIG="/tmp/test-config-$$.yaml"
cat > "$TEST_CONFIG" << 'EOF'
distributed:
  gpus_per_node: 2
EOF

if bash "$PROJECT_ROOT/runner/primus-cli" --config "$TEST_CONFIG" --show-config 2>&1 | grep -q "gpus_per_node: 2"; then
    test_pass "--config FILE loads correctly"
else
    test_fail "--config FILE loads correctly"
fi
rm -f "$TEST_CONFIG"
echo ""

# Test 8: Priority test (CLI > config)
LOG_INFO "Test 8: Testing configuration priority..."
# Create config with log_level=INFO
TEST_PRIORITY="/tmp/test-priority-$$.yaml"
cat > "$TEST_PRIORITY" << 'EOF'
global:
  log_level: INFO
EOF

# Override with --log-level DEBUG
if bash "$PROJECT_ROOT/runner/primus-cli" --config "$TEST_PRIORITY" --log-level DEBUG --show-config 2>&1 | grep -q "log_level: DEBUG"; then
    test_pass "CLI arguments override config file (priority works)"
else
    test_fail "CLI arguments override config file"
fi
rm -f "$TEST_PRIORITY"
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
    LOG_SUCCESS "All configuration tests passed! ✓"
    exit 0
else
    LOG_ERROR "Some configuration tests failed! ✗"
    exit 1
fi
