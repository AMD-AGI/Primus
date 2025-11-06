#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Unit tests for Primus CLI configuration system

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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
source "$PROJECT_ROOT/runner/lib/config.sh"

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
# Test 1: Config functions exist
# ============================================================================
test_config_functions_exist() {
    print_section "Test 1: Config Functions Existence"

    if type load_config &>/dev/null; then
        assert_pass "load_config function exists"
    else
        assert_fail "load_config function exists"
    fi

    if type get_config &>/dev/null; then
        assert_pass "get_config function exists"
    else
        assert_fail "get_config function exists"
    fi

    if type set_config &>/dev/null; then
        assert_pass "set_config function exists"
    else
        assert_fail "set_config function exists"
    fi
}

# ============================================================================
# Test 2: Default configuration values
# ============================================================================
test_default_config_values() {
    print_section "Test 2: Default Configuration Values"

    local default_gpus
    default_gpus=$(get_config "distributed.gpus_per_node" 2>/dev/null)
    assert_equals "8" "$default_gpus" "Default gpus_per_node is 8"
}

# ============================================================================
# Test 3: Set and get config
# ============================================================================
test_set_get_config() {
    print_section "Test 3: Set and Get Config"

    set_config "test.key" "test_value" 2>/dev/null
    local value
    value=$(get_config "test.key" 2>/dev/null)
    assert_equals "test_value" "$value" "set_config and get_config work"

    set_config "test.number" "42" 2>/dev/null
    value=$(get_config "test.number" 2>/dev/null)
    assert_equals "42" "$value" "Config handles numbers"

    set_config "test.string_with_spaces" "hello world" 2>/dev/null
    value=$(get_config "test.string_with_spaces" 2>/dev/null)
    assert_equals "hello world" "$value" "Config handles strings with spaces"
}

# ============================================================================
# Test 4: YAML config loading
# ============================================================================
test_yaml_config_loading() {
    print_section "Test 4: YAML Config Loading"

    local test_yaml="/tmp/test-primus-$$-yaml"
    cat > "$test_yaml" << 'EOF'
global:
  debug: true

distributed:
  gpus_per_node: 4
  master_port: 5678

container:
  image: "rocm/primus:test"
  options:
    cpus: "32"
    memory: "256G"
EOF

    if load_yaml_config "$test_yaml" 2>/dev/null; then
        assert_pass "YAML config loads without error"
    else
        assert_fail "YAML config loads without error"
    fi

    # Check if values were loaded
    local loaded_gpus
    loaded_gpus=$(get_config "distributed.gpus_per_node" 2>/dev/null)
    assert_equals "4" "$loaded_gpus" "YAML gpus_per_node value loaded"

    local loaded_cpus
    loaded_cpus=$(get_config "container.options.cpus" 2>/dev/null)
    assert_equals "32" "$loaded_cpus" "YAML nested container.options.cpus loaded"

    local loaded_image
    loaded_image=$(get_config "container.image" 2>/dev/null)
    assert_equals "rocm/primus:test" "$loaded_image" "YAML container.image loaded"

    rm -f "$test_yaml"
}

# ============================================================================
# Test 5: Shell config loading
# ============================================================================
test_shell_config_loading() {
    print_section "Test 5: Shell Config Loading"

    local test_shell="/tmp/test-primusrc-$$"
    cat > "$test_shell" << 'EOF'
# Test shell config
PRIMUS_DISTRIBUTED_GPUS_PER_NODE="16"
PRIMUS_CONTAINER_MEMORY="512G"
PRIMUS_CONTAINER_IMAGE="rocm/custom:latest"
EOF

    if load_shell_config "$test_shell" 2>/dev/null; then
        assert_pass "Shell config loads without error"
    else
        assert_fail "Shell config loads without error"
    fi

    # Check if values were loaded
    local loaded_memory
    loaded_memory=$(get_config "container.memory" 2>/dev/null)
    assert_equals "512G" "$loaded_memory" "Shell config memory loaded"

    local loaded_gpus
    loaded_gpus=$(get_config "distributed.gpus_per_node" 2>/dev/null)
    assert_equals "16" "$loaded_gpus" "Shell config gpus_per_node loaded"

    rm -f "$test_shell"
}

# ============================================================================
# Test 6: YAML array handling
# ============================================================================
test_yaml_array_handling() {
    print_section "Test 6: YAML Array Handling"

    local test_yaml="/tmp/test-array-$$.yaml"
    cat > "$test_yaml" << 'EOF'
container:
  mounts:
    - "/data:/data"
    - "/models:/models"
    - "/output:/output"
EOF

    if load_yaml_config "$test_yaml" 2>/dev/null; then
        assert_pass "YAML with arrays loads successfully"
    else
        assert_fail "YAML with arrays loads successfully"
    fi

    # Check if array values were loaded
    local mount1
    mount1=$(get_config "container.mounts.0" 2>/dev/null)
    assert_equals "/data:/data" "$mount1" "First array element loaded"

    local mount2
    mount2=$(get_config "container.mounts.1" 2>/dev/null)
    assert_equals "/models:/models" "$mount2" "Second array element loaded"

    rm -f "$test_yaml"
}

# ============================================================================
# Test 7: Config file priority (CLI > Project > Global)
# ============================================================================
test_config_priority() {
    print_section "Test 7: Config File Priority"

    # Create global config
    local global_config="/tmp/test-global-$$.yaml"
    cat > "$global_config" << 'EOF'
distributed:
  gpus_per_node: 2
container:
  image: "rocm/global:v1"
EOF

    # Create project config
    local project_config="/tmp/test-project-$$.yaml"
    cat > "$project_config" << 'EOF'
distributed:
  gpus_per_node: 4
EOF

    # Load in priority order (global first, then project)
    load_yaml_config "$global_config" 2>/dev/null
    load_yaml_config "$project_config" 2>/dev/null

    # Project config should override global for gpus_per_node
    local gpus
    gpus=$(get_config "distributed.gpus_per_node" 2>/dev/null)
    assert_equals "4" "$gpus" "Project config overrides global config"

    # Global config value should remain for image
    local image
    image=$(get_config "container.image" 2>/dev/null)
    assert_equals "rocm/global:v1" "$image" "Global config value preserved"

    rm -f "$global_config" "$project_config"
}

# ============================================================================
# Test 8: Nested configuration keys
# ============================================================================
test_nested_config_keys() {
    print_section "Test 8: Nested Configuration Keys"

    local test_yaml="/tmp/test-nested-$$.yaml"
    cat > "$test_yaml" << 'EOF'
container:
  options:
    cpus: "24"
    memory: "192G"
    user: "1000:1000"
    network: "host"
EOF

    load_yaml_config "$test_yaml" 2>/dev/null

    local cpus
    cpus=$(get_config "container.options.cpus" 2>/dev/null)
    assert_equals "24" "$cpus" "Nested key container.options.cpus"

    local memory
    memory=$(get_config "container.options.memory" 2>/dev/null)
    assert_equals "192G" "$memory" "Nested key container.options.memory"

    local user
    user=$(get_config "container.options.user" 2>/dev/null)
    assert_equals "1000:1000" "$user" "Nested key container.options.user"

    rm -f "$test_yaml"
}

# ============================================================================
# Run all tests
# ============================================================================
main() {
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Unit Tests for Primus CLI Configuration System             ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    test_config_functions_exist
    test_default_config_values
    test_set_get_config
    test_yaml_config_loading
    test_shell_config_loading
    test_yaml_array_handling
    test_config_priority
    test_nested_config_keys

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
