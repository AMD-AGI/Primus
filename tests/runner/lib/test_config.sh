#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Unit tests for Primus CLI configuration system

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

    if type load_config_auto &>/dev/null; then
        assert_pass "load_config_auto function exists"
    else
        assert_fail "load_config_auto function exists"
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

    if type extract_config_section &>/dev/null; then
        assert_pass "extract_config_section function exists"
    else
        assert_fail "extract_config_section function exists"
    fi
}

# ============================================================================
# Test 2: Set and get config
# ============================================================================
test_set_get_config() {
    print_section "Test 2: Set and Get Config"

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
# Test 3: YAML config loading
# ============================================================================
test_yaml_config_loading() {
    print_section "Test 3: YAML Config Loading"

    local test_yaml="/tmp/test-primus-$$-yaml"
    cat > "$test_yaml" << 'EOF'
main:
  debug: true
  mode: slurm

slurm:
  partition: gpu
  nodes: 2
  gpus_per_node: 4

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
    loaded_gpus=$(get_config "slurm.gpus_per_node" 2>/dev/null)
    assert_equals "4" "$loaded_gpus" "YAML slurm.gpus_per_node value loaded"

    local loaded_cpus
    loaded_cpus=$(get_config "container.options.cpus" 2>/dev/null)
    assert_equals "32" "$loaded_cpus" "YAML nested container.options.cpus loaded"

    local loaded_image
    loaded_image=$(get_config "container.image" 2>/dev/null)
    assert_equals "rocm/primus:test" "$loaded_image" "YAML container.image loaded"

    local loaded_debug
    loaded_debug=$(get_config "main.debug" 2>/dev/null)
    assert_equals "true" "$loaded_debug" "YAML main.debug loaded"

    rm -f "$test_yaml"
}

# ============================================================================
# Test 4: YAML array handling (newline-separated values)
# ============================================================================
test_yaml_array_handling() {
    print_section "Test 4: YAML Array Handling (Newline-Separated)"

    local test_yaml="/tmp/test-array-$$.yaml"
    cat > "$test_yaml" << 'EOF'
container:
  options:
    mounts:
      - "/data:/data"
      - "/models:/models"
      - "/output:/output"
    devices:
      - "/dev/kfd"
      - "/dev/dri"

direct:
  patch:
    - "/opt/patch1.sh"
    - "/opt/patch2.sh"
EOF

    if load_yaml_config "$test_yaml" 2>/dev/null; then
        assert_pass "YAML with arrays loads successfully"
    else
        assert_fail "YAML with arrays loads successfully"
    fi

    # Arrays are now stored as newline-separated strings
    local mounts
    mounts=$(get_config "container.options.mounts" 2>/dev/null)

    # Check if mounts contains the expected values
    if echo "$mounts" | grep -qF "/data:/data"; then
        assert_pass "Mounts array contains /data:/data"
    else
        assert_fail "Mounts array contains /data:/data"
    fi

    if echo "$mounts" | grep -qF "/models:/models"; then
        assert_pass "Mounts array contains /models:/models"
    else
        assert_fail "Mounts array contains /models:/models"
    fi

    # Check devices array
    local devices
    devices=$(get_config "container.options.devices" 2>/dev/null)

    if echo "$devices" | grep -qF "/dev/kfd"; then
        assert_pass "Devices array contains /dev/kfd"
    else
        assert_fail "Devices array contains /dev/kfd"
    fi

    # Check patch array
    local patches
    patches=$(get_config "direct.patch" 2>/dev/null)

    if echo "$patches" | grep -qF "/opt/patch1.sh"; then
        assert_pass "Patches array contains /opt/patch1.sh"
    else
        assert_fail "Patches array contains /opt/patch1.sh"
    fi

    if echo "$patches" | grep -qF "/opt/patch2.sh"; then
        assert_pass "Patches array contains /opt/patch2.sh"
    else
        assert_fail "Patches array contains /opt/patch2.sh"
    fi

    rm -f "$test_yaml"
}

# ============================================================================
# Test 5: load_config_auto function
# ============================================================================
test_load_config_auto() {
    print_section "Test 5: load_config_auto Function"

    # Test 1: Load specified config file
    local test_yaml="/tmp/test-auto-$$.yaml"
    cat > "$test_yaml" << 'EOF'
slurm:
  partition: test-gpu
  nodes: 3
EOF

    if load_config_auto "$test_yaml" "test" 2>/dev/null; then
        assert_pass "load_config_auto loads specified config file"
    else
        assert_fail "load_config_auto loads specified config file"
    fi

    local loaded_partition
    loaded_partition=$(get_config "slurm.partition" 2>/dev/null)
    assert_equals "test-gpu" "$loaded_partition" "Specified config value loaded"

    # Test 2: Fail on invalid config file
    if load_config_auto "/nonexistent/config.yaml" "test" 2>/dev/null; then
        assert_fail "load_config_auto should fail on nonexistent file"
    else
        assert_pass "load_config_auto fails on nonexistent file"
    fi

    rm -f "$test_yaml"
}

# ============================================================================
# Test 6: Config file priority (System overrides User)
# ============================================================================
test_config_priority() {
    print_section "Test 6: Config File Priority (System > User)"

    # Create user config (loaded first, lower priority)
    local user_config="/tmp/test-user-$$.yaml"
    cat > "$user_config" << 'EOF'
slurm:
  partition: user-gpu
  nodes: 2
container:
  image: "rocm/user:v1"
EOF

    # Create system config (loaded second, higher priority)
    local system_config="/tmp/test-system-$$.yaml"
    cat > "$system_config" << 'EOF'
slurm:
  partition: system-gpu
EOF

    # Load in order: user first (lowest priority), then system (highest priority)
    load_yaml_config "$user_config" 2>/dev/null
    load_yaml_config "$system_config" 2>/dev/null

    # System config should override user for partition
    local partition
    partition=$(get_config "slurm.partition" 2>/dev/null)
    assert_equals "system-gpu" "$partition" "System config overrides user config"

    # User config value should remain for nodes (not overridden)
    local nodes
    nodes=$(get_config "slurm.nodes" 2>/dev/null)
    assert_equals "2" "$nodes" "User config value preserved when not overridden"

    # User config value for image should remain
    local image
    image=$(get_config "container.image" 2>/dev/null)
    assert_equals "rocm/user:v1" "$image" "User config value preserved for different section"

    rm -f "$user_config" "$system_config"
}

# ============================================================================
# Test 7: Nested configuration keys
# ============================================================================
test_nested_config_keys() {
    print_section "Test 7: Nested Configuration Keys"

    local test_yaml="/tmp/test-nested-$$.yaml"
    cat > "$test_yaml" << 'EOF'
container:
  options:
    cpus: "24"
    memory: "192G"
    user: "1000:1000"
    network: "host"
slurm:
  resources:
    gpus: 8
    mem: "128G"
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

    local gpus
    gpus=$(get_config "slurm.resources.gpus" 2>/dev/null)
    assert_equals "8" "$gpus" "Nested key slurm.resources.gpus"

    rm -f "$test_yaml"
}

# ============================================================================
# Test 8: Extract config section
# ============================================================================
test_extract_config_section() {
    print_section "Test 8: Extract Config Section"

    # Create test config with multiple sections
    local test_yaml="/tmp/test-extract-$$.yaml"
    cat > "$test_yaml" << 'EOF'
slurm:
  partition: "gpu"
  nodes: 4
  time: "01:00:00"
  mem: "64G"
  debug: true
  dry_run: false

container:
  image: "rocm/primus:latest"
  cpus: "16"

direct:
  master_port: 29500
  gpus_per_node: 8
EOF

    load_yaml_config "$test_yaml" 2>/dev/null

    # Test extracting slurm section
    declare -A slurm_params
    if extract_config_section "slurm" slurm_params 2>/dev/null; then
        assert_pass "extract_config_section works for slurm"
    else
        assert_fail "extract_config_section works for slurm"
    fi

    # Verify extracted values
    assert_equals "gpu" "${slurm_params[partition]:-}" "Slurm partition extracted"
    assert_equals "4" "${slurm_params[nodes]:-}" "Slurm nodes extracted"
    assert_equals "01:00:00" "${slurm_params[time]:-}" "Slurm time extracted"
    assert_equals "64G" "${slurm_params[mem]:-}" "Slurm mem extracted"
    assert_equals "true" "${slurm_params[debug]:-}" "Slurm debug extracted"
    assert_equals "false" "${slurm_params[dry_run]:-}" "Slurm dry_run extracted"

    # Test extracting container section
    declare -A container_params
    extract_config_section "container" container_params 2>/dev/null
    assert_equals "rocm/primus:latest" "${container_params[image]:-}" "Container image extracted"
    assert_equals "16" "${container_params[cpus]:-}" "Container cpus extracted"

    # Test extracting direct section
    declare -A direct_params
    extract_config_section "direct" direct_params 2>/dev/null
    assert_equals "29500" "${direct_params[master_port]:-}" "Direct master_port extracted"
    assert_equals "8" "${direct_params[gpus_per_node]:-}" "Direct gpus_per_node extracted"

    # Test extracting non-existent section
    declare -A empty_params
    extract_config_section "nonexistent" empty_params 2>/dev/null
    if [[ ${#empty_params[@]} -eq 0 ]]; then
        assert_pass "Empty result for non-existent section"
    else
        assert_fail "Empty result for non-existent section"
    fi

    rm -f "$test_yaml"
}

# ============================================================================
# Test 9: YAML empty arrays
# ============================================================================
test_yaml_empty_arrays() {
    print_section "Test 9: YAML Empty Arrays"

    local test_yaml="/tmp/test-empty-array-$$.yaml"
    cat > "$test_yaml" << 'EOF'
container:
  options:
    mounts: []
    devices:
      - "/dev/kfd"
    capabilities: []
EOF

    if load_yaml_config "$test_yaml" 2>/dev/null; then
        assert_pass "YAML with empty arrays loads successfully"
    else
        assert_fail "YAML with empty arrays loads successfully"
    fi

    # Check empty array is represented as "[]"
    local empty_mounts
    empty_mounts=$(get_config "container.options.mounts" 2>/dev/null)
    assert_equals "[]" "$empty_mounts" "Empty mounts array stored as []"

    local empty_caps
    empty_caps=$(get_config "container.options.capabilities" 2>/dev/null)
    assert_equals "[]" "$empty_caps" "Empty capabilities array stored as []"

    # Check non-empty array works normally
    local devices
    devices=$(get_config "container.options.devices" 2>/dev/null)
    if echo "$devices" | grep -qF "/dev/kfd"; then
        assert_pass "Non-empty devices array contains expected value"
    else
        assert_fail "Non-empty devices array contains expected value"
    fi

    rm -f "$test_yaml"
}

# ============================================================================
# Test 10: YAML special characters and edge cases
# ============================================================================
test_yaml_edge_cases() {
    print_section "Test 10: YAML Special Characters and Edge Cases"

    local test_yaml="/tmp/test-edge-$$.yaml"
    cat > "$test_yaml" << 'EOF'
container:
  image: "rocm/primus:v1.0-beta"
  workdir: "/opt/workspace"
  path: "/data/models/llama-2-7b"
  hostname: "gpu-node-01"

slurm:
  account: "project-123"
  comment: "Test run #42"
  constraint: "mi300x&nvme"
EOF

    load_yaml_config "$test_yaml" 2>/dev/null

    local image
    image=$(get_config "container.image" 2>/dev/null)
    assert_equals "rocm/primus:v1.0-beta" "$image" "Image with version and dash"

    local workdir
    workdir=$(get_config "container.workdir" 2>/dev/null)
    assert_equals "/opt/workspace" "$workdir" "Path with slashes"

    local path
    path=$(get_config "container.path" 2>/dev/null)
    assert_equals "/data/models/llama-2-7b" "$path" "Path with dashes and numbers"

    local hostname
    hostname=$(get_config "container.hostname" 2>/dev/null)
    assert_equals "gpu-node-01" "$hostname" "Hostname with dashes"

    local account
    account=$(get_config "slurm.account" 2>/dev/null)
    assert_equals "project-123" "$account" "Account with dash and number"

    local constraint
    constraint=$(get_config "slurm.constraint" 2>/dev/null)
    assert_equals "mi300x&nvme" "$constraint" "Constraint with ampersand"

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
    test_set_get_config
    test_yaml_config_loading
    test_yaml_array_handling
    test_load_config_auto
    test_config_priority
    test_nested_config_keys
    test_extract_config_section
    test_yaml_empty_arrays
    test_yaml_edge_cases

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
