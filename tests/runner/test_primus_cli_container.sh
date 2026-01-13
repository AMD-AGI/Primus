#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Unit tests for primus-cli-container.sh
# Uses dry-run mode to verify functionality without actual execution

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER_DIR="$PROJECT_ROOT/runner"

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test helper functions
assert_equals() {
    local expected="$1"
    local actual="$2"
    local test_name="$3"

    ((TESTS_RUN++))

    if [[ "$expected" == "$actual" ]]; then
        echo -e "${GREEN}✓${NC} $test_name"
        ((TESTS_PASSED++)) || true
        return 0
    else
        echo -e "${RED}✗${NC} $test_name"
        echo "  Expected: $expected"
        echo "  Actual:   $actual"
        ((TESTS_FAILED++)) || true
        return 1
    fi
}

assert_contains() {
    local haystack="$1"
    local needle="$2"
    local test_name="$3"

    ((TESTS_RUN++))

    if echo "$haystack" | grep -qF -- "$needle"; then
        echo -e "${GREEN}✓${NC} $test_name"
        ((TESTS_PASSED++)) || true
        return 0
    else
        echo -e "${RED}✗${NC} $test_name"
        echo "  Expected to contain: $needle"
        echo "  Actual output (first 10 lines):"
        echo "$haystack" | head -10
        ((TESTS_FAILED++)) || true
        return 1
    fi
}

assert_not_contains() {
    local haystack="$1"
    local needle="$2"
    local test_name="$3"

    ((TESTS_RUN++))

    if ! echo "$haystack" | grep -qF -- "$needle"; then
        echo -e "${GREEN}✓${NC} $test_name"
        ((TESTS_PASSED++)) || true
        return 0
    else
        echo -e "${RED}✗${NC} $test_name"
        echo "  Should NOT contain: $needle"
        echo "  But found in output"
        ((TESTS_FAILED++)) || true
        return 1
    fi
}

assert_pass() {
    local test_name="$1"
    ((TESTS_RUN++))
    echo -e "${GREEN}✓${NC} $test_name"
    ((TESTS_PASSED++)) || true
}

assert_fail() {
    local test_name="$1"
    local message="${2:-Test failed}"
    ((TESTS_RUN++))
    echo -e "${RED}✗${NC} $test_name"
    echo "  $message"
    ((TESTS_FAILED++)) || true
}

# Print test section header
print_section() {
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# ============================================================================
# Test 1: Basic dry-run functionality
# ============================================================================
test_basic_dry_run() {
    print_section "Test 1: Basic Dry-Run Functionality"

    # Create a temporary config to provide required image
    local test_config="/tmp/test_container_config_$$.yaml"
    cat > "$test_config" << 'EOF'
container:
  options:
    image: "rocm/primus:test"
    device:
      - "/dev/kfd"
      - "/dev/dri"
EOF

    local output
    output=$(timeout 10 bash "$RUNNER_DIR/primus-cli-container.sh" --config "$test_config" --dry-run -- test 2>&1 || true)

    assert_contains "$output" "primus-cli-container.sh" "Container launcher header displayed"
    assert_contains "$output" "Launching container" "Launching message displayed"
    assert_contains "$output" "Runtime:" "Runtime displayed"
    assert_contains "$output" "Image:" "Image displayed"
    assert_contains "$output" "Dry-run mode: command not executed" "Dry-run footer displayed"

    rm -f "$test_config"
}

# ============================================================================
# Test 2: Image override via CLI
# ============================================================================
test_image_override() {
    print_section "Test 2: Image Override via CLI"

    local test_config="/tmp/test_container_config_$$.yaml"
    cat > "$test_config" << 'EOF'
container:
  options:
    image: "default_image:v1"
    device:
      - "/dev/kfd"
EOF

    local output
    output=$(timeout 10 bash "$RUNNER_DIR/primus-cli-container.sh" --config "$test_config" --dry-run --image custom_image:v2 -- test 2>&1 || true)

    assert_contains "$output" "Image: custom_image:v2" "CLI image overrides config"
    assert_not_contains "$output" "default_image:v1" "Default image not used"

    rm -f "$test_config"
}

# ============================================================================
# Test 3: Volume mounting
# ============================================================================
test_volume_mounting() {
    print_section "Test 3: Volume Mounting"

    local test_config="/tmp/test_container_config_$$.yaml"
    cat > "$test_config" << 'EOF'
container:
  options:
    image: "test_image:v1"
    device:
      - "/dev/kfd"
    volume:
      - "/data:/data"
EOF

    local output
    output=$(timeout 10 bash "$RUNNER_DIR/primus-cli-container.sh" --config "$test_config" --dry-run --volume /tmp:/tmp -- test 2>&1 || true)

    assert_contains "$output" "--volume /data:/data" "Config volume mounted"
    assert_contains "$output" "--volume /tmp:/tmp" "CLI volume mounted"

    rm -f "$test_config"
}

# ============================================================================
# Test 4: Environment variables
# ============================================================================
test_environment_variables() {
    print_section "Test 4: Environment Variables"

    local test_config="/tmp/test_container_config_$$.yaml"
    cat > "$test_config" << 'EOF'
container:
  options:
    image: "test_image:v1"
    device:
      - "/dev/kfd"
    env:
      - "CONFIG_VAR=from_config"
EOF

    local output
    output=$(timeout 10 bash "$RUNNER_DIR/primus-cli-container.sh" --config "$test_config" --dry-run --env CLI_VAR=from_cli -- test 2>&1 || true)

    assert_contains "$output" "--env CONFIG_VAR=from_config" "Config env var set"
    assert_contains "$output" "--env CLI_VAR=from_cli" "CLI env var set"

    rm -f "$test_config"
}

# ============================================================================
# Test 5: Device configuration
# ============================================================================
test_device_configuration() {
    print_section "Test 5: Device Configuration"

    local test_config="/tmp/test_container_config_$$.yaml"
    cat > "$test_config" << 'EOF'
container:
  options:
    image: "test_image:v1"
    device:
      - "/dev/kfd"
      - "/dev/dri"
EOF

    local output
    output=$(timeout 10 bash "$RUNNER_DIR/primus-cli-container.sh" --config "$test_config" --dry-run -- test 2>&1 || true)

    assert_contains "$output" "--device /dev/kfd" "Device /dev/kfd configured"
    assert_contains "$output" "--device /dev/dri" "Device /dev/dri configured"

    rm -f "$test_config"
}

# ============================================================================
# Test 6: Resource limits (memory, cpus)
# ============================================================================
test_resource_limits() {
    print_section "Test 6: Resource Limits"

    local test_config="/tmp/test_container_config_$$.yaml"
    cat > "$test_config" << 'EOF'
container:
  options:
    image: "test_image:v1"
    device:
      - "/dev/kfd"
EOF

    local output
    output=$(timeout 10 bash "$RUNNER_DIR/primus-cli-container.sh" --config "$test_config" --dry-run --memory 128G --cpus 16 -- test 2>&1 || true)

    assert_contains "$output" "--memory 128G" "Memory limit set"
    assert_contains "$output" "--cpus 16" "CPU limit set"

    rm -f "$test_config"
}

# ============================================================================
# Test 7: Network and IPC modes
# ============================================================================
test_network_ipc_modes() {
    print_section "Test 7: Network and IPC Modes"

    local test_config="/tmp/test_container_config_$$.yaml"
    cat > "$test_config" << 'EOF'
container:
  options:
    image: "test_image:v1"
    device:
      - "/dev/kfd"
    network: "host"
    ipc: "host"
EOF

    local output
    output=$(timeout 10 bash "$RUNNER_DIR/primus-cli-container.sh" --config "$test_config" --dry-run -- test 2>&1 || true)

    assert_contains "$output" "--network host" "Network mode set"
    assert_contains "$output" "--ipc host" "IPC mode set"

    rm -f "$test_config"
}

# ============================================================================
# Test 8: Boolean flags (privileged, rm)
# ============================================================================
test_boolean_flags() {
    print_section "Test 8: Boolean Flags"

    local test_config="/tmp/test_container_config_$$.yaml"
    cat > "$test_config" << 'EOF'
container:
  options:
    image: "test_image:v1"
    device:
      - "/dev/kfd"
EOF

    local output
    output=$(timeout 10 bash "$RUNNER_DIR/primus-cli-container.sh" --config "$test_config" --dry-run --privileged -- test 2>&1 || true)

    assert_contains "$output" "--privileged" "Privileged flag set"
    assert_contains "$output" "--rm" "Auto-remove flag set (always added)"

    rm -f "$test_config"
}

# ============================================================================
# Test 9: Config priority (CLI > Config > Default)
# ============================================================================
test_config_priority() {
    print_section "Test 9: Config Priority (CLI > Config > Default)"

    local test_config="/tmp/test_container_config_$$.yaml"
    cat > "$test_config" << 'EOF'
container:
  options:
    image: "config_image:v1"
    device:
      - "/dev/kfd"
    memory: "64G"
    env:
      - "VAR1=config_value"
EOF

    local output
    output=$(timeout 10 bash "$RUNNER_DIR/primus-cli-container.sh" --config "$test_config" --dry-run --image cli_image:v2 --memory 128G --env VAR1=cli_value -- test 2>&1 || true)

    assert_contains "$output" "Image: cli_image:v2" "CLI image overrides config"
    assert_contains "$output" "--memory 128G" "CLI memory overrides config"
    assert_contains "$output" "--env VAR1=cli_value" "CLI env appends to config"

    rm -f "$test_config"
}

# ============================================================================
# Test 10: Help output
# ============================================================================
test_help_output() {
    print_section "Test 10: Help Output"

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" --help 2>&1)

    assert_contains "$output" "Usage:" "Usage section displayed"
    assert_contains "$output" "--image" "Image option documented"
    assert_contains "$output" "--volume" "Volume option documented"
    assert_contains "$output" "--env" "Env option documented"
    assert_contains "$output" "--memory" "Memory option documented"
    assert_contains "$output" "--cpus" "CPUs option documented"
}

# ============================================================================
# Run all tests
# ============================================================================
main() {
    echo "Starting primus-cli-container.sh unit tests..."
    echo "Project root: $PROJECT_ROOT"
    echo ""

    test_basic_dry_run
    test_image_override
    test_volume_mounting
    test_environment_variables
    test_device_configuration
    test_resource_limits
    test_network_ipc_modes
    test_boolean_flags
    test_config_priority
    test_help_output

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
        exit 0
    else
        echo -e "${RED}✗ Some tests failed${NC}"
        exit 1
    fi
}

# Run tests if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
