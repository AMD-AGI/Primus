#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Unit tests for primus-cli-container.sh
# Uses dry-run mode and mock docker to verify functionality

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

    if echo "$haystack" | grep -qF "$needle"; then
        echo -e "${GREEN}✓${NC} $test_name"
        ((TESTS_PASSED++)) || true
        return 0
    else
        echo -e "${RED}✗${NC} $test_name"
        echo "  Expected to contain: $needle"
        echo "  Actual output:"
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

    if ! echo "$haystack" | grep -qF "$needle"; then
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
    print_section "Test 1: Basic Dry-run Functionality"

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" --dry-run -- train 2>&1)

    assert_contains "$output" "DRY-RUN" "Should contain DRY-RUN marker"
    assert_contains "$output" "docker run" "Should show docker run command"
    assert_contains "$output" "run --rm" "Should include --rm flag"
    assert_contains "$output" "ipc=host" "Should include --ipc=host"
    assert_contains "$output" "rocm/primus:v25.9_gfx942" "Should use default image"
}

# ============================================================================
# Test 2: CLI options parsing
# ============================================================================
test_cli_options() {
    print_section "Test 2: CLI Options Parsing"

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --dry-run \
        --cpus 32 \
        --memory 256G \
        --name test-container \
        -- train pretrain 2>&1)

    assert_contains "$output" "cpus 32" "Should include cpus option"
    assert_contains "$output" "memory 256G" "Should include memory option"
    assert_contains "$output" "name test-container" "Should include name option"
    assert_contains "$output" "train pretrain" "Should include command args"
}

# ============================================================================
# Test 3: Configuration file loading
# ============================================================================
test_config_file() {
    print_section "Test 3: Configuration File Loading"

    # Create test config and directories
    local test_config="/tmp/test-container-config-$$.yaml"
    mkdir -p /tmp/test-data-$$
    cat > "$test_config" << EOF
container:
  image: "rocm/primus:test-v1.0"
  options:
    cpus: "24"
    memory: "192G"
    user: "2000:2000"
  mounts:
    - "/tmp/test-data-$$:/data"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        -- train 2>&1)

    assert_contains "$output" "rocm/primus:test-v1.0" "Should use config image"
    assert_contains "$output" "cpus 24" "Should use config cpus"
    assert_contains "$output" "memory 192G" "Should use config memory"
    assert_contains "$output" "user 2000:2000" "Should use config user"
    assert_contains "$output" "/tmp/test-data-$$:/data" "Should mount config volume"

    rm -rf "$test_config" /tmp/test-data-$$
}

# ============================================================================
# Test 4: CLI overrides config
# ============================================================================
test_cli_overrides_config() {
    print_section "Test 4: CLI Overrides Config"

    # Create test config
    local test_config="/tmp/test-container-override-$$.yaml"
    cat > "$test_config" << 'EOF'
container:
  image: "rocm/primus:config-image"
  options:
    cpus: "16"
    memory: "128G"
    name: "config-name"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        --cpus 32 \
        --name cli-name \
        -- train 2>&1)

    assert_contains "$output" "cpus 32" "CLI should override config cpus"
    assert_contains "$output" "memory 128G" "Config memory should be preserved"
    assert_contains "$output" "name cli-name" "CLI should override config name"
    assert_not_contains "$output" "cpus 16" "Old cpus value should not appear"
    assert_not_contains "$output" "config-name" "Old name should not appear"

    rm -f "$test_config"
}

# ============================================================================
# Test 5: Mount handling
# ============================================================================
test_mount_handling() {
    print_section "Test 5: Mount Handling"

    # Create test directories
    mkdir -p /tmp/test-mount-$$/{data,output}

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --dry-run \
        --mount /tmp/test-mount-$$/data:/container/data \
        --mount /tmp/test-mount-$$/output \
        -- train 2>&1)

    assert_contains "$output" "/tmp/test-mount-$$/data:/container/data" "Should mount with custom path"
    assert_contains "$output" "/tmp/test-mount-$$/output" "Should mount to same path"

    rm -rf /tmp/test-mount-$$
}

# ============================================================================
# Test 6: Image specification
# ============================================================================
test_image_specification() {
    print_section "Test 6: Image Specification"

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --dry-run \
        --image "custom/image:v2.0" \
        -- train 2>&1)

    assert_contains "$output" "custom/image:v2.0" "Should use specified image"
    assert_not_contains "$output" "rocm/primus:v25.9_gfx942" "Should not use default image"
}

# ============================================================================
# Test 7: Multiple generic options
# ============================================================================
test_multiple_generic_options() {
    print_section "Test 7: Multiple Generic Options"

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --dry-run \
        --cpus 16 \
        --memory 128G \
        --shm-size 32G \
        --user 1000:1000 \
        --network bridge \
        --env NCCL_DEBUG=INFO \
        -- benchmark 2>&1)

    assert_contains "$output" "cpus 16" "Should include cpus"
    assert_contains "$output" "memory 128G" "Should include memory"
    assert_contains "$output" "shm-size 32G" "Should include shm-size"
    assert_contains "$output" "user 1000:1000" "Should include user"
    assert_contains "$output" "network bridge" "Should include network"
    assert_contains "$output" "env NCCL_DEBUG=INFO" "Should include env"
}

# ============================================================================
# Test 8: Help message
# ============================================================================
test_help_message() {
    print_section "Test 8: Help Message"

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" --help 2>&1)

    assert_contains "$output" "Usage:" "Should show usage"
    assert_contains "$output" "image" "Should document --image"
    assert_contains "$output" "mount" "Should document --mount"
    assert_contains "$output" "dry-run" "Should document --dry-run"
    assert_contains "$output" "Generic Docker/Podman Options" "Should document generic options"
}

# ============================================================================
# Test 9: Config with mounts array
# ============================================================================
test_config_mounts_array() {
    print_section "Test 9: Config with Mounts Array"

    # Create test directories
    mkdir -p /tmp/test-cfg-mount-$$/{data,models,output}

    # Create test config
    local test_config="/tmp/test-mounts-array-$$.yaml"
    cat > "$test_config" << EOF
container:
  options:
    cpus: "8"
  mounts:
    - "/tmp/test-cfg-mount-$$/data:/data"
    - "/tmp/test-cfg-mount-$$/models:/models"
    - "/tmp/test-cfg-mount-$$/output:/output"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        -- train 2>&1)

    assert_contains "$output" "/tmp/test-cfg-mount-$$/data:/data" "Should mount data"
    assert_contains "$output" "/tmp/test-cfg-mount-$$/models:/models" "Should mount models"
    assert_contains "$output" "/tmp/test-cfg-mount-$$/output:/output" "Should mount output"

    rm -f "$test_config"
    rm -rf /tmp/test-cfg-mount-$$
}

# ============================================================================
# Test 10: Config + CLI mounts combination
# ============================================================================
test_config_cli_mounts_combination() {
    print_section "Test 10: Config + CLI Mounts Combination"

    # Create test directories
    mkdir -p /tmp/test-combo-mount-$$/{config-data,cli-data}

    # Create test config
    local test_config="/tmp/test-combo-mounts-$$.yaml"
    cat > "$test_config" << EOF
container:
  mounts:
    - "/tmp/test-combo-mount-$$/config-data:/config-data"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        --mount /tmp/test-combo-mount-$$/cli-data:/cli-data \
        -- train 2>&1)

    assert_contains "$output" "/tmp/test-combo-mount-$$/config-data:/config-data" "Should mount config volume"
    assert_contains "$output" "/tmp/test-combo-mount-$$/cli-data:/cli-data" "Should mount CLI volume"

    rm -f "$test_config"
    rm -rf /tmp/test-combo-mount-$$
}

# ============================================================================
# Run all tests
# ============================================================================
main() {
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Unit Tests for primus-cli-container.sh                     ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    test_basic_dry_run
    test_cli_options
    test_config_file
    test_cli_overrides_config
    test_mount_handling
    test_image_specification
    test_multiple_generic_options
    test_help_message
    test_config_mounts_array
    test_config_cli_mounts_combination

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
