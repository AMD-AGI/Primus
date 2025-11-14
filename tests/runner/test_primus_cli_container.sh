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

    # Use grep with -- to prevent needle from being interpreted as option
    if echo "$haystack" | grep -qF -- "$needle"; then
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

    # Use grep with -- to prevent needle from being interpreted as option
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

    # Create minimal test config with a device that exists
    local test_config="/tmp/test-basic-$$.yaml"
    cat > "$test_config" << EOF
container:
  options:
    image: "rocm/primus:v25.9_gfx942"
    ipc: "host"
    device:
      - "/dev/null"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        -- train 2>&1)

    assert_contains "$output" "Launching container" "Should contain launching message"
    assert_contains "$output" "rocm/primus:v25.9_gfx942" "Should use default image"
    assert_contains "$output" "Runtime: docker" "Should show runtime"
    assert_contains "$output" "Image: rocm/primus:v25.9_gfx942" "Should show image"

    rm -f "$test_config"
}

# ============================================================================
# Test 2: CLI options parsing
# ============================================================================
test_cli_options() {
    print_section "Test 2: CLI Options Parsing"

    # Create minimal config
    local test_config="/tmp/test-cli-opts-$$.yaml"
    cat > "$test_config" << EOF
container:
  options:
    image: "rocm/primus:v25.9_gfx942"

    device:
      - "/dev/null"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        --cpus 32 \
        --memory 256G \
        --name test-container \
        -- train pretrain 2>&1)

    assert_contains "$output" "--cpus 32" "Should include cpus option"
    assert_contains "$output" "--memory 256G" "Should include memory option"
    assert_contains "$output" "--name test-container" "Should include name option"
    assert_contains "$output" "train pretrain" "Should include command args"

    rm -f "$test_config"
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
  options:
    image: "rocm/primus:test-v1.0"

    cpus: "24"
    memory: "192G"
    user: "2000:2000"
    device:
      - "/dev/null"
    volume:
      - "/tmp/test-data-$$:/data"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        -- train 2>&1)

    assert_contains "$output" "rocm/primus:test-v1.0" "Should use config image"
    assert_contains "$output" "--cpus 24" "Should use config cpus"
    assert_contains "$output" "--memory 192G" "Should use config memory"
    assert_contains "$output" "--user 2000:2000" "Should use config user"
    assert_contains "$output" "--volume /tmp/test-data-$$:/data" "Should mount config volume"

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
  options:
    image: "rocm/primus:config-image"

    cpus: "16"
    memory: "128G"
    name: "config-name"
    device:
      - "/dev/null"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        --cpus 32 \
        --name cli-name \
        -- train 2>&1)

    assert_contains "$output" "--cpus 32" "CLI should override config cpus"
    assert_contains "$output" "--memory 128G" "Config memory should be preserved"
    assert_contains "$output" "--name cli-name" "CLI should override config name"
    assert_not_contains "$output" "--cpus 16" "Old cpus value should not appear"
    assert_not_contains "$output" "config-name" "Old name should not appear"

    rm -f "$test_config"
}

# ============================================================================
# Test 5: Volume handling
# ============================================================================
test_volume_handling() {
    print_section "Test 5: Volume Handling"

    # Create test directories
    mkdir -p /tmp/test-volume-$$/{data,output}

    # Create minimal config
    local test_config="/tmp/test-vol-$$.yaml"
    cat > "$test_config" << EOF
container:
  options:
    image: "rocm/primus:v25.9_gfx942"

    device:
      - "/dev/null"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        --volume /tmp/test-volume-$$/data:/container/data \
        --volume /tmp/test-volume-$$/output \
        -- train 2>&1)

    assert_contains "$output" "--volume /tmp/test-volume-$$/data:/container/data" \
        "Should mount with custom path"
    assert_contains "$output" "--volume /tmp/test-volume-$$/output" \
        "Should mount to same path"

    rm -f "$test_config"
    rm -rf /tmp/test-volume-$$
}

# ============================================================================
# Test 6: Image specification
# ============================================================================
test_image_specification() {
    print_section "Test 6: Image Specification"

    # Create minimal config
    local test_config="/tmp/test-img-$$.yaml"
    cat > "$test_config" << EOF
container:
  options:
    image: "rocm/primus:v25.9_gfx942"

    device:
      - "/dev/null"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        --image "custom/image:v2.0" \
        -- train 2>&1)

    assert_contains "$output" "Image: custom/image:v2.0" "Should use specified image"
    assert_contains "$output" "custom/image:v2.0" "CLI image should override config"

    rm -f "$test_config"
}

# ============================================================================
# Test 7: Multiple generic options
# ============================================================================
test_multiple_generic_options() {
    print_section "Test 7: Multiple Generic Options"

    # Create minimal config
    local test_config="/tmp/test-multi-opts-$$.yaml"
    cat > "$test_config" << EOF
container:
  options:
    image: "rocm/primus:v25.9_gfx942"

    device:
      - "/dev/null"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        --cpus 16 \
        --memory 128G \
        --shm-size 32G \
        --user 1000:1000 \
        --network bridge \
        --env NCCL_DEBUG=INFO \
        -- benchmark 2>&1)

    assert_contains "$output" "--cpus 16" "Should include cpus"
    assert_contains "$output" "--memory 128G" "Should include memory"
    assert_contains "$output" "--shm-size 32G" "Should include shm-size"
    assert_contains "$output" "--user 1000:1000" "Should include user"
    assert_contains "$output" "--network bridge" "Should include network"
    assert_contains "$output" "--env NCCL_DEBUG=INFO" "Should include env"

    rm -f "$test_config"
}

# ============================================================================
# Test 8: Help message
# ============================================================================
test_help_message() {
    print_section "Test 8: Help Message"

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" --help 2>&1)

    assert_contains "$output" "Usage:" "Should show usage"
    assert_contains "$output" "--image" "Should document --image"
    assert_contains "$output" "--volume" "Should document --volume"
    assert_contains "$output" "--dry-run" "Should document --dry-run"
    assert_contains "$output" "--config" "Should document --config"
    assert_contains "$output" "--debug" "Should document --debug"
}

# ============================================================================
# Test 9: Config with volumes array
# ============================================================================
test_config_volumes_array() {
    print_section "Test 9: Config with Volumes Array"

    # Create test directories
    mkdir -p /tmp/test-cfg-volume-$$/{data,models,output}

    # Create test config
    local test_config="/tmp/test-volumes-array-$$.yaml"
    cat > "$test_config" << EOF
container:
  options:
    image: "rocm/primus:test"

    cpus: "8"
    device:
      - "/dev/null"
    volume:
      - "/tmp/test-cfg-volume-$$/data:/data"
      - "/tmp/test-cfg-volume-$$/models:/models"
      - "/tmp/test-cfg-volume-$$/output:/output"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        -- train 2>&1)

    assert_contains "$output" "--volume /tmp/test-cfg-volume-$$/data:/data" "Should mount data"
    assert_contains "$output" "--volume /tmp/test-cfg-volume-$$/models:/models" "Should mount models"
    assert_contains "$output" "--volume /tmp/test-cfg-volume-$$/output:/output" "Should mount output"

    rm -f "$test_config"
    rm -rf /tmp/test-cfg-volume-$$
}

# ============================================================================
# Test 10: Config + CLI volumes combination
# ============================================================================
test_config_cli_volumes_combination() {
    print_section "Test 10: Config + CLI Volumes Combination"

    # Create test directories
    mkdir -p /tmp/test-combo-volume-$$/{config-data,cli-data}

    # Create test config
    local test_config="/tmp/test-combo-volumes-$$.yaml"
    cat > "$test_config" << EOF
container:
  options:
    image: "rocm/primus:test"

    device:
      - "/dev/null"
    volume:
      - "/tmp/test-combo-volume-$$/config-data:/config-data"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        --volume /tmp/test-combo-volume-$$/cli-data:/cli-data \
        -- train 2>&1)

    assert_contains "$output" "--volume /tmp/test-combo-volume-$$/config-data:/config-data" \
        "Should mount config volume"
    assert_contains "$output" "--volume /tmp/test-combo-volume-$$/cli-data:/cli-data" \
        "Should mount CLI volume"

    rm -f "$test_config"
    rm -rf /tmp/test-combo-volume-$$
}

# ============================================================================
# Test 11: Validation - Missing image
# ============================================================================
test_validation_missing_image() {
    print_section "Test 11: Validation - Missing Image"

    # Create config without image
    local test_config="/tmp/test-no-image-$$.yaml"
    cat > "$test_config" << EOF
container:
  options:
    device:
      - "/dev/null"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        -- train 2>&1) || true

    assert_contains "$output" "Missing required parameter: --image" \
        "Should report missing image"

    rm -f "$test_config"
}

# ============================================================================
# Test 12: Validation - Missing devices
# ============================================================================
test_validation_missing_devices() {
    print_section "Test 12: Validation - Missing Devices"

    # Create config without devices
    local test_config="/tmp/test-no-devices-$$.yaml"
    cat > "$test_config" << EOF
container:
  options:
    image: "rocm/primus:test"

    cpus: "8"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        -- train 2>&1) || true

    assert_contains "$output" "No GPU devices configured" \
        "Should report missing devices"

    rm -f "$test_config"
}

# ============================================================================
# Test 13: Validation - Missing Primus commands
# ============================================================================
test_validation_missing_commands() {
    print_section "Test 13: Validation - Missing Primus Commands"

    # Create minimal config
    local test_config="/tmp/test-no-cmd-$$.yaml"
    cat > "$test_config" << EOF
container:
  options:
    image: "rocm/primus:test"

    device:
      - "/dev/null"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run 2>&1) || true

    assert_contains "$output" "Missing Primus commands after '--'" \
        "Should report missing commands"

    rm -f "$test_config"
}

# ============================================================================
# Test 14: Validation - Device does not exist
# ============================================================================
test_validation_device_not_exist() {
    print_section "Test 14: Validation - Device Does Not Exist"

    # Create minimal config
    local test_config="/tmp/test-bad-dev-$$.yaml"
    cat > "$test_config" << EOF
container:
  options:
    image: "rocm/primus:test"

    device:
      - "/dev/nonexistent-device-$$"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        -- train 2>&1) || true

    assert_contains "$output" "Device does not exist on host" \
        "Should report device not found"

    rm -f "$test_config"
}

# ============================================================================
# Test 15: Device and capability CLI arguments
# ============================================================================
test_device_and_capability_cli() {
    print_section "Test 15: Device and Capability CLI Arguments"

    # Create minimal config to avoid loading default config with /dev/kfd
    # CLI arguments will add additional devices and capabilities
    local test_config="/tmp/test-dev-cap-$$.yaml"
    cat > "$test_config" << EOF
container:
  options:
    image: "rocm/primus:test"

    device:
      - "/dev/null"
    cap-add:
      - "SYS_ADMIN"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        --device /dev/zero \
        --cap-add NET_ADMIN \
        -- train 2>&1)

    assert_contains "$output" "--device /dev/null" "Should include /dev/null from config"
    assert_contains "$output" "--device /dev/zero" "Should include /dev/zero from CLI"
    assert_contains "$output" "--cap-add SYS_ADMIN" "Should include SYS_ADMIN from config"
    assert_contains "$output" "--cap-add NET_ADMIN" "Should include NET_ADMIN from CLI"

    rm -f "$test_config"
}

# ============================================================================
# Test 16: Environment variable handling
# ============================================================================
test_env_variable_handling() {
    print_section "Test 16: Environment Variable Handling"

    # Create config with env vars
    local test_config="/tmp/test-env-$$.yaml"
    cat > "$test_config" << EOF
container:
  options:
    image: "rocm/primus:test"

    device:
      - "/dev/null"
    env:
      - "NCCL_DEBUG=INFO"
      - "TORCH_DISTRIBUTED_DEBUG=DETAIL"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
        --env WANDB_API_KEY=test-key \
        -- train 2>&1)

    assert_contains "$output" "--env NCCL_DEBUG=INFO" "Should include config env 1"
    assert_contains "$output" "--env TORCH_DISTRIBUTED_DEBUG=DETAIL" "Should include config env 2"
    assert_contains "$output" "--env CUDA_VISIBLE_DEVICES=0,1,2,3" "Should include CLI env 1"
    assert_contains "$output" "--env WANDB_API_KEY=test-key" "Should include CLI env 2"

    rm -f "$test_config"
}

# ============================================================================
# Test 17: Format validation - invalid memory
# ============================================================================
test_validation_invalid_memory() {
    print_section "Test 17: Format Validation - Invalid Memory"

    local test_config="/tmp/test-invalid-mem-$$.yaml"
    cat > "$test_config" << EOF
container:
  options:
    image: "rocm/primus:test"

    device:
      - "/dev/null"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        --memory "invalid-memory" \
        -- train 2>&1) || true

    assert_contains "$output" "Invalid memory format" \
        "Should report invalid memory format"

    rm -f "$test_config"
}

# ============================================================================
# Test 18: Format validation - invalid cpus
# ============================================================================
test_validation_invalid_cpus() {
    print_section "Test 18: Format Validation - Invalid CPUs"

    local test_config="/tmp/test-invalid-cpus-$$.yaml"
    cat > "$test_config" << EOF
container:
  options:
    image: "rocm/primus:test"

    device:
      - "/dev/null"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        --cpus "abc" \
        -- train 2>&1) || true

    assert_contains "$output" "Invalid cpus format" \
        "Should report invalid cpus format"

    rm -f "$test_config"
}

# ============================================================================
# Test 19: Format validation - invalid env
# ============================================================================
test_validation_invalid_env() {
    print_section "Test 19: Format Validation - Invalid Env"

    local test_config="/tmp/test-invalid-env-$$.yaml"
    cat > "$test_config" << EOF
container:
  options:
    image: "rocm/primus:test"

    device:
      - "/dev/null"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        --env "INVALID_FORMAT" \
        -- train 2>&1) || true

    assert_contains "$output" "Invalid env format" \
        "Should report invalid env format"

    rm -f "$test_config"
}

# ============================================================================
# Test 20: Format validation - invalid volume
# ============================================================================
test_validation_invalid_volume() {
    print_section "Test 20: Format Validation - Invalid Volume"

    local test_config="/tmp/test-invalid-vol-$$.yaml"
    cat > "$test_config" << EOF
container:
  options:
    image: "rocm/primus:test"

    device:
      - "/dev/null"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        --volume "/data:" \
        -- train 2>&1) || true

    assert_contains "$output" "Invalid volume format" \
        "Should report invalid volume format"

    rm -f "$test_config"
}

# ============================================================================
# Test 21: Dry-run output format
# ============================================================================
test_dry_run_output_format() {
    print_section "Test 21: Dry-run Output Format"

    local test_config="/tmp/test-dry-format-$$.yaml"
    cat > "$test_config" << EOF
container:
  options:
    image: "rocm/primus:test"

    device:
      - "/dev/null"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-container.sh" \
        --config "$test_config" \
        --dry-run \
        --cpus 16 \
        --memory 128G \
        -- train pretrain 2>&1)

    assert_contains "$output" "Launching container" "Should show launching message"
    assert_contains "$output" "Image: rocm/primus:test" "Should show image info"
    assert_contains "$output" "Runtime: docker" "Should show runtime"
    assert_contains "$output" "Container options" "Should show container options section"
    assert_contains "$output" "--cpus 16" "Should show cpus option"
    assert_contains "$output" "--memory 128G" "Should show memory option"

    rm -f "$test_config"
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
    test_volume_handling
    test_image_specification
    test_multiple_generic_options
    test_help_message
    test_config_volumes_array
    test_config_cli_volumes_combination
    test_validation_missing_image
    test_validation_missing_devices
    test_validation_missing_commands
    test_validation_device_not_exist
    test_device_and_capability_cli
    test_env_variable_handling
    test_validation_invalid_memory
    test_validation_invalid_cpus
    test_validation_invalid_env
    test_validation_invalid_volume
    test_dry_run_output_format

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
