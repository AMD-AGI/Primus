#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Unit tests for primus-cli-direct.sh
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
local_print_section() {
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# ============================================================================
# Test 1: Basic dry-run functionality
# ============================================================================
test_basic_dry_run() {
    local_print_section "Test 1: Basic Dry-Run Functionality"

    # Create a temporary config to avoid pip install
    local test_config="/tmp/test_direct_config_$$.yaml"
    cat > "$test_config" << 'EOF'
direct:
  script: "primus/cli/main.py"
  numa: "auto"
EOF

    local output
    output=$(timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --config "$test_config" --dry-run -- benchmark gemm 2>&1 || true)

    assert_contains "$output" "[DRY RUN] Direct Launch Configuration" "Dry-run header displayed"
    assert_contains "$output" "Run Mode" "Run mode displayed"
    assert_contains "$output" "Script Path" "Script path displayed"
    assert_contains "$output" "Full Command" "Full command displayed"
    assert_contains "$output" "End of Dry Run" "Dry-run footer displayed"

    rm -f "$test_config"
}

# ============================================================================
# Test 2: Environment variable handling
# ============================================================================
test_env_variables() {
    local_print_section "Test 2: Environment Variable Handling"

    local test_config="/tmp/test_direct_config_$$.yaml"
    cat > "$test_config" << 'EOF'
direct:
  script: "test.py"
  env:
    - "CONFIG_VAR=from_config"
EOF

    local output
    output=$(timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --config "$test_config" --dry-run --env CLI_VAR=from_cli -- test 2>&1 || true)

    assert_contains "$output" "Environment Variables:" "Environment variables section displayed"
    assert_contains "$output" "CONFIG_VAR=from_config" "Config env var displayed"
    assert_contains "$output" "CLI_VAR=from_cli" "CLI env var displayed"

    rm -f "$test_config"
}

# ============================================================================
# Test 3: Script path override
# ============================================================================
test_script_override() {
    local_print_section "Test 3: Script Path Override"

    local test_config="/tmp/test_direct_config_$$.yaml"
    cat > "$test_config" << 'EOF'
direct:
  script: "default_script.py"
EOF

    local output
    output=$(timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --config "$test_config" --dry-run --script custom_script.py -- test 2>&1 || true)

    assert_contains "$output" "Script Path     : custom_script.py" "CLI script overrides config"
    assert_not_contains "$output" "default_script.py" "Default script not used"

    rm -f "$test_config"
}

# ============================================================================
# Test 4: NUMA binding options
# ============================================================================
test_numa_binding() {
    local_print_section "Test 4: NUMA Binding Options"

    local test_config="/tmp/test_direct_config_$$.yaml"
    cat > "$test_config" << 'EOF'
direct:
  script: "test.py"
EOF

    # Test --numa flag
    local output_numa
    output_numa=$(timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --config "$test_config" --dry-run --numa -- test 2>&1 || true)
    assert_contains "$output_numa" "NUMA Binding    : true" "NUMA enabled with --numa"

    # Test --no-numa flag
    local output_no_numa
    output_no_numa=$(timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --config "$test_config" --dry-run --no-numa -- test 2>&1 || true)
    assert_contains "$output_no_numa" "NUMA Binding    : false" "NUMA disabled with --no-numa"

    rm -f "$test_config"
}

# ============================================================================
# Test 5: Single mode vs torchrun mode
# ============================================================================
test_run_modes() {
    local_print_section "Test 5: Run Modes (Single vs Torchrun)"

    local test_config="/tmp/test_direct_config_$$.yaml"
    cat > "$test_config" << 'EOF'
direct:
  script: "test.py"
EOF

    # Test single mode
    local output_single
    output_single=$(timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --config "$test_config" --dry-run --single -- test 2>&1 || true)
    assert_contains "$output_single" "Run Mode        : single" "Single mode set"
    assert_contains "$output_single" "python3 test.py" "Python3 command used"
    assert_not_contains "$output_single" "torchrun" "Torchrun not used in single mode"

    # Test torchrun mode (default)
    local output_torchrun
    output_torchrun=$(timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --config "$test_config" --dry-run -- test 2>&1 || true)
    assert_contains "$output_torchrun" "Run Mode        : torchrun" "Torchrun mode set"
    assert_contains "$output_torchrun" "torchrun" "Torchrun command used"
    assert_contains "$output_torchrun" "Distributed Settings:" "Distributed settings displayed"

    rm -f "$test_config"
}

# ============================================================================
# Test 6: Patch scripts handling
# ============================================================================
test_patch_scripts() {
    local_print_section "Test 6: Patch Scripts Handling"

    local test_config="/tmp/test_direct_config_$$.yaml"
    cat > "$test_config" << 'EOF'
direct:
  script: "test.py"
  patch:
    - "/tmp/patch1.sh"
EOF

    local output
    output=$(timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --config "$test_config" --dry-run --patch /tmp/patch2.sh -- test 2>&1 || true)

    assert_contains "$output" "Detected patch scripts: /tmp/patch1.sh /tmp/patch2.sh" "Both config and CLI patches displayed"

    rm -f "$test_config"
}

# ============================================================================
# Test 7: Env file handling via --env <file>
# ============================================================================
test_env_file_handling() {
    local_print_section "Test 7: Env File Handling via --env <file>"

    local test_config="/tmp/test_direct_config_$$.yaml"
    cat > "$test_config" << 'EOF'
direct:
  script: "test.py"
EOF

    local output
    output=$(timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --dry-run --config "$test_config" --env INVALID_ENV -- test 2>&1 || true)

    assert_contains "$output" "Env file not found or not readable: INVALID_ENV" "Non KEY=VALUE --env treated as env file and validated"

    rm -f "$test_config"
}

# ============================================================================
# Test 8: Debug mode output
# ============================================================================
test_debug_mode() {
    local_print_section "Test 8: Debug Mode Output"

    local test_config="/tmp/test_direct_config_$$.yaml"
    cat > "$test_config" << 'EOF'
direct:
  script: "test.py"
EOF

    local output
    output=$(timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --config "$test_config" --debug --dry-run -- test 2>&1 || true)

    assert_contains "$output" "Debug mode enabled" "Debug mode message displayed"
    assert_contains "$output" "[DEBUG]" "Debug logs present"
    assert_contains "$output" "Direct Launch Configuration" "Configuration displayed in debug mode"

    rm -f "$test_config"
}

# ============================================================================
# Test 9: Config file priority
# ============================================================================
test_config_priority() {
    local_print_section "Test 9: Config File Priority (CLI > Config > Default)"

    local test_config="/tmp/test_direct_config_$$.yaml"
    cat > "$test_config" << 'EOF'
direct:
  script: "config_script.py"
  numa: "false"
  env:
    - "VAR1=config_value"
EOF

    local output
    output=$(timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --config "$test_config" --dry-run --script cli_script.py --numa --env VAR1=cli_value -- test 2>&1 || true)

    assert_contains "$output" "Script Path     : cli_script.py" "CLI script overrides config"
    assert_contains "$output" "NUMA Binding    : true" "CLI numa overrides config"
    assert_contains "$output" "VAR1=cli_value" "CLI env appends to config"

    rm -f "$test_config"
}

# ============================================================================
# Test 10: Help output
# ============================================================================
test_help_output() {
    local_print_section "Test 10: Help Output"

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-direct.sh" --help 2>&1)

    assert_contains "$output" "Primus Direct Launcher" "Help header displayed"
    assert_contains "$output" "Usage:" "Usage section displayed"
    assert_contains "$output" "--single" "Single mode option documented"
    assert_contains "$output" "--numa" "NUMA option documented"
    assert_contains "$output" "--env" "Env option documented"
    assert_contains "$output" "--patch" "Patch option documented"
}

# ============================================================================
# Run all tests
# ============================================================================
main() {
    echo "Starting primus-cli-direct.sh unit tests..."
    echo "Project root: $PROJECT_ROOT"
    echo ""

    test_basic_dry_run
    test_env_variables
    test_script_override
    test_numa_binding
    test_run_modes
    test_patch_scripts
    test_env_file_handling
    test_debug_mode
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
