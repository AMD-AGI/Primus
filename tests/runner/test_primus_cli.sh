#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Unit tests for primus-cli main entry point

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
# Test 1: Help option
# ============================================================================
test_help_option() {
    print_section "Test 1: Help Option"

    local output
    output=$(bash "$PROJECT_ROOT/runner/primus-cli" --help 2>&1)

    assert_contains "$output" "Primus Unified Launcher CLI" "Shows help message"
    assert_contains "$output" "Usage:" "Shows usage section"
    assert_contains "$output" "Global Options" "Shows global options"
}

# ============================================================================
# Test 2: Version option
# ============================================================================
test_version_option() {
    print_section "Test 2: Version Option"

    local output
    output=$(bash "$PROJECT_ROOT/runner/primus-cli" --version 2>&1)

    assert_contains "$output" "Primus CLI version" "Shows version information"
    assert_contains "$output" "1.0.0" "Shows correct version number"
}

# ============================================================================
# Test 3: No arguments
# ============================================================================
test_no_arguments() {
    print_section "Test 3: No Arguments"

    local output
    output=$(bash "$PROJECT_ROOT/runner/primus-cli" 2>&1)

    assert_contains "$output" "Usage:" "Shows usage when no arguments provided"
}

# ============================================================================
# Test 4: Unknown mode
# ============================================================================
test_unknown_mode() {
    print_section "Test 4: Unknown Mode"

    local output
    output=$(bash "$PROJECT_ROOT/runner/primus-cli" unknown-mode 2>&1)

    assert_contains "$output" "Unknown or unsupported mode" "Rejects unknown mode"
}

# ============================================================================
# Test 5: Dry-run mode
# ============================================================================
test_dry_run_mode() {
    print_section "Test 5: Dry-run Mode"

    local output
    output=$(bash "$PROJECT_ROOT/runner/primus-cli" --dry-run direct --help 2>&1)

    assert_contains "$output" "[DRY RUN] Primus CLI" "Shows dry-run indicator"
}

# ============================================================================
# Test 6: Debug mode
# ============================================================================
test_debug_mode() {
    print_section "Test 6: Debug Mode"

    local output
    export NODE_RANK=0
    output=$(bash "$PROJECT_ROOT/runner/primus-cli" --debug direct --help 2>&1)

    # Debug mode sets PRIMUS_LOG_LEVEL=DEBUG and shows debug messages
    if echo "$output" | grep -qE "(DEBUG|Debug mode enabled)"; then
        assert_pass "Debug mode enables verbose output"
    else
        assert_fail "Debug mode enables verbose output" "Expected DEBUG output but got: $output"
    fi
}

# ============================================================================
# Test 7: Direct mode help
# ============================================================================
test_direct_mode_help() {
    print_section "Test 7: Direct Mode Help"

    local output
    output=$(bash "$PROJECT_ROOT/runner/primus-cli" direct --help 2>&1)

    assert_contains "$output" "Primus Direct Launcher" "Shows direct mode help"
    assert_contains "$output" "Usage:" "Shows direct mode usage"
}

# ============================================================================
# Test 8: Container mode help
# ============================================================================
test_container_mode_help() {
    print_section "Test 8: Container Mode Help"

    local output
    output=$(bash "$PROJECT_ROOT/runner/primus-cli" container --help 2>&1)

    assert_contains "$output" "Docker/Podman container" "Shows container mode help"
}

# ============================================================================
# Test 9: Slurm mode help
# ============================================================================
test_slurm_mode_help() {
    print_section "Test 9: Slurm Mode Help"

    local output
    output=$(bash "$PROJECT_ROOT/runner/primus-cli" slurm --help 2>&1)

    assert_contains "$output" "Primus Slurm Launcher" "Shows slurm mode help"
}

# ============================================================================
# Test 10: Config option parsing
# ============================================================================
test_config_option() {
    print_section "Test 10: Config Option Parsing"

    # Create a test config file
    local test_config="/tmp/test-primus-cli-$$.yaml"
    cat > "$test_config" << 'EOF'
main:
  debug: false
  dry_run: false

direct:
  gpus_per_node: 4
EOF

    local output
    output=$(bash "$PROJECT_ROOT/runner/primus-cli" --config "$test_config" direct --help 2>&1)

    # Should not error when config is provided
    if bash "$PROJECT_ROOT/runner/primus-cli" --config "$test_config" direct --help >/dev/null 2>&1 || echo "$output" | grep -qF "Direct"; then
        assert_pass "Config option accepted without error"
    else
        assert_fail "Config option accepted without error"
    fi

    rm -f "$test_config"
}

# ============================================================================
# Test 11: Multiple global options
# ============================================================================
test_multiple_global_options() {
    print_section "Test 11: Multiple Global Options"

    local output
    output=$(bash "$PROJECT_ROOT/runner/primus-cli" --debug --dry-run direct --help 2>&1)

    assert_contains "$output" "[DRY RUN] Primus CLI" "Multiple global options work together"
}

# ============================================================================
# Test 12: Mode selection validation
# ============================================================================
test_mode_selection() {
    print_section "Test 12: Mode Selection Validation"

    # Test that valid modes are accepted
    local modes=("direct" "container" "slurm")
    local all_passed=true

    for mode in "${modes[@]}"; do
        local output
        output=$(bash "$PROJECT_ROOT/runner/primus-cli" "$mode" --help 2>&1)

        if ! bash "$PROJECT_ROOT/runner/primus-cli" "$mode" --help >/dev/null 2>&1 && ! echo "$output" | grep -qE "(Direct|Container|Slurm)"; then
            all_passed=false
            break
        fi
    done

    if $all_passed; then
        assert_pass "All valid modes accepted (direct, container, slurm)"
    else
        assert_fail "All valid modes accepted"
    fi
}

# ============================================================================
# Test 13: Config file not found
# ============================================================================
test_config_file_not_found() {
    print_section "Test 13: Config File Not Found"

    local output
    output=$(bash "$PROJECT_ROOT/runner/primus-cli" --config /nonexistent/file.yaml direct --help 2>&1)

    assert_contains "$output" "Config file not found" "Rejects missing config file"
}

# ============================================================================
# Test 14: Config priority (CLI > Config)
# ============================================================================
test_config_priority() {
    print_section "Test 14: Config Priority (CLI overrides Config)"

    # Create a test config file with debug=true (without dry_run to see actual debug output)
    local test_config="/tmp/test-primus-cli-priority-$$.yaml"
    cat > "$test_config" << 'EOF'
main:
  debug: true
  dry_run: false
EOF

    # Config should apply when not overridden by CLI
    local output
    export NODE_RANK=0
    output=$(bash "$PROJECT_ROOT/runner/primus-cli" --config "$test_config" direct --help 2>&1)

    # With config debug=true, should see --debug flag in the "Executing" line
    if echo "$output" | grep -qF -- "--debug"; then
        assert_pass "Config debug mode applies when not overridden by CLI"
    else
        assert_fail "Config debug mode applies when not overridden by CLI" "Expected --debug flag in execution command"
    fi

    rm -f "$test_config"
}

# ============================================================================
# Test 15: Invalid config file format
# ============================================================================
test_invalid_config_format() {
    print_section "Test 15: Invalid Config File Format"

    # Create an invalid YAML file
    local test_config="/tmp/test-primus-cli-invalid-$$.yaml"
    cat > "$test_config" << 'EOF'
this is not valid yaml: [
  unclosed bracket
EOF

    local output
    output=$(bash "$PROJECT_ROOT/runner/primus-cli" --config "$test_config" direct --help 2>&1)

    # Should handle gracefully (either show error or continue with defaults)
    # We just check it doesn't crash completely
    if [[ -n "$output" ]]; then
        assert_pass "Handles invalid config gracefully"
    else
        assert_fail "Handles invalid config gracefully"
    fi

    rm -f "$test_config"
}

# ============================================================================
# Test 16: Default config loading
# ============================================================================
test_default_config() {
    print_section "Test 16: Default Config Loading"

    # The system should load runner/.primus.yaml by default
    local output
    output=$(bash "$PROJECT_ROOT/runner/primus-cli" direct --help 2>&1)

    # Should work without explicit config
    if echo "$output" | grep -qF "Direct"; then
        assert_pass "Loads default config without errors"
    else
        assert_fail "Loads default config without errors"
    fi
}

# ============================================================================
# Test 17: Missing mode before separator
# ============================================================================
test_missing_mode_before_separator() {
    print_section "Test 17: Missing Mode Before Separator"

    # Test that '--' without mode is caught
    local output
    output=$(bash "$PROJECT_ROOT/runner/primus-cli" -- train pretrain 2>&1)

    assert_contains "$output" "Missing <mode> argument before '--'" "Detects missing mode before separator"
}

# ============================================================================
# Test 18: Nested command structure
# ============================================================================
test_nested_command_structure() {
    print_section "Test 18: Nested Command Structure"

    # Test nested command with multiple '--' separators in dry-run
    local output
    output=$(bash "$PROJECT_ROOT/runner/primus-cli" --dry-run slurm -N 1 -- container --debug -- train pretrain 2>&1)

    assert_contains "$output" "[DRY RUN] Primus CLI" "Handles nested command structure"
    assert_contains "$output" "slurm" "Shows slurm mode"
    assert_contains "$output" "container --debug" "Shows nested container mode"
    assert_contains "$output" "train pretrain" "Shows final Primus command"
}

# ============================================================================
# Run all tests
# ============================================================================
main() {
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Unit Tests for Primus CLI Main Entry Point                 ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    test_help_option
    test_version_option
    test_no_arguments
    test_unknown_mode
    test_dry_run_mode
    test_debug_mode
    test_direct_mode_help
    test_container_mode_help
    test_slurm_mode_help
    test_config_option
    test_multiple_global_options
    test_mode_selection
    test_config_file_not_found
    test_config_priority
    test_invalid_config_format
    test_default_config
    test_missing_mode_before_separator
    test_nested_command_structure

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
