#!/bin/bash
###############################################################################
# Unit tests for primus-cli-slurm.sh
# Uses dry-run mode to verify functionality
###############################################################################

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
    local reason="${2:-}"
    ((TESTS_RUN++))
    echo -e "${RED}✗${NC} $test_name"
    if [[ -n "$reason" ]]; then
        echo "  Reason: $reason"
    fi
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
    print_section "Test 1: Basic Dry-run Functionality"

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" --dry-run srun -N 2 -- container -- train 2>&1)

    assert_contains "$output" "[DRY RUN]" "Should contain DRY RUN marker"
    assert_contains "$output" "srun" "Should show srun command"
    assert_contains "$output" "-N 2" "Should include node count"
    assert_contains "$output" "primus-cli-slurm-entry.sh" "Should call slurm-entry script"
    assert_contains "$output" "container" "Should pass container mode"
}

# ============================================================================
# Test 2: Slurm launcher selection (srun vs sbatch)
# ============================================================================
test_launcher_selection() {
    print_section "Test 2: Slurm Launcher Selection"

    local output_srun
    output_srun=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" --dry-run srun -- container -- train 2>&1)

    local output_sbatch
    output_sbatch=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" --dry-run sbatch -- container -- train 2>&1)

    local output_default
    output_default=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" --dry-run -- container -- train 2>&1)

    assert_contains "$output_srun" "srun" "Should use srun when specified"
    assert_contains "$output_sbatch" "sbatch" "Should use sbatch when specified"
    assert_contains "$output_default" "srun" "Should default to srun"
}

# ============================================================================
# Test 3: Slurm flags passing
# ============================================================================
test_slurm_flags() {
    print_section "Test 3: Slurm Flags Passing"

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" \
        --dry-run \
        srun -N 4 -p AIG_Model --output=run.log \
        -- container -- train 2>&1)

    assert_contains "$output" "-N 4" "Should include node count"
    assert_contains "$output" "-p AIG_Model" "Should include partition"
    assert_contains "$output" "--output=run.log" "Should include output file"
}

# ============================================================================
# Test 4: Configuration file loading
# ============================================================================
test_config_file() {
    print_section "Test 4: Configuration File Loading"

    # Create test config
    local test_config="/tmp/test-slurm-config-$$.yaml"
    cat > "$test_config" << 'EOF'
slurm:
  partition: "test_partition"
  nodes: 8
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" \
        --config "$test_config" \
        --dry-run \
        srun -- container -- train 2>&1)

    assert_contains "$output" "-p test_partition" "Should use config partition"
    assert_contains "$output" "-N 8" "Should use config nodes"
    assert_contains "$output" "--config $test_config" "Should pass config to entry script"

    rm -f "$test_config"
}

# ============================================================================
# Test 5: CLI overrides config (parameter priority)
# ============================================================================
test_cli_overrides_config() {
    print_section "Test 5: CLI Overrides Config"

    # Create test config
    local test_config="/tmp/test-slurm-priority-$$.yaml"
    cat > "$test_config" << 'EOF'
slurm:
  partition: "config_partition"
  nodes: 4
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" \
        --config "$test_config" \
        --dry-run \
        srun -N 16 -p cli_partition \
        -- container -- train 2>&1)

    # CLI should override config
    assert_contains "$output" "-p cli_partition" "CLI partition should override config"
    assert_contains "$output" "-N 16" "CLI nodes should override config"
    assert_not_contains "$output" "config_partition" "Config partition should not appear"
    # Note: -N 4 from config should not appear (overridden)

    rm -f "$test_config"
}

# ============================================================================
# Test 6: Config values used when CLI not provided
# ============================================================================
test_config_used_when_cli_not_provided() {
    print_section "Test 6: Config Used When CLI Not Provided"

    # Create test config
    local test_config="/tmp/test-slurm-fallback-$$.yaml"
    cat > "$test_config" << 'EOF'
slurm:
  partition: "fallback_partition"
  nodes: 12
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" \
        --config "$test_config" \
        --dry-run \
        srun \
        -- container -- train 2>&1)

    assert_contains "$output" "-p fallback_partition" "Should use config partition when CLI not provided"
    assert_contains "$output" "-N 12" "Should use config nodes when CLI not provided"

    rm -f "$test_config"
}

# ============================================================================
# Test 7: Partial override (only some parameters)
# ============================================================================
test_partial_override() {
    print_section "Test 7: Partial Override"

    # Create test config
    local test_config="/tmp/test-slurm-partial-$$.yaml"
    cat > "$test_config" << 'EOF'
slurm:
  partition: "config_part"
  nodes: 8
EOF

    # Override only partition, keep nodes from config
    local output
    output=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" \
        --config "$test_config" \
        --dry-run \
        srun -p cli_part \
        -- container -- train 2>&1)

    assert_contains "$output" "-p cli_part" "Should use CLI partition"
    assert_contains "$output" "-N 8" "Should use config nodes (not overridden)"
    assert_not_contains "$output" "config_part" "Config partition should not appear"

    rm -f "$test_config"
}

# ============================================================================
# Test 8: Debug and config flags passed to entry script
# ============================================================================
test_flags_passed_to_entry() {
    print_section "Test 8: Debug and Config Flags Passed to Entry"

    local test_config="/tmp/test-slurm-flags-$$.yaml"
    cat > "$test_config" << 'EOF'
slurm:
  partition: "test"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" \
        --config "$test_config" \
        --debug \
        --dry-run \
        srun \
        -- container -- train 2>&1)

    assert_contains "$output" "--config $test_config" "Should pass --config to entry"
    assert_contains "$output" "--debug" "Should pass --debug to entry"

    rm -f "$test_config"
}

# ============================================================================
# Test 9: Entry mode (container vs direct)
# ============================================================================
test_entry_mode() {
    print_section "Test 9: Entry Mode"

    local output_container
    output_container=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" \
        --dry-run \
        srun -N 2 -- container -- train 2>&1)

    local output_direct
    output_direct=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" \
        --dry-run \
        srun -N 2 -- direct -- train 2>&1)

    assert_contains "$output_container" "container" "Should pass container mode"
    assert_contains "$output_direct" "direct" "Should pass direct mode"
}

# ============================================================================
# Test 10: Help message
# ============================================================================
test_help_message() {
    print_section "Test 10: Help Message"

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" --help 2>&1)

    assert_contains "$output" "Usage:" "Should show usage"
    assert_contains "$output" "config" "Should document --config"
    assert_contains "$output" "debug" "Should document --debug"
    assert_contains "$output" "dry-run" "Should document --dry-run"
    assert_contains "$output" "srun" "Should document srun"
    assert_contains "$output" "sbatch" "Should document sbatch"
}

# ============================================================================
# Test 11: Multiple Slurm flags
# ============================================================================
test_multiple_slurm_flags() {
    print_section "Test 11: Multiple Slurm Flags"

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" \
        --dry-run \
        srun \
        -N 8 \
        -p gpu_partition \
        --time=01:00:00 \
        --mem=128G \
        --gres=gpu:8 \
        -- container -- train 2>&1)

    assert_contains "$output" "-N 8" "Should include nodes"
    assert_contains "$output" "-p gpu_partition" "Should include partition"
    assert_contains "$output" "--time=01:00:00" "Should include time limit"
    assert_contains "$output" "--mem=128G" "Should include memory"
    assert_contains "$output" "--gres=gpu:8" "Should include GPU resources"
}

# ============================================================================
# Test 12: Arguments after entry mode
# ============================================================================
test_arguments_after_entry() {
    print_section "Test 12: Arguments After Entry Mode"

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" \
        --dry-run \
        srun -N 2 \
        -- container -- train pretrain --config exp.yaml --batch-size 32 2>&1)

    assert_contains "$output" "train" "Should pass train command"
    assert_contains "$output" "pretrain" "Should pass pretrain argument"
    assert_contains "$output" "exp.yaml" "Should pass config argument"
    assert_contains "$output" "batch-size" "Should pass batch-size argument"
}

# ============================================================================
# Test 13: Long option format override
# ============================================================================
test_long_option_override() {
    print_section "Test 13: Long Option Format Override"

    # Create test config
    local test_config="/tmp/test-slurm-long-opt-$$.yaml"
    cat > "$test_config" << 'EOF'
slurm:
  partition: "config_partition"
  nodes: 4
EOF

    # Use long option format to override
    local output
    output=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" \
        --config "$test_config" \
        --dry-run \
        srun --nodes 16 --partition cli_partition \
        -- container -- train 2>&1)

    assert_contains "$output" "--partition cli_partition" "Long option should override config"
    assert_contains "$output" "--nodes 16" "Long option should override config"
    assert_not_contains "$output" "config_partition" "Config value should not appear"

    rm -f "$test_config"
}

# ============================================================================
# Test 14: Pure passthrough - arbitrary Slurm parameters
# ============================================================================
test_pure_passthrough() {
    print_section "Test 14: Pure Passthrough - Arbitrary Parameters"

    # Test with various Slurm parameters that are NOT hardcoded
    local output
    output=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" \
        --dry-run \
        srun \
        --exclusive \
        --ntasks-per-node=8 \
        --cpus-per-task=16 \
        --job-name=my_training \
        --account=research_team \
        -C gpu \
        -- container -- train 2>&1)

    assert_contains "$output" "--exclusive" "Should passthrough --exclusive"
    assert_contains "$output" "--ntasks-per-node=8" "Should passthrough --ntasks-per-node"
    assert_contains "$output" "--cpus-per-task=16" "Should passthrough --cpus-per-task"
    assert_contains "$output" "--job-name=my_training" "Should passthrough --job-name"
    assert_contains "$output" "--account=research_team" "Should passthrough --account"
    assert_contains "$output" "-C gpu" "Should passthrough -C constraint"
}

# ============================================================================
# Test 15: Parameter order (config first, CLI second)
# ============================================================================
test_parameter_order() {
    print_section "Test 15: Parameter Order (Config First, CLI Second)"

    # Create test config
    local test_config="/tmp/test-slurm-order-$$.yaml"
    cat > "$test_config" << 'EOF'
slurm:
  partition: "config_part"
  nodes: 4
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" \
        --config "$test_config" \
        --dry-run \
        srun --time=01:00:00 --mem=64G \
        -- container -- train 2>&1)

    # Verify both config params and CLI params are present
    # Config params should appear before CLI params in the command
    assert_contains "$output" "-p config_part" "Config partition should be present"
    assert_contains "$output" "-N 4" "Config nodes should be present"
    assert_contains "$output" "--time=01:00:00" "CLI time should be present"
    assert_contains "$output" "--mem=64G" "CLI mem should be present"
}

# ============================================================================
# Test 16: Mixed short and long options with override
# ============================================================================
test_mixed_short_long_override() {
    print_section "Test 16: Mixed Short and Long Options Override"

    # Create test config with both partition and nodes
    local test_config="/tmp/test-slurm-mixed-$$.yaml"
    cat > "$test_config" << 'EOF'
slurm:
  partition: "default_partition"
  nodes: 8
EOF

    # Override partition with short option, nodes with long option
    local output
    output=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" \
        --config "$test_config" \
        --dry-run \
        srun -p gpu_queue --nodes 32 \
        -- container -- train 2>&1)

    assert_contains "$output" "-p gpu_queue" "Short option should override partition"
    assert_contains "$output" "--nodes 32" "Long option should override nodes"
    assert_not_contains "$output" "default_partition" "Default partition should not appear"
    # Note: Both -N 8 and its equivalents should be overridden

    rm -f "$test_config"
}

# ============================================================================
# Test 17: Sbatch-specific parameters
# ============================================================================
test_sbatch_specific_params() {
    print_section "Test 17: Sbatch-Specific Parameters"

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" \
        --dry-run \
        sbatch \
        --job-name=training_job \
        --output=logs/job-%j.out \
        --error=logs/job-%j.err \
        --mail-type=END,FAIL \
        --mail-user=user@example.com \
        -N 4 \
        -- container -- train 2>&1)

    assert_contains "$output" "sbatch" "Should use sbatch"
    assert_contains "$output" "--job-name=training_job" "Should include job name"
    assert_contains "$output" "--output=logs/job-%j.out" "Should include output path"
    assert_contains "$output" "--error=logs/job-%j.err" "Should include error path"
    assert_contains "$output" "--mail-type=END,FAIL" "Should include mail type"
    assert_contains "$output" "--mail-user=user@example.com" "Should include mail user"
    assert_contains "$output" "-N 4" "Should include nodes"
}

# ============================================================================
# Test 18: Config-based dry_run
# ============================================================================
test_config_dry_run() {
    print_section "Test 18: Config-Based Dry Run"

    # Create test config with dry_run enabled
    local test_config="/tmp/test-slurm-config-dryrun-$$.yaml"
    cat > "$test_config" << 'EOF'
slurm:
  dry_run: true
  partition: "test_partition"
  nodes: 2
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" \
        --config "$test_config" \
        srun -- container -- train 2>&1)

    assert_contains "$output" "[DRY RUN]" "Should enable dry-run from config"
    assert_contains "$output" "srun" "Should show srun command"
    assert_contains "$output" "-p test_partition" "Should use config partition"

    rm -f "$test_config"
}

# ============================================================================
# Test 19: Config-based debug
# ============================================================================
test_config_debug() {
    print_section "Test 19: Config-Based Debug"

    # Create test config with debug enabled
    local test_config="/tmp/test-slurm-config-debug-$$.yaml"
    cat > "$test_config" << 'EOF'
slurm:
  debug: true
  dry_run: true
  partition: "test_partition"
EOF

    local output
    output=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" \
        --config "$test_config" \
        srun -N 2 -- container -- train 2>&1)

    # Debug mode should show debug log message
    if echo "$output" | grep -qE "(Debug mode enabled|PRIMUS_LOG_LEVEL=DEBUG)"; then
        assert_pass "Config debug mode enables verbose output"
    else
        assert_fail "Config debug mode enables verbose output" "No debug output found"
    fi

    rm -f "$test_config"
}

# ============================================================================
# Test 20: CLI overrides config debug/dry_run
# ============================================================================
test_cli_overrides_config_debug_dryrun() {
    print_section "Test 20: CLI Overrides Config Debug/Dry-Run"

    # Create test config with debug=false and dry_run=false
    local test_config="/tmp/test-slurm-override-debug-$$.yaml"
    cat > "$test_config" << 'EOF'
slurm:
  debug: false
  dry_run: false
  partition: "test_partition"
EOF

    # CLI should override config (enable both via CLI)
    local output
    output=$(bash "$RUNNER_DIR/primus-cli-slurm.sh" \
        --config "$test_config" \
        --debug \
        --dry-run \
        srun -N 2 -- container -- train 2>&1)

    assert_contains "$output" "[DRY RUN]" "CLI --dry-run should override config"

    # Debug mode should be enabled (check for debug log message)
    if echo "$output" | grep -qE "(Debug mode enabled|PRIMUS_LOG_LEVEL=DEBUG)"; then
        assert_pass "CLI --debug should override config"
    else
        assert_fail "CLI --debug should override config" "No debug output found"
    fi

    rm -f "$test_config"
}

# ============================================================================
# Run all tests
# ============================================================================
main() {
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Unit Tests for primus-cli-slurm.sh                         ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    test_basic_dry_run
    test_launcher_selection
    test_slurm_flags
    test_config_file
    test_cli_overrides_config
    test_config_used_when_cli_not_provided
    test_partial_override
    test_flags_passed_to_entry
    test_entry_mode
    test_help_message
    test_multiple_slurm_flags
    test_arguments_after_entry
    test_long_option_override
    test_pure_passthrough
    test_parameter_order
    test_mixed_short_long_override
    test_sbatch_specific_params
    test_config_dry_run
    test_config_debug
    test_cli_overrides_config_debug_dryrun

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
