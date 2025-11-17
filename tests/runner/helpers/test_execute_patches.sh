#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Unit tests for execute_patches.sh

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNNER_DIR="$PROJECT_ROOT/runner"

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Color codes
COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_YELLOW='\033[1;33m'
COLOR_RESET='\033[0m'

# Print section header
print_section() {
    echo ""
    echo -e "${COLOR_YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${COLOR_RESET}"
    echo -e "${COLOR_YELLOW}$1${COLOR_RESET}"
    echo -e "${COLOR_YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${COLOR_RESET}"
}

# Assert helper
assert_contains() {
    TESTS_RUN=$((TESTS_RUN + 1))
    local output="$1"
    local expected="$2"
    local description="$3"

    if echo "$output" | grep -q "$expected"; then
        echo -e "${COLOR_GREEN}✓${COLOR_RESET} $description"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${COLOR_RED}✗${COLOR_RESET} $description"
        echo "  Expected to contain: $expected"
        echo "  Actual output (first 10 lines):"
        echo "$output" | head -10 | sed 's/^/  /'
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

assert_not_contains() {
    TESTS_RUN=$((TESTS_RUN + 1))
    local output="$1"
    local unexpected="$2"
    local description="$3"

    if ! echo "$output" | grep -q "$unexpected"; then
        echo -e "${COLOR_GREEN}✓${COLOR_RESET} $description"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${COLOR_RED}✗${COLOR_RESET} $description"
        echo "  Should not contain: $unexpected"
        echo "  Actual output (first 10 lines):"
        echo "$output" | head -10 | sed 's/^/  /'
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

assert_exit_code() {
    TESTS_RUN=$((TESTS_RUN + 1))
    local actual_code="$1"
    local expected_code="$2"
    local description="$3"

    if [[ "$actual_code" -eq "$expected_code" ]]; then
        echo -e "${COLOR_GREEN}✓${COLOR_RESET} $description"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${COLOR_RED}✗${COLOR_RESET} $description"
        echo "  Expected exit code: $expected_code"
        echo "  Actual exit code: $actual_code"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# ============================================================================
# Test 1: No patch scripts (empty call)
# ============================================================================
test_no_patches() {
    print_section "Test 1: No Patch Scripts"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_patches.sh" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 0 "Exit code is 0 for no patches"
    assert_contains "$output" "No patch scripts specified" "No patches message displayed"
}

# ============================================================================
# Test 2: Single successful patch
# ============================================================================
test_single_success() {
    print_section "Test 2: Single Successful Patch"

    # Create test patch
    local test_patch="/tmp/test_patch_success_$$.sh"
    cat > "$test_patch" << 'EOF'
#!/bin/bash
echo "Patch executing successfully"
exit 0
EOF
    chmod +x "$test_patch"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_patches.sh" "$test_patch" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 0 "Exit code is 0 for successful patch"
    assert_contains "$output" "Detected patch scripts" "Detected message shown"
    assert_contains "$output" "Running patch: bash $test_patch" "Running message shown"
    assert_contains "$output" "Patch completed successfully" "Success message shown"
    assert_contains "$output" "All patch scripts executed successfully" "All patches success message shown"

    rm -f "$test_patch"
}

# ============================================================================
# Test 3: Patch with exit code 2 (skip)
# ============================================================================
test_patch_skip() {
    print_section "Test 3: Patch Skip (Exit Code 2)"

    # Create test patch that skips
    local test_patch="/tmp/test_patch_skip_$$.sh"
    cat > "$test_patch" << 'EOF'
#!/bin/bash
echo "Conditions not met, skipping"
exit 2
EOF
    chmod +x "$test_patch"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_patches.sh" "$test_patch" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 0 "Exit code is 0 when patch skips"
    assert_contains "$output" "Patch skipped (exit code 2)" "Skip message shown"
    assert_contains "$output" "All patch scripts executed successfully" "All patches success message shown"

    rm -f "$test_patch"
}

# ============================================================================
# Test 4: Patch failure (exit code 1)
# ============================================================================
test_patch_failure() {
    print_section "Test 4: Patch Failure"

    # Create test patch that fails
    local test_patch="/tmp/test_patch_fail_$$.sh"
    cat > "$test_patch" << 'EOF'
#!/bin/bash
echo "Patch failed"
exit 1
EOF
    chmod +x "$test_patch"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_patches.sh" "$test_patch" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 1 "Exit code is 1 for failed patch"
    assert_contains "$output" "Patch script failed" "Failure message shown"
    assert_contains "$output" "exit code: 1" "Exit code displayed"
    assert_not_contains "$output" "All patch scripts executed successfully" "Success message not shown"

    rm -f "$test_patch"
}

# ============================================================================
# Test 5: Multiple patches (success + skip + success)
# ============================================================================
test_multiple_patches() {
    print_section "Test 5: Multiple Patches (Success + Skip + Success)"

    # Create test patches
    local patch1="/tmp/test_patch_1_$$.sh"
    local patch2="/tmp/test_patch_2_$$.sh"
    local patch3="/tmp/test_patch_3_$$.sh"

    cat > "$patch1" << 'EOF'
#!/bin/bash
echo "Patch 1 executing"
exit 0
EOF
    chmod +x "$patch1"

    cat > "$patch2" << 'EOF'
#!/bin/bash
echo "Patch 2 skipping"
exit 2
EOF
    chmod +x "$patch2"

    cat > "$patch3" << 'EOF'
#!/bin/bash
echo "Patch 3 executing"
exit 0
EOF
    chmod +x "$patch3"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_patches.sh" "$patch1" "$patch2" "$patch3" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 0 "Exit code is 0 for mixed patches"
    assert_contains "$output" "Patch 1 executing" "Patch 1 executed"
    assert_contains "$output" "Patch completed successfully: $patch1" "Patch 1 success"
    assert_contains "$output" "Patch 2 skipping" "Patch 2 skipped"
    assert_contains "$output" "Patch skipped (exit code 2): $patch2" "Patch 2 skip message"
    assert_contains "$output" "Patch 3 executing" "Patch 3 executed"
    assert_contains "$output" "Patch completed successfully: $patch3" "Patch 3 success"
    assert_contains "$output" "All patch scripts executed successfully" "All patches success message"

    rm -f "$patch1" "$patch2" "$patch3"
}

# ============================================================================
# Test 6: Patch not found
# ============================================================================
test_patch_not_found() {
    print_section "Test 6: Patch Not Found"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_patches.sh" "/tmp/nonexistent_patch_$$.sh" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 1 "Exit code is 1 for missing patch"
    assert_contains "$output" "Patch script not found" "Not found error shown"
    assert_not_contains "$output" "All patch scripts executed successfully" "Success message not shown"
}

# ============================================================================
# Test 7: Patch not readable (no permission)
# ============================================================================
test_patch_not_readable() {
    print_section "Test 7: Patch Not Readable"

    # Skip this test if running as root (root can read any file)
    if [[ $EUID -eq 0 ]]; then
        echo -e "${COLOR_YELLOW}⚠${COLOR_RESET} Skipping test (running as root)"
        return 0
    fi

    # Create test patch without read permission
    local test_patch="/tmp/test_patch_noread_$$.sh"
    cat > "$test_patch" << 'EOF'
#!/bin/bash
echo "Should not run"
exit 0
EOF
    chmod 000 "$test_patch"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_patches.sh" "$test_patch" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 1 "Exit code is 1 for unreadable patch"
    assert_contains "$output" "Patch script not readable" "Not readable error shown"
    assert_not_contains "$output" "All patch scripts executed successfully" "Success message not shown"

    chmod 644 "$test_patch"
    rm -f "$test_patch"
}

# ============================================================================
# Test 8: Stop on first failure
# ============================================================================
test_stop_on_failure() {
    print_section "Test 8: Stop on First Failure"

    # Create test patches
    local patch1="/tmp/test_patch_ok_$$.sh"
    local patch2="/tmp/test_patch_fail_$$.sh"
    local patch3="/tmp/test_patch_never_run_$$.sh"

    cat > "$patch1" << 'EOF'
#!/bin/bash
echo "Patch 1 OK"
exit 0
EOF
    chmod +x "$patch1"

    cat > "$patch2" << 'EOF'
#!/bin/bash
echo "Patch 2 FAIL"
exit 1
EOF
    chmod +x "$patch2"

    cat > "$patch3" << 'EOF'
#!/bin/bash
echo "Patch 3 should not run"
exit 0
EOF
    chmod +x "$patch3"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_patches.sh" "$patch1" "$patch2" "$patch3" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 1 "Exit code is 1 when patch fails"
    assert_contains "$output" "Patch 1 OK" "Patch 1 executed"
    assert_contains "$output" "Patch 2 FAIL" "Patch 2 executed"
    assert_not_contains "$output" "Patch 3 should not run" "Patch 3 not executed"
    assert_contains "$output" "Patch script failed" "Failure message shown"

    rm -f "$patch1" "$patch2" "$patch3"
}

# ============================================================================
# Test 9: Patch with custom exit code (not 0, 1, or 2)
# ============================================================================
test_custom_exit_code() {
    print_section "Test 9: Patch with Custom Exit Code"

    # Create test patch with exit code 5
    local test_patch="/tmp/test_patch_custom_$$.sh"
    cat > "$test_patch" << 'EOF'
#!/bin/bash
echo "Patch with custom exit code"
exit 5
EOF
    chmod +x "$test_patch"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_patches.sh" "$test_patch" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 1 "Exit code is 1 for custom exit code"
    assert_contains "$output" "Patch script failed" "Failure message shown"
    assert_contains "$output" "exit code: 5" "Custom exit code displayed"

    rm -f "$test_patch"
}

# ============================================================================
# Run all tests
# ============================================================================
echo "Starting execute_patches.sh unit tests..."
echo "Project root: $PROJECT_ROOT"

test_no_patches
test_single_success
test_patch_skip
test_patch_failure
test_multiple_patches
test_patch_not_found
test_patch_not_readable
test_stop_on_failure
test_custom_exit_code

# Print summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test Summary:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Total:  $TESTS_RUN"
echo -e "  Passed: ${COLOR_GREEN}$TESTS_PASSED${COLOR_RESET}"
if [[ $TESTS_FAILED -gt 0 ]]; then
    echo -e "  Failed: ${COLOR_RED}$TESTS_FAILED${COLOR_RESET}"
else
    echo "  Failed: 0"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ $TESTS_FAILED -eq 0 ]]; then
    echo -e "${COLOR_GREEN}✓ All tests passed!${COLOR_RESET}"
    exit 0
else
    echo -e "${COLOR_RED}✗ Some tests failed${COLOR_RESET}"
    exit 1
fi
