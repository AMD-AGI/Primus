#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Unit tests for execute_hooks.sh

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNNER_DIR="$PROJECT_ROOT/runner"
HOOKS_BASE_DIR="$RUNNER_DIR/helpers/hooks"

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
# Test 1: No arguments (missing hook group and name)
# ============================================================================
test_no_arguments() {
    print_section "Test 1: No Arguments"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_hooks.sh" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 0 "Exit code is 0 for no arguments"
    assert_contains "$output" "No hook target specified" "No target message displayed"
}

# ============================================================================
# Test 2: Only one argument (missing hook name)
# ============================================================================
test_missing_hook_name() {
    print_section "Test 2: Missing Hook Name"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_hooks.sh" "train" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 0 "Exit code is 0 for missing name"
    assert_contains "$output" "No hook target specified" "No target message displayed"
}

# ============================================================================
# Test 3: Non-existent hook directory
# ============================================================================
test_nonexistent_hook_dir() {
    print_section "Test 3: Non-existent Hook Directory"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_hooks.sh" "nonexistent_group" "nonexistent_hook" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 0 "Exit code is 0 for non-existent directory"
    assert_contains "$output" "No hook directory for" "No directory message displayed"
}

# ============================================================================
# Test 4: Empty hook directory
# ============================================================================
test_empty_hook_dir() {
    print_section "Test 4: Empty Hook Directory"

    # Create empty hook directory
    local test_group="test_group_$$"
    local test_name="empty_hook"
    local test_hook_dir="$HOOKS_BASE_DIR/$test_group/$test_name"
    mkdir -p "$test_hook_dir"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_hooks.sh" "$test_group" "$test_name" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 0 "Exit code is 0 for empty directory"
    assert_contains "$output" "Detected hooks directory" "Directory detected message"
    assert_contains "$output" "No hook files found" "No files message displayed"

    # Cleanup
    rm -rf "${HOOKS_BASE_DIR:?}/$test_group"
}

# ============================================================================
# Test 5: Single successful bash hook
# ============================================================================
test_single_bash_hook() {
    print_section "Test 5: Single Successful Bash Hook"

    # Create test hook
    local test_group="test_group_$$"
    local test_name="bash_hook"
    local test_hook_dir="$HOOKS_BASE_DIR/$test_group/$test_name"
    mkdir -p "$test_hook_dir"

    cat > "$test_hook_dir/01_test.sh" << 'EOF'
#!/bin/bash
echo "Bash hook executing"
exit 0
EOF
    chmod +x "$test_hook_dir/01_test.sh"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_hooks.sh" "$test_group" "$test_name" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 0 "Exit code is 0 for successful hook"
    assert_contains "$output" "Detected hooks directory" "Directory detected"
    assert_contains "$output" "Executing hook.*01_test.sh" "Hook execution message"
    assert_contains "$output" "Bash hook executing" "Hook output shown"
    assert_contains "$output" "Hook.*finished in.*s" "Hook completion message"
    assert_contains "$output" "All hooks executed successfully" "Success message shown"

    # Cleanup
    rm -rf "${HOOKS_BASE_DIR:?}/$test_group"
}

# ============================================================================
# Test 6: Single successful Python hook
# ============================================================================
test_single_python_hook() {
    print_section "Test 6: Single Successful Python Hook"

    # Create test hook
    local test_group="test_group_$$"
    local test_name="python_hook"
    local test_hook_dir="$HOOKS_BASE_DIR/$test_group/$test_name"
    mkdir -p "$test_hook_dir"

    cat > "$test_hook_dir/01_test.py" << 'EOF'
#!/usr/bin/env python3
print("Python hook executing")
exit(0)
EOF
    chmod +x "$test_hook_dir/01_test.py"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_hooks.sh" "$test_group" "$test_name" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 0 "Exit code is 0 for successful Python hook"
    assert_contains "$output" "Executing hook.*01_test.py" "Python hook execution message"
    assert_contains "$output" "Python hook executing" "Python hook output shown"
    assert_contains "$output" "All hooks executed successfully" "Success message shown"

    # Cleanup
    rm -rf "${HOOKS_BASE_DIR:?}/$test_group"
}

# ============================================================================
# Test 7: Hook with arguments
# ============================================================================
test_hook_with_arguments() {
    print_section "Test 7: Hook with Arguments"

    # Create test hook that echoes its arguments
    local test_group="test_group_$$"
    local test_name="args_hook"
    local test_hook_dir="$HOOKS_BASE_DIR/$test_group/$test_name"
    mkdir -p "$test_hook_dir"

    cat > "$test_hook_dir/01_test.sh" << 'EOF'
#!/bin/bash
echo "Hook received args: $*"
exit 0
EOF
    chmod +x "$test_hook_dir/01_test.sh"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_hooks.sh" "$test_group" "$test_name" "arg1" "arg2" "arg3" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 0 "Exit code is 0 for hook with args"
    assert_contains "$output" "Executing hook.*arg1 arg2 arg3" "Arguments passed to hook"
    assert_contains "$output" "Hook received args: arg1 arg2 arg3" "Hook received arguments"

    # Cleanup
    rm -rf "${HOOKS_BASE_DIR:?}/$test_group"
}

# ============================================================================
# Test 8: Multiple hooks (sorted execution)
# ============================================================================
test_multiple_hooks() {
    print_section "Test 8: Multiple Hooks (Sorted Execution)"

    # Create multiple test hooks
    local test_group="test_group_$$"
    local test_name="multi_hook"
    local test_hook_dir="$HOOKS_BASE_DIR/$test_group/$test_name"
    mkdir -p "$test_hook_dir"

    cat > "$test_hook_dir/03_third.sh" << 'EOF'
#!/bin/bash
echo "Third hook"
exit 0
EOF
    chmod +x "$test_hook_dir/03_third.sh"

    cat > "$test_hook_dir/01_first.sh" << 'EOF'
#!/bin/bash
echo "First hook"
exit 0
EOF
    chmod +x "$test_hook_dir/01_first.sh"

    cat > "$test_hook_dir/02_second.py" << 'EOF'
#!/usr/bin/env python3
print("Second hook")
exit(0)
EOF
    chmod +x "$test_hook_dir/02_second.py"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_hooks.sh" "$test_group" "$test_name" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 0 "Exit code is 0 for multiple hooks"

    # Check execution order by finding line numbers
    local first_line
    local second_line
    local third_line
    first_line=$(echo "$output" | grep -n "First hook" | cut -d: -f1)
    second_line=$(echo "$output" | grep -n "Second hook" | cut -d: -f1)
    third_line=$(echo "$output" | grep -n "Third hook" | cut -d: -f1)

    if [[ "$first_line" -lt "$second_line" ]] && [[ "$second_line" -lt "$third_line" ]]; then
        echo -e "${COLOR_GREEN}✓${COLOR_RESET} Hooks executed in sorted order"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${COLOR_RED}✗${COLOR_RESET} Hooks executed in sorted order"
        echo "  First: line $first_line, Second: line $second_line, Third: line $third_line"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    TESTS_RUN=$((TESTS_RUN + 1))

    assert_contains "$output" "All hooks executed successfully" "All hooks success message"

    # Cleanup
    rm -rf "${HOOKS_BASE_DIR:?}/$test_group"
}

# ============================================================================
# Test 9: Hook failure (bash)
# ============================================================================
test_bash_hook_failure() {
    print_section "Test 9: Bash Hook Failure"

    # Create failing hook
    local test_group="test_group_$$"
    local test_name="fail_hook"
    local test_hook_dir="$HOOKS_BASE_DIR/$test_group/$test_name"
    mkdir -p "$test_hook_dir"

    cat > "$test_hook_dir/01_fail.sh" << 'EOF'
#!/bin/bash
echo "Hook is failing"
exit 1
EOF
    chmod +x "$test_hook_dir/01_fail.sh"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_hooks.sh" "$test_group" "$test_name" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 1 "Exit code is 1 for failed hook"
    assert_contains "$output" "Hook failed.*01_fail.sh" "Failure message shown"
    assert_not_contains "$output" "All hooks executed successfully" "Success message not shown"

    # Cleanup
    rm -rf "${HOOKS_BASE_DIR:?}/$test_group"
}

# ============================================================================
# Test 10: Hook failure (python)
# ============================================================================
test_python_hook_failure() {
    print_section "Test 10: Python Hook Failure"

    # Create failing Python hook
    local test_group="test_group_$$"
    local test_name="fail_python_hook"
    local test_hook_dir="$HOOKS_BASE_DIR/$test_group/$test_name"
    mkdir -p "$test_hook_dir"

    cat > "$test_hook_dir/01_fail.py" << 'EOF'
#!/usr/bin/env python3
print("Python hook is failing")
exit(1)
EOF
    chmod +x "$test_hook_dir/01_fail.py"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_hooks.sh" "$test_group" "$test_name" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 1 "Exit code is 1 for failed Python hook"
    assert_contains "$output" "Hook failed.*01_fail.py" "Python failure message shown"
    assert_not_contains "$output" "All hooks executed successfully" "Success message not shown"

    # Cleanup
    rm -rf "${HOOKS_BASE_DIR:?}/$test_group"
}

# ============================================================================
# Test 11: Stop on first failure
# ============================================================================
test_stop_on_failure() {
    print_section "Test 11: Stop on First Failure"

    # Create multiple hooks where second one fails
    local test_group="test_group_$$"
    local test_name="stop_on_fail"
    local test_hook_dir="$HOOKS_BASE_DIR/$test_group/$test_name"
    mkdir -p "$test_hook_dir"

    cat > "$test_hook_dir/01_ok.sh" << 'EOF'
#!/bin/bash
echo "First hook OK"
exit 0
EOF
    chmod +x "$test_hook_dir/01_ok.sh"

    cat > "$test_hook_dir/02_fail.sh" << 'EOF'
#!/bin/bash
echo "Second hook FAIL"
exit 1
EOF
    chmod +x "$test_hook_dir/02_fail.sh"

    cat > "$test_hook_dir/03_never.sh" << 'EOF'
#!/bin/bash
echo "Third hook should not run"
exit 0
EOF
    chmod +x "$test_hook_dir/03_never.sh"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_hooks.sh" "$test_group" "$test_name" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 1 "Exit code is 1 when hook fails"
    assert_contains "$output" "First hook OK" "First hook executed"
    assert_contains "$output" "Second hook FAIL" "Second hook executed"
    assert_not_contains "$output" "Third hook should not run" "Third hook not executed"
    assert_contains "$output" "Hook failed" "Failure message shown"

    # Cleanup
    rm -rf "${HOOKS_BASE_DIR:?}/$test_group"
}

# ============================================================================
# Test 12: Mixed hook types (bash and python)
# ============================================================================
test_mixed_hook_types() {
    print_section "Test 12: Mixed Hook Types"

    # Create mixed hooks
    local test_group="test_group_$$"
    local test_name="mixed_hooks"
    local test_hook_dir="$HOOKS_BASE_DIR/$test_group/$test_name"
    mkdir -p "$test_hook_dir"

    cat > "$test_hook_dir/01_bash.sh" << 'EOF'
#!/bin/bash
echo "Bash hook"
exit 0
EOF
    chmod +x "$test_hook_dir/01_bash.sh"

    cat > "$test_hook_dir/02_python.py" << 'EOF'
#!/usr/bin/env python3
print("Python hook")
exit(0)
EOF
    chmod +x "$test_hook_dir/02_python.py"

    cat > "$test_hook_dir/03_bash2.sh" << 'EOF'
#!/bin/bash
echo "Another bash hook"
exit 0
EOF
    chmod +x "$test_hook_dir/03_bash2.sh"

    local output
    output=$(bash "$RUNNER_DIR/helpers/execute_hooks.sh" "$test_group" "$test_name" 2>&1)
    local exit_code=$?

    assert_exit_code "$exit_code" 0 "Exit code is 0 for mixed hooks"
    assert_contains "$output" "Bash hook" "Bash hook executed"
    assert_contains "$output" "Python hook" "Python hook executed"
    assert_contains "$output" "Another bash hook" "Second bash hook executed"
    assert_contains "$output" "All hooks executed successfully" "All hooks success message"

    # Cleanup
    rm -rf "${HOOKS_BASE_DIR:?}/$test_group"
}

# ============================================================================
# Run all tests
# ============================================================================
echo "Starting execute_hooks.sh unit tests..."
echo "Project root: $PROJECT_ROOT"
echo "Hooks base directory: $HOOKS_BASE_DIR"

test_no_arguments
test_missing_hook_name
test_nonexistent_hook_dir
test_empty_hook_dir
test_single_bash_hook
test_single_python_hook
test_hook_with_arguments
test_multiple_hooks
test_bash_hook_failure
test_python_hook_failure
test_stop_on_failure
test_mixed_hook_types

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
