#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Unit tests for Primus CLI helper modules

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
source "$PROJECT_ROOT/runner/helpers/execute_hooks.sh"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/runner/helpers/execute_patches.sh"

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
# Test 1: execute_hooks function exists
# ============================================================================
test_execute_hooks_exists() {
    print_section "Test 1: execute_hooks Function Definition"

    if type execute_hooks &>/dev/null; then
        assert_pass "execute_hooks function is defined"
    else
        assert_fail "execute_hooks function is defined"
    fi
}

# ============================================================================
# Test 2: execute_patches function exists
# ============================================================================
test_execute_patches_exists() {
    print_section "Test 2: execute_patches Function Definition"

    if type execute_patches &>/dev/null; then
        assert_pass "execute_patches function is defined"
    else
        assert_fail "execute_patches function is defined"
    fi
}

# ============================================================================
# Test 3: execute_hooks with non-existent hook
# ============================================================================
test_execute_hooks_nonexistent() {
    print_section "Test 3: execute_hooks with Non-existent Hook"

    local output
    output=$(execute_hooks "nonexistent" "nonexistent" 2>&1)

    assert_contains "$output" "No hook directory" "Handles non-existent hook gracefully"
}

# ============================================================================
# Test 4: execute_patches with no patches
# ============================================================================
test_execute_patches_no_patches() {
    print_section "Test 4: execute_patches with No Patches"

    local output
    output=$(execute_patches 2>&1)

    assert_contains "$output" "No patch scripts" "Handles no patches gracefully"
}

# ============================================================================
# Test 5: execute_patches with non-existent patch
# ============================================================================
test_execute_patches_nonexistent() {
    print_section "Test 5: execute_patches with Non-existent Patch"

    local output
    output=$(execute_patches "/nonexistent/patch.sh" 2>&1)

    assert_contains "$output" "not found" "Handles non-existent patch file"
}

# ============================================================================
# Test 6: execute_patches with valid patch
# ============================================================================
test_execute_patches_valid() {
    print_section "Test 6: execute_patches with Valid Patch"

    local test_patch="/tmp/test-patch-$$.sh"
    cat > "$test_patch" << 'EOF'
#!/bin/bash
echo "Test patch executed"
exit 0
EOF
    chmod +x "$test_patch"

    local output
    output=$(execute_patches "$test_patch" 2>&1)

    if echo "$output" | grep -qF "All patch scripts executed successfully"; then
        assert_pass "Executes valid patch successfully"
    else
        assert_fail "Executes valid patch successfully"
    fi

    rm -f "$test_patch"
}

# ============================================================================
# Test 7: execute_patches with failing patch
# ============================================================================
test_execute_patches_failing() {
    print_section "Test 7: execute_patches with Failing Patch"

    local test_patch_fail="/tmp/test-patch-fail-$$.sh"
    cat > "$test_patch_fail" << 'EOF'
#!/bin/bash
echo "Failing patch"
exit 1
EOF
    chmod +x "$test_patch_fail"

    local output
    output=$(execute_patches "$test_patch_fail" 2>&1)

    if ! echo "$output" | grep -qF "All patch scripts executed successfully"; then
        assert_pass "Handles failing patch correctly"
    else
        assert_fail "Handles failing patch correctly"
    fi

    rm -f "$test_patch_fail"
}

# ============================================================================
# Test 8: Hooks directory structure
# ============================================================================
test_hooks_directory() {
    print_section "Test 8: Hooks Directory Structure"

    local hooks_dir="$PROJECT_ROOT/runner/helpers/hooks"

    if [[ -d "$hooks_dir" ]]; then
        assert_pass "Hooks directory exists at $hooks_dir"
    else
        assert_fail "Hooks directory exists" "Directory not found: $hooks_dir"
    fi
}

# ============================================================================
# Test 9: Multiple patches execution
# ============================================================================
test_multiple_patches() {
    print_section "Test 9: Multiple Patches Execution"

    local test_patch1="/tmp/test-patch1-$$.sh"
    local test_patch2="/tmp/test-patch2-$$.sh"

    cat > "$test_patch1" << 'EOF'
#!/bin/bash
echo "Patch 1 executed"
exit 0
EOF

    cat > "$test_patch2" << 'EOF'
#!/bin/bash
echo "Patch 2 executed"
exit 0
EOF

    chmod +x "$test_patch1" "$test_patch2"

    local output
    output=$(execute_patches "$test_patch1" "$test_patch2" 2>&1)

    if echo "$output" | grep -qF "Patch 1 executed" && \
       echo "$output" | grep -qF "Patch 2 executed"; then
        assert_pass "Executes multiple patches in order"
    else
        assert_fail "Executes multiple patches in order"
    fi

    rm -f "$test_patch1" "$test_patch2"
}

# ============================================================================
# Run all tests
# ============================================================================
main() {
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Unit Tests for Primus CLI Helper Modules                   ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    test_execute_hooks_exists
    test_execute_patches_exists
    test_execute_hooks_nonexistent
    test_execute_patches_no_patches
    test_execute_patches_nonexistent
    test_execute_patches_valid
    test_execute_patches_failing
    test_hooks_directory
    test_multiple_patches

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
