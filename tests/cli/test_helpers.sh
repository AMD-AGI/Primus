#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Test script for helper modules

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source common library
# shellcheck disable=SC1091
source "$PROJECT_ROOT/runner/lib/common.sh"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/runner/helpers/execute_hooks.sh"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/runner/helpers/execute_patches.sh"

export NODE_RANK=0

echo "========================================="
echo "  Primus CLI Helper Modules Tests"
echo "========================================="
echo ""

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

test_pass() {
    LOG_SUCCESS "✓ $1"
    ((TESTS_PASSED++)) || true
}

test_fail() {
    LOG_ERROR "✗ $1"
    ((TESTS_FAILED++)) || true
}

# Test 1: execute_hooks function exists
LOG_INFO "Test 1: Testing execute_hooks function..."
if type execute_hooks &>/dev/null; then
    test_pass "execute_hooks function is defined"
else
    test_fail "execute_hooks function is defined"
fi
echo ""

# Test 2: execute_patches function exists
LOG_INFO "Test 2: Testing execute_patches function..."
if type execute_patches &>/dev/null; then
    test_pass "execute_patches function is defined"
else
    test_fail "execute_patches function is defined"
fi
echo ""

# Test 3: execute_hooks with non-existent hook
LOG_INFO "Test 3: Testing execute_hooks with non-existent hook..."
if execute_hooks "nonexistent" "nonexistent" 2>&1 | grep -q "No hook directory"; then
    test_pass "execute_hooks handles non-existent hooks"
else
    test_fail "execute_hooks handles non-existent hooks"
fi
echo ""

# Test 4: execute_patches with no patches
LOG_INFO "Test 4: Testing execute_patches with no patches..."
if execute_patches 2>&1 | grep -q "No patch scripts"; then
    test_pass "execute_patches handles no patches"
else
    test_fail "execute_patches handles no patches"
fi
echo ""

# Test 5: execute_patches with non-existent patch
LOG_INFO "Test 5: Testing execute_patches with non-existent patch..."
if execute_patches "/nonexistent/patch.sh" 2>&1 | grep -q "not found"; then
    test_pass "execute_patches handles non-existent patches"
else
    test_fail "execute_patches handles non-existent patches"
fi
echo ""

# Test 6: Create and execute a test patch
LOG_INFO "Test 6: Testing execute_patches with valid patch..."
TEST_PATCH="/tmp/test-patch-$$.sh"
cat > "$TEST_PATCH" << 'EOF'
#!/bin/bash
echo "Test patch executed"
exit 0
EOF
chmod +x "$TEST_PATCH"

if execute_patches "$TEST_PATCH" 2>&1 | grep -q "All patch scripts executed successfully"; then
    test_pass "execute_patches executes valid patches"
else
    test_fail "execute_patches executes valid patches"
fi
rm -f "$TEST_PATCH"
echo ""

# Test 7: Test patch failure handling
LOG_INFO "Test 7: Testing execute_patches with failing patch..."
TEST_PATCH_FAIL="/tmp/test-patch-fail-$$.sh"
cat > "$TEST_PATCH_FAIL" << 'EOF'
#!/bin/bash
echo "Failing patch"
exit 1
EOF
chmod +x "$TEST_PATCH_FAIL"

if ! execute_patches "$TEST_PATCH_FAIL" 2>&1 | grep -q "All patch scripts executed successfully"; then
    test_pass "execute_patches handles failing patches"
else
    test_fail "execute_patches handles failing patches"
fi
rm -f "$TEST_PATCH_FAIL"
echo ""

# Test 8: Test hook directory structure
LOG_INFO "Test 8: Testing hooks directory structure..."
HOOKS_DIR="$PROJECT_ROOT/runner/helpers/hooks"
if [[ -d "$HOOKS_DIR" ]]; then
    test_pass "Hooks directory exists"
else
    test_fail "Hooks directory exists"
fi
echo ""

# Summary
echo "========================================="
echo "  Test Summary"
echo "========================================="
LOG_SUCCESS "Passed: $TESTS_PASSED"
if [[ "$TESTS_FAILED" -gt 0 ]]; then
    LOG_ERROR "Failed: $TESTS_FAILED"
else
    LOG_INFO "Failed: $TESTS_FAILED"
fi
echo "Total: $((TESTS_PASSED + TESTS_FAILED))"
echo "========================================="

if [[ "$TESTS_FAILED" -eq 0 ]]; then
    LOG_SUCCESS "All helper tests passed! ✓"
    exit 0
else
    LOG_ERROR "Some helper tests failed! ✗"
    exit 1
fi
