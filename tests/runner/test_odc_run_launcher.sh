#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Unit tests for the ODC LB-mini launcher (primus/core/odc/rocshmem_runtime/
# scripts/run_odc.sh).
#
# These are CPU-only / no-GPU tests: they exercise run_odc.sh's argument parsing
# and environment / PYTHONPATH wiring WITHOUT actually launching training. The
# real examples/run_pretrain.sh is stubbed by pointing PRIMUS_ROOT at a fake
# tree whose examples/run_pretrain.sh just dumps the environment run_odc.sh
# exported, so we can assert on it deterministically.
#
# What is asserted (real ODC launch/config wiring, not trivial always-true):
#   * backend selection: `mori` (default) vs `rocshmem` sets ODC_P2P_BACKEND and
#     the matching rocSHMEM / MORI infra env (heap size, bootstrap ifname,
#     Triton cache policy).
#   * the pad|nopad positional arg is a NO-OP (retained for backwards-compatible
#     invocation): it does not change the backend or introduce any aligned-vs-
#     decoupled env; only the echoed PAD token differs.
#   * PYTHONPATH is wired so `import odc` resolves (odc_early shim + primus/core),
#     and PRIMUS_TURBO_PATH is prepended when provided.
#   * extra KEY=VAL trailing args are exported verbatim.
#   * EXP / PRIMUS_EXP_NAME / MASTER_PORT plumbing.
#   * ODC feature switches (enable_odc, odc_phase, enable_odc_lb_mini, ...) are
#     NOT exported as env by the launcher -- they are now CONFIG items read from
#     the EXP yaml. This guards the env->config migration from regressing.

# Get project root (tests/runner/ -> tests/ -> repo root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ODC="$PROJECT_ROOT/primus/core/odc/rocshmem_runtime/scripts/run_odc.sh"

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Scratch state (populated by run_odc_capture)
FAKE_ROOT=""
CAPTURE=""
STDOUT_LOG=""

# ---------------------------------------------------------------------------
# Test helpers (mirrors tests/runner/test_primus_cli.sh conventions)
# ---------------------------------------------------------------------------
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

# assert_line FILE EXACT_LINE TEST_NAME -- assert FILE contains an exact line.
assert_line() {
    local file="$1"
    local line="$2"
    local test_name="$3"
    ((TESTS_RUN++))
    if grep -qxF -- "$line" "$file"; then
        echo -e "${GREEN}✓${NC} $test_name"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗${NC} $test_name"
        echo "  Expected exact line: $line"
        echo "  In file: $file"
        ((TESTS_FAILED++))
    fi
}

# assert_contains HAYSTACK NEEDLE TEST_NAME
assert_contains() {
    local haystack="$1"
    local needle="$2"
    local test_name="$3"
    ((TESTS_RUN++))
    if echo "$haystack" | grep -qF -- "$needle"; then
        echo -e "${GREEN}✓${NC} $test_name"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗${NC} $test_name"
        echo "  Expected to contain: $needle"
        ((TESTS_FAILED++))
    fi
}

print_section() {
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

cap_value() {
    # Echo the value of KEY from the capture file (or empty if absent).
    local key="$1"
    grep -m1 "^${key}=" "$CAPTURE" 2>/dev/null | cut -d= -f2-
}

# ---------------------------------------------------------------------------
# Fixture: a fake PRIMUS_ROOT whose examples/run_pretrain.sh dumps the env that
# run_odc.sh exported, instead of launching real training.
# ---------------------------------------------------------------------------
setup_fake_root() {
    FAKE_ROOT="$(mktemp -d)"
    # Export CAPTURE once here (not inside the run subshell) so the stubbed
    # run_pretrain.sh inherits it, while shellcheck does not flag a subshell-local
    # modification (SC2030/SC2031).
    export CAPTURE="$FAKE_ROOT/capture.env"
    STDOUT_LOG="$FAKE_ROOT/stdout.log"
    mkdir -p "$FAKE_ROOT/examples"
    cat > "$FAKE_ROOT/examples/run_pretrain.sh" << 'STUB'
#!/bin/bash
# Stub standing in for the real trainer launch. Dump the env run_odc.sh set so
# the test can assert on it, then attempt `import odc` to prove PYTHONPATH wiring.
{
    for _k in ODC_P2P_BACKEND EXP PRIMUS_EXP_NAME MASTER_PORT TRITON_CACHE_DIR \
              MORI_SHMEM_HEAP_SIZE ROCSHMEM_HEAP_SIZE ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME \
              PYTHONPATH FUSED_LINEAR_CE GLOO_SOCKET_IFNAME NCCL_IB_DISABLE FOO \
              enable_odc ODC_ENABLE odc_phase enable_odc_lb_mini; do
        if [[ -n "${!_k+x}" ]]; then
            echo "${_k}=${!_k}"
        else
            echo "${_k}=<unset>"
        fi
    done
    if command -v python >/dev/null 2>&1; then
        if python -c "import odc" >/dev/null 2>&1; then
            echo "IMPORT_ODC=ok"
        else
            echo "IMPORT_ODC=fail"
        fi
    elif command -v python3 >/dev/null 2>&1; then
        if python3 -c "import odc" >/dev/null 2>&1; then
            echo "IMPORT_ODC=ok"
        else
            echo "IMPORT_ODC=fail"
        fi
    else
        echo "IMPORT_ODC=no-python"
    fi
} > "$CAPTURE"
STUB
}

teardown_fake_root() {
    [[ -n "$FAKE_ROOT" && -d "$FAKE_ROOT" ]] && rm -rf "$FAKE_ROOT"
}

# run_odc_capture <backend> <pad> <exp> <name> [extra KEY=VAL ...]
# Optional overrides via env before the call:
#   ODC_TURBO_PATH_OVERRIDE  -> PRIMUS_TURBO_PATH
#   ODC_MASTER_PORT_OVERRIDE -> MASTER_PORT
run_odc_capture() {
    rm -f "$CAPTURE" "$STDOUT_LOG"
    (
        # Start from a clean ODC-related env so absence assertions are meaningful.
        unset PRIMUS_TURBO_PATH ODC_P2P_BACKEND ROCSHMEM_HEAP_SIZE TRITON_CACHE_DIR \
            MORI_SHMEM_HEAP_SIZE ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME MASTER_PORT \
            enable_odc ODC_ENABLE odc_phase enable_odc_lb_mini FOO
        [[ -n "${ODC_TURBO_PATH_OVERRIDE:-}" ]] && export PRIMUS_TURBO_PATH="$ODC_TURBO_PATH_OVERRIDE"
        [[ -n "${ODC_MASTER_PORT_OVERRIDE:-}" ]] && export MASTER_PORT="$ODC_MASTER_PORT_OVERRIDE"
        export PRIMUS_ROOT="$FAKE_ROOT"
        export PRIMUS_PACK_CACHE_DIR="$FAKE_ROOT/pack"
        export TRAIN_LOG_DIR="$FAKE_ROOT/logs"
        bash "$RUN_ODC" "$@"
    ) > "$STDOUT_LOG" 2>&1
}

# ============================================================================
# Test 1: mori backend (default) env wiring
# ============================================================================
test_mori_backend_env() {
    print_section "Test 1: MORI backend (default) env wiring"
    run_odc_capture mori pad some/exp.yaml myexp

    assert_line "$CAPTURE" "ODC_P2P_BACKEND=mori" "mori sets ODC_P2P_BACKEND=mori"
    assert_line "$CAPTURE" "MORI_SHMEM_HEAP_SIZE=8G" "mori sets MORI symmetric heap 8G"
    assert_line "$CAPTURE" "TRITON_CACHE_DIR=/tmp/tcache_mori" "mori uses stable /tmp/tcache_mori"
    assert_line "$CAPTURE" "ROCSHMEM_HEAP_SIZE=<unset>" "mori does NOT set ROCSHMEM_HEAP_SIZE"
    assert_line "$CAPTURE" "ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME=<unset>" \
        "mori does NOT set ROCSHMEM bootstrap ifname"
    assert_contains "$(cat "$STDOUT_LOG")" "P2P=mori" "launcher banner reports P2P=mori"
}

# ============================================================================
# Test 2: rocshmem backend env wiring
# ============================================================================
test_rocshmem_backend_env() {
    print_section "Test 2: rocSHMEM backend env wiring"
    run_odc_capture rocshmem nopad other/exp.yaml exp2

    assert_line "$CAPTURE" "ODC_P2P_BACKEND=rocshmem" "rocshmem sets ODC_P2P_BACKEND=rocshmem"
    # Decimal-only heap parser: 8 GiB in raw bytes, NOT a K/M/G suffix.
    assert_line "$CAPTURE" "ROCSHMEM_HEAP_SIZE=8589934592" "rocshmem sets 8 GiB heap in raw bytes"
    assert_line "$CAPTURE" "ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME=lo" "rocshmem sets bootstrap ifname=lo"

    local triton
    triton="$(cap_value TRITON_CACHE_DIR)"
    if [[ "$triton" == /tmp/tcache_rocshmem_* ]]; then
        assert_pass "rocshmem uses a fresh per-run Triton cache (/tmp/tcache_rocshmem_*)"
    else
        assert_fail "rocshmem uses a fresh per-run Triton cache" "got TRITON_CACHE_DIR=$triton"
    fi
    assert_contains "$(cat "$STDOUT_LOG")" "P2P=rocshmem" "launcher banner reports P2P=rocshmem"
}

# ============================================================================
# Test 3: pad|nopad positional arg is a no-op
# ============================================================================
test_pad_arg_is_noop() {
    print_section "Test 3: pad|nopad positional arg is a no-op"

    run_odc_capture mori pad e.yaml n
    local pad_backend pad_triton pad_mori
    pad_backend="$(cap_value ODC_P2P_BACKEND)"
    pad_triton="$(cap_value TRITON_CACHE_DIR)"
    pad_mori="$(cap_value MORI_SHMEM_HEAP_SIZE)"
    local pad_banner; pad_banner="$(grep -o 'PAD=[^ ]*' "$STDOUT_LOG" | head -1)"

    run_odc_capture mori nopad e.yaml n
    local nopad_backend nopad_triton nopad_mori
    nopad_backend="$(cap_value ODC_P2P_BACKEND)"
    nopad_triton="$(cap_value TRITON_CACHE_DIR)"
    nopad_mori="$(cap_value MORI_SHMEM_HEAP_SIZE)"
    local nopad_banner; nopad_banner="$(grep -o 'PAD=[^ ]*' "$STDOUT_LOG" | head -1)"

    if [[ "$pad_backend" == "$nopad_backend" && "$pad_triton" == "$nopad_triton" \
          && "$pad_mori" == "$nopad_mori" ]]; then
        assert_pass "pad vs nopad produce identical backend/env wiring"
    else
        assert_fail "pad vs nopad produce identical backend/env wiring" \
            "pad=($pad_backend,$pad_triton,$pad_mori) nopad=($nopad_backend,$nopad_triton,$nopad_mori)"
    fi

    if [[ "$pad_banner" == "PAD=pad" && "$nopad_banner" == "PAD=nopad" ]]; then
        assert_pass "only the echoed PAD token reflects the arg"
    else
        assert_fail "only the echoed PAD token reflects the arg" \
            "pad_banner=$pad_banner nopad_banner=$nopad_banner"
    fi

    # No aligned/decoupled A/B env should be introduced by the pad arg.
    assert_line "$CAPTURE" "enable_odc_lb_mini=<unset>" "pad arg does not toggle any LB-mini env"
}

# ============================================================================
# Test 4: PYTHONPATH wiring for `import odc`
# ============================================================================
test_pythonpath_wiring() {
    print_section "Test 4: PYTHONPATH wiring for import odc"
    run_odc_capture mori pad e.yaml n

    local pp; pp="$(cap_value PYTHONPATH)"
    assert_contains "$pp" "primus/core/odc/odc_early" "PYTHONPATH includes the odc_early load-order shim"
    if [[ "$pp" == *"/primus/core" ]]; then
        assert_pass "PYTHONPATH ends with primus/core (parent of the odc package)"
    else
        assert_fail "PYTHONPATH ends with primus/core" "PYTHONPATH=$pp"
    fi

    local imp; imp="$(cap_value IMPORT_ODC)"
    if [[ "$imp" == "ok" ]]; then
        assert_pass "import odc resolves via the wired PYTHONPATH"
    elif [[ "$imp" == "no-python" ]]; then
        assert_pass "import odc check skipped (no python interpreter on host)"
    else
        assert_fail "import odc resolves via the wired PYTHONPATH" "IMPORT_ODC=$imp"
    fi
}

# ============================================================================
# Test 5: PRIMUS_TURBO_PATH is prepended to PYTHONPATH
# ============================================================================
test_turbo_path_prepend() {
    print_section "Test 5: PRIMUS_TURBO_PATH prepend"
    ODC_TURBO_PATH_OVERRIDE="/fake/turbo/build" run_odc_capture mori pad e.yaml n

    local pp; pp="$(cap_value PYTHONPATH)"
    if [[ "$pp" == /fake/turbo/build:* ]]; then
        assert_pass "PRIMUS_TURBO_PATH is prepended to PYTHONPATH"
    else
        assert_fail "PRIMUS_TURBO_PATH is prepended to PYTHONPATH" "PYTHONPATH=$pp"
    fi
    assert_contains "$(cat "$STDOUT_LOG")" "TURBO_PATH=/fake/turbo/build" \
        "launcher banner reports the turbo build path"
}

# ============================================================================
# Test 6: extra KEY=VAL trailing args are exported verbatim
# ============================================================================
test_extra_kv_exported() {
    print_section "Test 6: extra KEY=VAL args exported verbatim"
    run_odc_capture mori pad e.yaml n FOO=bar

    assert_line "$CAPTURE" "FOO=bar" "trailing FOO=bar is exported to the trainer env"
}

# ============================================================================
# Test 7: EXP / PRIMUS_EXP_NAME / MASTER_PORT plumbing
# ============================================================================
test_positional_and_port_plumbing() {
    print_section "Test 7: EXP / exp-name / MASTER_PORT plumbing"

    run_odc_capture mori pad path/to/exp.yaml my-run-name
    assert_line "$CAPTURE" "EXP=path/to/exp.yaml" "EXP is set from the 3rd positional arg"
    assert_line "$CAPTURE" "PRIMUS_EXP_NAME=my-run-name" "PRIMUS_EXP_NAME is set from the 4th positional arg"
    assert_line "$CAPTURE" "MASTER_PORT=29600" "MASTER_PORT defaults to 29600"

    ODC_MASTER_PORT_OVERRIDE="29777" run_odc_capture mori pad e.yaml n
    assert_line "$CAPTURE" "MASTER_PORT=29777" "MASTER_PORT is overridable via env"
}

# ============================================================================
# Test 8: ODC feature switches are config, not env
# ============================================================================
test_switches_are_config_not_env() {
    print_section "Test 8: ODC feature switches are config, not env"
    run_odc_capture rocshmem pad e.yaml n

    # The launcher must NOT export any of the ODC feature switches: post-#864
    # they live in the EXP yaml config, read by the before_train patches.
    assert_line "$CAPTURE" "enable_odc=<unset>" "launcher does not export enable_odc"
    assert_line "$CAPTURE" "ODC_ENABLE=<unset>" "launcher does not export the legacy ODC_ENABLE env"
    assert_line "$CAPTURE" "odc_phase=<unset>" "launcher does not export odc_phase"
    assert_line "$CAPTURE" "enable_odc_lb_mini=<unset>" "launcher does not export enable_odc_lb_mini"
}

# ============================================================================
# Run all tests
# ============================================================================
main() {
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Unit Tests for the ODC LB-mini launcher (run_odc.sh)        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    if [[ ! -f "$RUN_ODC" ]]; then
        echo -e "${RED}✗ run_odc.sh not found at $RUN_ODC${NC}"
        return 1
    fi

    setup_fake_root
    trap teardown_fake_root EXIT

    test_mori_backend_env
    test_rocshmem_backend_env
    test_pad_arg_is_noop
    test_pythonpath_wiring
    test_turbo_path_prepend
    test_extra_kv_exported
    test_positional_and_port_plumbing
    test_switches_are_config_not_env

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

main
