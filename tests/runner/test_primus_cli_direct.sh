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
    assert_contains "$output" "--silent" "Silent flag documented"
    assert_contains "$output" "VENV_ACTIVATE" "VENV_ACTIVATE documented"
    assert_contains "$output" "SLURM" "SLURM auto-derivation documented"
}

# ============================================================================
# Test 11: VENV_ACTIVATE handling (R1 -- consolidate-preflight-direct-wrappers)
# ============================================================================
test_venv_activate() {
    local_print_section "Test 11: VENV_ACTIVATE (R1)"

    # Sub-test 11a: VENV_ACTIVATE unset = no-op (dry-run succeeds without
    # any "VENV_ACTIVATE" error, and the script never tries to source a file).
    local out_unset
    out_unset=$(unset VENV_ACTIVATE; timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --dry-run -- benchmark gemm 2>&1 || true)
    assert_contains "$out_unset" "End of Dry Run" "Unset VENV_ACTIVATE is a no-op (dry-run completes)"
    assert_not_contains "$out_unset" "VENV_ACTIVATE is set but" "No spurious 'missing file' error when unset"
    assert_not_contains "$out_unset" "Activated virtualenv:" "No 'Activated virtualenv' message when unset"

    # Sub-test 11b: VENV_ACTIVATE set + valid file = sourced.
    local tmpvenv
    tmpvenv="$(mktemp)"
    echo 'export PRIMUS_TEST_VENV_SOURCED=yes' > "$tmpvenv"
    local out_valid
    out_valid=$(VENV_ACTIVATE="$tmpvenv" timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --dry-run -- benchmark gemm 2>&1 || true)
    assert_contains "$out_valid" "Activated virtualenv:" "Valid VENV_ACTIVATE is sourced"
    rm -f "$tmpvenv"

    # Sub-test 11c: VENV_ACTIVATE set + missing file = fail-fast (LOG_ERROR
    # on stderr, non-zero exit).
    local ec_missing=0
    VENV_ACTIVATE=/does/not/exist/anywhere timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --dry-run -- benchmark gemm >/dev/null 2>/tmp/test_venv_err_$$ || ec_missing=$?
    if [[ "$ec_missing" -ne 0 ]] && grep -q "VENV_ACTIVATE is set but file does not exist" /tmp/test_venv_err_$$; then
        assert_pass "Missing VENV_ACTIVATE file fails fast with clear error"
    else
        assert_fail "Missing VENV_ACTIVATE file should fail fast" \
            "exit=$ec_missing stderr=$(cat /tmp/test_venv_err_$$)"
    fi
    rm -f /tmp/test_venv_err_$$
}

# ============================================================================
# Test 12: SLURM env derivation (R2)
# ============================================================================
test_slurm_env_derivation() {
    local_print_section "Test 12: SLURM env derivation (R2)"

    # Sub-test 12a: SLURM_JOB_ID + SLURM_NNODES + SLURM_NODEID -> NNODES /
    # NODE_RANK derived.
    local out_slurm
    out_slurm=$(SLURM_JOB_ID=999 SLURM_NNODES=4 SLURM_NODEID=0 SLURM_NODELIST=tus1-p3-g25 \
        timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --dry-run -- benchmark gemm 2>&1 || true)
    assert_contains "$out_slurm" "SLURM detected" "SLURM detection log fires"
    assert_contains "$out_slurm" "NNODES=4" "NNODES derived from SLURM_NNODES"
    assert_contains "$out_slurm" "--nnodes 4" "torchrun gets --nnodes 4"

    # Sub-test 12b: pre-exported NNODES wins over SLURM_NNODES.
    local out_preset
    out_preset=$(SLURM_JOB_ID=999 SLURM_NNODES=4 NNODES=7 NODE_RANK=0 SLURM_NODELIST=tus1-p3-g25 \
        timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --dry-run -- benchmark gemm 2>&1 || true)
    assert_contains "$out_preset" "NNODES=7" "Pre-exported NNODES=7 wins over SLURM_NNODES=4"
    assert_contains "$out_preset" "--nnodes 7" "torchrun honors pre-exported NNODES=7"

    # Sub-test 12c: sanity check rejects NODE_RANK >= NNODES.
    local ec_sanity=0
    NNODES=2 NODE_RANK=5 timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --dry-run -- benchmark gemm >/dev/null 2>/tmp/test_slurm_err_$$ || ec_sanity=$?
    if [[ "$ec_sanity" -ne 0 ]] && grep -q "NODE_RANK (5) must be < NNODES (2)" /tmp/test_slurm_err_$$; then
        assert_pass "Sanity check rejects NODE_RANK >= NNODES"
    else
        assert_fail "Sanity check should reject NODE_RANK >= NNODES" \
            "exit=$ec_sanity stderr=$(cat /tmp/test_slurm_err_$$)"
    fi
    rm -f /tmp/test_slurm_err_$$
}

# ============================================================================
# Test 13: Auto-single run_mode for node_smoke
# ============================================================================
test_auto_single_for_node_smoke() {
    local_print_section "Test 13: Auto-single run_mode for node_smoke"

    # Sub-test 13a: preflight stays in torchrun mode by default.
    local out_preflight
    out_preflight=$(timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --dry-run -- preflight --quick 2>&1 || true)
    assert_contains "$out_preflight" "Run Mode        : torchrun" "preflight defaults to torchrun"
    assert_contains "$out_preflight" "torchrun --nproc_per_node" "preflight uses torchrun command"

    # Sub-test 13b: node_smoke auto-selects single mode.
    local out_smoke
    out_smoke=$(timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --dry-run -- node_smoke --tier2-perf 2>&1 || true)
    assert_contains "$out_smoke" "Auto-selected run_mode=single for subcommand 'node_smoke'" \
        "node_smoke auto-detect fires"
    assert_contains "$out_smoke" "Run Mode        : single" "node_smoke ends up in single mode"
    assert_contains "$out_smoke" "python3" "node_smoke uses python3 launcher"
    assert_not_contains "$out_smoke" "torchrun --nproc_per_node" "node_smoke does NOT use torchrun"

    # Sub-test 13c: explicit --single on a non-node_smoke subcommand still
    # works (regression guard).
    local out_explicit
    out_explicit=$(timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --single --dry-run -- benchmark gemm 2>&1 || true)
    assert_contains "$out_explicit" "Run Mode        : single" "Explicit --single still works for benchmark"
}

# ============================================================================
# Test 14: --silent contract (bash-level)
# ============================================================================
test_silent_flag() {
    local_print_section "Test 14: --silent flag contract"

    # Sub-test 14a: --silent produces empty stdout.
    local out_silent="/tmp/test_silent_out_$$"
    local err_silent="/tmp/test_silent_err_$$"
    timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --silent --dry-run -- benchmark gemm >"$out_silent" 2>"$err_silent" || true
    if [[ ! -s "$out_silent" ]]; then
        assert_pass "--silent silences stdout (out file is empty)"
    else
        assert_fail "--silent should silence stdout" "size=$(wc -c <"$out_silent") head=$(head -3 "$out_silent")"
    fi
    rm -f "$out_silent" "$err_silent"

    # Sub-test 14b: --silent does NOT silence launcher errors (stderr survives).
    local err_err_silent="/tmp/test_silent_err2_$$"
    VENV_ACTIVATE=/does/not/exist timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --silent --dry-run -- benchmark gemm >/dev/null 2>"$err_err_silent" || true
    if grep -q "VENV_ACTIVATE is set but file does not exist" "$err_err_silent"; then
        assert_pass "--silent preserves launcher LOG_ERROR on stderr"
    else
        assert_fail "--silent should preserve launcher errors on stderr" \
            "stderr=$(cat "$err_err_silent")"
    fi
    rm -f "$err_err_silent"

    # Sub-test 14c: --silent is consumed by the launcher and NOT forwarded
    # to the python tool. The forwarded args list (logged before silencing
    # applied? actually after -- it shouldn't appear at all). Easiest check:
    # the dry-run "Would Execute" line must NOT contain --silent.
    local out_no_forward
    out_no_forward=$(timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --dry-run -- benchmark gemm 2>&1 || true)
    # Sanity: this baseline run has no --silent anywhere.
    assert_not_contains "$out_no_forward" "--silent" "Baseline dry-run has no --silent leakage"
}

# ============================================================================
# Test 15: slurm-entry direct mode dispatch
# ============================================================================
test_slurm_entry_direct_dispatch() {
    local_print_section "Test 15: slurm-entry direct/container dispatch"

    # The slurm-entry script needs SLURM_NODELIST to run; mock it.
    # Sub-test 15a: -- direct -- preflight ... routes through primus-cli-direct.sh.
    local out_direct
    out_direct=$(SLURM_NODELIST=tus1-p3-g25 SLURM_JOB_ID=1 SLURM_NNODES=1 SLURM_NODEID=0 \
        timeout 5 bash "$RUNNER_DIR/primus-cli-slurm-entry.sh" --dry-run -- direct -- preflight --quick 2>&1 || true)
    assert_contains "$out_direct" "Entry mode: direct" "slurm-entry parses 'direct' keyword"
    assert_contains "$out_direct" "primus-cli-direct.sh" "slurm-entry dispatches to primus-cli-direct.sh"

    # Sub-test 15b: -- container -- ... routes through primus-cli-container.sh (existing path).
    local out_container
    out_container=$(SLURM_NODELIST=tus1-p3-g25 SLURM_JOB_ID=1 SLURM_NNODES=1 SLURM_NODEID=0 \
        timeout 5 bash "$RUNNER_DIR/primus-cli-slurm-entry.sh" --dry-run -- container -- train pretrain 2>&1 || true)
    assert_contains "$out_container" "Entry mode: container" "slurm-entry parses 'container' keyword"
    assert_contains "$out_container" "primus-cli-container.sh" "slurm-entry dispatches to primus-cli-container.sh"

    # Sub-test 15c: terse form `-- preflight` (no keyword) defaults to container.
    local out_terse
    out_terse=$(SLURM_NODELIST=tus1-p3-g25 SLURM_JOB_ID=1 SLURM_NNODES=1 SLURM_NODEID=0 \
        timeout 5 bash "$RUNNER_DIR/primus-cli-slurm-entry.sh" --dry-run -- preflight --quick 2>&1 || true)
    assert_contains "$out_terse" "Entry mode: container" "Terse form defaults to container"
    assert_contains "$out_terse" "primus-cli-container.sh" "Terse form dispatches to primus-cli-container.sh"
}

# ============================================================================
# Test 16: RUN_MODE env override (framework prepare-hook contract)
#
# Framework hooks (e.g. runner/helpers/hooks/train/pretrain/maxtext/prepare.py)
# emit `env.RUN_MODE=single` so JAX/MaxText runs as `python3 ...` instead of
# `torchrun ...`. This used to launch correctly but display incorrectly:
# STEP 10 showed `Run Mode: torchrun` and printed the Distributed Settings
# block, because both reads went against the pre-hook direct_config[run_mode]
# instead of the final $RUN_MODE.
#
# This test pins the post-fix contract: when RUN_MODE is exported in the env
# before primus-cli-direct.sh runs (a sufficient stand-in for "a hook
# exported it"), every visible knob -- the displayed Run Mode, the
# Distributed Settings gate, and the actual launch command -- reflects the
# env-override, not the pre-hook default.
# ============================================================================
test_run_mode_env_override() {
    local_print_section "Test 16: RUN_MODE env override (framework hook contract)"

    # Baseline: `train pretrain` with NO hook override -- the launcher's
    # auto-default (torchrun) wins and the display shows torchrun + the
    # Distributed Settings block. This anchors the "before" state we are
    # protecting users from.
    local out_baseline
    out_baseline=$(timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" --dry-run -- train pretrain 2>&1 || true)
    assert_contains "$out_baseline" "Run Mode        : torchrun" \
        "Baseline train pretrain shows torchrun (no env override)"
    assert_contains "$out_baseline" "Distributed Settings:" \
        "Baseline train pretrain prints Distributed Settings"
    assert_contains "$out_baseline" "torchrun --nproc_per_node" \
        "Baseline train pretrain uses torchrun in Full Command"

    # Sub-test 16a: pre-exporting RUN_MODE=single (the MaxText hook's
    # effect, modeled in-process so the test doesn't depend on the
    # framework hook actually firing) must flip ALL three views:
    #   - displayed "Run Mode" line
    #   - "Distributed Settings" gate
    #   - launch command (`python3 ...`, not `torchrun ...`)
    local out_override
    out_override=$(RUN_MODE=single timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" \
        --dry-run -- train pretrain 2>&1 || true)
    assert_contains "$out_override" "Run Mode        : single" \
        "RUN_MODE=single env override surfaces in displayed Run Mode"
    assert_not_contains "$out_override" "Distributed Settings:" \
        "RUN_MODE=single env override suppresses Distributed Settings block"
    assert_not_contains "$out_override" "torchrun --nproc_per_node" \
        "RUN_MODE=single env override drops torchrun from Full Command"
    assert_contains "$out_override" "python3" \
        "RUN_MODE=single env override uses python3 launcher in Full Command"

    # Sub-test 16b: env-override beats an EXPLICIT --single on the CLI.
    # This matches the launcher's documented precedence at STEP 9:
    #   RUN_MODE="${RUN_MODE:-${direct_config[run_mode]:-torchrun}}"
    # The env layer wins because the hook layer (which is where RUN_MODE
    # actually originates in real runs) knows things the user / config
    # can't and gets the final word. We use --single here purely as a
    # convenient stand-in for "user/config set direct_config[run_mode]"
    # -- there is no symmetric --torchrun CLI flag.
    local out_vs_cli
    out_vs_cli=$(RUN_MODE=torchrun timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" \
        --single --dry-run -- benchmark gemm 2>&1 || true)
    assert_contains "$out_vs_cli" "Run Mode        : torchrun" \
        "RUN_MODE=torchrun env override wins over --single CLI flag"
    assert_contains "$out_vs_cli" "torchrun --nproc_per_node" \
        "RUN_MODE=torchrun env override produces torchrun launcher despite --single"

    # Sub-test 16c: exporting RUN_MODE must override even the auto-single
    # detection for node_smoke. Auto-detect sets direct_config[run_mode]
    # BEFORE the env-override check at line 610, so env still wins. This
    # is the corner case behind the bug we fixed: the OLD code printed
    # `Run Mode: single` here (from direct_config[run_mode]) even though
    # the launch command was actually torchrun -- a display divergence
    # in the exact opposite direction from the MaxText case.
    local out_reverse
    out_reverse=$(RUN_MODE=torchrun timeout 5 bash "$RUNNER_DIR/primus-cli-direct.sh" \
        --dry-run -- node_smoke --tier2-perf 2>&1 || true)
    assert_contains "$out_reverse" "Run Mode        : torchrun" \
        "RUN_MODE=torchrun env override surfaces in display for node_smoke"
    assert_contains "$out_reverse" "Distributed Settings:" \
        "RUN_MODE=torchrun env override re-enables Distributed Settings for node_smoke"
    assert_contains "$out_reverse" "torchrun --nproc_per_node" \
        "RUN_MODE=torchrun env override produces torchrun launch for node_smoke"
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
    # New tests from the consolidate-preflight-direct-wrappers refactor:
    test_venv_activate
    test_slurm_env_derivation
    test_auto_single_for_node_smoke
    test_silent_flag
    test_slurm_entry_direct_dispatch
    test_run_mode_env_override

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
