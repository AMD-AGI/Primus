# Pull Request: Hook Execution System with Comprehensive Test Suite

## 🎯 Overview

This PR introduces a flexible hook execution system for the Primus runner, enabling users to inject custom scripts at various stages of command execution. The system supports both Bash and Python hooks with comprehensive test coverage.

## 📝 Changes

### New Features

#### 1. Hook Execution System (`runner/helpers/execute_hooks.sh`)
- **Flexible hook discovery**: Automatically discovers and executes hooks based on command group and name
- **Multi-language support**: Supports both `.sh` (Bash) and `.py` (Python) hook files
- **Sorted execution**: Hooks are executed in alphabetical order (useful for numbered prefixes like `01_`, `02_`)
- **Argument forwarding**: Passes additional arguments to all hook scripts
- **Fail-fast behavior**: Stops execution on first hook failure
- **Fallback logging**: Provides logging functions when `common.sh` is not available

**Usage:**
```bash
execute_hooks <hook_group> <hook_name> [args...]
```

**Example:**
```bash
# Execute pre-training hooks with arguments
execute_hooks train pretrain --model-size 7B

# Hook directory structure:
# runner/helpers/hooks/
#   └── train/
#       └── pretrain/
#           ├── 01_setup.sh
#           ├── 02_validate.py
#           └── 03_prepare.sh
```

#### 2. Comprehensive Test Suite (`tests/runner/helpers/test_execute_hooks.sh`)
- **41 tests** covering all scenarios
- **Test coverage includes:**
  - Missing arguments and edge cases
  - Single and multiple hooks
  - Bash and Python hooks
  - Hook failures and error handling
  - Stop-on-first-failure behavior
  - Argument passing
  - Sorted execution order
  - Mixed hook types

### Core Library Updates

#### `runner/lib/common.sh`
- **Added `LOG_ERROR_RANK0()` function**
  - Mirrors the pattern of `LOG_INFO_RANK0` and `LOG_SUCCESS_RANK0`
  - Logs errors only on rank 0 for distributed environments
  - Properly exported for use in subshells

### Test Infrastructure Updates

#### `tests/runner/run_all_tests.sh`
- Updated to include `test_execute_hooks.sh` in the test suite
- Now runs 4 test suites with **126 total tests**

## 🔧 Technical Details

### Subshell Function Inheritance Fix

**Problem:** Environment variables are inherited by subshells, but function definitions are not. When `run_all_tests.sh` sources `common.sh`, the guard variable `__PRIMUS_COMMON_SOURCED` is exported, but functions aren't available in subshells spawned by test scripts.

**Solution:** Force re-sourcing of `common.sh` in subshells by unsetting the guard variable:

```bash
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Always source common.sh when called directly, as functions are not inherited by subshells
    # Unset the guard variable to force re-sourcing in this new shell instance
    unset __PRIMUS_COMMON_SOURCED
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/../lib/common.sh" || {
        echo "[ERROR] Failed to load common library" >&2
        exit 1
    }
    execute_hooks "$@"
fi
```

## ✅ Testing

### Test Results
```
=========================================
  Final Test Results
=========================================
Total test suites: 4
Passed: 4
Failed: 0
=========================================
🎉 All test suites passed! ✓
```

### Test Breakdown
| Test Suite | Tests | Status |
|-----------|-------|--------|
| `test_common.sh` | 15 | ✅ All passed |
| `test_validation.sh` | 19 | ✅ All passed |
| `test_config.sh` | 51 | ✅ All passed |
| `test_execute_hooks.sh` | 41 | ✅ All passed |
| **Total** | **126** | **✅ All passed** |

### Running Tests
```bash
# Run all tests
bash tests/runner/run_all_tests.sh

# Run hook tests only
bash tests/runner/helpers/test_execute_hooks.sh
```

## 📋 Files Changed

### Added Files
- `runner/helpers/execute_hooks.sh` - Hook execution system (111 lines)
- `tests/runner/helpers/test_execute_hooks.sh` - Comprehensive test suite (515 lines)

### Modified Files
- `runner/lib/common.sh` - Added `LOG_ERROR_RANK0()` function and export
- `tests/runner/run_all_tests.sh` - Added hook tests to test suite

## 🎨 Hook System Design

### Directory Structure
```
runner/helpers/hooks/
├── <group>/              # e.g., "train", "benchmark"
│   └── <name>/           # e.g., "pretrain", "gemm"
│       ├── 01_*.sh       # Bash hooks (executed in order)
│       ├── 02_*.py       # Python hooks (executed in order)
│       └── 03_*.sh       # More hooks...
```

### Hook Exit Codes
- **0**: Success, continue to next hook
- **Non-zero**: Failure, stop execution and return error

### Hook Capabilities
- ✅ Access to all environment variables
- ✅ Receive command-line arguments
- ✅ Execute in sorted order
- ✅ Full stdout/stderr output
- ✅ Timing information logged
- ✅ Fail-fast on errors

## 🔍 Use Cases

### 1. Pre-training Setup
```bash
# hooks/train/pretrain/01_validate_config.sh
#!/bin/bash
# Validate training configuration before starting
```

### 2. Environment Preparation
```bash
# hooks/benchmark/gemm/01_setup_env.py
#!/usr/bin/env python3
# Set up benchmark environment
```

### 3. Post-execution Cleanup
```bash
# hooks/train/cleanup/01_collect_logs.sh
#!/bin/bash
# Collect and archive training logs
```

## 🚀 Future Enhancements

- [ ] Support for async/parallel hook execution
- [ ] Hook dependency management
- [ ] Conditional hook execution based on environment
- [ ] Hook timeout support
- [ ] Integration with CI/CD pipelines

## ✅ Pre-commit Checks

All pre-commit checks passed:
- ✅ Trim Trailing Whitespace
- ✅ Fix End of Files
- ✅ Check for added large files
- ✅ Check for merge conflicts
- ✅ shellcheck

## 📚 Documentation

Hook system usage will be documented in:
- Runner documentation
- Developer guide
- Example hook templates

## 🤝 Review Checklist

- [x] All tests pass (126/126)
- [x] Code follows project style guidelines
- [x] shellcheck passes
- [x] No breaking changes to existing functionality
- [x] Comprehensive test coverage for new features
- [x] Functions properly exported in common.sh
- [x] Subshell inheritance issues resolved

## 🙏 Acknowledgments

This implementation provides a robust foundation for extensible command execution in the Primus runner system.

---

**Ready for review!** 🎉
