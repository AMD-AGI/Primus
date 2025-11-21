# feat(runner): add flexible hook execution system with comprehensive test suite

## 📝 Description

This PR introduces a flexible hook execution system for the Primus runner, enabling users to inject custom Bash and Python scripts at various stages of command execution.

### What's Changed

- ✅ New hook execution engine supporting `.sh` and `.py` files
- ✅ Comprehensive test suite with 41 tests
- ✅ Added `LOG_ERROR_RANK0()` function to common library
- ✅ Fixed subshell function inheritance issues
- ✅ Updated test infrastructure to include hook tests

## 🎯 Motivation

Enable extensible command execution by allowing users to inject custom logic at various stages without modifying core runner code. This provides:

1. **Flexibility**: Users can add setup, validation, cleanup scripts
2. **Modularity**: Hooks are isolated and independently testable
3. **Maintainability**: Easier to extend functionality without touching core code
4. **Reusability**: Hooks can be shared across different commands

## 🚀 Key Features

### Hook Execution System (`execute_hooks.sh`)

- **Auto-discovery**: Finds all `.sh` and `.py` files in hook directories
- **Sorted execution**: Executes hooks in alphabetical order (01_, 02_, 03_...)
- **Multi-language**: Supports both Bash and Python scripts
- **Argument forwarding**: Passes additional arguments to all hooks
- **Fail-fast**: Stops on first hook failure
- **Timing**: Logs execution duration for each hook
- **Fallback logging**: Works even when common.sh isn't loaded

### Usage

```bash
# Execute hooks for a command
execute_hooks <group> <name> [args...]

# Example: Execute pre-training hooks
execute_hooks train pretrain --model-size 7B
```

### Hook Directory Structure

```
runner/helpers/hooks/
├── train/
│   └── pretrain/
│       ├── 01_validate_config.sh
│       ├── 02_setup_env.py
│       └── 03_prepare_data.sh
├── benchmark/
│   └── gemm/
│       └── 01_setup.sh
└── ...
```

## 🧪 Testing

### Test Coverage

- **Total tests**: 126 (up from 85)
- **New hook tests**: 41
- **Test suites**: 4
- **Coverage**: All hook scenarios including failures, arguments, mixed types

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

| Test Suite | Tests | Status |
|-----------|-------|--------|
| test_common.sh | 15 | ✅ PASSED |
| test_validation.sh | 19 | ✅ PASSED |
| test_config.sh | 51 | ✅ PASSED |
| **test_execute_hooks.sh** | **41** | **✅ PASSED** |

### Running Tests

```bash
# Run all tests
bash tests/runner/run_all_tests.sh

# Run hook tests only
bash tests/runner/helpers/test_execute_hooks.sh
```

## 🔧 Technical Details

### Subshell Function Inheritance Fix

**Problem**: Environment variables are inherited by subshells, but function definitions are not. When tests spawn subshells, functions from `common.sh` weren't available.

**Solution**: Force re-sourcing by unsetting the guard variable:

```bash
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    unset __PRIMUS_COMMON_SOURCED  # Force re-source in new shell
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/../lib/common.sh"
    execute_hooks "$@"
fi
```

### New Common Library Function

Added `LOG_ERROR_RANK0()` for consistent error logging in distributed environments:

```bash
LOG_ERROR_RANK0() {
    if [[ "${NODE_RANK:-0}" == "0" ]]; then
        LOG_ERROR "$@"
    fi
}
```

## 📋 Files Changed

### Added
- `runner/helpers/execute_hooks.sh` (111 lines)
  - Hook discovery and execution engine
  - Bash and Python support
  - Fallback logging functions

- `tests/runner/helpers/test_execute_hooks.sh` (515 lines)
  - 41 comprehensive tests
  - Tests for success, failure, arguments, sorting, mixed types

### Modified
- `runner/lib/common.sh` (+6 lines)
  - Added `LOG_ERROR_RANK0()` function definition
  - Added `LOG_ERROR_RANK0` to function export list

- `tests/runner/run_all_tests.sh` (1 line changed)
  - Updated to include `test_execute_hooks.sh`

## 🎨 Usage Examples

### Example 1: Training Pre-flight Checks

```bash
# hooks/train/pretrain/01_validate.sh
#!/bin/bash
# Validate configuration before training
if [[ ! -f config.yaml ]]; then
    echo "Error: config.yaml not found"
    exit 1
fi
exit 0
```

### Example 2: Environment Setup

```bash
# hooks/benchmark/gemm/01_setup.py
#!/usr/bin/env python3
# Set up benchmark environment
import os
os.environ['BENCHMARK_MODE'] = '1'
print("Benchmark environment configured")
exit(0)
```

### Example 3: Multi-step Workflow

```bash
# hooks/train/pretrain/
├── 01_validate_config.sh     # Validate configuration
├── 02_check_gpus.sh           # Check GPU availability
├── 03_setup_paths.py          # Set up data paths
└── 04_log_environment.sh      # Log environment info
```

## ✅ Checklist

- [x] All tests pass (126/126)
- [x] Code follows project style guidelines
- [x] shellcheck passes
- [x] Pre-commit hooks pass
- [x] No breaking changes
- [x] Comprehensive test coverage
- [x] Functions properly exported
- [x] Documentation updated

## 🔍 Review Focus Areas

1. **Subshell handling**: Verify the guard variable unset approach
2. **Error handling**: Check fail-fast behavior
3. **Test coverage**: Review edge case testing
4. **Function exports**: Confirm all functions are exported

## 🚀 Future Enhancements

- [ ] Async/parallel hook execution
- [ ] Hook dependency management
- [ ] Conditional execution based on environment
- [ ] Hook timeout support
- [ ] Integration examples in documentation

## 📚 Related Issues

Closes: N/A (New feature)

## 🙏 Additional Notes

This hook system provides a solid foundation for extensible command execution. The comprehensive test suite ensures reliability, and the design allows for future enhancements without breaking existing functionality.

---

**Ready for review!** 🎉
