# feat: Add shared runner library with comprehensive test suite and CI integration

## Summary
Add shared library system for Primus CLI with comprehensive test suite (19 tests passing).

## What's New
- 🔧 **Runner Library** (`runner/lib/`)
  - `common.sh`: Logging, error handling, utilities
  - `validation.sh`: Parameter validation for distributed training
  - `config.sh`: Configuration parsing and validation

- 🧪 **Test Suite** (`tests/runner/`)
  - 19/19 tests passing ✅
  - Coverage: validation, logging, config handling
  - Master test runner: `run_all_tests.sh`

- 🚀 **CI Integration**
  - Shell tests now run in GitHub Actions
  - Automatic validation on every PR

## Test Results
```
✅ test_common.sh PASSED
✅ test_validation.sh PASSED (19/19 tests)
✅ test_config.sh PASSED

Total: 3/3 test suites passing
```

## Key Features

### Structured Logging
```bash
LOG_INFO "Starting process..."
LOG_ERROR "Something went wrong"
LOG_SUCCESS "Completed!"
```

### Parameter Validation
```bash
validate_distributed_params
validate_gpus_per_node
validate_nnodes
validate_node_rank
```

### Error Handling
```bash
require_command "python3"
require_file "config.yaml"
die "Fatal error"
```

## Testing
```bash
# Run all tests
bash tests/runner/run_all_tests.sh

# Run specific test
bash tests/runner/lib/test_validation.sh
```

## Changes
- ✅ 8 new files
- ✅ ~2,900 lines added
- ✅ All shellcheck/pre-commit hooks pass
- ✅ Backward compatible
- ✅ No breaking changes

## Checklist
- [x] Code passes shellcheck
- [x] All tests passing (19/19)
- [x] CI integration complete
- [x] Documentation included
- [x] Backward compatible

---
**Branch**: `feature/cli/lib`
**Files Changed**: +8 new
**Tests**: 19/19 ✅
