# Primus CLI Test Suite

This directory contains integration and unit tests for Primus CLI.

## Test Files

### Unit Tests

1. **`test_common.sh`** - Tests for common library functions
   - Logging functions
   - Path utilities
   - String utilities
   - System utilities
   - Environment utilities
   - Command validation
   - Environment file loading

2. **`test_validation.sh`** - Tests for validation library
   - Distributed training parameter validation
   - Numeric validation (integer, range, positive)
   - Container runtime detection
   - NNODES, NODE_RANK, GPUS_PER_NODE validation
   - MASTER_ADDR, MASTER_PORT validation

### Integration Tests

3. **`test_primus_cli.sh`** - Tests for main entry point
   - `--help` option
   - `--version` option
   - `--dry-run` option
   - `--debug` option
   - `--log-level` option
   - Mode routing (slurm, container, direct)
   - Error handling

4. **`test_helpers.sh`** - Tests for helper modules
   - `execute_hooks()` function
   - `execute_patches()` function
   - Hook execution with various scenarios
   - Patch execution with success/failure cases

### Test Runner

5. **`run_all_tests.sh`** - Master test runner
   - Runs all test suites in order
   - Collects and reports results
   - Exit with appropriate status code

## Running Tests

### Run All Tests

```bash
# From project root
bash tests/cli/run_all_tests.sh

# Or with explicit path
cd /path/to/Primus-CLI
bash tests/cli/run_all_tests.sh
```

### Run Individual Test Suites

```bash
# Test common library
bash tests/cli/test_common.sh

# Test validation library
bash tests/cli/test_validation.sh

# Test main CLI entry
bash tests/cli/test_primus_cli.sh

# Test helper modules
bash tests/cli/test_helpers.sh
```

## Test Requirements

- **Bash**: Version 4.0 or later
- **Environment Variables**:
  - `NODE_RANK` - Set to 0 for testing (automatically set in tests)
- **Dependencies**:
  - Common library (`runner/lib/common.sh`)
  - Validation library (`runner/lib/validation.sh`)
  - Helper modules (`runner/helpers/*.sh`)

## Test Structure

Each test file follows this structure:

```bash
#!/bin/bash
set -euo pipefail

# Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$PROJECT_ROOT/runner/lib/common.sh"

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Test functions
test_pass() { ... }
test_fail() { ... }

# Run tests
LOG_INFO "Test 1: ..."
if condition; then
    test_pass "Test name"
else
    test_fail "Test name"
fi

# Summary
echo "Passed: $TESTS_PASSED"
echo "Failed: $TESTS_FAILED"
exit $TESTS_FAILED
```

## Expected Output

### Successful Test Run

```
=========================================
  Primus CLI Test Suite Runner
=========================================

Running: test_common.sh
=========================================
✓ test_common.sh PASSED

Running: test_validation.sh
=========================================
✓ test_validation.sh PASSED

Running: test_primus_cli.sh
=========================================
✓ test_primus_cli.sh PASSED

Running: test_helpers.sh
=========================================
✓ test_helpers.sh PASSED

=========================================
  Final Test Results
=========================================
Total test suites: 4
Passed: 4
Failed: 0
=========================================
🎉 All test suites passed! ✓
```

### Failed Test Run

If any test fails, the output will show which test failed and the exit code will be non-zero:

```
=========================================
  Final Test Results
=========================================
Total test suites: 4
Passed: 3
Failed: 1
=========================================
❌ Some test suites failed! ✗
```

## Adding New Tests

To add a new test:

1. Create a new test file: `tests/cli/test_<feature>.sh`
2. Follow the standard test structure
3. Make it executable: `chmod +x tests/cli/test_<feature>.sh`
4. Add it to `run_all_tests.sh` in the `TEST_SCRIPTS` array
5. Update this README

Example:

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$PROJECT_ROOT/runner/lib/common.sh"

export NODE_RANK=0

echo "Testing new feature..."
# Your tests here
```

## CI/CD Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: bash tests/cli/run_all_tests.sh
```

## Troubleshooting

### Tests Fail to Find Libraries

**Problem**: Tests can't find `common.sh` or `validation.sh`

**Solution**: Ensure you're running tests from the project root or the paths in test scripts are correct.

### Permission Denied

**Problem**: `Permission denied` when running tests

**Solution**: Make test scripts executable:
```bash
chmod +x tests/cli/*.sh
```

### Environment Variable Issues

**Problem**: Tests fail due to missing `NODE_RANK`

**Solution**: Tests automatically set `NODE_RANK=0`, but if you're running manually, ensure:
```bash
export NODE_RANK=0
```

## Test Coverage

Current test coverage:

- ✅ Common library functions (8 tests)
- ✅ Validation library functions (8 tests)
- ✅ Main CLI entry (10 tests)
- ✅ Helper modules (8 tests)
- ⏳ Container mode (TODO)
- ⏳ Slurm mode (TODO - requires mock)
- ⏳ Direct mode execution (TODO - requires environment)

**Total**: 34 tests across 4 test suites

## Future Improvements

1. **Mock Framework**: Add mocking for Docker/Podman/Slurm
2. **Coverage Report**: Generate test coverage metrics
3. **Performance Tests**: Add performance benchmarks
4. **Stress Tests**: Test with large-scale scenarios
5. **Regression Tests**: Automated regression testing
6. **CI Integration**: Full CI/CD pipeline integration

## Contributing

When adding new features to Primus CLI, please:

1. Write corresponding tests
2. Ensure all existing tests still pass
3. Update this README
4. Document any new test requirements

## Contact

For questions or issues with tests, please contact the Primus CLI development team.
