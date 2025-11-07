# Primus CLI Testing Guide

## Quick Start

Run all tests:
```bash
cd /path/to/Primus-CLI
bash tests/cli/run_all_tests.sh
```

Run individual test:
```bash
bash tests/cli/test_primus_cli.sh
```

## Test Suite Structure

```
tests/cli/
├── README.md                 # Detailed documentation
├── TESTING_GUIDE.md         # This file (quick reference)
├── run_all_tests.sh         # Master test runner
├── test_common.sh           # Common library tests
├── test_validation.sh       # Validation library tests
├── test_primus_cli.sh       # Main CLI tests
└── test_helpers.sh          # Helper modules tests
```

## Test Categories

### 1. Unit Tests
- **test_common.sh**: Tests for `runner/lib/common.sh`
  - Logging, path utilities, string utilities, system utilities

- **test_validation.sh**: Tests for `runner/lib/validation.sh`
  - Parameter validation, distributed training settings

### 2. Integration Tests
- **test_primus_cli.sh**: Tests for main `primus-cli` entry point
  - Global options, mode routing, error handling

- **test_helpers.sh**: Tests for helper modules
  - `execute_hooks.sh`, `execute_patches.sh`

## Running Tests

### All Tests (Recommended)
```bash
bash tests/cli/run_all_tests.sh
```

Expected output on success:
```
🎉 All test suites passed! ✓
Exit code: 0
```

### Individual Test Suites
```bash
# Common library
bash tests/cli/test_common.sh

# Validation library
bash tests/cli/test_validation.sh

# Main CLI
bash tests/cli/test_primus_cli.sh

# Helper modules
bash tests/cli/test_helpers.sh
```

## Test Requirements

- Bash 4.0+
- Project must be in valid state (all files present)
- No special permissions required

## Environment Variables

Tests automatically set:
- `NODE_RANK=0` - For distributed testing
- `PRIMUS_LOG_LEVEL=INFO` - Default log level

Override if needed:
```bash
export PRIMUS_LOG_LEVEL=DEBUG
bash tests/cli/run_all_tests.sh
```

## Test Output

### Success
```
[2025-11-05 20:28:12] [NODE-0(hostname)] [SUCCESS] ✓ Test name
```

### Failure
```
[2025-11-05 20:28:12] [NODE-0(hostname)] [ERROR] ✗ Test name
```

### Summary
```
Passed: 34
Failed: 0
Total: 34
```

## Common Issues

### Issue: Permission Denied
**Solution**: Make scripts executable
```bash
chmod +x tests/cli/*.sh
```

### Issue: Module Not Found
**Solution**: Ensure you're in project root
```bash
cd /path/to/Primus-CLI
bash tests/cli/run_all_tests.sh
```

### Issue: Tests Hang
**Solution**: Use timeout
```bash
timeout 60 bash tests/cli/run_all_tests.sh
```

## Test Coverage

Current coverage (34 tests):

| Component | Tests | Status |
|-----------|-------|--------|
| Common library | 8 | ✅ |
| Validation library | 8 | ✅ |
| Main CLI entry | 10 | ✅ |
| Helper modules | 8 | ✅ |
| Container mode | 0 | ⏳ TODO |
| Slurm mode | 0 | ⏳ TODO |
| Direct execution | 0 | ⏳ TODO |

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run CLI tests
        run: bash tests/cli/run_all_tests.sh
```

### GitLab CI Example
```yaml
test:
  script:
    - bash tests/cli/run_all_tests.sh
```

## Adding New Tests

1. Create test file: `tests/cli/test_<feature>.sh`
2. Follow the template structure (see existing tests)
3. Make it executable: `chmod +x tests/cli/test_<feature>.sh`
4. Add to `run_all_tests.sh`
5. Update README.md

## Quick Reference

| Command | Description |
|---------|-------------|
| `bash tests/cli/run_all_tests.sh` | Run all tests |
| `bash tests/cli/test_common.sh` | Test common library |
| `bash tests/cli/test_validation.sh` | Test validation |
| `bash tests/cli/test_primus_cli.sh` | Test main CLI |
| `bash tests/cli/test_helpers.sh` | Test helpers |

## Troubleshooting

### Debug Mode
```bash
export PRIMUS_LOG_LEVEL=DEBUG
bash tests/cli/test_primus_cli.sh
```

### Verbose Output
```bash
bash -x tests/cli/test_primus_cli.sh
```

### Check Specific Test
```bash
# Run only Test 3
bash tests/cli/test_primus_cli.sh 2>&1 | grep "Test 3"
```

## Best Practices

1. ✅ Run tests before committing
2. ✅ Add tests for new features
3. ✅ Keep tests fast and focused
4. ✅ Use descriptive test names
5. ✅ Clean up temporary files

## Support

For issues or questions:
- Check `tests/cli/README.md` for detailed documentation
- Review existing test examples
- Contact Primus CLI development team

## Change Log

### v1.1.0 (Week 2)
- ✅ Created comprehensive test suite
- ✅ Added 4 test files with 34 tests
- ✅ Moved tests to `tests/cli/` directory
- ✅ Added documentation (README.md, TESTING_GUIDE.md)

### v1.0.0 (Week 1)
- ✅ Initial test files in `runner/lib/`
- ✅ Basic common and validation tests
