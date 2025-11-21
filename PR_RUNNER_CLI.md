# Runner CLI - Initial Integration

## Overview

First integration of the refactored Primus Runner CLI into the main branch. This CLI provides a unified command-line interface supporting three execution modes: container, Slurm, and direct execution.

## Key Features

### Three Execution Modes
- **Container**: Containerized execution (Docker/Podman support)
- **Slurm**: Cluster scheduling execution (srun/sbatch support)
- **Direct**: Direct execution (single/multi-process support)

### Core Capabilities
- ✅ Unified configuration management (YAML config file support)
- ✅ Flexible parameter passing (CLI overrides config file)
- ✅ Dry-run mode (preview commands before execution)
- ✅ Debug mode (detailed logging output)
- ✅ Comprehensive validation and error messages

## Usage Examples

### Container Mode
```bash
primus-cli container --image rocm/primus:latest -- train
```

### Slurm Mode
```bash
primus-cli slurm -N 4 -p gpu -- container -- train
```

### Direct Mode
```bash
primus-cli direct -- train
```

### With Configuration File
```bash
primus-cli --config .primus.yaml container -- train
```

## File Structure

```
runner/
├── primus-cli                    # Main entry script
├── primus-cli-container.sh       # Container mode
├── primus-cli-slurm.sh          # Slurm mode
├── primus-cli-slurm-entry.sh    # Slurm entry script
├── primus-cli-direct.sh         # Direct mode
├── .primus.yaml                 # Default config example
├── lib/
│   ├── common.sh                # Common utilities
│   ├── config.sh                # Configuration management
│   └── validation.sh            # Parameter validation
└── helpers/
    ├── detect_gpu_model.sh      # GPU detection
    └── envs/                    # Environment configs

tests/runner/
├── test_primus_cli.sh           # CLI main logic tests
├── test_primus_cli_container.sh # Container mode tests
├── test_primus_cli_slurm.sh     # Slurm mode tests
├── test_primus_cli_direct.sh    # Direct mode tests
└── lib/
    ├── test_common.sh           # Common utilities tests
    ├── test_config.sh           # Config management tests
    └── test_validation.sh       # Validation functions tests
```

## Test Results

- ✅ **8/10** test suites passing
- ✅ **317** unit tests passing
- ⚠️ 2 test suites fail due to environment limitations (require Docker/Podman)

## Change Statistics

- **Added**: 5 files (direct mode, tests, configs)
- **Modified**: 13 files (scripts, libraries, tests)
- **Deleted**: 1 file (deprecated entrypoint)
- **Lines of code**: +3,357 / -818

## Important Notes

### Slurm Argument Format
Only `--key value` format is supported, not `--key=value`

✅ Correct: `primus-cli slurm --nodes 4`
❌ Incorrect: `primus-cli slurm --nodes=4`

### Configuration File
Boolean values use `true`/`false`, not `0`/`1`

```yaml
# Recommended
debug: true
dry_run: false

# Not recommended
debug: 1
dry_run: 0
```

## Future Plans

- [ ] Add more documentation and examples
- [ ] Improve error messages and hints
- [ ] Support more customization options
- [ ] Optimize performance and user experience

---

**This marks the initial integration of Runner CLI, officially launching the Primus command-line tool.**
