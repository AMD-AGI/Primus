# PR Title
Preflight: simplify CLI + unify info/perf + improve report

# PR Description
This PR refactors `primus-cli preflight` to make it simpler to use and easier to maintain.

- **Simplified CLI flags**: use `--host/--gpu/--network` (keeps `--check-*` aliases for backward compatibility) and adds `--perf-test`.
- **Clear dispatch/defaults**:
  - No flags → run **all info + all perf tests**
  - `--host/--gpu/--network` → run **only selected info**
  - `--perf-test` → run **perf tests only**
- **Unified implementation**: argument parsing moved to `primus/tools/preflight/preflight_args.py`; lightweight info logic merged into `preflight_perf_test.py` and `preflight_check.py` removed; distributed init/finalize unified.
- **Better report**: section titles updated to **Host Info / GPU Info / Network Info**; GEMM sanity aggregates **min/max/avg** across ranks and shows GEMM shape as a description.
