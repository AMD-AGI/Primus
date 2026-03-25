# Documentation Gaps and Verification Needed

This section lists areas where the documentation is based on inference from code structure, where evidence is incomplete, or where maintainer verification would improve accuracy. Items are organized by priority.

---

## High Priority

### Primus-Turbo Installation

The codebase references an external `primus_turbo` Python package extensively, but no installation instructions exist in this repository. The CI pipeline installs it from a specific commit (`PRIMUS_TURBO_COMMIT`), but the steps for end users are not documented.

**Action needed**: Document how users install the `primus_turbo` package (pip from GitHub? pre-installed in Docker image? manual build?).

### Replay Feature Status

The CLI Architecture document (`docs/cli/CLI-ARCHITECTURE.md`) describes a `--replay` feature for reproducing experiments from saved snapshots. It is unclear whether this feature is fully implemented or aspirational.

**Action needed**: Verify if `--replay` is functional and document its current status.

### MaxText Parameter Completeness

Most MaxText configuration parameters come from upstream MaxText's `base.yml`, which is loaded at runtime. The Primus-side documentation covers only the overlay parameters defined in `primus/configs/modules/maxtext/`. The full parameter surface is much larger.

**Action needed**: Consider generating or linking to a complete MaxText parameter reference, or clearly state which parameters are user-serviceable vs internal.

### Megatron Bridge Recipe Parameters

Megatron Bridge loads default parameters dynamically from `megatron.bridge.recipes` at runtime. The exact parameter set depends on the recipe and flavor selected. Not all parameters are statically visible in Primus config files.

**Action needed**: Document the recipe system more completely, or provide a way to dump the effective config for a given recipe/flavor.

---

## Medium Priority

### License Discrepancy

The root `README.md` states "Apache 2.0 License" but the `LICENSE` file contains an MIT license (Copyright AMD). Third-party components have their own licenses (Megatron-LM: MIT/NVIDIA, TorchTitan: Apache 2.0, etc.).

**Action needed**: Clarify the actual license. Update README or LICENSE to be consistent.

### MI325X Support

MI325X is mentioned in documentation as supported hardware, but no example configurations exist under `examples/*/configs/MI325X/`. Only MI300X and MI355X directories are present.

**Action needed**: Clarify MI325X support status. Are MI300X configs expected to work on MI325X? Add MI325X example configs if tested.

### GPU-Specific Environment Files

The GPU environment files (`runner/helpers/envs/MI300X.sh`, `MI325X.sh`, `MI355X.sh`) are mostly commented out with optional overrides. It is unclear which settings are recommended for production.

**Action needed**: Document which GPU-specific environment variables are recommended for each GPU model.

### Experiment Snapshot Directory Structure

The CLI Architecture document describes an auto-saved experiment snapshot structure (env/, config/, logs/, metadata.json), but the exact implementation may differ from the description.

**Action needed**: Verify the snapshot directory structure matches what the code actually produces.

### Missing Documentation Pages

The existing `docs/README.md` links to pages that do not exist:
- `docs/configuration.md`
- `docs/slurm-container.md`
- `docs/experiments.md`
- `docs/advanced.md`
- `docs/faq.md`

**Action needed**: Either create these pages, remove the links, or redirect to corresponding `production_docs/` pages.

---

## Low Priority

### HummingbirdXT Backend Maturity

The HummingbirdXT backend is registered but has minimal configuration (one model preset, one module preset). Documentation coverage is thin.

**Action needed**: Document HummingbirdXT capabilities, limitations, and supported workflows if it is user-facing.

### Primus-SaFE Integration

Primus-SaFE is referenced as part of the ecosystem but no integration details are visible in this repository.

**Action needed**: Document how Primus-LM and Primus-SaFE interact, if applicable.

### Security Audit

No evidence of a security audit, dependency vulnerability scanning, or automated secrets detection in the CI pipeline.

**Action needed**: Consider adding dependency scanning (e.g., `pip-audit`, Dependabot) and document security practices.

### Release and Versioning Process

No release workflow, changelog generation, or version tagging strategy is documented.

**Action needed**: Document the release process, versioning scheme, and how Docker image versions correspond to code versions.

### Reinforcement Learning Workflow

The Megatron trainer_base.yaml includes RL/GRPO parameters (`perform_rl_step`, `rl_grpo`, etc.) but no user-facing documentation or examples for RL workflows exist.

**Action needed**: Document RL capabilities when they are ready for users.

### Undocumented Environment Variables

A small number of environment variables found in code may not be fully documented:
- `CONTI_PARAMS` (used in some Megatron example YAMLs)
- `SAFE_NFS_PATH`, `SAFE_NFS_INPUT`, `SAFE_NFS_OUTPUT` (CI/benchmark infrastructure)
- Various `PRIMUS_TURBO_*` internal variables

**Action needed**: Review these for user relevance and document if needed.

### Test Coverage

The `tests/README.md` is empty. While testing information is now in `production_docs/06-developer-guide/testing.md`, the in-tree README should at minimum point to the production docs.

**Action needed**: Add content to `tests/README.md` and `tools/README.md`.

---

## Methodology Note

This documentation was generated by analyzing:
- All 42 first-party Markdown files in the repository
- Python source code under `primus/`, including config files, CLI definitions, backend adapters, and patches
- Shell scripts under `runner/`, `examples/`, and `benchmark/`
- YAML configuration files under `primus/configs/` and `examples/`
- CI/CD workflows under `.github/workflows/`
- Third-party submodule references in `.gitmodules`
- Requirements files at the repository root
- Web references for Megatron-LM, TorchTitan, MaxText, NCCL, RCCL, and ROCm documentation

Where repository evidence was insufficient, statements are marked with qualifiers such as "needs verification" or "based on code structure".
