# Production Documentation Verification Report

Date: 2026-06-08

## Scope and Method

Reviewed all 35 Markdown files under `production_docs/` against repository evidence: runner scripts, Python CLI code, config loaders, YAML presets, backend adapters and argument builders, examples, tests, CI, and reviewed documentation outside `production_docs/`.

Source precedence used for conflicts:

1. Code, YAML configs, runner scripts, CI, and checked-in examples.
2. Tests and runnable examples.
3. Reviewed docs outside `production_docs/`.
4. Upstream third-party docs for backend semantics only.
5. Existing `production_docs/` claims are draft material and were not treated as evidence.

## Executive Summary

The documentation set is structurally strong and most architectural, CLI, config, benchmark, preflight, security, and operations claims are directionally correct. It is not publication-ready yet because several high-visibility facts are stale or incorrect.

Highest-priority fixes before publication:

- Replace all `rocm/primus:v26.1` / `docker.io/rocm/primus:v26.1` references with the current `v26.2` default, except where a site-specific historical image is intentionally documented.
- Correct the Megatron parameter reference so defaults reflect the effective `pre_trainer.yaml` preset, not only `trainer_base.yaml`.
- Fix invalid command examples missing the launcher mode, missing the launcher `--` separator, using unsupported Slurm entry shapes, or missing `train pretrain` / `train posttrain`.
- Remove or caveat `--export_config` guidance for the default core runtime until export is implemented there.
- Refresh MI325X support claims and gap entries; MI325X examples now exist under Megatron and TorchTitan.
- Fix the backend-extension sample so the adapter implements `detect_backend_version()` and consumes `params` correctly.

## Severity-Ranked Findings

### Incorrect

| ID | Production doc | Claim | Evidence | Recommended correction |
| --- | --- | --- | --- | --- |
| I-1 | `production_docs/03-configuration-reference/megatron-parameters.md` | Megatron defaults are presented as fully merged effective defaults, but many values come from `trainer_base.yaml` and ignore `pre_trainer.yaml` overrides. | `primus/configs/modules/megatron/pre_trainer.yaml`, `primus/configs/modules/megatron/trainer_base.yaml`; effective examples include `train_iters: 1000`, `global_batch_size: 16`, `use_flash_attn: true`, `save_interval: 1000`, while the doc lists base values such as `train_iters: null`, `global_batch_size: 128`, `use_flash_attn: false`, `save_interval: 20000`. | Regenerate or manually correct Megatron defaults from the resolved `pre_trainer.yaml` extends chain. If documenting base-only defaults, state that explicitly and separate from effective pretrain defaults. |
| I-2 | `production_docs/02-user-guide/configuration-system.md` | The mental model says `model preset` takes precedence over `module preset`. | `primus/core/launcher/parser.py`, `primus/core/utils/yaml_utils.py`; `merge_namespace(module_config, model_config, allow_override=False, excepts=["name"])` keeps duplicate module keys and only adds non-duplicate model keys. | Change precedence text to: CLI overrides > experiment overrides > module preset with model preset additions; duplicate top-level module keys win over model keys during parser merge. |
| I-3 | `production_docs/02-user-guide/posttraining.md` | Direct-mode examples use `./runner/primus-cli direct train posttrain ...` without `--`. | `runner/primus-cli`, `runner/primus-cli-direct.sh`, `production_docs/02-user-guide/cli-reference.md`; shell launcher syntax requires `direct -- train posttrain ...`. | Update direct examples to `./runner/primus-cli direct -- train posttrain --config ...`. |
| I-4 | `production_docs/02-user-guide/posttraining.md` | Container example omits `--` and uses `rocm/primus:latest`. | `runner/primus-cli-container.sh`, `runner/.primus.yaml`, `.github/workflows/ci.yaml`; default image is `rocm/primus:v26.2`, and container syntax requires `container [options] -- train ...`. | Use `./runner/primus-cli container --image rocm/primus:v26.2 -- train posttrain --config ...`, or omit `--image` to use the default. |
| I-5 | `production_docs/06-developer-guide/extending-backends.md` | The `DummyAdapter` sample does not implement abstract `detect_backend_version()` and treats the `convert_config()` input as if it has `.params`. | `primus/core/backend/backend_adapter.py`, `primus/core/runtime/train_runtime.py`; runtime calls `adapter.convert_config(module_config.params)`, and `detect_backend_version()` is abstract. | Add `detect_backend_version()` to the sample and change `convert_config(self, params)` to consume the passed params directly. |
| I-6 | `production_docs/03-configuration-reference/environment-variables.md` | Defaults for `HSA_NO_SCRATCH_RECLAIM` and `NVTE_CK_USES_BWD_V3` are documented as `0`. | `runner/helpers/envs/base_env.sh`; current defaults are `HSA_NO_SCRATCH_RECLAIM=1` and `NVTE_CK_USES_BWD_V3=1`. | Correct the defaults and note any workflow-specific override guidance separately. |
| I-7 | `production_docs/04-technical-guides/multi-node-networking.md` | References `runner/helpers/envs/enable_ainic.sh` as an available legacy script. | `runner/helpers/envs/` has no `enable_ainic.sh`; supported hook is `runner/helpers/hooks/03_enable_ainic.sh`, whose comments mention the old removed file. | Remove the missing path as an available source file; describe it only as historical if needed. |
| I-8 | `production_docs/02-user-guide/configuration-system.md`, `production_docs/02-user-guide/cli-reference.md`, `production_docs/05-operations/monitoring-logging.md`, `production_docs/05-operations/troubleshooting.md` | `--export_config` is documented as usable with normal train commands. | `primus/core/launcher/parser.py` defines the flag, but the default core pretrain path (`primus/cli/subcommands/train.py` -> `PrimusRuntime`) does not read or export it. Export handling exists in legacy `primus/pretrain.py`. | Either document it as legacy-only/currently nonfunctional on the core runtime, or implement export support in `PrimusRuntime` before publishing this guidance. |
| I-9 | `production_docs/02-user-guide/cli-reference.md`, `production_docs/05-operations/deployment.md` | Slurm examples and text recommend `slurm ... -- container -- train ...` as the normal entry pattern. | `runner/primus-cli-slurm-entry.sh` already invokes `primus-cli-container.sh`; a leading `container` token can be forwarded into the Python CLI as an unknown command. Reviewed CLI docs use `slurm ... -- train pretrain ...`, with container flags placed before the inner `--` when needed. | Normalize Slurm examples to the actually supported entry shape and document where container flags belong. |
| I-10 | `production_docs/04-technical-guides/performance-tuning.md` | Megatron loss-fusion defaults are cited as `cross_entropy_loss_fusion: true` and `cross_entropy_fusion_impl: "te"`. | `primus/configs/models/megatron/language_model.yaml` sets `cross_entropy_loss_fusion: false` and `cross_entropy_fusion_impl: "native"`. | Correct the default and make any TE fused-cross-entropy discussion an optional override. |
| I-11 | `production_docs/04-technical-guides/performance-tuning.md` | `recompute_layer_ids` is described using global layer IDs. | `primus/configs/modules/megatron/primus_megatron_module.yaml` and `primus/backends/megatron/patches/recompute_layer_patches.py` treat IDs as per-pipeline-stage indices (`0` to `num_layers_per_pp_stage - 1`). | Document per-stage indexing and required `recompute_granularity: full` / method constraints. |
| I-12 | `production_docs/04-technical-guides/performance-tuning.md` | `use_turbo_grouped_mlp` is presented as the active Megatron Turbo grouped-MoE flag. | `primus/configs/modules/megatron/primus_turbo.yaml` marks it deprecated in favor of `use_turbo_grouped_gemm`; runtime utilities emit a deprecation warning. | Replace with `use_turbo_grouped_gemm` and mention the deprecated alias only for migration. |

### Stale

| ID | Production doc | Claim | Evidence | Recommended correction |
| --- | --- | --- | --- | --- |
| S-1 | Multiple: `overview.md`, `installation.md`, `quickstart.md`, `cli-reference.md`, `pretraining.md`, `deployment.md` | Current/default image is `rocm/primus:v26.1` or `docker.io/rocm/primus:v26.1`. | `runner/.primus.yaml`, `runner/primus-cli-container.sh`, `README.md`, `docs/quickstart.md`, `docs/cli/PRIMUS-CLI-GUIDE.md`, `.github/workflows/ci.yaml`, `examples/README.md`; all use `v26.2`. | Replace production-doc default/reference image tags with `v26.2`. Also update Kubernetes script reference in `deployment.md` to `docker.io/rocm/primus:v26.2`. |
| S-2 | `production_docs/appendix/gaps-and-verification.md`, `production_docs/06-developer-guide/model-support-matrix.md` | MI325X examples do not exist. | `examples/megatron/configs/MI325X/*.yaml` and `examples/torchtitan/configs/MI325X/*.yaml` exist. | Mark Megatron and TorchTitan MI325X examples as present; keep MaxText and Megatron Bridge MI325X as absent unless examples are added. |
| S-3 | `production_docs/appendix/gaps-and-verification.md` | `tests/README.md` content expectations remain unresolved. | `tests/README.md` now contains a test overview and commands. | Close or update this gap; note any remaining details only if the current README is insufficient. |
| S-4 | `production_docs/appendix/gaps-and-verification.md` | Missing old docs pages are listed under `docs/README.md`. | `docs/README.md` now links to production troubleshooting; stale missing links are in `docs/cli/README.md` (`../configuration.md`, `../slurm-container.md`). | Point the gap at `docs/cli/README.md` and the actual missing pages, or update the reference doc. |
| S-5 | `production_docs/02-user-guide/projection.md` | Performance projection command reference omits several current options. | `primus/cli/subcommands/projection.py`; current parser includes `--pipeline-schedule-algorithm`, `--target-num-nodes`, `--target-ep-size`, `--enable-zero-bubble`, `--enable-deepep`, `--sync-free-stage`, `--num-virtual-stages-per-pipeline-rank`, `--micro-batch-size`, and `--global-batch-size`. | Add a projection advanced-options subsection or state the page lists common options only. |
| S-6 | `production_docs/05-operations/monitoring-logging.md`, `production_docs/05-operations/troubleshooting.md` | `primus-cli train --config ... --export_config ...` is shown as a valid export command. | `primus/cli/subcommands/train.py`, `primus/core/launcher/parser.py`, `runner/primus-cli`; shell launcher needs a mode and Python train needs a suite. | Use `./runner/primus-cli direct -- train pretrain --config ... --export_config ...` or `train posttrain` for post-training configs. |
| S-7 | `production_docs/02-user-guide/pretraining.md`, `production_docs/06-developer-guide/model-support-matrix.md` | Example inventories claim to list every checked-in config/preset but omit many current Megatron, TorchTitan, and Megatron Bridge examples and presets. | Filesystem scans show expanded `examples/megatron/configs/MI300X`, `examples/torchtitan/configs/MI300X`, and `examples/megatron_bridge/configs/MI300X` coverage, including Qwen3, LLaMA4, Zebra LLaMA, and Mamba examples not represented in the tables. | Regenerate inventories from the filesystem or remove "every"/"authoritative" wording and label tables as curated samples. |
| S-8 | `production_docs/06-developer-guide/adding-models.md` | TorchTitan Qwen3 8B is presented as a model the reader may need to add. | `primus/configs/models/torchtitan/qwen3_8b.yaml` and matching examples already exist. | Replace with a model not already present, or frame it as an already-shipped pattern to copy. |
| S-9 | `production_docs/06-developer-guide/testing.md`, `tests/README.md` | Unit-test layout is shown as top-level `config/` and `patches/` directories. | Current tests place those under `tests/unit_tests/core/`, and there is also a top-level `tests/unit_tests/megatron/` tree. | Update the test tree diagrams to match the current directory layout. |

### Unsupported or Needs Maintainer Confirmation

| ID | Production doc | Claim | Evidence | Recommended correction |
| --- | --- | --- | --- | --- |
| U-1 | `production_docs/01-getting-started/overview.md` | Workflows include "RL-oriented pipelines." | RL/GRPO parameters exist in Megatron YAML, but no user-facing workflow examples were found. | Downgrade to "experimental RL-related knobs exist" or keep as a documented gap until examples and support status are confirmed. |
| U-2 | `production_docs/01-getting-started/overview.md`, `production_docs/README.md`, `production_docs/appendix/gaps-and-verification.md` | Primus-SaFE is described as an ecosystem layer. | `tools/daily/safe_wrapper.py` references SaFE protocol environment, but no production integration guide exists in this repo. | Keep as external/ecosystem context, not a supported integration claim, unless maintainers provide details. |
| U-3 | `production_docs/05-operations/deployment.md` | Kubernetes deployment is described through `examples/run_k8s_pretrain.sh`. | Script exists and uses an API client, but there is no Helm chart/operator or in-repo server-side implementation. | Current caveat is good; keep the feature framed as a reference script rather than supported Kubernetes deployment product. |
| U-4 | `production_docs/appendix/gaps-and-verification.md`, `production_docs/05-operations/troubleshooting.md` | HummingbirdXT maturity is thin. | `primus/backends/hummingbirdxt/` has adapter, argument builder, and posttrain trainer; `examples/hummingbirdxt/configs/wan22_posttrain.yaml` exists. | Update from "minimal config only" to "adapter/trainer plus one example exists; user-facing support level still needs maintainer confirmation." |
| U-5 | `production_docs/04-technical-guides/checkpoint-management.md`, `production_docs/03-configuration-reference/maxtext-parameters.md` | MaxText checkpoint defaults are presented as Primus defaults without clearly distinguishing upstream MaxText defaults. | Primus overlay sets `enable_checkpointing: false` / `async_checkpointing: false`, while upstream MaxText `base.yml` defaults are `true`. | Clarify whether the table is documenting Primus overlay defaults or upstream MaxText runtime defaults after `base_config: "base.yml"` is loaded. |
| U-6 | `production_docs/03-configuration-reference/torchtitan-parameters.md` | The page says it lists all TorchTitan keys. | It covers Primus presets well, but not every upstream `JobConfig` field, and some documented keys such as `training.mock_data` are Primus extensions. | Reword as "Primus preset keys and common TorchTitan JobConfig fields"; label Primus-only extensions explicitly. |
| U-7 | `production_docs/03-configuration-reference/megatron-bridge-parameters.md` | Megatron Bridge is framed mainly as `sft_trainer.yaml` / post-training. | Repo also contains `primus/configs/modules/megatron_bridge/pretrain_trainer.yaml` and pretrain examples under `examples/megatron_bridge/configs/MI300X/`. | Add a pretrain subsection or explicitly scope the page to SFT/post-training. |

### Publication Readiness

| ID | Area | Evidence | Recommended correction |
| --- | --- | --- | --- |
| P-1 | Broken links | Link audit found one broken production-doc link: `production_docs/01-getting-started/quickstart.md` -> `../02-user-guide/pretraining-workflows.md`. | Change to `../02-user-guide/pretraining.md`. |
| P-2 | Repeated command syntax | `posttraining.md`, `monitoring-logging.md`, and `troubleshooting.md` have invalid command forms; most other CLI pages use the right `mode -- command` pattern. | Normalize all command examples to the CLI reference grammar. |
| P-3 | Repeated image tag | `v26.1` appears across getting-started, CLI, workflow, and deployment docs; reviewed docs and code use `v26.2`. | Replace consistently to avoid user confusion. |
| P-4 | Terminology | Security and deployment docs correctly avoid strong guarantees. Overview language for RL and Primus-SaFE is broader than repository evidence. | Add caveats or move those claims into gaps until confirmed. |
| P-5 | Generated reference scope | `environment-variables.md` says it catalogs variables users may encounter, but omits several `base_env.sh` exports and container passthrough keys. | Either make the table exhaustive or clearly label it as representative. |
| P-6 | CLI/path consistency | Docs mix `./primus-cli` and `./runner/primus-cli`, and some examples use short GEMM flags (`-M`, `-N`, `-K`) not registered by the parser. | Explain root wrapper equivalence once, and use parser-supported `--M`, `--N`, `--K`. |

## Traceability Matrix

| Area | Production docs checked | Evidence paths | Status |
| --- | --- | --- | --- |
| Inventory | All `production_docs/**/*.md` | 35 Markdown files found under `production_docs/` | Verified corpus |
| CLI syntax and modes | `cli-reference.md`, `quickstart.md`, `installation.md`, `pretraining.md`, `posttraining.md`, `deployment.md` | `runner/primus-cli`, `runner/primus-cli-direct.sh`, `runner/primus-cli-container.sh`, `runner/primus-cli-slurm.sh`, `runner/primus-cli-slurm-entry.sh`, `primus/cli/main.py` | Several command fixes needed; Slurm entry syntax and `--export_config` need special attention |
| Config pipeline | `configuration-system.md`, `architecture.md`, `adding-models.md` | `primus/core/launcher/parser.py`, `primus/core/config/yaml_loader.py`, `primus/core/utils/yaml_utils.py`, `primus/core/config/preset_loader.py`, `primus/core/config/primus_config.py` | One precedence correction needed |
| Megatron parameters | `megatron-parameters.md`, related technical docs | `primus/configs/modules/megatron/pre_trainer.yaml`, `trainer_base.yaml`, `primus_turbo.yaml`, `primus_pipeline.yaml`, `zero_bubble.yaml`, `primus/backends/megatron/argument_builder.py` | Incorrect defaults; missing newer keys |
| TorchTitan parameters | `torchtitan-parameters.md` | `primus/configs/modules/torchtitan/pre_trainer.yaml`, `quantize.yaml`, `primus/backends/torchtitan/argument_builder.py` | Leaf defaults verified |
| MaxText parameters | `maxtext-parameters.md` | `primus/configs/modules/maxtext/*.yaml`, `primus/configs/models/maxtext/model_base.yaml`, `primus/backends/maxtext/argument_builder.py`, upstream MaxText `base.yml` | Primus overlay values verified; upstream-vs-overlay defaults need clearer scoping |
| Megatron Bridge parameters | `megatron-bridge-parameters.md`, `posttraining.md` | `primus/configs/modules/megatron_bridge/sft_trainer.yaml`, `pretrain_trainer.yaml`, `primus/configs/models/megatron_bridge/`, `examples/megatron_bridge/configs/`, `primus/backends/megatron_bridge/argument_builder.py` | SFT path mostly verified; pretrain path underdocumented |
| Environment variables | `environment-variables.md`, networking/performance/deployment docs | `runner/helpers/envs/base_env.sh`, `runner/.primus.yaml`, `runner/helpers/hooks/*.sh`, Python `os.getenv` usage, CI env | Partial; default and completeness fixes needed |
| Examples and model support | `model-support-matrix.md`, `pretraining.md`, `posttraining.md`, gaps appendix | `examples/*/configs/MI300X`, `MI325X`, `MI355X`, `primus/configs/models/*` | Stale MI325X and some matrix entries |
| Benchmark/preflight/projection | `benchmarking.md`, `preflight.md`, `projection.md`, collective ops | `primus/cli/subcommands/benchmark.py`, `preflight.py`, `projection.py`, `primus/tools/benchmark/*_args.py`, `primus/tools/preflight/preflight_args.py` | Benchmark/preflight verified; projection incomplete |
| Performance/networking/checkpoint/data | `04-technical-guides/*.md` | `runner/helpers/envs/base_env.sh`, `runner/helpers/hooks/03_enable_ainic.sh`, `examples/offline_tune/`, `primus/configs/modules/*`, backend patches | Mostly verified; stale AINIC legacy path |
| Deployment/operations/security | `05-operations/*.md` | `runner/.primus.yaml`, `runner/use_ainic.yaml`, `examples/run_k8s_pretrain.sh`, `.github/workflows/ci.yaml`, `LICENSE` | Conservative security language verified; image and command fixes needed |
| Developer guide | `architecture.md`, `testing.md`, `contributing.md`, `extending-backends.md`, `adding-models.md` | `primus/core/backend/*`, `primus/core/trainer/base_trainer.py`, `.pre-commit-config.yaml`, `.github/workflows/ci.yaml`, `CONTRIBUTING.md`, `tests/README.md` | Mostly verified; backend sample bug and CI wording drift |

## Verified Areas

- The production-doc corpus contains 35 Markdown files, matching the intended scope.
- CLI discovery, subcommand registration, and `parse_known_args()` behavior in `architecture.md` match `primus/cli/main.py`.
- Runtime architecture in `architecture.md` matches `PrimusRuntime`, `BackendRegistry`, `BackendAdapter`, and patch phases at a high level.
- YAML environment substitution and `extends:` inheritance behavior match `yaml_loader.py`.
- TorchTitan, MaxText, and Megatron Bridge parameter docs are broadly aligned with current checked-in YAML where they document Primus overlays, but need stronger wording around upstream scope, Primus-only extensions, and Megatron Bridge pretrain coverage.
- Benchmark suite names and most flags in `benchmarking.md` match `primus/cli/subcommands/benchmark.py` and benchmark argument builders.
- Preflight flags, output defaults, and no-flag behavior match `preflight_args.py` and the preflight subcommand.
- Security docs are appropriately caveated: they do not claim built-in auth, encryption, or audit coverage.
- License discrepancy is correctly identified in the overview and gaps appendix: `README.md` says Apache 2.0 while `LICENSE` is MIT.

## Recommended Correction Order

1. Fix publication blockers: image tags, broken link, invalid command examples, Slurm entry syntax, and unsupported `--export_config` guidance.
2. Correct Megatron defaults, missing keys, and technical tuning defaults (`cross_entropy_loss_fusion`, `recompute_layer_ids`, deprecated Turbo flags).
3. Update environment-variable defaults and decide whether the reference is exhaustive or representative.
4. Refresh example/model inventories, especially MI325X, Megatron Bridge, Megatron, and TorchTitan tables.
5. Fix backend-extension sample code and developer/test-tree examples.
6. Add caveats for RL, Primus-SaFE, HummingbirdXT, Kubernetes, projection advanced options, and upstream-vs-Primus parameter scopes.

## Residual Risk

This review was repository-evidence based. It did not execute training jobs, build containers, contact external services, validate all upstream third-party default values dynamically, or verify private cluster-specific assumptions. Claims requiring maintainer confirmation should stay explicitly marked as gaps until tested or approved.
