# recommendation backend

`framework: recommendation`. The model preset's `model_family` picks the trainer
(`recommendation_adapter.py`, `MODEL_FAMILIES`).

| Family | Model | Recipe | Docs |
|---|---|---|---|
| `dlrm_v4` | HSTU ranker on yambda-5b (MLPerf Training `dlrmv4`) | `examples/mlperf/dlrm_v4/` | `docs/04-technical-guides/recommendation/` |

## dlrm_v4

| Path | What |
|---|---|
| `dlrm_v4/src/generative_recommenders/` | The trainer package: model, data loader, Triton / FlyDSL / HIP kernels, gin configs, per-stack tables, and its own kernel tests. |
| `dlrm_v4/trainer.py` | Primus lifecycle: run-shape environment, gin file, page-cache flush and warm, launch; environment defaults (`ENV_DEFAULTS`). |
| `dlrm_v4/run_config.py` | Config to environment / gin translation (no GPU; unit-tested). |
| `dlrm_v4/pagecache.py` | Evict / verify / warm the dataset in the page cache. |

`dlrm_v4/src/` holds only the package (and its license) because it goes on
`sys.path`: anything else in it would become a top-level module. A different
checkout of the package can be used with `trainer_src:` in the config, or
`--backend_path` / `BACKEND_PATH`.

The trainer starts one process per GPU itself, so the prepare hook
(`runner/helpers/hooks/train/pretrain/recommendation/prepare.py`) makes
`primus-cli direct` run one Primus process per node. The install hook next to
it adds the trainer's extra packages to the `rocm/primus` image.

Elsewhere: data scripts in `examples/mlperf/dlrm_v4/data/`, developer tools
(BLAS / HIP preload table generators, trace tools) in `tools/recommendation/dlrm_v4/`,
Primus unit tests in `tests/unit_tests/backends/recommendation/`.

### Origin

Imported from the MLPerf reference fork `training_dlrmv4` (`recommendation/`),
branch `dlrmv4_opt-mi355-rocm10.2` plus T29, commit `7879b4b`
(2026-11-06): the ROCm 10.2 port validated on MI355X. The import commit keeps
every file's content; later commits are Primus development. Code comments
naming `scripts/launch_slurm.sh`, `scripts/cluster_a/...` or `scripts/cluster_b/...`
refer to that repository's launchers, which were not imported.

The package is excluded from the pre-commit formatters (kept diffable with its
origin). Some of its tests are text-sensitive: `ops/tests/outproj_mx6_seam_test.py`
allow-lists exact bytes of `ops/triton/bwd_fp8.py`, and several gin tests check
the comment blocks of `yambda_5b.gin` (80 columns, item headers). Both files keep
their imported text, including doc paths of the original repository
(`docs/perf_opt.md` is now `docs/04-technical-guides/recommendation/dlrm_v4/perf-notes.md`,
`docs/rocm_10_port.md` -> `rocm-port.md`, `docs/fp8/fp8.md` -> `fp8.md`). Even a
comment edit in the package needs its tests run again (in the image, on a GPU node). License: Apache-2.0
(`dlrm_v4/src/LICENSE`).
