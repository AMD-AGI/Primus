# MegaMoE 2×2 comparison (MI355X, 1 node × 8 GPU)

Four arms on the same DeepSeek-V3 4-layer mock-data setup:

| precision | MegaMoE | baseline (stock turbo MoE) |
|-----------|---------|----------------------------|
| bf16      | yes     | yes                        |
| mxfp8     | yes     | yes                        |

## Prerequisites

- Container with Primus + MegaMoE bind mounts:
  - `/home/xiaompen/Primus` → this repo (override: `REPO=` / `PRIMUS_ROOT=`)
  - `/home/xiaompen/MegaMoE` → MegaMoE working tree (override: `MEGAMOE=` / `PRIMUS_MEGAMOE_SRC=`)
  - `PRIMUS_MEGAMOE_SRC` default is set in `run.sh`
- SSH from the driver host to the training node (`BatchMode=yes`).
- **wgrad fix**: mxfp8 baseline needs `primus_turbo.py` to accumulate wgrad on every microbatch (already on `main`; cherry-pick `fix/turbo-wgrad-accum-every-microbatch` if missing).
- **FP8 yaml**: this branch sets `fp8: e4m3`, `fp8_recipe: mxfp8`, `use_turbo_grouped_gemm: true`, `moe_use_legacy_grouped_gemm: false` so the mxfp8 baseline is a same-precision turbo path, not hybrid/legacy bf16 experts.

## Run the full 2×2 matrix

**On the training node itself** (local container, no ssh):

```bash
cd /home/xiaompen/Primus
LOCAL=1 CONTAINER=xiaoming-dev ./run_ab_matrix.sh
```

From a remote driver host that can `ssh` + `docker exec`:

```bash
cd /home/xiaompen/Primus
NODE=<mi355-node> CONTAINER=xiaoming-dev ./run_ab_matrix.sh
```

Outputs land under `ab_2x2/<MMDD-HHMMSS>/`:

- `bf16.mega.log`, `bf16.baseline.log`, `mxfp8.mega.log`, `mxfp8.baseline.log`
- `launch.txt` (node, image, git SHAs, config diff)
- `summary.md` (throughput + loss tables)

Useful overrides:

```bash
LOCAL=1 CONTAINER=xiaoming-dev TRAIN_ITERS=50 ./run_ab_matrix.sh   # defaults
LOCAL=1 ONLY="mxfp8 False" ./run_ab_matrix.sh                      # single arm
LOCAL=1 TRAIN_ITERS=3 ONLY="mxfp8 False" ./run_ab_matrix.sh        # smoke test
```

### Performance mode (default)

DeepEP + sync-free stage 1 on — matches production-ish throughput. Loss gaps between arms can be noisy (DeepEP atomics); use for **ms/iter and TFLOP/s** only.

### Accuracy mode (deterministic loss)

```bash
COMMON_EXTRA="--turbo_sync_free_moe_stage 0 --use_turbo_deepep False" \
  LOCAL=1 CONTAINER=xiaoming-dev ./run_ab_matrix.sh
```

MegaMoE ignores these flags internally but receives the same CLI for symmetry.

## Single-arm shortcuts

```bash
# inside container
PRECISION=bf16  USE_MEGA_MOE=False bash run.sh   # bf16 baseline
PRECISION=mxfp8 USE_MEGA_MOE=True  bash run.sh   # mxfp8 MegaMoE
```

Or use `run_baseline_bf16.sh` / `run_baseline_mxfp8.sh`.

## Parse and plot

```bash
python3 tools/parse_ab_matrix.py ab_2x2/<run-dir> | tee ab_2x2/<run-dir>/summary.md
python3 tools/plot_ab_loss.py ab_2x2/<run-dir>
# → ab_2x2/<run-dir>/loss_curves.png
```

Throughput means iterations **> 20** (JIT + warmup excluded). Parser flag: `--skip 20`.

## Determinism probe (optional)

Diagnose bf16 baseline loss jitter (DeepEP vs sync-free):

```bash
NODE=<node> ./run_determinism_probe.sh
```

## Notes

- `run.sh` runs `pkill -9 python` — **one arm at a time**; do not overlap jobs.
- Logs use `LOG=` override; default names are `log.<precision>.<mega|baseline>` under the repo root.
- Arms are interleaved: mxfp8 baseline → bf16 mega → mxfp8 mega → bf16 baseline.
