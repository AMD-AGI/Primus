# Determinism and reproducibility

Reproducibility — getting bit-identical (or run-to-run stable) results — matters for debugging divergence, validating optimizations, and regression testing. This guide covers Primus's deterministic mode, the environment variables it sets, the per-backend seed/determinism knobs, and the performance trade-offs. Parameters and behavior are grounded in `examples/run_pretrain.sh`, `primus/configs/modules/megatron/trainer_base.yaml`, and `primus/configs/modules/torchtitan/pre_trainer.yaml`.

---

## 1. What "deterministic" means here

There are two distinct goals:

- **Reproducible (seeded)**—same seed + same config + same hardware/software gives the *same trajectory*. Achieved with fixed seeds; cheap.
- **Bitwise-deterministic**—kernels avoid non-deterministic reductions/atomics and tuning so results don't vary between runs. Requires deterministic algorithms and disabling autotuning; **slower**.

Full determinism also generally requires the **same world size, parallelism layout, and library versions**. Changing TP/PP/DP, GPU count, or ROCm/Megatron versions can change numerics even with everything else fixed.

---

## 2. Primus deterministic mode (`PRIMUS_DETERMINISTIC`)

Setting `PRIMUS_DETERMINISTIC=1` configures the GPU/communication stack for deterministic behavior. The CLI/runner path applies this through the hook `runner/helpers/hooks/05_deterministic.sh`; the `examples/run_pretrain.sh` script applies an equivalent inline block. The exported variables are:

```bash
# when PRIMUS_DETERMINISTIC=1 (runner/helpers/hooks/05_deterministic.sh)
export NCCL_ALGO="Ring"                    # deterministic collective algorithm
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0  # Transformer Engine: forbid non-deterministic kernels
export ROCBLAS_DEFAULT_ATOMICS_MODE=0      # rocBLAS: disable atomic (non-deterministic) reductions
export TORCH_COMPILE_DISABLE=1             # avoid torch.compile/Triton race conditions
export PRIMUS_TURBO_AUTO_TUNE=0            # disable Primus-Turbo autotuning (stable kernel choice)
```

> `PRIMUS_TURBO_AUTO_TUNE` also defaults to `0` in `runner/helpers/envs/base_env.sh`. The inline block in `examples/run_pretrain.sh` sets the first four variables and relies on that default for the fifth.

Additionally, **HipBLASLt autotuning is disabled** in deterministic mode: tuning only runs when `PRIMUS_DETERMINISTIC != 1` *and* `PRIMUS_HIPBLASLT_TUNING=1` (`examples/run_pretrain.sh`). This prevents run-to-run kernel-selection differences. See [Performance tuning](./performance-tuning.md).

`PRIMUS_DETERMINISTIC` is on the container passthrough allowlist (`runner/.primus.yaml`), so it reaches the training container. See [Environment variables](../03-configuration-reference/environment-variables.md).

```bash
export PRIMUS_DETERMINISTIC=1
./runner/primus-cli direct -- train pretrain \
  --config examples/megatron/configs/MI300X/llama2_7B-BF16-pretrain.yaml
```

> The MoE example scripts explicitly set `PRIMUS_DETERMINISTIC=0` because deterministic mode disables the performance kernels/tuning they rely on.

---

## 3. Seeds & deterministic algorithms (Megatron)

In `primus/configs/modules/megatron/trainer_base.yaml`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `seed` | `1234` | Master RNG seed (Python/NumPy/Torch, data order, init). |
| `deterministic_mode` | `false` | Force deterministic kernels/algorithms inside Megatron (slower; pairs with `PRIMUS_DETERMINISTIC`). |
| `data_parallel_random_init` | `false` | When `false`, parameters are initialized identically and broadcast across DP ranks; keep `false` for reproducible init. |

For a fully reproducible Megatron run: set a fixed `seed`, `deterministic_mode: true`, and launch with `PRIMUS_DETERMINISTIC=1`.

> **Startup assertion.** When `deterministic_mode: true`, Primus validates (`primus/modules/trainer/megatron/utils.py`, `validate_args_on_rocm`) that these environment variables are set, and **fails fast** otherwise: `TORCH_COMPILE_DISABLE=1`, `ROCBLAS_DEFAULT_ATOMICS_MODE=0`, `PRIMUS_TURBO_AUTO_TUNE=0`, and `PRIMUS_DETERMINISTIC=1`. Launching with `PRIMUS_DETERMINISTIC=1` (above) sets all of them, so always pair `deterministic_mode: true` with `PRIMUS_DETERMINISTIC=1`.

---

## 4. Seeds & determinism (TorchTitan)

Under `training:` in `primus/configs/modules/torchtitan/pre_trainer.yaml`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `seed` | `null` | RNG seed; set an integer for reproducible runs. |
| `deterministic` | `false` | Enable deterministic algorithms (disables some optimized kernels; slower). |

Related: `checkpoint.create_seed_checkpoint` (`false`) creates a deterministic seed checkpoint that all ranks load, ensuring identical initialization across a distributed run.

Note `compile.enable: true` is the TorchTitan default; for strict determinism prefer launching with `PRIMUS_DETERMINISTIC=1` (which sets `TORCH_COMPILE_DISABLE=1`) or disable compilation.

---

## 5. MaxText

MaxText determinism is governed by the upstream MaxText seed/data options surfaced through the MaxText config (see [MaxText parameters](../03-configuration-reference/maxtext-parameters.md)). The GPU-stack environment effects of `PRIMUS_DETERMINISTIC` (rocBLAS atomics, deterministic collectives) still apply at the launcher level.

---

## 6. Performance trade-offs

Determinism is not free:

| Setting | Cost |
|---------|------|
| `NCCL_ALGO=Ring` | Forgoes faster topology-aware collective algorithms. |
| `ROCBLAS_DEFAULT_ATOMICS_MODE=0` | Disables atomic reductions—slower GEMMs. |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | Restricts TE to deterministic (often slower) kernels. |
| `TORCH_COMPILE_DISABLE=1` | No `torch.compile` fusion/codegen speedups. |
| `PRIMUS_TURBO_AUTO_TUNE=0` | No Primus-Turbo kernel autotuning. |
| HipBLASLt tuning disabled | No autotuned GEMM kernels. |
| `deterministic_mode` / `deterministic` | Deterministic algorithm variants are generally slower. |

**Use deterministic mode for debugging and validation, not production throughput runs.** Once a result is reproduced/diagnosed, disable it to recover performance.

---

## 7. Reproducibility checklist

1. **Pin the environment**—same container image, ROCm version, and backend (Megatron/TorchTitan) commit.
2. **Fix seeds**—Megatron `seed`; TorchTitan `training.seed`.
3. **Hold the layout constant**—same world size and TP/PP/DP/EP/CP degrees.
4. **Enable determinism**—`PRIMUS_DETERMINISTIC=1` plus backend `deterministic_mode`/`deterministic`.
5. **Disable autotuning**—automatic in deterministic mode (HipBLASLt tuning off).
6. **Use mock or fixed data ordering**—ensure the data pipeline is seeded; see [Data preparation](./data-preparation.md).
7. **Record everything**—log the full resolved config and env (see [Logging & experiment tracking](./logging-and-experiment-tracking.md)).

---

## See also

- [Performance tuning](./performance-tuning.md)—HipBLASLt tuning and its interaction with deterministic mode.
- [Environment variables](../03-configuration-reference/environment-variables.md)—`PRIMUS_DETERMINISTIC` and related flags.
- [Data preparation](./data-preparation.md)—deterministic data ordering.
- [Megatron parameters](../03-configuration-reference/megatron-parameters.md) and [TorchTitan parameters](../03-configuration-reference/torchtitan-parameters.md).
