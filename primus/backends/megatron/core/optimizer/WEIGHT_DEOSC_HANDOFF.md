# MXFP4 Weight De-oscillation — Handoff

> Audience: the next engineer / Cursor session that will **run and validate** this
> feature on a GPU node. The author implemented it without GPU access, so it is
> compile-checked and unit-tested (CPU, mocked) but **not yet run end-to-end**.

---

## 1. TL;DR

We ported ALTO's *weight de-oscillation* to the **Primus Megatron backend** for
**Primus-Turbo MXFP4** training. After each distributed-optimizer step, a
`DistRatio` detector finds weight elements whose MXFP4-quantized value
oscillates while the fp32 master barely moves, and "snaps" them to their
quantization-bin center. This is a convergence-stability trick for MXFP4
training.

- **Branch:** `feat/mxfp4-weight-deosc`
- **Commits:** `042718ae` (core), `c5fbe2e4` (persistence + overlap sync + MoE align + tests)
- **Hardware:** MXFP4 kernels require **gfx950 / MI355X**. **MI300/MI325 (gfx942) cannot run native MXFP4** (and therefore cannot run this feature). See §6.

---

## 2. Files in this change

| File | Role |
|---|---|
| `primus/backends/megatron/core/optimizer/weight_deosc.py` | Core: `qdq_mxfp4`, `WeightDeOscRunner`, `install_weight_deosc`, checkpoint sidecar helpers |
| `primus/backends/megatron/patches/turbo/deosc_patches.py` | Patch: wraps `get_megatron_optimizer` to install de-osc; wraps `save_checkpoint`/`load_checkpoint` for persistence |
| `primus/configs/modules/megatron/primus_turbo.yaml` | New config flags `weight_deosc*` |
| `tests/unit_tests/backends/megatron/test_weight_deosc.py` | CPU unit tests (mocked QDQ + fake distributed optimizer) |

---

## 3. How it works (essential mental model)

ALTO's algorithm, per de-osc-eligible weight, over a `period`-step window:

```
dist_w     = Σ_t |w_t     − w_{t-1}|        # fp32 master movement (denominator)
dist_w_qdq = Σ_t |Q(w_t)  − Q(w_{t-1})|     # quantized-value movement (numerator)
# period end: snap elements with dist_w_qdq/dist_w >= ratio_threshold to Q(w)
```

**The hard part (why this lives in the optimizer):** `Q(w) = dequant(quant(w))`
must use the *same* grid as the forward GEMM. The Primus-Turbo MXFP4 weight path
uses **2D 32×32 block scaling** (`ScalingRecipe(use_2d_block=True)`, `axis=-1`,
`block_size=32`), so `Q(w)[i,j]` needs the **full 2D weight tile**. But the
Megatron **distributed optimizer flattens each param and splits it into
arbitrary contiguous 1D slices across DP ranks** — a slice can cut a row/tile in
half, so you cannot block-quantize a shard directly.

**Our solution (zero extra communication):**
- Primus-Turbo linears run with `tensor_model_parallel_size == 1`, and the
  distributed optimizer **all-gathers the bf16 model weight every step**
  (`step_with_ready_grads` → `_copy_main_params_to_model_params` →
  `start_param_sync`). So right after the step, `model_param.data` is the
  **full 2D weight** on every rank.
- Compute `Q_full = qdq_mxfp4(model_param)` on the full weight (block-correct).
- Use the optimizer's `_get_model_param_range_map(model_param)["param"]` to find
  this rank's flat slice `[start:end)`, and keep **only local-shard-sized**
  tracking state (`prev`, `prev_q`, `dist_w`, `dist_w_qdq`).
- `dist_w` is measured on the **local fp32 master shard** (`shard_main_param`),
  `dist_w_qdq` on the local slice of `Q_full`.
- The snap writes back into the **local fp32 master shard**; it propagates to the
  model weight on the next `_copy_main_params_to_model_params` + all-gather
  (a harmless 1-step delay vs ALTO).

Insertion point: `install_weight_deosc` wraps each `DistributedOptimizer`
instance's `step_with_ready_grads` (after the inner step + copy + all-gather).

**Eligibility** is detected at runtime: a weight is de-osc eligible iff its
module has `quantized_weight_buffer is not None` (set by the Primus-Turbo FP4
forward). This automatically excludes bf16 first/last layers and any layer whose
FP4 path never ran.

---

## 4. How to enable

Requires `fp4 + use_turbo_fp4_autocast + use_distributed_optimizer` and the
Primus-Turbo FP4 autocast path (NOT the TE recipe path). Add to the megatron
module config (e.g. in an example under `examples/megatron/configs/MI355X/`):

```yaml
# Mixed precision (MXFP4 via Primus-Turbo)
fp4: e2m1
fp4_recipe: mxfp4
enable_primus_turbo: true
use_turbo_fp4_autocast: true        # MUST be true (TE recipe path is not de-osc'd)
use_distributed_optimizer: true
overlap_param_gather: false         # optional; if true, de-osc force-syncs anyway

# Weight de-oscillation
weight_deosc: true
weight_deosc_period: 200            # observe/reset window in optimizer steps
weight_deosc_ratio: 4.0             # DistRatio threshold
weight_deosc_start_step: 2000       # start tracking after this many steps
weight_deosc_log_freq: 1            # log snap stats every N periods (0 = off)
```

Config flags are declared in `primus/configs/modules/megatron/primus_turbo.yaml`
and read via `getattr(args, ...)`.

---

## 5. How to verify (do this first on the GPU node)

1. **Unit tests (no GPU needed, just torch):**
   ```bash
   pytest tests/unit_tests/backends/megatron/test_weight_deosc.py -v
   ```
   These mock the QDQ and a fake distributed optimizer; they validate snap
   masking, fp32-shard write-back, period reset, eligibility, and checkpoint
   state round-trip.

2. **Smoke run** on MI355X with a small dense model (e.g. `llama3.1_8B`),
   `tensor_model_parallel_size=1`, `data_parallel_size>=2`,
   `use_distributed_optimizer=true`, turbo MXFP4 autocast, `weight_deosc: true`,
   `weight_deosc_start_step: 1`, `weight_deosc_log_freq: 1`.
   - Expect a startup log:
     `[Patch:megatron.turbo.weight_deosc] Patched get_megatron_optimizer ...`
     and `[WeightDeOsc] enabled on N distributed optimizer instance(s): ...`
   - After each `period`, expect `[WeightDeOsc] step=… period=… snapped X/Y elems`.
   - Training must not crash; loss should stay sane.

3. **Checkpoint round-trip:** save a checkpoint, confirm
   `<ckpt_dir>/iter_XXXXXXX/weight_deosc/rank_<R>.pt` files appear; resume and
   confirm no errors and de-osc continues (global_step preserved).

4. **Convergence A/B:** run identical configs with `weight_deosc: true` vs
   `false`; compare MXFP4 loss curves (this is the actual point of the feature).

---

## 6. Hardware reality check

- `check_mxfp4_support()` requires `get_device_compute_capability() >= (9, 5)`
  (`~/Primus-Turbo/primus_turbo/pytorch/core/low_precision.py:45`). gfx950 =
  CDNA4 = **MI355X**. On gfx942 (MI300/MI325) `float4_e2m1fn_x2 is None` and the
  FP4 quantize kernel asserts — native MXFP4 will not run.
- **Simulation on MI300:** possible in principle (fake-quant QDQ in bf16, which
  is exactly what ALTO does with Triton kernels), but **Primus megatron has no
  fake-quant MXFP4 path today** — it only wires the real Primus-Turbo/TE FP4
  kernels. If MI300 validation is needed, either use ALTO directly, or add an
  emulated-MXFP4 forward + emulated `qdq_mxfp4` (would also need to bypass
  `float4_e2m1fn_x2`). Not implemented.

---

## 7. Known limitations / open work

1. **MoE experts are not runnable yet.** `PrimusTurboGroupedLinear.forward_internal`
   still has `assert False, "FP4 is not supported in PrimusTurboGroupedLinear"`
   (`primus/backends/megatron/core/extensions/primus_turbo.py`, ~line 1782).
   Primus-Turbo **PR #398** adds the grouped MXFP4 GEMM (`grouped_gemm_fp4`), but
   it is **not wired into Primus' `PrimusTurboGroupedLinear` yet**. The de-osc
   3D path (`qdq_mxfp4` per-expert, `axis=-1`, `use_2d_block=True`) is already
   correct for PR #398's weight quantization (verified against
   `grouped_gemm_fp4.py::_quant_weight_dual`), and grouped weights become
   eligible **automatically** once the grouped FP4 forward sets
   `quantized_weight_buffer`. **Next step for MoE:** wire
   `PrimusTurboGroupedLinear` to `primus_turbo.pytorch.ops.grouped_gemm_fp4`
   (mirror the fp8 grouped path) and set `quantized_weight_buffer` like the dense
   linears do.

2. **Checkpoint persistence is per-rank, same-layout only.** Sidecars are keyed
   by `<param_name>|<start>:<end>` and saved per global rank. Resume with a
   **different** parallel layout (resharding, different world size) will drop
   mismatched shards and re-seed those windows (harmless — just loses one
   period's accumulation). Full resharding-correct persistence would require
   integrating with Megatron's `ShardedTensor`/`get_parameter_state` machinery
   (deliberately avoided to keep risk low).

3. **QDQ redundancy.** Each DP rank recomputes the full-weight QDQ for every
   eligible param each step (no extra comm, but redundant compute). Cheap vs the
   GEMMs; could later be optimized to only the tiles covering the local shard, or
   to reuse the forward's cached quantized weight.

4. **`overlap_param_gather=True`** triggers a forced synchronous param
   all-gather inside de-osc each step (correctness over the overlap perf win).
   Prefer `overlap_param_gather: false` when de-osc is on.

5. **Incompatible with `use_precision_aware_optimizer` (bf16 main params).**
   In that mode the distributed optimizer keeps no fp32 master shard
   (`shard_fp32_from_float16_groups` is all `None`; main params live inside
   FusedAdam), so de-osc has nothing to track/snap. `install_weight_deosc`
   detects this and **skips with a clear warning** (it does NOT silently no-op).
   Keep `use_precision_aware_optimizer: false` for de-osc runs. Supporting it
   would require reaching into FusedAdam's master and would also weaken the
   DistRatio denominator (bf16 master loses sub-bf16 movements). Note: a bf16
   master does not *conceptually* break de-osc (bf16 is still ~16x finer than
   mxfp4), but it is unsupported in code today.

---

## 8. Key external references (read these to understand the integration)

| What | Where |
|---|---|
| ALTO de-osc origin | `~/ALTO` branch `han/weight-deosc`, `alto/components/optimizer.py` |
| Primus-Turbo MXFP4 dense forward (weight quant we mirror) | `primus/backends/megatron/core/extensions/primus_turbo.py` (FP4 branches in `forward_internal`, `ScalingRecipe(use_2d_block=True)`, `axis=-1`) |
| Primus-Turbo MXFP4 quant config | `primus/backends/megatron/core/fp4_utils.py::get_fp4_quant_config` (MX_BLOCKWISE, E2M1_X2, block=32, E8M0) |
| Primus-Turbo grouped MXFP4 (MoE) | PR #398 `AMD-AGI/Primus-Turbo`; commits exist locally on `~/Primus-Turbo` (`710f9ff3`, `03927e18`, `3ad1228e`); see `grouped_gemm_fp4.py::_quant_weight_dual` |
| Distributed optimizer internals | `third_party/Megatron-LM/megatron/core/optimizer/distrib_optimizer.py` — `shard_fp32_from_float16_groups`, `model_float16_groups`, `_get_model_param_range_map`, `step_with_ready_grads`, `_copy_main_params_to_model_params` |
| QuantizedTensor API | `~/Primus-Turbo/primus_turbo/pytorch/core/quantized_tensor.py` (`quantize`/`dequantize`) |
| Patch mechanism reference | `primus/backends/megatron/patches/muon_optimizer_patches.py` (same `get_megatron_optimizer` wrap pattern) |

---

## 9. Parity vs ALTO (for reference)

- **Same:** DistRatio detector, per-element snap to bin center, period reset,
  `prev` refresh after snap, config params (`period`/`ratio`/`start`/`log_freq`).
- **Intentional differences (Megatron-specific):** `dist_w` from fp32 master
  shard; `Q` from the all-gathered bf16 model weight (sliced to the local
  shard); 1-step delay; eligibility via runtime `quantized_weight_buffer`
  instead of ALTO's wrapper-type check; one QDQ/step (cached `prev_q`) vs ALTO's
  two.
- **Not applicable:** NVFP4 (Primus-Turbo has no NVFP4); 1D-block toggle
  (Primus weight forward is always 2D-block).

---

## 10. If something breaks

- De-osc never throws into training: `run()` is wrapped in try/except (logs
  `[WeightDeOsc] skipped this step due to error: …`). If you see that spam,
  read the exception — most likely `qdq_mxfp4` config mismatch or a non-2D/3D
  eligible weight.
- No `[WeightDeOsc] enabled …` log → check the enable condition: `fp4` +
  `use_turbo_fp4_autocast` + `weight_deosc` all true, and `tensor_model_parallel_size == 1`
  (required by `is_primus_turbo_can_patch`).
- Snap count always 0 → `start_step` not reached yet, or `ratio_threshold` too
  high, or the eligible set is empty (FP4 forward never ran → check the model is
  actually using Primus-Turbo FP4 linears).
- Eligible set empty → confirm dense linears are `PrimusTurbo*Linear` and that
  `quantized_weight_buffer` is set after a forward (it is set on
  `is_first_microbatch`).
