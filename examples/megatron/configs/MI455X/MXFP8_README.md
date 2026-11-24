# MXFP8 pretraining on MI455X (gfx1250): Llama-3.1-8B, GPT-OSS-120B, Qwen3-30B-A3B

MXFP8 here means E4M3 data with one E8M0 scale per 32 elements, on every transformer-layer linear
GEMM: forward, dgrad and wgrad. **Primus-Turbo does all of it. Transformer Engine runs no MXFP8 and
no FP8 at all**:

- Megatron runs without `fp8` / `fp8_recipe`.
- `turbo_precision: mxfp8` makes every Primus-Turbo linear quantize its own operands:
  - dense linears (`use_turbo_gemm`): Turbo's FlyDSL gfx1250 WMMA-scale MXFP8 GEMM;
  - expert grouped linears (`use_turbo_grouped_gemm`): Turbo's Triton `MX_BLOCKWISE` grouped GEMM.
- These stay BF16: the embedding, the output layer (LM head), the router, the norms and the attention
  core. Master weights stay FP32, as in the BF16 configs.

| config | source | changes |
|---|---|---|
| `llama3.1_8B-MXFP8-pretrain.yaml` | `llama3.1_8B-BF16-pretrain.yaml` (MI455X, via `extends`) | `turbo_precision: mxfp8`, `use_turbo_gemm: true` |
| `gpt_oss_120B-MXFP8-pretrain.yaml` | `gpt_oss_120B-BF16-pretrain.yaml` (MI455X, via `extends`) | the above, plus `use_turbo_grouped_gemm: true`, `turbo_grouped_gemm_wgrad_accum_fusion: false`, `turbo_grouped_gemm_without_padding: true` (the MXFP8 settings of `muse_spark_816B_A42B-MXFP8`) |
| `qwen3_30B_A3B-MXFP8-pretrain.yaml` | `MI355X/qwen3_30B_A3B-FP8-pretrain.yaml` (there is no MI455X one) | the gpt-oss changes, with `fp8` / `fp8_recipe` removed, plus DeepEP and sync-free MoE off (as in the MI455X gpt-oss / DeepSeek-V3 configs), `turbo_fused_grouped_gemm: false`, `turbo_fp8_permute: false` |

Model sizes, batch sizes and parallelism are the same as in the source configs. MI455X memory did not
force any change. Qwen3 keeps EP8 from MI355X; on a 4-GPU node, set `PRIMUS_EP=4`.

## The switches

| switch | values | effect |
|---|---|---|
| `turbo_precision` (yaml, `primus_turbo.yaml`) or env `PRIMUS_TURBO_PRECISION` (wins) | `null` (default), `mxfp8` | Turbo-only MXFP8 on every Turbo dense and grouped linear. Requires `fp8` / `fp4` unset, and `turbo_fused_gemm` / `turbo_fused_grouped_gemm` off. `PRIMUS_TURBO_PRECISION=bf16` turns it off for a BF16 twin with the same modules. |
| env `PRIMUS_TURBO_DENSE_RECIPE` / `PRIMUS_TURBO_GROUPED_RECIPE` | `inherit` (default), `bf16`, `mxfp8` | Overrides one kind, e.g. MXFP8 experts with BF16 dense linears. |
| env `PRIMUS_TE_MXFP8_FALLBACK` | `bf16` (default), `tensorwise` | Safety net only. If someone sets `fp8_recipe: mxfp8` on gfx1250, TE gets no MXFP8 recipe: TE GEMMs run BF16 (or FP8 current scaling with `tensorwise`). One rank-0 `[fp8]` warning. The Turbo GEMMs keep MXFP8. |

Every linear in these three models is a Turbo module, so there is nothing left for TE to run in FP8.
`use_turbo_gemm` makes QKV (fused with its input norm) and fc1 (fused with its norm)
`PrimusTurboLayerNormColumnParallelLinear`, and O / fc2 `PrimusTurboRowParallelLinear`.
`use_turbo_grouped_gemm` makes the experts `PrimusTurbo{Column,Row}ParallelGroupedLinear`.
The output layer is Megatron's `ColumnParallelLinear` and runs BF16 (7% of Llama-3.1-8B's linear FLOPs).

## Which Primus-Turbo

The run needs a Primus-Turbo with a correct dense MXFP8 GEMM for gfx1250, built in place for gfx1250:

- **AMD-AGI/Primus-Turbo branch `fix/gemm/mxfp8-gfx1250-flydsl`**: `dev/mi455x_v6` (8e760cbe) plus
  - the gfx1250 FlyDSL GEMM;
  - the dense MXFP8 backend on it;
  - hipBLASLt MXFP8 refused on gfx1250. Dense MX cases the FlyDSL kernel cannot take raise
    "No compatible backend" instead of falling back.

**The Turbo shipped in the images is not safe for MXFP8 on gfx1250.** That is 0.6.1.dev31 (126a485b)
in `amdprimus/amdprimus:tas-20261109` and `tas-20261109-fix1`.
Upstream `dev/mi455x_v6` is not safe either. In both:

- dense MXFP8 on gfx1250 runs hipBLASLt `MX_BLOCKWISE`;
- that GEMM gives about 5 dB SNR against BF16 (relL2 ~0.52), for fwd, dgrad and wgrad;
- it raises no error.

Build the checkout in place, inside a container of the image. This is CPU only and takes about 3 minutes:

```bash
cd /path/to/Primus-Turbo && git submodule update --init --recursive --depth 1
GPU_ARCHS=gfx1250 PYTORCH_ROCM_ARCH=gfx1250 MAX_JOBS=64 pip3 install --no-build-isolation --no-deps -e .
```

Then pass the checkout as `TURBO_DIR`, or put it first on `PYTHONPATH`.

## 1. Pre-flight check (1 GPU, under 1 min)

Run it inside the training container, with the same `PYTHONPATH` the training will use:

```bash
PYTHONPATH=${TURBO_DIR}:${PWD}:${PWD}/third_party/Megatron-LM HIP_VISIBLE_DEVICES=0 \
    python examples/megatron/configs/MI455X/mxfp8_preflight.py            # Llama dense + Qwen3 experts
PYTHONPATH=... python examples/megatron/configs/MI455X/mxfp8_preflight.py --gpt-oss  # + gpt-oss K=2880 shapes
```

The script does four things:

1. It prints the GPU arch, `primus_turbo.__file__`, the Turbo version and commit, and whether the
   gfx1250 MX backend is present.
2. It runs the ops the yamls select, forward and backward:
   - `gemm_fp8` on an 8192x4096x14336 linear (and with `--gpt-oss`, 4096x2880 -> 5120);
   - `grouped_gemm_fp8` on 16 Qwen3 experts (2048 tokens each, 2048 -> 1536), plus a correctness-only
     case with unaligned per-expert counts (and with `--gpt-oss`, 32 experts 2880 -> 5760).
   Both use the MXFP8 quant config of `turbo_precision: mxfp8`: E4M3, `MX_BLOCKWISE`, block 32, E8M0.
3. It compares fwd, dgrad and wgrad against BF16 on identical inputs (SNR, relL2) and reports
   TFLOP/s for MXFP8 and BF16.
4. It names the Turbo backend that actually ran.

Exit code: 0 = PASS, 1 = FAIL, 2 = could not run. Every check needs SNR >= 20 dB and finite values.

Measured on MI455X, 2026-11-24, `--gpt-oss`, image `amdprimus/amdprimus:tas-20261109`:

| Primus-Turbo | dense backend | dense SNR fwd / dgrad / wgrad | grouped SNR | verdict |
|---|---|---|---|---|
| `fix/gemm/mxfp8-gfx1250-flydsl` (69955b8) | `GEMMFP8FlyDSLBackend` | 28.3 / 28.3 / 28.1 dB (Llama and gpt-oss) | 28.1-28.3 dB (Triton MX) | **PASS** (15 checks, worst 28.1 dB) |
| `dev/mi455x_v6` (8e760cbe), unpatched | `GEMMFP8HipBLASLtBackend` | 5.7 / 5.6 / 4.6 dB (Llama), 5.3 / 5.8 / 4.6 dB (gpt-oss) | 28.1-28.3 dB (Triton MX) | **FAIL** (6/15, `bad-fallback signature`) |
| image 0.6.1.dev31 (126a485b), 2026-11-23 | `GEMMFP8HipBLASLtBackend` | 5.7 / 5.6 / 4.6 dB (Llama), 5.3 / 5.8 / 4.6 dB (gpt-oss) | 28.1-28.3 dB (Triton MX) | **FAIL** (6/15) |

Do not start an MXFP8 training run unless the preflight prints PASS.

## 2. Training

Use the standard Primus CLI, inside a container of the image, from the Primus root. The
`--env PYTHONPATH` is applied after Primus builds its own `PYTHONPATH`, so the Turbo checkout wins.

```bash
# Llama-3.1-8B (the MI455X launcher; it mounts TURBO_DIR and puts it first)
EXP=examples/megatron/configs/MI455X/llama3.1_8B-MXFP8-pretrain.yaml TURBO_DIR=/path/to/Primus-Turbo \
    PRIMUS_LAUNCHER=direct bash examples/megatron/models/llama3.1-8b/run_llama3.1_8b_mi455x.sh

# GPT-OSS-120B proxy (4 GPUs, EP4) and Qwen3-30B-A3B (EP8; PRIMUS_EP=4 on 4 GPUs)
for EXP in gpt_oss_120B-MXFP8-pretrain qwen3_30B_A3B-MXFP8-pretrain; do
  ./primus-cli direct \
      --env "PYTHONPATH=${TURBO_DIR}:${TURBO_DIR}/build/pydeps:${PWD}:${PWD}/third_party/Megatron-LM" \
      --env GPUS_PER_NODE=4 --env TRITON_HIP_USE_EXPERT_SCHEDULING=0 \
      -- train pretrain --config examples/megatron/configs/MI455X/${EXP}.yaml
done
```

`examples/megatron/models/gpt-oss/run_gpt_oss_mi455x.sh` hard-codes the BF16 yaml and has no
`TURBO_DIR`, so use `primus-cli` directly for GPT-OSS.

Check the first iterations of the log:

- `[precision] turbo_precision=mxfp8 (Transformer Engine FP8 off): Turbo dense linears=mxfp8, Turbo
  grouped linears=mxfp8; primus_turbo imported from <your checkout>`.
- `[Patch:megatron.turbo.turbo_precision_first_microbatch]`: the patch that re-quantizes the Turbo
  weight cache every step (needed when a step has more than one microbatch).
- No `[fp8]` warning, because TE is not asked for FP8.
- `lm loss` at iteration 1 is within ~1e-2 of the BF16 config. A non-learning loss means the wrong
  Turbo was imported.
- To see which backend each GEMM shape ran, set `PRIMUS_TURBO_DISPATCH_REPORT=/tmp/dispatch_{rank}.json`.
  The log then prints `[dispatch] GEMMFP8KernelDispatcher (...) -> FLYDSL` per shape. It must never
  show `HIPBLASLT` for an MX GEMM.

## 3. Smoke runs (1x MI455X, 20 iterations, mock data, seed 1234)

These configs with Primus-Turbo `fix/gemm/mxfp8-gfx1250-flydsl`, image `amdprimus/amdprimus:tas-20261109`, MI455X,
2026-11-23. Each MXFP8 run has a twin that differs only in precision: `PRIMUS_TURBO_PRECISION=bf16`
keeps the same Turbo modules on BF16 GEMMs. Llama also has the plain BF16 yaml (TE linears).
Step time is the median of iterations 11-20.

| model, reduced for 1 GPU | run | loss@1 | loss@20 | mean loss, it 11-20 | max \|loss diff\| vs Turbo BF16 | step ms |
|---|---|---|---|---|---|---|
| Llama-3.1-8B, 32 layers, seq 4096, MBS 1, GBS 2 | BF16 yaml (TE linears) | 11.89094 | 10.65697 | 10.9695 | 0.009 | 313.7 |
| | Turbo BF16 | 11.89090 | 10.65664 | 10.9681 | - | 317.8 |
| | **Turbo MXFP8** | 11.89295 | 10.79598 | 11.0007 | 0.178 (it 1: 2.0e-3) | 326.9 |
| Qwen3-30B-A3B, 8 layers, 128 experts top-8, EP1, MBS 4, GBS 8 | Turbo BF16 | 11.99380 | 11.55876 | 11.5555 | - | 393.7 |
| | **Turbo MXFP8** | 11.99338 | 11.55820 | 11.5551 | 0.002 | 411.9 |
| GPT-OSS-120B proxy, 4 layers, 32 experts top-4, EP1, MBS 1, GBS 2, Triton attention | Turbo BF16 | 12.30648 | 11.79777 | 11.7698 | - | 134.5 |
| | **Turbo MXFP8** | 12.30659 | 11.79826 | 11.7703 | 0.002 | 148.1 |

- No run asserted. No `[fp8]` warning appeared, because TE was not asked for FP8.
- Every transformer-layer linear ran Turbo MXFP8, and no linear GEMM ran on TE:
  - Llama: QKV and fc1 ran on `PrimusTurboLayerNormColumnParallelLinear`; O and fc2 on
    `PrimusTurboRowParallelLinear`. That is 93% of the linear FLOPs.
  - MoE models: the experts ran on `PrimusTurbo{Column,Row}ParallelGroupedLinear`.
  - Dense GEMMs dispatched to `FLYDSL`; grouped GEMMs to `TRITON`.
- These stayed BF16:
  - the output layer (Megatron `ColumnParallelLinear`). At full depth that is 7% of the linear
    FLOPs for Llama-3.1-8B, ~10% for Qwen3-30B-A3B and ~11% for GPT-OSS-120B.
  - the router: below 0.5%.
- The Llama run (gradient norms 30-500 on mock data) drifts from BF16 more than the MoE runs do. Its
  BF16-vs-BF16 twins agree to 0.009, but MXFP8 is 0.03 higher on average over iterations 11-20.
  Check convergence on real data before a long run.
- MXFP8 is 3-10% slower per step than BF16 in all three. The quantization kernels cost more than
  the GEMM saves at these sizes. The Triton MX grouped GEMM is slower than BF16, and so are the
  K=2880 / N=2880 GPT-OSS dense shapes. Dense MXFP8 vs BF16 fwd+bwd, including quantization:
  - Llama fc1 / fc2 at M=4096: 1.14x / 1.28x; at M=8192: 1.33x / 1.31x;
  - QKV / O: 0.98x / 0.93x;
  - GPT-OSS QKV / O at M=4096: 0.57x / 0.59x.
- A plain `fp8: e4m3`, `fp8_recipe: mxfp8` run (Llama, 4 layers, 6 iterations, no `turbo_precision`)
  also runs. TE gets BF16 with one `[fp8]` warning, and the Turbo linears run MXFP8.

## Known pitfalls

1. **Silent hipBLASLt fallback.** With the image Turbo (0.6.1.dev31) or upstream `dev/mi455x_v6`,
   MXFP8 "works" on gfx1250 at ~5 dB SNR, with no error. In a DeepSeek-V4 proxy run an MX `lm_head` did not
   learn because of it (rel err 0.37).
2. **The wrong Turbo gets imported.** On `dev/mi455x_v6`, `runner/helpers/envs/base_env.sh` puts the
   image site-packages ahead of the caller's `PYTHONPATH`, so a mounted Turbo is shadowed. Pass it
   with `primus-cli direct --env PYTHONPATH=...` (or `TURBO_DIR` with the Llama launcher). Confirm
   with the preflight and the `[precision]` log line that `primus_turbo` comes from your checkout.
3. **Do not combine `turbo_precision` with `fp8` / `fp8_recipe`.** The run refuses it. TE 2.19 cannot
   run MXFP8 on gfx1250 even with `NVTE_ROCM_ENABLE_MXFP8=1`. A plain `fp8_recipe: mxfp8` run no
   longer asserts: TE runs BF16 and the Turbo modules run MXFP8. But that path relies on TE's FP8
   autocast state, so prefer `turbo_precision`.
4. **The fused Turbo MLP ops are not covered.** `turbo_fused_grouped_gemm` (MXFP8 fused-GLU,
   gfx950-only anyway) and `turbo_fused_gemm` (MXFP4) bypass the per-module switch. `turbo_precision`
   refuses them. Keep `turbo_fp8_permute` off too: it depends on the fused grouped MLP.
5. **MX grouped GEMM on gfx1250 is the generic Triton `MX_BLOCKWISE` kernel.** It is not a tuned
   FlyDSL kernel, so the expert GEMMs are slower than BF16 (preflight: 0.87-0.92x fwd+bwd).
6. **GPT-OSS `hidden_size` 2880 is not a multiple of 128.** The gfx1250 FlyDSL GEMM zero-pads K and N
   to its tile grid. These shapes are slower than BF16 (0.57x at 4096x2880 -> 5120).
