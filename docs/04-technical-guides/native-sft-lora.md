# Primus native SFT LoRA—quick start

> **Branch**: `feat/megatron/support-sft-native` (PR701)
> **Backend**: Megatron-LM **native** (no Megatron-Bridge runtime dependency)
> **Hardware**: AMD MI355X / MI300X
> **Models verified**: Llama2-70B, Llama3-8B, Llama3-70B, Qwen3-30B-A3B, Qwen3-235B-A22B, DeepSeek-V2-Lite

This README walks through how to launch training on Primus's **native SFT LoRA** path, and explains exactly which fields to change when switching from BF16 / FP8 to FP4 (NVFP4 / MXFP4).

---

## 1. Overview

PR701 adds a **complete, Megatron-Bridge-free SFT training stack** to Primus, sitting alongside the existing pretrain path. Core layout:

```
primus/backends/megatron/
├── sft/                              # SFT data + forward + packing
│   ├── dataset.py                    # Multi-source datasets (HF / local JSONL / OpenAI messages)
│   ├── forward_step.py               # SFT-specific forward (per-token loss masking)
│   ├── formatters.py                 # squad / alpaca / openai prompt templates
│   ├── gpt_sft_chat_dataset.py       # Multi-turn conversation dataset
│   ├── packing.py                    # Sequence packing (FFD + cu_seqlens)
│   ├── mlperf_packed_dataset.py      # Direct loader for mlperf .npy artifacts
│   ├── preprocessing.py              # Tokenization + on-disk cache
│   ├── runtime.py                    # SFT runtime helpers
│   └── schema.py                     # Data schema validation
└── peft/                             # PEFT (LoRA / Recompute)
    ├── lora.py                       # LoRA main entry point
    ├── lora_layers.py                # TE-compatible LoRA layers
    ├── adapter_wrapper.py            # adapter wrap utilities
    ├── module_matcher.py             # target_modules matcher
    ├── recompute.py                  # Input-grad recompute (OOM rescue)
    └── walk_utils.py                 # module-tree walker
```

Entry point: `primus/backends/megatron/megatron_sft_trainer.py` (`MegatronSFTTrainer`) + the stage-based `BackendRegistry`.

---

## 2. Runtime environment

### 2.1 Docker container

| Container | Image | Notes |
|---|---|---|
| **`sft_primus_0507_native`** | `rocm/primus:v26.3` | Recommended; verified |

Container mounts (set once at container start):

| Inside container | Host path | Purpose |
|---|---|---|
| `/workspace/Primus` | `/home/botahu/sft_primus_0507/Primus` | Source code |
| `/data/mlperf_llama2` | `/data/mlperf_llama2` (NVMe) | mlperf data + HF cache |
| `/workspace` | container overlay-fs | Large model weights + cache_persist |

### 2.2 Required environment variables

```bash
# Recommended: export from the outer host shell
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"           # required for gated models
export EXP_NAME="llama2_70b_native_$(date +%Y%m%d_%H%M%S)"
```

Set automatically inside the container by `examples/run_pretrain.sh` (you don't need to touch these):
- `TRITON_CACHE_DIR`, `MIOPEN_USER_DB_PATH`, `PRIMUS_CACHE_ROOT`—persistent JIT cache
- `NCCL_*` / `RCCL_*`—communication tuning
- `HSA_*` / `GPU_MAX_HW_QUEUES`—AMD GPU performance tuning

---

## 3. Launch commands (verified)

### 3.1 BF16 / FP8 (existing yaml configs, ready to run)

```bash
docker exec \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_HOME=/data/mlperf_llama2/.cache/huggingface \
  -e PRIMUS_EXP_NAME="$EXP_NAME" \
  -e NCCL_SOCKET_IFNAME=lo \
  -e GLOO_SOCKET_IFNAME=lo \
  -e NCCL_IB_DISABLE=1 \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  -e EXP=examples/megatron/configs/MI355X/llama2_70B-BF16-sft-packed-mlperf_aligned.yaml \
  sft_primus_0507_native \
  bash -c 'cd /workspace/Primus && bash examples/run_pretrain.sh' \
  2>&1 | tee /home/botahu/llama2_70b_500iter_runs/${EXP_NAME}.log
```

Replace the `EXP=` path with any yaml from the table below.

### 3.2 Available yaml configurations (verified)

| Model | Precision | Data | yaml file |
|---|---|---|---|
| Llama2-70B | BF16 | mlperf packed (8192 seq) | `llama2_70B-BF16-sft-packed-mlperf_aligned.yaml` |
| Llama2-70B | BF16 | SQuAD packed (Bridge-aligned) | `llama2_70B-BF16-sft-packed-bridge_aligned.yaml` |
| Llama2-70B | **FP8 hybrid** | SQuAD packed | `llama2_70B-FP8-sft-packed-perf.yaml` |
| Llama3-70B | BF16 | SQuAD packed (Bridge-aligned) | `llama3_70B-BF16-sft-packed-bridge_aligned.yaml` |
| Llama3-8B | BF16 | Alpaca packed | `llama3_8B-BF16-sft-packed.yaml` |
| Llama3-8B | BF16 | SQuAD packed (Bridge-aligned) | `llama3_8B-BF16-sft-packed-bridge_aligned.yaml` |
| Llama3-8B | BF16 | Alpaca + LoRA only | `llama3_8B-BF16-lora-sft.yaml` |
| Llama3-8B | BF16 | OpenAI messages multi-turn | `llama3_8B-BF16-multiturn-sft.yaml` |
| DeepSeek-V2-Lite | BF16 | Alpaca packed | `deepseek_v2_lite-BF16-sft-packed.yaml` |
| Qwen3-30B-A3B (MoE) | BF16 | Alpaca packed | `qwen3_30B_A3B-BF16-sft-packed.yaml` |
| Qwen3-235B-A22B (MoE) | BF16 | Alpaca | `qwen3_235B_A22B-BF16-sft.yaml` |

### 3.3 Log locations after launch

| Type | Path |
|---|---|
| Main stdout | `/home/botahu/llama2_70b_500iter_runs/${EXP_NAME}.log` |
| In-container training logs | `/workspace/Primus/output/amd/root/${EXP_NAME}/logs/post_trainer/rank-{0..7}/debug.log` |
| Checkpoint | `/workspace/Primus/output/amd/root/${EXP_NAME}/checkpoints/` (16-137 GB by default; see yaml `disable_last_saving`) |

---

## 4. Switching to FP4 SFT LoRA (NVFP4 / MXFP4)

### 4.1 FP4 support status in Megatron-LM (the facts first)

The plumbing is already in place:

| Location | Key flag |
|---|---|
| `third_party/Megatron-LM/megatron/training/arguments.py:851` | `--fp4` / `--fp4-format` / `--fp4-param` / `--fp4-recipe` |
| `third_party/Megatron-LM/megatron/core/transformer/transformer_config.py:585` | `fp4_recipe: Optional[Literal['nvfp4', 'custom']] = "nvfp4"` |
| `primus/backends/megatron/core/extensions/primus_turbo.py:165` | `mxfp4_scaling()` detection (MX_BLOCKWISE + DYNAMIC + E2M1_X2 + E8M0) |

Hard constraints:

1. **TransformerEngine ≥ 2.7.0.dev0** required (the `rocm/primus:v26.3` image already satisfies this)
2. **FP4 and FP8 are mutually exclusive**: `args.fp4 and args.fp8` raises in Megatron (`arguments.py:885-887`)
3. **`fp4_param` must be paired with `fp4`**: enabling `fp4_param` alone raises (`arguments.py:889-891`)

### 4.2 Deriving an FP4 yaml from an FP8 yaml (minimum diff)

Copy `llama2_70B-FP8-sft-packed-perf.yaml` as the starting point, change exactly **5 yaml fields**:

```diff
  modules:
    pre_trainer:
      framework: megatron
      config: sft_trainer.yaml
      model: llama2_70B.yaml

      overrides:
        ...
        # ============================================================
        # Precision: FP8 hybrid → FP4 (NVFP4)
        # ============================================================
        bf16: true                            # primary precision stays BF16 (master weights / LoRA adapter)

-       # FP8 hybrid (E4M3 fwd / E5M2 bwd)
-       fp8: hybrid
-       no_fp8_weight_transpose_cache: true

+       # FP4 NVFP4 recipe (E2M1 + E8M0 block scale)
+       fp4: e2m1                             # FP4 main switch
+       fp4_format: e2m1                      # quantization format (E2M1 elements + E8M0 scale)
+       fp4_recipe: nvfp4                     # NVFP4 recipe (default; the other option is 'custom')
+       fp4_param: false                      # FP4 optimizer master weights (keep off; verify training first)
+       fp4_param_gather: false               # FP4 param gather (depends on fp4_param)

        # primus_turbo auto-detects the FP4 quant config (mxfp4_scaling path)
        enable_primus_turbo: true
        ...

        # LoRA target modules (FP4 does not affect LoRA configuration)
        lora:
          enabled: true
          dim: 16
          alpha: 32
          target_modules:
            - linear_qkv
            - linear_proj
            - linear_fc1
            - linear_fc2
```

**Field mapping**:

| yaml field | Megatron CLI flag | FP4 value | Purpose |
|---|---|---|---|
| `fp4` | `--fp4` | `e2m1` | Main switch; any non-empty string enables FP4 |
| `fp4_format` | `--fp4-format` | `e2m1` | Element format |
| `fp4_recipe` | `--fp4-recipe` | `nvfp4` | Recipe selection (NVFP4 = 4-bit elements + E8M0 block scale) |
| `fp4_param` | `--fp4-param` | `false` | Whether to also FP4-quantize optimizer master weights (advanced) |
| `fp4_param_gather` | `--fp4-param-gather` | `false` | Whether to compress to FP4 during all-gather |
| `fp8` / `fp8_*` | (mutually exclusive) | **must be removed/commented out** | Otherwise startup raises |

### 4.3 Complete FP4 yaml template (drop-in usable)

Create `examples/megatron/configs/MI355X/llama2_70B-FP4-sft-packed-perf.yaml`:

```yaml
work_group: ${PRIMUS_TEAM:amd}
user_name: ${PRIMUS_USER:root}
# =============================================================================
# Llama-2 70B LoRA SFT — Native FP4 (NVFP4) performance variant
# Derived from llama2_70B-FP8-sft-packed-perf.yaml; only precision flags differ.
#
# Recommended invocation:
#   export PRIMUS_EXP_NAME=native_llama2_70b_fp4_perf_$(date +%Y%m%d_%H%M%S)
#   EXP=examples/megatron/configs/MI355X/llama2_70B-FP4-sft-packed-perf.yaml \
#       bash examples/run_pretrain.sh
# =============================================================================

exp_name: ${PRIMUS_EXP_NAME:llama2_70B-FP4-sft-packed-perf}
workspace: ${PRIMUS_WORKSPACE:./output}

modules:
  pre_trainer:
    framework: megatron
    config: sft_trainer.yaml
    model: llama2_70B.yaml

    overrides:
      data_path: null
      sft_dataset_name: rajpurkar/squad
      sft_dataset_formatter: squad
      sft_bridge_compat_inline_bos: false

      num_workers: 1
      dataloader_type: cyclic

      stderr_sink_level: DEBUG
      log_avg_skip_iterations: 2
      log_avg_reset_interval: 10

      # ---------- Training schedule ----------
      train_iters: ${PRIMUS_TRAIN_ITERS:200}
      micro_batch_size: ${PRIMUS_MBS:1}
      global_batch_size: ${PRIMUS_GBS:16}
      seq_length: ${PRIMUS_SEQ_LENGTH:4096}
      max_position_embeddings: ${PRIMUS_MAX_POSITION_EMBEDDINGS:4096}

      lr: ${PRIMUS_LR:1.0e-4}
      min_lr: 0.0
      lr_warmup_iters: 10
      lr_decay_iters: 200
      lr_decay_style: cosine
      weight_decay: 0.1
      adam_beta1: 0.9
      adam_beta2: 0.98
      adam_eps: 1.0e-8

      seed: 5678
      eod_mask_loss: false
      init_method_std: 0.008
      norm_epsilon: 1.0e-6

      # ---------- Parallelism ----------
      tensor_model_parallel_size: ${PRIMUS_TP:2}
      pipeline_model_parallel_size: ${PRIMUS_PP:1}
      expert_model_parallel_size: 1
      context_parallel_size: 1
      sequence_parallel: false

      # ---------- DDP / optimizer ----------
      use_distributed_optimizer: true
      overlap_grad_reduce: true
      overlap_param_gather: true
      gradient_accumulation_fusion: true
      use_precision_aware_optimizer: false
      apply_rope_fusion: true
      masked_softmax_fusion: true
      attention_softmax_in_fp32: false
      align_param_gather: false

      # ---------- Activation recompute ----------
      recompute_granularity: full
      recompute_method: block
      recompute_num_layers: 8

      # ---------- Checkpoint ----------
      pretrained_checkpoint: /workspace/megatron_checkpoints/Llama-2-70b-hf
      finetune: true
      load: null
      save: null
      save_interval: 1000000
      eval_interval: 1000000
      no_save_optim: null
      no_save_rng: null
      disable_last_saving: true
      ckpt_format: torch_dist
      dist_ckpt_strictness: log_all

      # =====================================================================
      # FP4: ENABLED (NVFP4 recipe)
      # ---------------------------------------------------------------------
      # NVFP4 = 4-bit elements (E2M1) with per-block FP8 scale (E8M0). At
      # 70B, FP4 cuts effective weight memory ~2x vs FP8 hybrid; on AMD MI355X
      # the matmul also runs at FP4 throughput for the linear / fc1 / fc2 /
      # qkv layers. bf16 stays as the master / LoRA-adapter precision so the
      # PEFT path is unaffected.
      #
      # NOTE: fp8 must NOT be set; Megatron-LM raises if both are on.
      # =====================================================================
      bf16: true
      fp4: e2m1                              # main switch
      fp4_format: e2m1                       # element format
      fp4_recipe: nvfp4                      # 'nvfp4' (default) | 'custom'
      fp4_param: false                       # keep master weights in BF16 first
      fp4_param_gather: false                # depends on fp4_param

      # =====================================================================
      # PRIMUS-TURBO: enabled (mxfp4_scaling auto-detected via quant_config)
      # =====================================================================
      enable_primus_turbo: true
      use_turbo_attention: false
      use_turbo_grouped_gemm: false
      use_turbo_rms_norm: false

      # ---------- Cross-entropy fusion ----
      cross_entropy_fusion_impl: "te"
      cross_entropy_loss_fusion: true

      # ---------- Manual GC ----
      manual_gc: true
      manual_gc_interval: 100

      # ---------- Profiling: OFF for perf measurement ----
      profile: false
      use_pytorch_profiler: false
      use_nsys_profiler: false
      record_shapes: false
      record_memory_history: false
      nvtx_ranges: false

      eval_iters: 0

      lora:
        enabled: true
        dim: 16
        alpha: 32
        dropout: 0.0
        dropout_position: pre
        lora_A_init_method: xavier
        lora_B_init_method: zero
        target_modules:
          - linear_qkv
          - linear_proj
          - linear_fc1
          - linear_fc2
```

### 4.4 Launching the FP4 experiment

Just swap the yaml path in the launch command:

```bash
export EXP_NAME="llama2_70b_native_fp4_$(date +%Y%m%d_%H%M%S)"

docker exec \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_HOME=/data/mlperf_llama2/.cache/huggingface \
  -e PRIMUS_EXP_NAME="$EXP_NAME" \
  -e NCCL_SOCKET_IFNAME=lo \
  -e GLOO_SOCKET_IFNAME=lo \
  -e NCCL_IB_DISABLE=1 \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  -e EXP=examples/megatron/configs/MI355X/llama2_70B-FP4-sft-packed-perf.yaml \
  sft_primus_0507_native \
  bash -c 'cd /workspace/Primus && bash examples/run_pretrain.sh' \
  2>&1 | tee /home/botahu/llama2_70b_500iter_runs/${EXP_NAME}.log
```

### 4.5 Verifying that FP4 is actually engaged

After the experiment starts, grep the rank-0 debug.log for these 5 pieces of evidence:

```bash
RANK0=/home/botahu/sft_primus_0507/Primus/output/amd/root/${EXP_NAME}/logs/post_trainer/rank-0/debug.log

# 1. yaml-parsed args dump
grep -E "fp4\s+:" "$RANK0" | head -3
# Expected: fp4: e2m1 (str)

# 2. fp8 should be None / False
grep -E "fp8\s+:" "$RANK0" | head -3
# Expected: fp8: None (NoneType)

# 3. TE internal dispatch
grep -E "Float4Tensor|NVFP4|fp4_recipe|MX_BLOCKWISE.*E2M1" "$RANK0" | head -5
# Expected: NVFP4 / Float4Tensor strings appear

# 4. Memory drop vs FP8 hybrid
grep -E "rocm mem usage" "$RANK0" | head -3
# Expected: 70B FP4 + LoRA at TP=2 ~50-70 GB / GPU
#           (FP8 same setup ~80-100 GB)

# 5. TFLOPS uplift
grep -E "throughput per GPU" "$RANK0" | head -5
# Expected: FP4 on MI355X is ~30-50% faster than FP8
#           (theoretical FP4 throughput is 2x FP8)
```

---

## 5. Troubleshooting

### Q1: `--fp4-format requires Transformer Engine >= 2.7.0.dev0`

Upgrade TE inside the container, or switch to image `rocm/primus:v26.3`+.

### Q2: `--fp4-format and --fp8-format cannot be used simultaneously`

Leftover `fp8: hybrid` / `fp8: e4m3` in the yaml—must be removed.

### Q3: `--fp4-param-gather must be used together with --fp4-format`

Either `fp4_param_gather: true` is set while `fp4` is empty, or `fp4_param: true` is set while `fp4` is empty. Both must be enabled together.

### Q4: NFS quota full / `OSError: [Errno 122] Disk quota exceeded`

A 70B LoRA last-iteration checkpoint is ~16 GB; running multiple `EXP_NAME`s for a long time will fill NFS (`/home/botahu` is typically ~100 GB).
- Quick fix: set `disable_last_saving: true` in the yaml
- Permanent fix: point `workspace` to the container overlay-fs (the default `output/` already lives inside the container)

### Q5: Loss is huge for the first 2 iters, stabilizes from iter 3

Expected. `log_avg_skip_iterations: 2` is already configured to skip warmup; TFLOPS reporting starts from iter 3.

### Q6: How do I compare FP4 vs FP8 vs BF16 accuracy on the same data?

Pin `seed`, `train_iters`, `global_batch_size`, `lr` identical across all three yamls, then diff per-iter `lm loss`:

```bash
for prec in BF16 FP8 FP4; do
  log=/home/botahu/sft_primus_0507/Primus/output/amd/root/llama2_70b_${prec,,}_*/logs/post_trainer/rank-0/debug.log
  echo "=== $prec ==="
  grep "lm loss" $log | tail -10
done
```

---

## 6. References / further reading

- **Post-training overview**: [Post-Training (SFT / LoRA / DPO)](../02-user-guide/posttraining.md)—how this native SFT LoRA path fits into the broader fine-tuning workflow.
- **PR #701**—Full implementation of this native SFT stack:
  https://github.com/AMD-AGI/Primus/pull/701
- **Megatron-LM FP4 design**:
  `third_party/Megatron-LM/megatron/core/fp4_utils.py` +
  `third_party/Megatron-LM/megatron/core/transformer/transformer_config.py:585`
- **Primus Turbo MXFP4 path**:
  `primus/backends/megatron/core/extensions/primus_turbo.py:160-180`
- **NVFP4 / MXFP4 format**:
  - NVFP4: NVIDIA Blackwell 4-bit elements (E2M1) + FP8 block scale (E8M0)
  - MXFP4: OCP MX standard (same as NVFP4; AMD ships this format too)

---

## 7. Maintainers

- @wenxie-amd—PR #701 main author
- @Xiaoming-AMD—co-author (trainer + dataset core)
- @botaohu001—packing / mlperf-aligned recipe / diagnostic tools
