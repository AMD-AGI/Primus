# Native HyLo training — change notes

This note records the Primus implementation of the public HyLo 1B procedure.
The runbook is [hylo-native-training.md](hylo-native-training.md). The reference
is [AMD-Hybrid-Models `feat/HyLo`](https://github.com/AMD-AGI/AMD-Hybrid-Models/tree/feat/HyLo),
especially `train_hybrid/train_distill.py` and
`configs/llama3.2_1B/zebra_4MLA12GDN_8bT_SFT_8k_combined.yaml`.

No Megatron-LM submodule files were edited, and the reference TRL trainer is
not vendored. Distillation is a Primus forward step behind
`hylo_distill_stage`.

## Files

| File | Why it is here |
| --- | --- |
| `primus/backends/megatron/training/hylo_distillation.py` | Frozen dense teacher, Enhanced-ILD, and masked chunked `KL(teacher \|\| student)`. |
| `primus/backends/megatron/megatron_sft_trainer.py` | Enables that forward step for both training stages. |
| `primus/backends/megatron/megatron_pretrain_trainer.py` | Same hook, used by the zero-learning-rate constructor. |
| `primus/backends/megatron/argument_builder.py` | Keeps the `hylo_*` recipe fields through Megatron argument merge. |
| `primus/backends/megatron/checkpoint/hybrid_upcycle.py` | From-teacher MLA and GDN init, plus stage-1 checkpoint assembly. |
| `primus/backends/megatron/checkpoint/native_loader_llama.py` | Preserves Llama-3 `rope_scaling` when converting the teachers. |
| `primus/backends/megatron/core/models/hybrid/hybrid_mamba_mla_layer_specs.py` | `hylo_gdn_hybrid_stack_spec`: RMSNorm, with MLA LoRA latent norms off. |
| `primus/configs/models/megatron/hylo_gdn_1B_hybrid.yaml` | 1B 4MLA12GDN widths. MLA blocks are 1, 5, 10, and 14. |
| `examples/megatron/configs/MI300X/hylo_1B-constructor.yaml` | One mock step that saves a hybrid init checkpoint. |
| `examples/megatron/configs/MI300X/hylo_gdn_1B-enhanced-ild.yaml` | All-GDN Enhanced-ILD. |
| `examples/megatron/configs/MI300X/hylo_mla_1B-enhanced-ild.yaml` | All-MLA Enhanced-ILD. |
| `examples/megatron/configs/MI300X/hylo_4MLA12GDN_1B-8K-sft.yaml` | 8K pure-KL stage 2. |
| `tools/hybrid/prepare_hylo_assets.py` | Downloads and converts the gated Llama teachers and exports the public SFT set. |
| `tools/hybrid/assemble_hylo_ild.py` | Copies trained GDN and MLA blocks into the target layout. |
| `tools/hybrid/run_hylo_1b_4mla12gdn.sh` | Constructor, upcycle, both ILD stages, assembly, then stage 2. |
| `tests/unit_tests/backends/megatron/test_hylo_distillation.py` | Loss, mask, and argument checks. |
| `tests/unit_tests/backends/megatron/test_hybrid_upcycle.py` | Upcycle and assembly rules. |
| `tests/unit_tests/backends/megatron/test_hybrid_upcycle_hylo_reference.py` | Exact match against HyLo `re_init_q`, `re_init_kv`, and `_copy_llama_attn_to_gdn`. |
| `tests/unit_tests/backends/megatron/test_native_loader_llama_rope.py` | Llama-3 rope-scaling conversion. |
| `tests/fixtures/hylo_smoke.jsonl` | One chat example for a one-step container run. |
| `docs/04-technical-guides/hybrid-models/upcycling-changes.md` | Corrects the MLA RoPE copy: those rows are a full replace. |

## What the runtime does

Enhanced-ILD sums per-layer hidden L2 and token-mixer-output L2. During that
stage each student mixer is forced to the matching teacher layer input. Stage 2
uses pure KL with `hylo_kl_weight: 1.0` and `hylo_ce_weight: 0.0`. The KL is
chunked at 4096 tokens and masked.

The teacher is a frozen Megatron GPT built from the converted checkpoint. Each
data-parallel rank owns its own copy. Tensor and pipeline parallelism must be 1.
The 1B recipes assume eight-way data parallel: ILD global batch 96 is eight
times micro-batch 12, and stage-2 global batch 32 is eight times micro-batch 4.

`hylo_gdn_hybrid_stack_spec` is separate from the Zebra GDN spec because the
released model sets `use_lora_layer_norm: false`.

## Recipe match

These values follow `configs/llama3.2_1B/`:

| | Enhanced-ILD | 8K stage 2 |
| --- | --- | --- |
| Context | 4096 | 8192 |
| Learning rate | 2e-4 | 6e-5 |
| Warmup | 0.01 | 0.01 |
| Schedule | cosine | cosine |
| Micro-batch | 12 | 4 |
| Teacher | Llama-3.2-1B-Instruct | Llama-3.1-8B-Instruct |
| Objective | Enhanced-ILD | KL 1, CE 0 |
| YaRN | MLA ILD factor 32 over an 8192 window | factor 4 over a 2048 window |
| Save interval | 2000 | 1000 |

The reference stage-1 configs also set `data_ratio: 0.2` and
`num_train_epochs: 1`. The Primus recipes default to the full public dataset
for `train_iters: 1000`, so they exercise the same objective without consuming
one epoch of a 20% subset. Stage 2 in the reference is also one epoch, and its
released run used four AMD-preprocessed datasets that are not in the public
repository. `sftdatasetv3` alone cannot reproduce those weights.

## What was run

On `rocm/primus:v26.8`, branch `dev/clairlee/hybrid-model-upcycling`:

- Real Llama-3.2-1B-Instruct tensors match the HyLo from-teacher init.
- Unit and config checks passed, including 519 example-config tests.
- One all-GDN constructor step, its upcycle, one Enhanced-ILD step, and one
  pure-KL stage-2 step completed on one GPU at sequence length 128.

On `useocpm2m-097-017` (the jump host for `tw044` did not accept SSH), one-step
smokes at the recipe context lengths also completed in `rocm/primus:v26.8`:

| Step | Context | Loss | Checkpoint reload |
| --- | --- | --- | --- |
| all-GDN Enhanced-ILD | 4096 | total 360.000000, hidden 168.000000, mixer 192.000000 | iteration 1, 243 tensors, all finite |
| all-MLA Enhanced-ILD | 4096 | total 169.000000, hidden 66.000000, mixer 103.000000 | iteration 1, 275 tensors, all finite |
| assembled 4MLA12GDN pure KL | 8192 | summed masked KL 72.599213 | iteration 1, 251 tensors, all finite |

Each step used micro-batch 1, global batch 1, and one optimizer step. Assembly
checked that every mixer/MLP pair came from the matching ILD checkpoint.
`run_hylo_1b_4mla12gdn.sh` was not executed as a full 1000-iteration job.
