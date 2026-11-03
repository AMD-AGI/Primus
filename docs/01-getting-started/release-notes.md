# Release notes

AMD publishes two families of Primus training Docker image:

| Image | Backends | Documented in |
| ----- | -------- | ------------- |
| `rocm/primus:<version>` | Megatron-LM, TorchTitan, Megatron Bridge | [Megatron-LM](../02-user-guide/megatron-lm-training.md), [TorchTitan](../02-user-guide/torchtitan-training.md) |
| `rocm/jax-training:maxtext-<version>` | MaxText (JAX) | [JAX MaxText](../02-user-guide/jax-maxtext-training.md) |

The two families share a version number but are **not** built in lockstep — the same `vNN.N` tag can be weeks apart between families, and the MaxText family also ships patch releases (`v26.3.1`, `v26.3.2`) with no `rocm/primus` counterpart. Always read the section for the exact tag you are running.

**This page is the single source of truth for image contents.** Other pages link here instead of repeating version tables, so there is exactly one place to update per release. If you add a page that names an image tag, link to the relevant section below rather than restating the stack.

Every version below was read out of the published image itself. From v26.4 on, the values were additionally cross-checked against the release Dockerfiles in [`.github/workflows/docker-release/`](https://github.com/AMD-AGI/Primus/tree/main/.github/workflows/docker-release); earlier releases predate those files and are image-derived only. See [Verifying the stack in an image](#verifying-the-stack-in-an-image) to reproduce any table.

---

## Highlights for v26.9

**Both families move to ROCm nightlies.** `rocm/primus` moves from the ROCm 10.1.0 release to a ROCm 10.2 nightly (`10.2.0a20261020`), with a PyTorch 2.16 pre-release and a Transformer Engine 2.20 development build, still built from source. `rocm/jax-training:maxtext` moves to a ROCm 10.3 nightly (`10.3.0a20261026`) with JAX 0.12.0, a Transformer Engine 2.19 development build, and MaxText `release/v26.9`, and **ships a copy of Primus again**. Both images were built from the same Primus commit, `db1a4160` on `release/v26.9`—see [Primus source for v26.9](#primus-source-for-v269).

### Both families

- **A convergence test for Megatron and MaxText.** `tools/convergence_test/` builds a dataset from a real corpus, lints the config for settings that silently spoil a convergence run (mock or synthetic data, forced load balancing, vocab mismatches), trains through the normal `primus-cli` path, plots the loss, and gates it against a previous release's baseline ([#1229](https://github.com/AMD-AGI/Primus/pull/1229)). It ships Llama-3.2-1B, Llama-2-7B, and Mixtral-8x7B recipes for both backends, plus a MaxText Llama-2-7B FP8 recipe for MI355X ([#1261](https://github.com/AMD-AGI/Primus/pull/1261)).
- **The `tools/perf` batch runner is ready for unattended runs.** Each run is bounded by `MAX_RUN_SECONDS` (default 1 h) and `MAX_IDLE_SECONDS` (default 20 min); busy GPUs and bad network interfaces are refused up front; failures are classified with the log line that shows them; every result records the image, host, and platform it came from; diffusion and DLRM benchmarks join the batch; and results go into an Excel report that can compare against a baseline ([#1260](https://github.com/AMD-AGI/Primus/pull/1260), [#1266](https://github.com/AMD-AGI/Primus/pull/1266), [#1299](https://github.com/AMD-AGI/Primus/pull/1299)). Two misreported throughput values are fixed: TorchTitan values printed with a thousands separator, and Megatron-Bridge's post-training estimate. Note the behavior changes: single-node runs now time out, output files have fixed names, and the runners exit non-zero when any run fails.
- **Release benchmark suites in the auto-benchmark menu.** One choice detects the GPU and runs the whole Primus suite (Megatron, Megatron Bridge, TorchTitan) or JAX suite (MaxText, MaxDiffusion), recording the image digest, Primus commit, and submodule pins with every result ([#1251](https://github.com/AMD-AGI/Primus/pull/1251)).
- **A preset `HIP_VISIBLE_DEVICES` is kept.** The runner used to overwrite it with `0..GPUS_PER_NODE-1`, so on a shared node a job could land on another user's GPU; it now keeps your selection and warns when its count differs from `GPUS_PER_NODE` ([#1237](https://github.com/AMD-AGI/Primus/pull/1237)).

### `rocm/primus`: Megatron-LM and TorchTitan

#### Changes you might need to act on

- **Mock-data runs use the model's real vocabulary.** Megatron mock data used to size every model for a 131,072-token vocabulary; each model config now sets its real `vocab_size`, so embeddings, output layer, and logits have their true size. Mock-data throughput and memory are therefore not comparable with earlier releases ([#1249](https://github.com/AMD-AGI/Primus/pull/1249)).
- **Megatron-only example material moved under `examples/megatron/`.** For example `examples/deepseek-v4/` is now `examples/megatron/models/deepseek-v4/`, and `examples/moe_package/` is now `examples/megatron/guides/moe_package/`; `examples/mlperf/` and `examples/hardware_configs/` stay where they were. Update scripts that name the old paths ([#1130](https://github.com/AMD-AGI/Primus/pull/1130)).
- **Some environment switches are now config keys, and the old variables do nothing.** `PRIMUS_FUSED_QK_RMSNORM_ROPE` is replaced by `use_turbo_fused_qk_rmsnorm_rope` ([#1247](https://github.com/AMD-AGI/Primus/pull/1247)); `PRIMUS_TURBO_CROSS_ENTROPY` and `PRIMUS_TURBO_CE_OVERWRITE_INPUT` by `use_turbo_cross_entropy` and `turbo_ce_overwrite_input` ([#1245](https://github.com/AMD-AGI/Primus/pull/1245)); the MoE gather-quantization variables by `moe_permute_quant_fusion`, `moe_backward_permute_quant_fusion`, and `moe_skip_identity_sort` ([#1241](https://github.com/AMD-AGI/Primus/pull/1241)).
- **Transformer Engine fused cross entropy reuses the logits buffer by default** (`te_ce_overwrite_input: true`), so after the loss backward the logits tensor holds its gradient. Set it to `false` when the same logits feed anything else in the step, such as a distillation or auxiliary loss, metrics, or hooks ([#1287](https://github.com/AMD-AGI/Primus/pull/1287)).
- **TorchTitan uses a Triton fused AdamW by default** (`optimizer.use_triton_fused_adam: true`). Checkpoints stay interchangeable with `torch.optim.AdamW`; set the flag to `false` to go back to it. Reading `optimizer.state` directly between steps now shows stale `step` values ([#1233](https://github.com/AMD-AGI/Primus/pull/1233)).
- **Turbo grouped GEMM expects expandable segments.** Primus warns when `use_turbo_grouped_gemm` is on without `expandable_segments:True`, and a runner hook stops the launch when `expandable_segments:True` is requested but not in effect on the PyTorch build ([#1257](https://github.com/AMD-AGI/Primus/pull/1257)).

#### New features

- **GLM-5.3-Flash** (`model_type: glm5_next`): KDA linear attention, DSA sparse attention, and manifold-constrained hyper-connections with a 288-expert MoE, an MI355X BF16 recipe, and online loading of the released Hugging Face checkpoint for any TP / EP / PP layout ([#1230](https://github.com/AMD-AGI/Primus/pull/1230)). Packed sequences now run under context parallelism ([#1265](https://github.com/AMD-AGI/Primus/pull/1265)) and text-only checkpoint exports load ([#1254](https://github.com/AMD-AGI/Primus/pull/1254)). Training was validated on an 8-layer proxy; the full 45-layer model was checked forward-only against SGLang, and its DSA indexer is frozen.
- **ZAYA1-8B FP8 recipes** for MI300X, MI325X, and MI355X (`fp8: hybrid`, tensorwise scaling, accurate from a cold start) and an MI325X BF16 recipe ([#1269](https://github.com/AMD-AGI/Primus/pull/1269)); the MI355X micro-batch sizes now fill that GPU's memory ([#1279](https://github.com/AMD-AGI/Primus/pull/1279)).
- **HunyuanWorld-Mirror post-training**, stage 1 (geometry heads) and stage 2 (Gaussian head), with MI355X configs launched through `train posttrain`. The upstream trainer is the new `third_party/HunyuanWorld-Mirror-rocm` submodule; Primus applies ROCm fixes at runtime ([#1242](https://github.com/AMD-AGI/Primus/pull/1242)).
- **MiniMax-M3 on MegaMoE.** The fused MegaMoE expert layer accepts MiniMax-M3's `swigluoai` activation and matches the DeepEP path's loss ([#1243](https://github.com/AMD-AGI/Primus/pull/1243)).
- **Opt-in fusions:** Primus-Turbo fused cross entropy for TP1 training ([#1245](https://github.com/AMD-AGI/Primus/pull/1245)); FlyDSL Q/K RMSNorm + RoPE and residual RMSNorm + MXFP4 QKV fusions for GPT-OSS ([#1208](https://github.com/AMD-AGI/Primus/pull/1208)), with the fused Q/K RMSNorm + RoPE now also covering head dimension 128 ([#1247](https://github.com/AMD-AGI/Primus/pull/1247)); and gradient reduce-scatter on the GPU's copy engines (`MEGATRON_GRAD_REDUCE_BACKEND=rccl_sdma`, which needs an RCCL build with CE ReduceScatter) ([#1207](https://github.com/AMD-AGI/Primus/pull/1207)).
- **MI455X FP8 hybrid LoRA SFT proxies** for Llama-2-70B, Qwen2.5-72B, and Qwen3-235B-A22B on one GPU ([#1248](https://github.com/AMD-AGI/Primus/pull/1248)).

#### Performance

- **Optimizer and MoE kernel work, opt-in:** gradient clipping and BF16 Adam moments fused into the Adam update ([#1219](https://github.com/AMD-AGI/Primus/pull/1219)), fused MXFP4 weight de-oscillation ([#1222](https://github.com/AMD-AGI/Primus/pull/1222)), BF16 router backward GEMMs (`moe_router_bwd_bf16`, [#1240](https://github.com/AMD-AGI/Primus/pull/1240)), and MoE gathers folded into Primus-Turbo MXFP4 quantization ([#1241](https://github.com/AMD-AGI/Primus/pull/1241)).
- **Qwen3-30B-A3B FP8 on MI355X** moves to Primus-Turbo tensorwise FP8 with the Turbo grouped-MLP, activation, and permute fusions ([#1247](https://github.com/AMD-AGI/Primus/pull/1247)).
- **Tuned configs:** MI355X GPT-OSS-20B, DeepSeek-V2-Lite, Llama 70B, Qwen, Mixtral, and TorchTitan Llama recipes ([#1218](https://github.com/AMD-AGI/Primus/pull/1218), [#1252](https://github.com/AMD-AGI/Primus/pull/1252), [#1349](https://github.com/AMD-AGI/Primus/pull/1349)); MI300X GPT-OSS-20B and Llama 70B ([#1307](https://github.com/AMD-AGI/Primus/pull/1307), [#1332](https://github.com/AMD-AGI/Primus/pull/1332), [#1304](https://github.com/AMD-AGI/Primus/pull/1304)); MI325X Llama 70B and ZAYA1-8B ([#1318](https://github.com/AMD-AGI/Primus/pull/1318)). Two of these change more than speed: MI355X DeepSeek-V2-Lite BF16 now uses MLA, the model's real attention, and MI355X GPT-OSS-20B FP8 trains with global batch 96 instead of 80 ([#1252](https://github.com/AMD-AGI/Primus/pull/1252)).

#### Bug fixes

- **Stale expert weights with the fused grouped MLP.** With `turbo_fused_grouped_gemm` and `overlap_param_gather`, the fused op skipped the hooks that wait for the expert-weight all-gather, so whenever expert data parallelism exceeded one it read partly stale weights: no error and no NaN, just a drifting loss ([#1252](https://github.com/AMD-AGI/Primus/pull/1252)).
- **Forced "even" routing under DeepEP.** With DeepEP on, most routed slots received weight 0 and the MoE output collapsed toward the shared expert; the router now does the remap itself ([#1243](https://github.com/AMD-AGI/Primus/pull/1243)).
- **MoE recipes no longer force the DeepEP cheap fence.** v26.8 recipes set `PRIMUS_TURBO_DEEPEP_DISABLE_CHEAP_FENCE=0`, which could corrupt DeepEP dispatch-backward rows; the variable is removed from the 13 MI355X and MI325X recipes that set it ([#1236](https://github.com/AMD-AGI/Primus/pull/1236)).
- **MoE models no longer hang on hosts with an amdgpu DRM interface older than 3.64.** A runner hook detects such hosts and keeps HIP memory pools off virtual-memory mappings ([#1302](https://github.com/AMD-AGI/Primus/pull/1302)).
- **DeepSeek-V4 hyper-connections** contract on the same axis as the inference kernels; the old convention drifted from SGLang by more with every layer ([#1040](https://github.com/AMD-AGI/Primus/pull/1040)).
- **TorchTitan:** DeepSeek blocks are compiled inside their activation-checkpoint wrappers, so with full checkpointing the MoE block no longer runs eagerly ([#1320](https://github.com/AMD-AGI/Primus/pull/1320)); and the Primus-Turbo attention converter applies only when both `enable_primus_turbo` and `use_turbo_attention` are on, fixing a `TypeError` on the first forward otherwise ([#1234](https://github.com/AMD-AGI/Primus/pull/1234)).
- Smaller fixes: the Turbo FP8 weight cache is stored in the operator's padded layout ([#1333](https://github.com/AMD-AGI/Primus/pull/1333)), and `enable_turbo_attention_float8` no longer fails on its first call ([#1247](https://github.com/AMD-AGI/Primus/pull/1247)).

### `rocm/jax-training:maxtext`: MaxText

- **The image ships Primus again.** `/workspace/Primus` is at `db1a4160`, the commit the image was built from; a `release/v26.9` checkout is still recommended—see [Primus source for v26.9](#primus-source-for-v269).
- **MaxText moves to `release/v26.9`** (`8d1d43c0`), carrying the same ROCm patches as v26.8 on newer upstream. Upstream drops the `retry_when_tokens_dropped` config key, so remove it from any local config, and renames the `cosmos3-*-reasoner` model configs to `weaver-max` / `weaver-mini` ([#1316](https://github.com/AMD-AGI/Primus/pull/1316)).
- **XLA autotuning defaults to level 3.** At Primus's previous default, level 4, the autotuner's correctness checks intermittently aborted compilation of the train step on GPU 0; level 3 keeps autotuning and drops only those checks. Set `XLA_GPU_AUTOTUNE_LEVEL=4` to restore the old default ([#1316](https://github.com/AMD-AGI/Primus/pull/1316)).
- **Transformer Engine's per-call `return_max_logit` warning is silenced.** It was printed on every backward attention call, per layer, per GPU, and per step; Primus now passes the same fallback value TE was using, so numerics are unchanged ([#1316](https://github.com/AMD-AGI/Primus/pull/1316)).
- **MaxText runs on real data no longer stop before the first step.** Grain data-loader workers read absl flags that Primus had left unparsed, and MaxText reported the error as a graceful stop: zero steps trained, exit status 0. Synthetic-data runs, including the benchmark configs, were not affected ([#1229](https://github.com/AMD-AGI/Primus/pull/1229)).
- **Recipe fixes:** the MI300X Wan2.1-1.3B and MI355X Mixtral-8x7B FP8 out-of-memory errors are fixed, and FLUX.1-dev is retuned on MI300X and MI355X ([#1338](https://github.com/AMD-AGI/Primus/pull/1338), [#1349](https://github.com/AMD-AGI/Primus/pull/1349)).
- **Primus Projection gains a MaxText/JAX backend** ([#1232](https://github.com/AMD-AGI/Primus/pull/1232)). Its benchmark path has so far been verified on CPU only.

---

## v26.9 (current)

### `rocm/primus:v26.9`

Megatron-LM, TorchTitan, and Megatron Bridge backends.

| | |
| --- | --- |
| Image ID | `cd678296ff5b` |
| Built | 2026-10-30 |
| Size | 60.5 GB |
| Manifest | `8c8b5b3d977cfd4f746ba6cdd96731a894995bc0` |
| Dockerfile | [`Dockerfile.primus-v26.9`](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/docker-release/Dockerfile.primus-v26.9) |

| Software component | Version |
| ------------------ | ------- |
| ROCm | 10.2.0a20261020 |
| Python | 3.12.3 |
| PyTorch | 2.16.0a0+rocm10.2.0a20261020 |
| Transformer Engine | 2.20.0.dev0+023639b51 |
| Flash Attention | 2.8.4 |
| hipBLASLt | 1.5.0-cb261897 |
| Triton | 3.9.0+gitfc8d31a5.rocm10.2.0a20261020 |
| RCCL | 2.32.3 |
| torchvision | 0.30.0a0+rocm10.2.0a20261020 |
| torchaudio | 2.11.0a0+rocm10.2.0a20261020 |
| APEX | 1.16.0a0+rocm10.3.0a20261029 |
| AITER | 0.1.27 |
| Primus-Turbo | 0.6.1.dev4 |
| torchao | 0.15.0+gite9c7bead9 |
| FBGEMM | 2026.10.30 |
| mamba-ssm / causal-conv1d / grouped_gemm | 2.3.1 / 1.5.0.post8 / 1.1.4 |
| transformers / datasets | 5.10.0 / 5.0.1 |
| NumPy | 2.5.4 |

> **Both families run on ROCm nightlies this release, and not the same one.** `rocm/primus:v26.9` takes ROCm `10.2.0a20261020` from `nightly.repo.amd.com`, where v26.8 used the 10.1.0 release from `stable.repo.amd.com`; PyTorch, torchvision, and torchaudio are the matching pre-release builds from the nightly PyTorch index. `rocm/jax-training:maxtext-v26.9` takes ROCm `10.3.0a20261026`. hipBLASLt differs between the two images as a result; RCCL is 2.32.3 in both.

> **Transformer Engine is still built from source.** It comes from `ROCm/TransformerEngine` at `023639b5` (it reports `2.20.0.dev0+023639b51`). Flash Attention is unchanged: the `ROCm/flash-attention` fork at tag `v2.8.4.1-cktile`, reported as `2.8.4`.

> **The APEX row reads ahead of the image's ROCm.** The Dockerfile installs `apex` unpinned from the nightly PyTorch index, which resolved to `1.16.0a0+rocm10.3.0a20261029`, a build against a ROCm 10.3 nightly, on this ROCm 10.2 image.

### `rocm/jax-training:maxtext-v26.9`

MaxText (JAX) backend.

| | |
| --- | --- |
| Image ID | `2b6f7537faff` |
| Built | 2026-10-30 |
| Size | 50.0 GB |
| Manifest | `aa44dbf5a9a7edc0fc76694adc624023df6c50e7` |
| Dockerfile | [`Dockerfile.jax-v26.9`](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/docker-release/Dockerfile.jax-v26.9) |

| Software component | Version |
| ------------------ | ------- |
| ROCm | 10.3.0a20261026 |
| Python | 3.12.3 |
| JAX / jaxlib | 0.12.0 |
| jax-rocm10-pjrt / jax-rocm10-plugin | 0.12.0+rocm10.3.0a20261026 |
| Transformer Engine | 2.19.0.dev0+rocm10.2.0a20261002.bfe94fb |
| hipBLASLt | 1.5.0-565c4c74 |
| RCCL | 2.32.3 |
| Flax | 0.12.10 |
| TensorFlow | 2.21.0 |
| Optax / Orbax / Grain / tensorstore | 0.2.8 / 0.12.8 / 0.2.18 / 0.1.86 |
| MaxText | 8d1d43c0 |
| transformers / datasets | 4.57.3 / 4.8.5 |
| NumPy | 2.5.4 |

> **`rocm/jax-training:maxtext-v26.9` contains Primus again.** v26.8 shipped without it; this image has `/workspace/Primus` at `db1a4160`, the commit both images were built from, next to MaxText at `/workspace/maxtext` (`8d1d43c0`) and MaxDiffusion at `/workspace/maxdiffusion` (`68e06965`). A `release/v26.9` checkout is still the recommended way to run it—see [Primus source for v26.9](#primus-source-for-v269).

> **Transformer Engine is a development build against an older nightly.** TE `2.19.0.dev0` comes from the ROCm frameworks staging index and is tagged `rocm10.2.0a20261002`, while the image's ROCm and its JAX plugin pair are `10.3.0a20261026`.

> **Note:** `transformers` is 4.57.3 in the MaxText image because the published tag includes the MaxDiffusion stage, which pins transformers back to 4.x for Flax CLIP / T5. The build-time manifest records 5.17.0, captured before that stage runs; 4.57.3 is what you actually get (`huggingface-hub` 0.36.2 and `tokenizers` 0.22.2 move back with it). The PyTorch image is on 5.10.0.

> **Primus enables Shardy by default** (`shardy: true` in its MaxText module config). Leave it on. See the [Shardy guide for JAX users](https://openxla.org/shardy/getting_started_jax).

### Primus source for v26.9

Use the **`release/v26.9`** branch for both images:

```bash
git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus
git checkout release/v26.9
git submodule update --init --recursive
```

| | |
| --- | --- |
| Build commit | `db1a4160` (2026-10-30), on `release/v26.9` |
| Branch tip | `93dd9201` (2026-10-31) |
| Megatron-LM | `d3528a21` |
| TorchTitan | `73a0e697` |
| Megatron Bridge | `9577b128` |
| MaxText | `8d1d43c0` |
| Emerging-Optimizers | `93d9eb3a` |
| HummingbirdXT | `ed7b7bd0` |
| HunyuanWorld-Mirror | `c61300e4` |

MaxText moved to `release/v26.9` and HunyuanWorld-Mirror (`third_party/HunyuanWorld-Mirror-rocm`) is new since v26.8; Megatron-LM, TorchTitan, Megatron Bridge, Emerging-Optimizers, and HummingbirdXT are on the same commits.

> **Use a `release/v26.9` checkout rather than anything baked into the images.**
>
> - `rocm/primus:v26.9` and `rocm/jax-training:maxtext-v26.9` were both built from `db1a4160`, and both carry that commit at `/workspace/Primus`.
> - `release/v26.9` has moved since the build: it is now at `93dd9201` (2026-10-31), which adds MI325X GPT-OSS-20B and ZAYA1-8B recipe tuning ([#1355](https://github.com/AMD-AGI/Primus/pull/1355)). The branch checkout includes it; the in-image copy does not.
> - The MaxText image's `/workspace/maxtext` (`8d1d43c0`) matches the branch's `third_party/maxtext`.

### Changes since v26.8

`rocm/primus`:

| Component | v26.8 | v26.9 |
| --------- | ----- | ----- |
| ROCm | 10.1.0 | 10.2.0a20261020 |
| PyTorch | 2.14.0+rocm10.1.0 | 2.16.0a0+rocm10.2.0a20261020 |
| Transformer Engine | 2.17.1+130099ce4 | 2.20.0.dev0+023639b51 |
| hipBLASLt | 1.4.1-39d8d603 | 1.5.0-cb261897 |
| Triton | 3.8.0+git669b31ac.rocm10.1.0 | 3.9.0+gitfc8d31a5.rocm10.2.0a20261020 |
| RCCL | 2.30.7 | 2.32.3 |
| torchvision | 0.29.0a0+rocm10.1.0 | 0.30.0a0+rocm10.2.0a20261020 |
| torchaudio | 2.11.0+rocm10.1.0 | 2.11.0a0+rocm10.2.0a20261020 |
| APEX | 1.14.0+rocm10.1.0 | 1.16.0a0+rocm10.3.0a20261029 |
| AITER | 0.1.22.post1 | 0.1.27 |
| Primus-Turbo | 0.5.1.dev7 | 0.6.1.dev4 |
| FBGEMM | 2026.10.8 | 2026.10.30 |
| transformers / datasets | 5.10.0 / 3.6.0 | 5.10.0 / 5.0.1 |
| NumPy | 2.5.3 | 2.5.4 |
| Image size | 56.4 GB | 60.5 GB |

`rocm/jax-training:maxtext`:

| Component | v26.8 | v26.9 |
| --------- | ----- | ----- |
| ROCm | 10.2.0a20260923 | 10.3.0a20261026 |
| JAX / jaxlib | 0.11.1 | 0.12.0 |
| jax-rocm10-pjrt / jax-rocm10-plugin | 0.11.1+rocm10.2.0a20260923 | 0.12.0+rocm10.3.0a20261026 |
| Transformer Engine | 2.18.0.dev0+rocm10.2.0a20260929.03afd8f | 2.19.0.dev0+rocm10.2.0a20261002.bfe94fb |
| hipBLASLt | 1.5.0-9bb8b541 | 1.5.0-565c4c74 |
| RCCL | 2.31.2 | 2.32.3 |
| Flax | 0.12.9 | 0.12.10 |
| Optax / Orbax / Grain / tensorstore | 0.2.8 / 0.12.6 / 0.2.18 / 0.1.85 | 0.2.8 / 0.12.8 / 0.2.18 / 0.1.86 |
| MaxText | 07c240fa | 8d1d43c0 |
| NumPy | 2.5.3 | 2.5.4 |
| Image size | 43.9 GB | 50.0 GB |

---

## v26.8

### Highlights for v26.8

**The two families split on ROCm.** `rocm/primus` moves to the ROCm 10.1.0 release with PyTorch 2.14, and builds Transformer Engine and Flash Attention from source again. `rocm/jax-training:maxtext` moves to a ROCm 10.2 nightly (`10.2.0a20260923`) with JAX 0.11.1 and a Transformer Engine 2.18 development build, and **no longer ships a copy of Primus**: run it with a `release/v26.8` checkout, which `primus-cli container` mounts for you. Bare-metal installs changed accordingly — see [Bare-metal installation](./bare-metal-installation.md) and the [JAX guide](./bare-metal-installation-jax.md).

#### Both families

- **The `examples/run_*` launchers are retired** in favour of `./primus-cli` ([#999](https://github.com/AMD-AGI/Primus/pull/999)). Callers that relied on the `EXP` / `NNODES` / `DATA_PATH` contract can use `runner/helpers/launch/slurm_pretrain.sh`. The Kubernetes submission client (`examples/run_k8s_pretrain.sh`) is removed, and the dataset-split seed flag is renamed `--split_seed` so Megatron's `--seed` stays the training RNG.
- **Scientific-notation CLI overrides are parsed as numbers.** `--lr 1e-5` and similar used to reach the config as a string ([#1112](https://github.com/AMD-AGI/Primus/pull/1112)).
- **`amd-smi` is supported** alongside the deprecated `rocm-smi` ([#1134](https://github.com/AMD-AGI/Primus/pull/1134)).
- **Primus Projection gains a TorchTitan backend** ([#1202](https://github.com/AMD-AGI/Primus/pull/1202)).
- **New tooling:** a non-interactive batch benchmark runner and result extractor for every backend, in single-node and Slurm modes (`tools/perf`, [#1173](https://github.com/AMD-AGI/Primus/pull/1173)).

#### `rocm/primus`: Megatron-LM and TorchTitan

##### Changes you might need to act on

- **Hybrid configs are named `zebra_*` again.** v26.7 shipped them as `hylo_*`; v26.8 restores the Zebra-Llama name (for example `hylo_llama_mamba_8B_BF16-pretrain.yaml` is now `zebra_llama_mamba_8B_BF16-pretrain.yaml`, and the `HyloLlama*` classes are `ZebraLlama*`). Update any local config or script that names the old files ([#1167](https://github.com/AMD-AGI/Primus/pull/1167)).

##### New features

- **ZAYA1-8B** Megatron pretraining, with MI300X and MI355X mock-data configs. This ships on `release/v26.8` from [#1199](https://github.com/AMD-AGI/Primus/pull/1199), which is not yet merged to `main`.
- **MiniMax-M3 (text tower)** with MiniMax Sparse Attention, in a reference PyTorch backend and a gfx950 FlyDSL backend. On an 8-layer proxy on 8× MI355X the FlyDSL backend runs at 380 TFLOP/s/GPU, twice as fast as the reference backend with an identical loss curve ([#1197](https://github.com/AMD-AGI/Primus/pull/1197)).
- **SDMA parameter gather for the Megatron distributed optimizer** ([#1128](https://github.com/AMD-AGI/Primus/pull/1128)), enabled by default in the 1×8 MI355X GPT-OSS MLPerf recipe ([#1168](https://github.com/AMD-AGI/Primus/pull/1168)).
- **Fused grouped GEMM.** `turbo_fused_grouped_gemm` now replaces the fc1 → GLU → fc2 chain with a single Primus-Turbo FP8/FP4 grouped-MLP call that also owns WGRAD ([#1165](https://github.com/AMD-AGI/Primus/pull/1165)).
- **Fused MXFP4 weight-gradient accumulation** for non-preshuffled MXFP4 ([#1174](https://github.com/AMD-AGI/Primus/pull/1174)).
- **`qk_clip` (MuonClip) for hybrid models and the distributed optimizer** ([#1164](https://github.com/AMD-AGI/Primus/pull/1164)), with the per-head max attention logit now exposed on the FLA fused attention path ([#1162](https://github.com/AMD-AGI/Primus/pull/1162)).
- **GPT-OSS-20B dense QKVO GEMMs run on Turbo FlyDSL** in both the FP8 and MXFP4 recipes ([#1127](https://github.com/AMD-AGI/Primus/pull/1127)), on top of BF16 de-oscillation snapshots and a simplified 20B recipe ([#1070](https://github.com/AMD-AGI/Primus/pull/1070)).
- **Llama-3.1-8B FP4 building blocks** — fused FP4 dense MLP, residual RMSNorm and skip-y — upstreamed from MLPerf, default-off ([#1156](https://github.com/AMD-AGI/Primus/pull/1156)).
- **New recipes:** Qwen3-30B-A3B and Qwen2.5-72B MXFP4 pretraining on MI355X ([#1119](https://github.com/AMD-AGI/Primus/pull/1119)), Llama-3.1-8B BF16 LoRA-SFT on MI355X ([#1113](https://github.com/AMD-AGI/Primus/pull/1113)), and native LoRA-SFT proxies for gfx1250 ([#1155](https://github.com/AMD-AGI/Primus/pull/1155)).
- **SpecForge backend** for speculative-decoding draft training, offline ([#1139](https://github.com/AMD-AGI/Primus/pull/1139)) and online with Mooncake + SGLang ([#1191](https://github.com/AMD-AGI/Primus/pull/1191)). It runs on its own ROCm runtime image; see the [SpecForge guide](../04-technical-guides/specforge/introduction.md).

##### Performance

- **Fewer gradient-buffer clears and an isolated gradient reduce-scatter.** Slices that Primus-Turbo's expert WGRAD fully overwrites are no longer zeroed first, and the reduce-scatter runs on a dedicated RCCL group ([#1186](https://github.com/AMD-AGI/Primus/pull/1186)).
- **Cheaper MXFP4 de-oscillation**, now quantized on the same scale grid as the forward pass ([#1184](https://github.com/AMD-AGI/Primus/pull/1184)).
- **Tuned configs:** MI355X pretraining ([#1218](https://github.com/AMD-AGI/Primus/pull/1218)), and MI325X 70B/72B and MoE recipes ([#1227](https://github.com/AMD-AGI/Primus/pull/1227)).

##### Bug fixes

- **Silent gradient corruption from a Triton buffer-store miscompile.** Affected kernels are now detected and recompiled with buffer ops disabled, and TorchTitan restores eager precision semantics under `torch.compile` ([#962](https://github.com/AMD-AGI/Primus/pull/962)).
- **DeepSeek-V4 reported TFLOPs 3.63× too high** at the V4-Flash shape. Reporting only; training math is unchanged ([#1158](https://github.com/AMD-AGI/Primus/pull/1158)).
- **DeepSeek-V4 MegaMoE ignored `turbo_mega_moe_precision`**, so `mxfp8` runs silently stayed on BF16 experts ([#1196](https://github.com/AMD-AGI/Primus/pull/1196)).
- **TorchTitan GPT-OSS router bias** is initialized after meta-device materialization, fixing non-finite router scores ([#1183](https://github.com/AMD-AGI/Primus/pull/1183)), and TorchTitan DeepSeek `torch.compile` is bounded at the routed experts ([#1216](https://github.com/AMD-AGI/Primus/pull/1216), on `release/v26.8`).
- **Zebra-Llama KDA 1B no longer NaNs at iteration 1 on MI355X** ([#1215](https://github.com/AMD-AGI/Primus/pull/1215)), and Zebra-Llama-Mamba 3B no longer OOMs on MI300X ([#1209](https://github.com/AMD-AGI/Primus/pull/1209)).
- **MI350X gets its own GPU env file**, so `PRIMUS_GPU_MODEL=MI350X` no longer silently skips the gfx950 WarpSpeed setting and NaNs ([#1178](https://github.com/AMD-AGI/Primus/pull/1178)); the MXFP4 Transformer Engine settings move into the recipes' `env:` blocks so `primus-cli` launches pick them up ([#1179](https://github.com/AMD-AGI/Primus/pull/1179)); and `primus-cli` now applies the FLA Triton autotune patch that prevents hybrid-model hangs ([#1181](https://github.com/AMD-AGI/Primus/pull/1181)). All three were also backported to `release/v26.7`.
- Smaller fixes: the Flux validation loss is no longer rescaled in place ([#1079](https://github.com/AMD-AGI/Primus/pull/1079)), and Transformer Engine's `requantize_data` no longer raises `UnboundLocalError` ([#1108](https://github.com/AMD-AGI/Primus/pull/1108)).

#### `rocm/jax-training:maxtext`: MaxText

- **The image ships no Primus.** Use `release/v26.8` — see [Primus source for v26.8](#primus-source-for-v268).
- **MaxText moves to `release/v26.8`** (`07c240fa`, about 290 upstream commits). Upstream removed the Linen modules and with them the `pure_nnx_decoder` config key, which MaxText now rejects; the three MI325X `nanoo_fp8` configs that still set it are fixed. Drop the key from any local config ([#1228](https://github.com/AMD-AGI/Primus/pull/1228)).
- **Three performance regressions on JAX 0.11.1 fixed with Primus-managed defaults:** CUB radix sort is disabled for MoE top-k (about 0.8 s/step on DeepSeek-V2-16B), stream priority is honoured again so collectives overlap compute, and dynamic-slice fusion is disabled (about 10% on Llama-2-7B FP8). They need a `release/v26.8` checkout, and can be overridden per config ([#1228](https://github.com/AMD-AGI/Primus/pull/1228)).
- **`XLA_FLAGS` precedence fixed.** A config's own `XLA_FLAGS` now beats the managed defaults instead of being silently overridden, and `XLA_FLAGS_APPEND` adds flags without replacing them ([#1054](https://github.com/AMD-AGI/Primus/pull/1054)).

### `rocm/primus:v26.8`

Megatron-LM, TorchTitan, and Megatron Bridge backends.

| | |
| --- | --- |
| Image ID | `e86c9bb570ec` |
| Built | 2026-10-08 |
| Size | 56.4 GB |
| Manifest | `6025d2033abe4288c190d2db916bfeb2c9585d06` |
| Dockerfile | [`Dockerfile.primus-v26.8`](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/docker-release/Dockerfile.primus-v26.8) |

| Software component | Version |
| ------------------ | ------- |
| ROCm | 10.1.0 |
| Python | 3.12.3 |
| PyTorch | 2.14.0+rocm10.1.0 |
| Transformer Engine | 2.17.1+130099ce4 |
| Flash Attention | 2.8.4 |
| hipBLASLt | 1.4.1-39d8d603 |
| Triton | 3.8.0+git669b31ac.rocm10.1.0 |
| RCCL | 2.30.7 |
| torchvision | 0.29.0a0+rocm10.1.0 |
| torchaudio | 2.11.0+rocm10.1.0 |
| APEX | 1.14.0+rocm10.1.0 |
| AITER | 0.1.22.post1 |
| Primus-Turbo | 0.5.1.dev7 |
| torchao | 0.15.0+gite9c7bead9 |
| FBGEMM | 2026.10.8 |
| mamba-ssm / causal-conv1d / grouped_gemm | 2.3.1 / 1.5.0.post8 / 1.1.4 |
| transformers / datasets | 5.10.0 / 3.6.0 |
| NumPy | 2.5.3 |

### `rocm/jax-training:maxtext-v26.8`

MaxText (JAX) backend.

| | |
| --- | --- |
| Image ID | `8267d6a4a016` |
| Built | 2026-10-01 |
| Size | 43.9 GB |
| Manifest | `73710dda449ad7aaf12207a984f986a53d1234e2` |
| Dockerfile | [`Dockerfile.jax-v26.8`](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/docker-release/Dockerfile.jax-v26.8) |

| Software component | Version |
| ------------------ | ------- |
| ROCm | 10.2.0a20260923 |
| Python | 3.12.3 |
| JAX / jaxlib | 0.11.1 |
| jax-rocm10-pjrt / jax-rocm10-plugin | 0.11.1+rocm10.2.0a20260923 |
| Transformer Engine | 2.18.0.dev0+rocm10.2.0a20260929.03afd8f |
| hipBLASLt | 1.5.0-9bb8b541 |
| RCCL | 2.31.2 |
| Flax | 0.12.9 |
| TensorFlow | 2.21.0 |
| Optax / Orbax / Grain / tensorstore | 0.2.8 / 0.12.6 / 0.2.18 / 0.1.85 |
| MaxText | 07c240fa |
| transformers / datasets | 4.57.3 / 4.8.5 |
| NumPy | 2.5.3 |

> **The two families are on different ROCm builds this release.** `rocm/primus:v26.8` uses the ROCm 10.1.0 release from `stable.repo.amd.com`. `rocm/jax-training:maxtext-v26.8` uses a ROCm 10.2 nightly, `10.2.0a20260923` from `nightly.repo.amd.com`, and its JAX plugin pair and Transformer Engine are built against that nightly. hipBLASLt and RCCL differ between the two images as a result.

> **`rocm/primus` builds Transformer Engine and Flash Attention from source again.** Transformer Engine comes from `ROCm/TransformerEngine` at `130099ce` (it reports `2.17.1+130099ce4`), replacing the three wheel distributions v26.7 installed; CK fused attention is JIT-compiled at run time from the aiter tree kept at `/workspace/deps/te-aiter`. Flash Attention is back on the `ROCm/flash-attention` fork (tag `v2.8.4.1-cktile`, reported as `2.8.4`) instead of PyPI `flash-attn 2.8.1`. The image also turns on the HIP device-queue path (`GPU_USE_DEVICE_QUEUE=1`, `DEBUG_CLR_AQL_DEV_QUEUE=1`).

> **`rocm/jax-training:maxtext-v26.8` no longer contains Primus.** There is no `/workspace/Primus`; run the image with a Primus checkout mounted, which `primus-cli container` does for you. See [Primus source for v26.8](#primus-source-for-v268). The MaxText checkout at `/workspace/maxtext` (`07c240fa`) is still in the image.

> **Note:** `transformers` is 4.57.3 in the MaxText image because the published tag includes the MaxDiffusion stage, which pins transformers back to 4.x for Flax CLIP / T5. The build-time manifest records 5.16.1, captured before that stage runs; 4.57.3 is what you actually get. The PyTorch image is on 5.10.0.

> **JAX 0.11.1 uses Shardy, and Primus enables it by default** (`shardy: true` in its MaxText module config). Leave it on. See the [Shardy guide for JAX users](https://openxla.org/shardy/getting_started_jax).

### Primus source for v26.8

Use the **`release/v26.8`** branch for both images:

```bash
git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus
git checkout release/v26.8
git submodule update --init --recursive
```

| | |
| --- | --- |
| Branch tip | `f487a934` (2026-10-07) |
| Megatron-LM | `d3528a21` |
| TorchTitan | `73a0e697` |
| Megatron Bridge | `9577b128` |
| MaxText | `07c240fa` |
| Emerging-Optimizers | `93d9eb3a` |
| HummingbirdXT | `ed7b7bd0` |

Only the MaxText pin moved since v26.7; Megatron-LM, TorchTitan and Megatron Bridge are on the same commits.

> **Use a `release/v26.8` checkout rather than anything baked into the images.**
>
> - `rocm/primus:v26.8` was built from `f487a934`, which is the current `release/v26.8` tip, so the in-image `/workspace/Primus` currently matches, including the MI325X tuning for the 70B/72B and MoE recipes ([#1227](https://github.com/AMD-AGI/Primus/pull/1227)). Cloning the branch still keeps you current if later commits land on it.
> - `rocm/jax-training:maxtext-v26.8` ships no Primus at all. `release/v26.8` carries the MaxText environment defaults that restore performance on JAX 0.11.1 ([#1228](https://github.com/AMD-AGI/Primus/pull/1228)); an older checkout runs without them. Its `third_party/maxtext` matches the image's `/workspace/maxtext`.

### Changes since v26.7

`rocm/primus`:

| Component | v26.7 | v26.8 |
| --------- | ----- | ----- |
| ROCm | 10.0.0 | 10.1.0 |
| PyTorch | 2.12.0+rocm10.0.0 | 2.14.0+rocm10.1.0 |
| Transformer Engine | 2.17.0+rocm10.0.0 | 2.17.1+130099ce4 |
| Flash Attention | 2.8.1 | 2.8.4 |
| hipBLASLt | 1.4.1-8d1ae90e | 1.4.1-39d8d603 |
| Triton | 3.8.0+git4cff872c.rocm10.0.0 | 3.8.0+git669b31ac.rocm10.1.0 |
| RCCL | 2.30.4 | 2.30.7 |
| torchvision | 0.27.0+rocm10.0.0 | 0.29.0a0+rocm10.1.0 |
| torchaudio | 2.11.0+rocm10.0.0 | 2.11.0+rocm10.1.0 |
| APEX | 1.13.0+rocm10.0.0 | 1.14.0+rocm10.1.0 |
| AITER | 0.1.14.post1 | 0.1.22.post1 |
| Primus-Turbo | 0.4.1.dev33 | 0.5.1.dev7 |
| FBGEMM | 2026.9.9 | 2026.10.8 |
| Image size | 53.8 GB | 56.4 GB |

`rocm/jax-training:maxtext`:

| Component | v26.7 | v26.8 |
| --------- | ----- | ----- |
| ROCm | 10.0.0 | 10.2.0a20260923 |
| JAX / jaxlib | 0.11.0 | 0.11.1 |
| jax-rocm10-pjrt / jax-rocm10-plugin | 0.11.0+rocm10.0.0 | 0.11.1+rocm10.2.0a20260923 |
| Transformer Engine | 2.17.0+rocm10.0.0 | 2.18.0.dev0+rocm10.2.0a20260929.03afd8f |
| hipBLASLt | 1.4.1-8d1ae90e | 1.5.0-9bb8b541 |
| RCCL | 2.30.4 | 2.31.2 |
| Flax | 0.12.8 | 0.12.9 |
| Optax / Orbax / Grain / tensorstore | 0.2.8 / 0.12.4 / 0.2.18 / 0.1.85 | 0.2.8 / 0.12.6 / 0.2.18 / 0.1.85 |
| MaxText | b3c53763 | 07c240fa |
| Image size | 43.7 GB | 43.9 GB |

---

## v26.7

### Highlights for v26.7

**ROCm 10.0.0.** Both image families move off the ROCm 7.x line. This is the defining change of the release and it reaches everything: PyTorch, Transformer Engine, Triton, torchvision/torchaudio and APEX are all rebuilt against it, and the JAX plugin pair is renamed `jax-rocm10-pjrt` / `jax-rocm10-plugin`. If you install bare metal, note that the wheel indexes moved too — see [Bare-metal installation](./bare-metal-installation.md).

Beyond that, the backend work in this release is weighted towards the PyTorch
family; MaxText picks up the ROCm 10 stack plus config tuning rather than new
capability.

#### Both families

- **gfx1250 multi-GPU enablement.** Primus recognises gfx1250 by ISA rather than by MI number and ships an arch env file for it, so `primus-cli` applies real settings instead of falling through to `Detected GPU model: unknown` ([#1043](https://github.com/AMD-AGI/Primus/pull/1043)).
- **`primus-cli` honours an explicit flag** whose value happens to equal the parser default ([#1002](https://github.com/AMD-AGI/Primus/pull/1002)).
- **`*_SOCKET_IFNAME` no longer falls back to an IP address** ([#1063](https://github.com/AMD-AGI/Primus/pull/1063)).
- **The `primus` wheel builds again.** The `mlperf-logging` / `mlperf-common` VCS pins moved to an optional `mlperf` extra, because Hatchling 1.32 rejects PEP 508 direct references in `project.dependencies` ([#1061](https://github.com/AMD-AGI/Primus/pull/1061)).

#### `rocm/primus` — Megatron-LM and TorchTitan

##### New features

- **DeepSeek-V4 on gfx942 at long context.** V4 upstream targeted gfx950 and in practice only ran at `seq_length=4096`; it now trains on CDNA3 with context parallelism at 128k, full model ([#1047](https://github.com/AMD-AGI/Primus/pull/1047)).
- **Packed-sequence (THD) SFT for DeepSeek-V4** — many samples per window delimited by `cu_seqlens`, with attention isolated so no query sees another sample; full model at 4k and 128k across three nodes ([#1062](https://github.com/AMD-AGI/Primus/pull/1062)).
- **DLRM-v4 (TorchRec/HSTU) projection workload** with first-principles MI350X calibration ([#1059](https://github.com/AMD-AGI/Primus/pull/1059)).
- **MLPerf Llama 3.1 8B launcher** and refreshed MI355X benchmark configs ([#1012](https://github.com/AMD-AGI/Primus/pull/1012)).

##### Performance

- **Fused SwiGLU + fc2** (opt-in, via FLA `swiglu_linear`) removes one FFN-wide saved tensor per MLP layer by recomputing the activation in the backward pass ([#1051](https://github.com/AMD-AGI/Primus/pull/1051)).
- **Fused cross-entropy no longer copies the full logits tensor** ([#1049](https://github.com/AMD-AGI/Primus/pull/1049)).
- **DeepSeek-V4 indexer distillation loss** is fused and enabled ([#992](https://github.com/AMD-AGI/Primus/pull/992)).
- **Tuned configs:** GDN/KDA 1B on MI355X for higher occupancy ([#1080](https://github.com/AMD-AGI/Primus/pull/1080)), and MI355X Llama 3.1 8B / GPT-OSS 20B defaults ([#1045](https://github.com/AMD-AGI/Primus/pull/1045)).

##### Bug fixes

- **Weight gradients were dropped after the first microbatch** on Turbo's non-fused `_bridge_weight_grad` path, which gated accumulation on `grad_added_to_main_grad`. Any run using gradient accumulation on that path was training on partial gradients ([#1046](https://github.com/AMD-AGI/Primus/pull/1046)).
- **Out-of-bounds top-k index** in the V4 DSA forward kernel is now bounded ([#1044](https://github.com/AMD-AGI/Primus/pull/1044)).
- **`fused_softcap` stays finite** on saturation-range logits ([#1057](https://github.com/AMD-AGI/Primus/pull/1057)).
- **KDA fused `in_proj` is padded** past the hipBLASLt bf16 dead zone ([#1050](https://github.com/AMD-AGI/Primus/pull/1050)).
- **Pipeline warmup is clamped** for short batches ([#1032](https://github.com/AMD-AGI/Primus/pull/1032)).
- **TorchTitan Turbo grouped-GEMM config renamed** ([#1041](https://github.com/AMD-AGI/Primus/pull/1041)). Update any local config that sets the old key.

#### `rocm/jax-training:maxtext` — MaxText

No new MaxText capability this release; the change is the ROCm 10.0.0 stack itself,
plus config tuning:

- **MI300X batch-size tuning** across the MaxText recipes ([#1020](https://github.com/AMD-AGI/Primus/pull/1020)).
- **`gemma4_26B-fp8` retuned** and the `pure_nnx_decoder` fp8 workaround dropped, now that it is no longer needed ([#1019](https://github.com/AMD-AGI/Primus/pull/1019)).

### `rocm/primus:v26.7`

Megatron-LM, TorchTitan, and Megatron Bridge backends.

| | |
| --- | --- |
| Image ID | `68b7eb7d4db9` |
| Built | 2026-09-09 |
| Size | 53.8 GB |
| Manifest | `db0de753f37d6a6b782bc836b2be54d8ae362258` |
| Dockerfile | [`Dockerfile.primus-v26.7`](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/docker-release/Dockerfile.primus-v26.7) |

| Software component | Version |
| ------------------ | ------- |
| ROCm | 10.0.0 |
| Python | 3.12.3 |
| PyTorch | 2.12.0+rocm10.0.0 |
| Transformer Engine | 2.17.0+rocm10.0.0 |
| Flash Attention | 2.8.1 |
| hipBLASLt | 1.4.1-8d1ae90e |
| Triton | 3.8.0+git4cff872c.rocm10.0.0 |
| RCCL | 2.30.4 |
| torchvision | 0.27.0+rocm10.0.0 |
| torchaudio | 2.11.0+rocm10.0.0 |
| APEX | 1.13.0+rocm10.0.0 |
| AITER | 0.1.14.post1 |
| Primus-Turbo | 0.4.1.dev33 |
| torchao | 0.15.0+gite9c7bead9 |
| FBGEMM | 2026.9.9 |
| mamba-ssm / causal-conv1d / grouped_gemm | 2.3.1 / 1.5.0.post8 / 1.1.4 |
| transformers / datasets | 5.10.0 / 3.6.0 |
| NumPy | 2.5.3 |

### `rocm/jax-training:maxtext-v26.7`

MaxText (JAX) backend.

| | |
| --- | --- |
| Image ID | `b5117e775594` |
| Built | 2026-09-09 |
| Size | 43.7 GB |
| Manifest | `8da24470ccb3b16c60ddb6f8dfc1a1526bc2a408` |
| Dockerfile | [`Dockerfile.jax-v26.7`](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/docker-release/Dockerfile.jax-v26.7) |

| Software component | Version |
| ------------------ | ------- |
| ROCm | 10.0.0 |
| Python | 3.12.3 |
| JAX / jaxlib | 0.11.0 |
| jax-rocm10-pjrt / jax-rocm10-plugin | 0.11.0+rocm10.0.0 |
| Transformer Engine | 2.17.0+rocm10.0.0 |
| hipBLASLt | 1.4.1-8d1ae90e |
| RCCL | 2.30.4 |
| Flax | 0.12.8 |
| TensorFlow | 2.21.0 |
| Optax / Orbax / Grain / tensorstore | 0.2.8 / 0.12.4 / 0.2.18 / 0.1.85 |
| MaxText | b3c53763 |
| transformers / datasets | 4.57.3 / 4.8.5 |
| NumPy | 2.5.3 |

> **ROCm 10.0.0 changes where the wheels come from.** Both families now install ROCm core from `stable.repo.amd.com/rocm/core/whl-next`, with PyTorch, Transformer Engine and the JAX plugin on sibling indexes. The `rocm.nightlies.amd.com/whl-multi-arch` index used through v26.6 does not carry these wheels. Transformer Engine also ships as three distributions (`transformer_engine`, `transformer_engine_rocm10`, and the torch or jax flavour) where v26.6 had two.

> **RCCL is no longer rebuilt.** v26.6 overrode the SDK's RCCL with a `rocm-systems` build; v26.7 uses RCCL 2.30.4 as shipped in the ROCm 10.0.0 SDK. The version is unchanged, the provenance is not.

> **Two rows in the delta table need reading carefully.** APEX appears to go *backwards*, from `1.15.0a0+rocm10.1.0a20260822` to `1.13.0+rocm10.0.0`: v26.6 pulled a `rocm10.1` nightly onto a ROCm 7.15 base, whereas v26.7 uses the APEX built against its own ROCm 10.0.0. And the JAX plugin row is labelled with the v26.7 spelling on both sides — the v26.6 value `0.11.0.post1` shipped as `jax-rocm7-pjrt` / `jax-rocm7-plugin`, before the rename.

> **Note:** `transformers` is 4.57.3 in the MaxText image because the published tag includes the MaxDiffusion stage, which pins transformers back to 4.x for Flax CLIP / T5. The build-time manifest records 5.14.1, captured before that stage runs; 4.57.3 is what you actually get. The PyTorch image is on 5.10.0.

> **JAX 0.11.0 still requires Shardy.** Set `shardy=True` during the training run. See the [Shardy guide for JAX users](https://openxla.org/shardy/getting_started_jax).


### Primus source for v26.7

Use the **`release/v26.7`** branch for both images:

```bash
git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus
git checkout release/v26.7
git submodule update --init --recursive
```

| | |
| --- | --- |
| Branch tip | `2631e68d` (2026-09-02), which is also the `v26.7.0` tag |
| Megatron-LM | `d3528a21` |
| TorchTitan | `73a0e697` |
| Megatron Bridge | `9577b128` |
| MaxText | `b3c53763` |
| Emerging-Optimizers | `93d9eb3a` |
| HummingbirdXT | `ed7b7bd0` |

> **Prefer a `release/v26.7` checkout over the Primus copy baked into the images.**
>
> - `rocm/primus:v26.7` was built from `2631e68d`, which is the current `release/v26.7` tip, so the in-image `/workspace/Primus` currently matches. Cloning the branch still keeps you current if later commits land on it.
> - `rocm/jax-training:maxtext-v26.7` was built from `main` at `e7968675` (2026-09-09), because its Dockerfile pins `PRIMUS_BRANCH=main` rather than a commit. Use `release/v26.7` anyway, so you get the Primus recipes the release was validated against rather than whatever `main` held on the build day. Its bundled `third_party/maxtext` matches `/workspace/maxtext` (`b3c53763`).

### Changes since v26.6

`rocm/primus`:

| Component | v26.6 | v26.7 |
| --------- | ----- | ----- |
| ROCm | 7.15.0a20260727 | 10.0.0 |
| PyTorch | 2.12.0+rocm7.15.0a20260727 | 2.12.0+rocm10.0.0 |
| Transformer Engine | 2.17.0+rocm7.15.0a20260727.e028a6c | 2.17.0+rocm10.0.0 |
| hipBLASLt | 1.4.1-bbb68174 | 1.4.1-8d1ae90e |
| Triton | 3.8.0+git4cff872c.rocm7.15.0a20260727 | 3.8.0+git4cff872c.rocm10.0.0 |
| torchvision | 0.27.0+rocm7.15.0a20260727 | 0.27.0+rocm10.0.0 |
| torchaudio | 2.11.0+rocm7.15.0a20260728 | 2.11.0+rocm10.0.0 |
| APEX | 1.15.0a0+rocm10.1.0a20260822 | 1.13.0+rocm10.0.0 |
| Primus-Turbo | 0.4.1.dev26 | 0.4.1.dev33 |
| FBGEMM | 2026.8.25 | 2026.9.9 |
| transformers / datasets | 5.5.0 / 3.6.0 | 5.10.0 / 3.6.0 |
| NumPy | 2.5.2 | 2.5.3 |
| Image size | 54.0 GB | 53.8 GB |

`rocm/jax-training:maxtext`:

| Component | v26.6 | v26.7 |
| --------- | ----- | ----- |
| ROCm | 7.14.0 | 10.0.0 |
| jax-rocm10-pjrt / jax-rocm10-plugin | 0.11.0.post1 | 0.11.0+rocm10.0.0 |
| Transformer Engine | 2.17.0+rocm7.14.0.50a84ad | 2.17.0+rocm10.0.0 |
| hipBLASLt | 1.4.1-cd957402 | 1.4.1-8d1ae90e |
| MaxText | b47d74bf | b3c53763 |
| NumPy | 2.5.2 | 2.5.3 |
| Image size | 42.6 GB | 43.7 GB |

---

## Earlier releases

Headline versions only. Read [the in-image manifest](#verifying-the-stack-in-an-image) for the full stack of any image below — except `v26.2` and `v26.1`, which predate the manifest (use `pip list` there).

The three MaxText v26.3.x images ship an identical software stack; they differ only in MaxText and Primus content.

| Image | Python | ROCm | Framework | Transformer Engine | RCCL |
| ----- | ------ | ---- | --------- | ------------------ | ---- |
| `rocm/primus:v26.6` | 3.12.3 | 7.15.0 | PyTorch 2.12.0+rocm7.15.0a20260727 | 2.17.0+rocm7.15.0a20260727.e028a6c | 2.30.4 |
| `rocm/primus:v26.5` | 3.12.3 | 7.15.0 | PyTorch 2.12.0+rocm7.15.0a20260720 | 2.15.0.dev0+rocm7.15.0a20260716.a07e607 | 2.30.4 |
| `rocm/primus:v26.4` | 3.12.3 | 7.14.0 | PyTorch 2.12.0+rocm7.14.0a20260608 | 2.14.0.dev0+e6ede467 | 2.29.7 |
| `rocm/primus:v26.3` | 3.12.3 | 7.2.1 | PyTorch 2.10.0+git94c6e04 | 2.12.0.dev0+40434cf6 | 2.27.7 |
| `rocm/primus:v26.2` | 3.12.3 | 7.2.0 | PyTorch 2.10.0a0+git449b176 | 2.8.0.dev0+51f74fa7 | 2.27.7 |
| `rocm/primus:v26.1` | 3.10.12 | 7.1.0 | PyTorch 2.10.0.dev20251112+rocm7.1 | 2.6.0.dev0+f141f34b | 2.27.7 |
| `rocm/jax-training:maxtext-v26.6` | 3.12.3 | 7.14.0 | JAX 0.11.0 | 2.17.0+rocm7.14.0.50a84ad | 2.30.4 |
| `rocm/jax-training:maxtext-v26.5` | 3.12.3 | 7.14.0 | JAX 0.10.0 | 2.15.0.dev0+rocm7.15.0a20260707.72d01a0 | 2.30.4 |
| `rocm/jax-training:maxtext-v26.4` | 3.12.3 | 7.14.0 | JAX 0.9.1 | 2.12.0.dev0+635d7c08 | 2.28.9 |
| `rocm/jax-training:maxtext-v26.3.2` | 3.12.3 | 7.2.1 | JAX 0.8.2 | 2.8.0.dev0+9b312832 | 2.27.7 |
| `rocm/jax-training:maxtext-v26.3.1` | 3.12.3 | 7.2.1 | JAX 0.8.2 | 2.8.0.dev0+9b312832 | 2.27.7 |
| `rocm/jax-training:maxtext-v26.3` | 3.12.3 | 7.2.1 | JAX 0.8.2 | 2.8.0.dev0+9b312832 | 2.27.7 |
| `rocm/jax-training:maxtext-v26.2` | 3.12.3 | 7.1.1 | JAX 0.8.2 | 2.8.0.dev0+aec00a7f | 2.27.7 |

---

## Verifying the stack in an image

Images from v26.3 onward ship a manifest at `/workspace/.manifest/` recording exactly what was installed at build time:

| File | Contents |
| ---- | -------- |
| `requirements.txt` | full `pip list` |
| `dpkg-list.txt` | full `dpkg -l` |
| `env.txt` | every environment variable baked into the image |
| `training_docker_version` | the build's commit tag |
| `docker-build-recipe.txt` | the Dockerfile the image was built from (named `Dockerfile` in `rocm/primus` images before v26.8 and in `rocm/jax-training:maxtext` images before v26.6) |

```bash
docker run --rm --entrypoint bash rocm/primus:v26.8 -c 'cat /workspace/.manifest/requirements.txt'
```

Native library versions are not pip packages; read them from the ROCm headers:

```bash
docker run --rm --entrypoint bash rocm/primus:v26.8 -c '
  grep -E "HIPBLASLT_VERSION_(MAJOR|MINOR|PATCH|TWEAK)" $(find $ROCM_PATH /opt/rocm -name hipblaslt-version.h 2>/dev/null | head -1)
  grep -E "define NCCL_(MAJOR|MINOR|PATCH)"              $(find $ROCM_PATH /opt/rocm -name rccl.h            2>/dev/null | head -1)'
```

Images older than v26.3 predate the manifest; query them with `pip list` directly.

---

## Related documentation

- [Installation and setup](./installation.md)
- [Quickstart](./quickstart.md)
- [Megatron-LM training performance validation](../02-user-guide/megatron-lm-training.md)
- [TorchTitan training performance validation](../02-user-guide/torchtitan-training.md)
- [JAX MaxText training performance validation](../02-user-guide/jax-maxtext-training.md)
