# Release notes

AMD publishes two families of Primus training Docker image:

| Image | Backends | Documented in |
| ----- | -------- | ------------- |
| `rocm/primus:<version>` | Megatron-LM, TorchTitan, Megatron Bridge | [Megatron-LM](../02-user-guide/megatron-lm-training.md), [TorchTitan](../02-user-guide/torchtitan-training.md) |
| `rocm/jax-training:maxtext-<version>` | MaxText (JAX) | [JAX MaxText](../02-user-guide/jax-maxtext-training.md) |

The two families share a version number but are **not** built in lockstep — the same `vNN.N` tag can be weeks apart between families, and the MaxText family also ships patch releases (`v26.3.1`, `v26.3.2`) with no `rocm/primus` counterpart. Always read the section for the exact tag you are running.

**This page is the single source of truth for image contents.** Other pages link here instead of repeating version tables, so there is exactly one place to update per release. If you add a page that names an image tag, link to the relevant section below rather than restating the stack.

Every version below was read out of the published image itself. For v26.4 through v26.6 the values were additionally cross-checked against the release Dockerfiles in [`.github/workflows/docker-release/`](https://github.com/AMD-AGI/Primus/tree/main/.github/workflows/docker-release); earlier releases predate those files and are image-derived only. See [Verifying the stack in an image](#verifying-the-stack-in-an-image) to reproduce any table.

---

## Highlights for v26.7

**ROCm 10.0.0.** Both image families move off the ROCm 7.x line. This is the defining change of the release and it reaches everything: PyTorch, Transformer Engine, Triton, torchvision/torchaudio and APEX are all rebuilt against it, and the JAX plugin pair is renamed `jax-rocm10-pjrt` / `jax-rocm10-plugin`. If you install bare metal, note that the wheel indexes moved too — see [Bare-metal installation](./bare-metal-installation.md).

Beyond that, the backend work in this release is weighted towards the PyTorch
family; MaxText picks up the ROCm 10 stack plus config tuning rather than new
capability.

### Both families

- **gfx1250 multi-GPU enablement.** Primus recognises gfx1250 by ISA rather than by MI number and ships an arch env file for it, so `primus-cli` applies real settings instead of falling through to `Detected GPU model: unknown` ([#1043](https://github.com/AMD-AGI/Primus/pull/1043)).
- **`primus-cli` honours an explicit flag** whose value happens to equal the parser default ([#1002](https://github.com/AMD-AGI/Primus/pull/1002)).
- **`*_SOCKET_IFNAME` no longer falls back to an IP address** ([#1063](https://github.com/AMD-AGI/Primus/pull/1063)).
- **The `primus` wheel builds again.** The `mlperf-logging` / `mlperf-common` VCS pins moved to an optional `mlperf` extra, because Hatchling 1.32 rejects PEP 508 direct references in `project.dependencies` ([#1061](https://github.com/AMD-AGI/Primus/pull/1061)).

### `rocm/primus` — Megatron-LM and TorchTitan

#### New features

- **DeepSeek-V4 on gfx942 at long context.** V4 upstream targeted gfx950 and in practice only ran at `seq_length=4096`; it now trains on MI308X/CDNA3 with context parallelism at 128k, full model ([#1047](https://github.com/AMD-AGI/Primus/pull/1047)).
- **Packed-sequence (THD) SFT for DeepSeek-V4** — many samples per window delimited by `cu_seqlens`, with attention isolated so no query sees another sample; full model at 4k and 128k across three nodes ([#1062](https://github.com/AMD-AGI/Primus/pull/1062)).
- **DLRM-v4 (TorchRec/HSTU) projection workload** with first-principles MI350X calibration ([#1059](https://github.com/AMD-AGI/Primus/pull/1059)).
- **MLPerf Llama 3.1 8B launcher** and refreshed MI355X benchmark configs ([#1012](https://github.com/AMD-AGI/Primus/pull/1012)).

#### Performance

- **Fused SwiGLU + fc2** (opt-in, via FLA `swiglu_linear`) removes one FFN-wide saved tensor per MLP layer by recomputing the activation in the backward pass ([#1051](https://github.com/AMD-AGI/Primus/pull/1051)).
- **Fused cross-entropy no longer copies the full logits tensor** ([#1049](https://github.com/AMD-AGI/Primus/pull/1049)).
- **DeepSeek-V4 indexer distillation loss** is fused and enabled ([#992](https://github.com/AMD-AGI/Primus/pull/992)).
- **Tuned configs:** GDN/KDA 1B on MI355X for higher occupancy ([#1080](https://github.com/AMD-AGI/Primus/pull/1080)), and MI355X Llama 3.1 8B / GPT-OSS 20B defaults ([#1045](https://github.com/AMD-AGI/Primus/pull/1045)).

#### Bug fixes

- **Weight gradients were dropped after the first microbatch** on Turbo's non-fused `_bridge_weight_grad` path, which gated accumulation on `grad_added_to_main_grad`. Any run using gradient accumulation on that path was training on partial gradients ([#1046](https://github.com/AMD-AGI/Primus/pull/1046)).
- **Out-of-bounds top-k index** in the V4 DSA forward kernel is now bounded ([#1044](https://github.com/AMD-AGI/Primus/pull/1044)).
- **`fused_softcap` stays finite** on saturation-range logits ([#1057](https://github.com/AMD-AGI/Primus/pull/1057)).
- **KDA fused `in_proj` is padded** past the hipBLASLt bf16 dead zone ([#1050](https://github.com/AMD-AGI/Primus/pull/1050)).
- **Pipeline warmup is clamped** for short batches ([#1032](https://github.com/AMD-AGI/Primus/pull/1032)).
- **TorchTitan Turbo grouped-GEMM config renamed** ([#1041](https://github.com/AMD-AGI/Primus/pull/1041)). Update any local config that sets the old key.

### `rocm/jax-training:maxtext` — MaxText

No new MaxText capability this release; the change is the ROCm 10.0.0 stack itself,
plus config tuning:

- **MI300X batch-size tuning** across the MaxText recipes ([#1020](https://github.com/AMD-AGI/Primus/pull/1020)).
- **`gemma4_26B-fp8` retuned** and the `pure_nnx_decoder` fp8 workaround dropped, now that it is no longer needed ([#1019](https://github.com/AMD-AGI/Primus/pull/1019)).

---

## v26.7 (current)

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

> **JAX 0.11.0 still requires Shardy.** Set `shardy=True` during the training run. See the [Shardy migration guide](https://docs.jax.dev/en/latest/shardy_jax_migration.html).

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

## v26.6

### `rocm/primus:v26.6`

Megatron-LM, TorchTitan, and Megatron Bridge backends.

| | |
| --- | --- |
| Image ID | `4fcb3f210dc6` |
| Built | 2026-08-25 |
| Size | 54.0 GB |
| Manifest | `f756f2279d8ab57b6549bcbd50d249755b69a407` |
| Dockerfile | [`Dockerfile.primus-v26.6`](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/docker-release/Dockerfile.primus-v26.6) |

| Software component | Version |
| ------------------ | ------- |
| ROCm | 7.15.0 (`rocm-sdk` 7.15.0a20260727) |
| Python | 3.12.3 |
| PyTorch | 2.12.0+rocm7.15.0a20260727 |
| Transformer Engine | 2.17.0+rocm7.15.0a20260727.e028a6c |
| Flash Attention | 2.8.1 |
| hipBLASLt | 1.4.1-bbb68174 |
| Triton | 3.8.0+git4cff872c.rocm7.15.0a20260727 |
| RCCL | 2.30.4 |
| torchvision | 0.27.0+rocm7.15.0a20260727 |
| torchaudio | 2.11.0+rocm7.15.0a20260728 |
| APEX | 1.15.0a0+rocm10.1.0a20260822 |
| AITER | 0.1.14.post1 |
| Primus-Turbo | 0.4.1.dev26 |
| torchao | 0.15.0+gite9c7bead9 |
| FBGEMM | 2026.8.25 |
| mamba-ssm / causal-conv1d / grouped_gemm | 2.3.1 / 1.5.0.post8 / 1.1.4 |
| transformers / datasets | 5.5.0 / 3.6.0 |
| NumPy | 2.5.2 |

### `rocm/jax-training:maxtext-v26.6`

MaxText (JAX) backend. The published tag is the MaxDiffusion-combined image (`/workspace/maxdiffusion` at `68e06965`); MaxText remains at `/workspace/maxtext`.

| | |
| --- | --- |
| Image ID | `a71d8dbb045e` |
| Built | 2026-08-28 |
| Size | 42.6 GB |
| Manifest | `31a3a21d0a37cbd0b5de0342535f65d557ff77ba` |
| Dockerfile | [`Dockerfile.jax-v26.6`](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/docker-release/Dockerfile.jax-v26.6) |

| Software component | Version |
| ------------------ | ------- |
| ROCm | 7.14.0 |
| Python | 3.12.3 |
| JAX / jaxlib | 0.11.0 |
| jax-rocm7-pjrt / jax-rocm7-plugin | 0.11.0.post1 |
| Transformer Engine | 2.17.0+rocm7.14.0.50a84ad |
| hipBLASLt | 1.4.1-cd957402 |
| RCCL | 2.30.4 (built from rocm-systems `9e5e4084`) |
| Flax | 0.12.8 |
| TensorFlow | 2.21.0 (CPU-only, rebuilt from the ROCm fork) |
| Optax / Orbax / Grain / tensorstore | 0.2.8 / 0.12.4 / 0.2.18 / 0.1.85 |
| MaxText | `b47d74bf` (`release/v26.6`) |
| transformers / datasets | 4.57.3 / 4.8.5 |
| NumPy | 2.5.2 |

> **Note:** `transformers` is 4.57.3 because the published image includes the MaxDiffusion stage, which pins transformers back to 4.x (Flax CLIP / T5). That is a step down from v26.5 (`5.9.0`). Transformer Engine is tagged `rocm7.14.0` and matches the image ROCm, unlike v26.5 where the TE wheel was a `rocm7.15` build on ROCm 7.14.0.

### Primus source for v26.6

Use the **`release/v26.6`** branch for both images:

```bash
git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus
git checkout release/v26.6
git submodule update --init --recursive
```

| | |
| --- | --- |
| Branch tip | `2aa05ead` (2026-08-25) |
| Megatron-LM | `d3528a21` |
| TorchTitan | `73a0e697` |
| Megatron Bridge | `9577b128` |
| MaxText | `b47d74bf` |
| Emerging-Optimizers | `93d9eb3a` |
| HummingbirdXT | `ed7b7bd0` |

> **Prefer a `release/v26.6` checkout over the Primus copy baked into the images.**
>
> - `rocm/primus:v26.6` was built from `2aa05ead` (2026-08-25), which is the current `release/v26.6` tip, so the in-image `/workspace/Primus` currently matches. Cloning the branch still keeps you current if later commits land on it.
> - `rocm/jax-training:maxtext-v26.6` was built from `main` at `4d2f7a74` (2026-08-28), because its Dockerfile pins `PRIMUS_BRANCH=main` rather than a commit. Unlike v26.5, its bundled `third_party/maxtext` **does** match `/workspace/maxtext` (`b47d74bf`). Use `release/v26.6` for MaxText training so you get the Primus recipes validated against this image rather than whatever `main` was on the build day.

### Changes since v26.5

`rocm/primus`:

| Component | v26.5 | v26.6 |
| --------- | ----- | ----- |
| ROCm nightly | 7.15.0a20260720 | 7.15.0a20260727 |
| PyTorch | 2.12.0+rocm7.15.0a20260720 | 2.12.0+rocm7.15.0a20260727 |
| Transformer Engine | 2.15.0.dev0+rocm7.15.0a20260716.a07e607 | 2.17.0+rocm7.15.0a20260727.e028a6c |
| Flash Attention | 2.8.3 | 2.8.1 |
| Triton | 3.7.1+git0263a6a6 | 3.8.0+git4cff872c |
| hipBLASLt | 1.4.1-1aa46415 | 1.4.1-bbb68174 |
| Primus-Turbo | 0.3.2.dev48 | 0.4.1.dev26 |
| APEX | 1.14.0a0+rocm7.15.0a20260721 | 1.15.0a0+rocm10.1.0a20260822 |
| transformers | 4.55.0 | 5.5.0 |
| FBGEMM | 2026.7.22 | 2026.8.25 |
| NumPy | 2.5.1 | 2.5.2 |
| Image size | 54.7 GB | 54.0 GB |

`rocm/jax-training:maxtext`:

| Component | v26.5 | v26.6 |
| --------- | ----- | ----- |
| JAX / jaxlib | 0.10.0 | 0.11.0 |
| jax-rocm7-pjrt / jax-rocm7-plugin | 0.10.0+rocm7.14.0 | 0.11.0.post1 |
| Transformer Engine | 2.15.0.dev0+rocm7.15.0a20260707.72d01a0 | 2.17.0+rocm7.14.0.50a84ad |
| Flax | 0.12.2 | 0.12.8 |
| Orbax / Grain / tensorstore | 0.11.39 / 0.2.16 / 0.1.82 | 0.12.4 / 0.2.18 / 0.1.85 |
| MaxText | `a7c6c7e5` | `b47d74bf` |
| transformers | 5.9.0 | 4.57.3 |
| NumPy | 2.0.2 | 2.5.2 |
| Image size | 45.7 GB | 42.6 GB |

> **JAX 0.11.0 still requires Shardy.** Set `shardy=True` during the training run on v26.6. See the [Shardy migration guide](https://docs.jax.dev/en/latest/shardy_jax_migration.html).

---

## v26.5

### `rocm/primus:v26.5`

Megatron-LM, TorchTitan, and Megatron Bridge backends.

| | |
| --- | --- |
| Image ID | `3040bf42974d` |
| Built | 2026-07-22 |
| Size | 54.7 GB |
| Manifest | `8e124a76fbe33cbcc26062f05da3ca5e6419b163` |
| Dockerfile | [`Dockerfile.primus-v26.5`](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/docker-release/Dockerfile.primus-v26.5) |

| Software component | Version |
| ------------------ | ------- |
| ROCm | 7.15.0 (`rocm-sdk` 7.15.0a20260720) |
| Python | 3.12.3 |
| PyTorch | 2.12.0+rocm7.15.0a20260720 |
| Transformer Engine | 2.15.0.dev0+rocm7.15.0a20260716.a07e607 |
| Flash Attention | 2.8.3 |
| hipBLASLt | 1.4.1-1aa46415 |
| Triton | 3.7.1+git0263a6a6.rocm7.15.0a20260720 |
| RCCL | 2.30.4 |
| torchvision | 0.27.0+rocm7.15.0a20260720 |
| torchaudio | 2.11.0+rocm7.15.0a20260721 |
| APEX | 1.14.0a0+rocm7.15.0a20260721 |
| AITER | 0.1.14.post1 |
| Primus-Turbo | 0.3.2.dev48 |
| torchao | 0.15.0+gite9c7bead9 |
| FBGEMM | 2026.7.22 |
| mamba-ssm / causal-conv1d / grouped_gemm | 2.3.1 / 1.5.0.post8 / 1.1.4 |
| transformers / datasets | 4.55.0 / 3.6.0 |
| NumPy | 2.5.1 |

### `rocm/jax-training:maxtext-v26.5`

MaxText (JAX) backend.

| | |
| --- | --- |
| Image ID | `b034a6769b58` |
| Built | 2026-07-17 |
| Size | 45.7 GB |
| Manifest | `5ffe026e7bb969c42ce9f3c9f5c4a3c3164c8a8c` |
| Dockerfile | [`Dockerfile.jax-v26.5`](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/docker-release/Dockerfile.jax-v26.5) |

| Software component | Version |
| ------------------ | ------- |
| ROCm | 7.14.0 |
| Python | 3.12.3 |
| JAX / jaxlib | 0.10.0 |
| jax-rocm7-pjrt / jax-rocm7-plugin | 0.10.0+rocm7.14.0 |
| Transformer Engine | 2.15.0.dev0+rocm7.15.0a20260707.72d01a0 |
| hipBLASLt | 1.4.1-cd957402 |
| RCCL | 2.30.4 (built from rocm-systems `9e5e4084`) |
| Flax | 0.12.2 |
| TensorFlow | 2.21.0 (CPU-only, rebuilt from the ROCm fork) |
| Optax / Orbax / Grain / tensorstore | 0.2.8 / 0.11.39 / 0.2.16 / 0.1.82 |
| MaxText | `a7c6c7e5` (`release/v26.5`) |
| transformers / datasets | 5.9.0 / 4.8.5 |
| NumPy | 2.0.2 |
| amdsmi | 7.0.2 |

> **Note:** The Transformer Engine wheel is tagged `rocm7.15.0a…` even though the image ships ROCm 7.14.0. This is intentional and matches the Dockerfile.

### Primus source for v26.5

Use the **`release/v26.5`** branch for both images:

```bash
git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus
git checkout release/v26.5
git submodule update --init --recursive
```

| | |
| --- | --- |
| Branch tip | `ca7ddda3` (2026-07-30) |
| Megatron-LM | `d3528a21` |
| TorchTitan | `73a0e697` |
| Megatron Bridge | `9577b128` |
| MaxText | `a7c6c7e5` |
| Emerging-Optimizers | `93d9eb3a` |
| HummingbirdXT | `ed7b7bd0` |

> **Do not use the Primus checkout baked into the images.** Both images contain a `/workspace/Primus`, but neither tracks `release/v26.5`:
>
> - `rocm/primus:v26.5` was built from `b511d1b6` (2026-07-22). Three commits have landed on `release/v26.5` since, which bumped the MaxText submodule from `80f431d0` to `a7c6c7e5` and added support for the v26.5 MaxText API.
> - `rocm/jax-training:maxtext-v26.5` was built from `main` at `8b5e8091` (2026-07-17), because its Dockerfile pins `PRIMUS_BRANCH=main` rather than a commit. Its bundled `third_party/maxtext` is `80f431d0`, which does **not** match the `/workspace/maxtext` checkout (`a7c6c7e5`) the image was validated against.
>
> Cloning `release/v26.5` yourself avoids both problems and gives you the MaxText the image was built around.

### Changes since v26.4

`rocm/primus`:

| Component | v26.4 | v26.5 |
| --------- | ----- | ----- |
| ROCm | 7.14.0 | 7.15.0 |
| PyTorch | 2.12.0+rocm7.14.0a20260608 | 2.12.0+rocm7.15.0a20260720 |
| Transformer Engine | 2.14.0.dev0+e6ede467 | 2.15.0.dev0+rocm7.15.0a20260716.a07e607 |
| Triton | 3.7.0+gitb4e20bbe | 3.7.1+git0263a6a6 |
| RCCL | 2.29.7 | 2.30.4 |
| hipBLASLt | 1.4.1-be5adb9b | 1.4.1-1aa46415 |
| AITER | 0.1.12.post2.dev214+gb5e03ed19 | 0.1.14.post1 |
| Primus-Turbo | 0.3.0+3c39ef2 | 0.3.2.dev48 |
| APEX | 1.11.0+rocm7.14.0a20260618 | 1.14.0a0+rocm7.15.0a20260721 |
| NumPy | 2.4.6 | 2.5.1 |
| Image size | 75.9 GB | 54.7 GB |

`rocm/jax-training:maxtext`:

| Component | v26.4 | v26.5 |
| --------- | ----- | ----- |
| JAX / jaxlib | 0.9.1 | 0.10.0 |
| jax-rocm7-pjrt / jax-rocm7-plugin | 0.9.1+rocm7.14.0a20260526 | 0.10.0+rocm7.14.0 |
| Transformer Engine | 2.12.0.dev0+635d7c08 | 2.15.0.dev0+rocm7.15.0a20260707.72d01a0 |
| hipBLASLt | 1.4.0-807283e5 | 1.4.1-cd957402 |
| RCCL | 2.28.9 | 2.30.4 |
| TensorFlow | 2.20.0 | 2.21.0 |
| MaxText | `80f431d0` | `a7c6c7e5` |
| transformers | 5.8.1 | 5.9.0 |
| Image size | 63.6 GB | 45.7 GB |

> **JAX 0.10.0 requires Shardy.** Set `shardy=True` during the training run on v26.5. See the [Shardy migration guide](https://docs.jax.dev/en/latest/shardy_jax_migration.html).

---

## Earlier releases

Headline versions only. Read [the in-image manifest](#verifying-the-stack-in-an-image) for the full stack of any image below — except `v26.2` and `v26.1`, which predate the manifest (use `pip list` there).

The three MaxText v26.3.x images ship an identical software stack; they differ only in MaxText and Primus content.

| Image | Python | ROCm | Framework | Transformer Engine | RCCL |
| ----- | ------ | ---- | --------- | ------------------ | ---- |
| `rocm/primus:v26.4` | 3.12.3 | 7.14.0 | PyTorch 2.12.0+rocm7.14.0a20260608 | 2.14.0.dev0+e6ede467 | 2.29.7 |
| `rocm/primus:v26.3` | 3.12.3 | 7.2.1 | PyTorch 2.10.0+git94c6e04 | 2.12.0.dev0+40434cf6 | 2.27.7 |
| `rocm/primus:v26.2` | 3.12.3 | 7.2.0 | PyTorch 2.10.0a0+git449b176 | 2.8.0.dev0+51f74fa7 | 2.27.7 |
| `rocm/primus:v26.1` | 3.10.12 | 7.1.0 | PyTorch 2.10.0.dev20251112+rocm7.1 | 2.6.0.dev0+f141f34b | 2.27.7 |
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
| `Dockerfile` | the Dockerfile the image was built from (`docker-build-recipe.txt` in `rocm/jax-training:maxtext-v26.6`) |

```bash
docker run --rm --entrypoint bash rocm/primus:v26.6 -c 'cat /workspace/.manifest/requirements.txt'
```

Native library versions are not pip packages; read them from the ROCm headers:

```bash
docker run --rm --entrypoint bash rocm/primus:v26.6 -c '
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
