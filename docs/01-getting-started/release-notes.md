# Release notes

AMD publishes two families of Primus training Docker image:

| Image | Backends | Documented in |
| ----- | -------- | ------------- |
| `rocm/primus:<version>` | Megatron-LM, TorchTitan, Megatron Bridge | [Megatron-LM](../02-user-guide/megatron-lm-training.md), [TorchTitan](../02-user-guide/torchtitan-training.md) |
| `rocm/jax-training:maxtext-<version>` | MaxText (JAX) | [JAX MaxText](../02-user-guide/jax-maxtext-training.md) |

The two families share a version number but are **not** built in lockstep — the same `vNN.N` tag can be weeks apart between families, and the MaxText family also ships patch releases (`v26.3.1`, `v26.3.2`) with no `rocm/primus` counterpart. Always read the section for the exact tag you are running.

**This page is the single source of truth for image contents.** Other pages link here instead of repeating version tables, so there is exactly one place to update per release. If you add a page that names an image tag, link to the relevant section below rather than restating the stack.

Every version below was read out of the published image itself. For v26.4 and v26.5 the values were additionally cross-checked against the release Dockerfiles in [`.github/workflows/docker-release/`](https://github.com/AMD-AGI/Primus/tree/main/.github/workflows/docker-release); earlier releases predate those files and are image-derived only. See [Verifying the stack in an image](#verifying-the-stack-in-an-image) to reproduce any table.

---

## v26.5 (current)

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

## v26.4

### `rocm/primus:v26.4`

| | |
| --- | --- |
| Image ID | `8c8ecc6fe14b` |
| Built | 2026-06-18 |
| Size | 75.9 GB |
| Manifest | `a2d4c7239d874501f864080baf2a8131236b0221` |
| Also published as | `rocm/primus:pytorch-2.12.0-rocm7.14.0a20260608_te-2.14.0.dev0-e6ede467_v26.4` (same image) |
| Dockerfile | [`Dockerfile.primus-v26.4`](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/docker-release/Dockerfile.primus-v26.4) |

| Software component | Version |
| ------------------ | ------- |
| ROCm | 7.14.0 (`rocm-sdk` 7.14.0a20260608) |
| Python | 3.12.3 |
| PyTorch | 2.12.0+rocm7.14.0a20260608 |
| Transformer Engine | 2.14.0.dev0+e6ede467 |
| Flash Attention | 2.8.3 |
| hipBLASLt | 1.4.1-be5adb9b |
| Triton | 3.7.0+gitb4e20bbe.rocm7.14.0a20260608 |
| RCCL | 2.29.7 |
| torchvision | 0.27.0+rocm7.14.0a20260608 |
| torchaudio | 2.11.0+rocm7.14.0a20260617 |
| APEX | 1.11.0+rocm7.14.0a20260618 |
| AITER | 0.1.12.post2.dev214+gb5e03ed19 |
| Primus-Turbo | 0.3.0+3c39ef2 |
| torchao | 0.15.0+gite9c7bead9 |
| FBGEMM | 2026.6.18 |
| mamba-ssm / causal-conv1d / grouped_gemm | 2.3.1 / 1.5.0.post8 / 1.1.4 |
| transformers / datasets | 4.55.0 / 3.6.0 |
| NumPy | 2.4.6 |

### `rocm/jax-training:maxtext-v26.4`

| | |
| --- | --- |
| Image ID | `9b263b38c159` |
| Built | 2026-05-27 |
| Size | 63.6 GB |
| Manifest | `b0b705ce0533ead5490070daa402ca625f1489b5` |
| Also published as | `rocm/jax-training:maxtext-v26.4-jax0.9.1-te2.12.0` (same image) |
| Dockerfile | [`Dockerfile.jax-v26.4`](https://github.com/AMD-AGI/Primus/blob/main/.github/workflows/docker-release/Dockerfile.jax-v26.4) |

| Software component | Version |
| ------------------ | ------- |
| ROCm | 7.14.0 |
| Python | 3.12.3 |
| JAX / jaxlib | 0.9.1 |
| jax-rocm7-pjrt / jax-rocm7-plugin | 0.9.1+rocm7.14.0a20260526 |
| Transformer Engine | 2.12.0.dev0+635d7c08 |
| hipBLASLt | 1.4.0-807283e5 |
| RCCL | 2.28.9 |
| Flax | 0.12.2 |
| TensorFlow | 2.20.0 (CPU-only) |
| Optax / Orbax / Grain / tensorstore | 0.2.8 / 0.11.39 / 0.2.16 / 0.1.82 |
| MaxText | `80f431d0` |
| transformers / datasets | 5.8.1 / 4.8.5 |
| NumPy | 2.0.2 |
| amdsmi | 7.0.2 |

### Primus source for v26.4

`rocm/primus:v26.4` bakes Primus at `236cfa9d`, which is the commit the published v26.4 benchmark recipes pin:

```bash
git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus
git checkout 236cfa9d
git submodule update --init --recursive
```

---

## Earlier releases

Headline versions only. Read [the in-image manifest](#verifying-the-stack-in-an-image) for the full stack of any image below — except `v26.2` and `v26.1`, which predate the manifest (use `pip list` there).

The three MaxText v26.3.x images ship an identical software stack; they differ only in MaxText and Primus content.

| Image | Python | ROCm | Framework | Transformer Engine | RCCL |
| ----- | ------ | ---- | --------- | ------------------ | ---- |
| `rocm/primus:v26.3` | 3.12.3 | 7.2.1 | PyTorch 2.10.0+git94c6e04 | 2.12.0.dev0+40434cf6 | 2.27.7 |
| `rocm/primus:v26.2` | 3.12.3 | 7.2.0 | PyTorch 2.10.0a0+git449b176 | 2.8.0.dev0+51f74fa7 | 2.27.7 |
| `rocm/primus:v26.1` | 3.10.12 | 7.1.0 | PyTorch 2.10.0.dev20251112+rocm7.1 | 2.6.0.dev0+f141f34b | 2.27.7 |
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
| `Dockerfile` | the Dockerfile the image was built from |

```bash
docker run --rm --entrypoint bash rocm/primus:v26.5 -c 'cat /workspace/.manifest/requirements.txt'
```

Native library versions are not pip packages; read them from the ROCm headers:

```bash
docker run --rm --entrypoint bash rocm/primus:v26.5 -c '
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
