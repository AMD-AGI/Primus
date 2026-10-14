# Inkling (Megatron backend) — P1 parity runs

P1 checks the Megatron implementation of the Inkling text model against the
official transformers `models/inkling` code (the numerical reference). The
suite covers:

- the config presets, the release/Megatron weight mapping, the checkpoint
  converter and model builds;
- weight initialization statistics;
- the FlexAttention backend against eager, layer by layer;
- forward/backward parity against the reference with both attention backends,
  on a tiny fp32 model and on the real Inkling-Small layers 0–5. Those six
  layers are one full attention period (sliding 0–4, global 5) and include
  both MLP kinds (dense 0–1, MoE 2–5);
- bf16: a training step, and the distance from the fp32 reference compared
  with transformers' own bf16 run, with and without `inkling_fp32_activations`.

## Last run

Tuesday Oct 6, 2026, on one MI355X GPU (`HIP_VISIBLE_DEVICES=0`), spur job
195681 on `crsuse2-m2m-031`.

| Item | Value |
| --- | --- |
| Docker image | `unifiedtrainingdockers.azurecr.io/utd/ci:primus_ci_ac8b18c_20261005` (container `primus_v26.8rc_rocm10.1`) |
| torch | `2.14.0+rocm10.1.0rc3` |
| Container transformers | 5.10.0 (has no Inkling) |
| Reference transformers | 5.19.0.dev0 at commit `263c5eb913849847dcff1885b7b5e387c483f844` |
| Primus base commit | `bc8fd627`, with the Inkling files uncommitted |

### pytest

`50 passed in 100.38s`, with nothing skipped. The slowest tests are
`test_inkling_small_first_layers_match_reference` (69 s, both backends) and
`test_tiny_bf16_training_step_tracks_fp32` (7 s). Expect another 1–2 minutes
on a fresh container while aiter JIT-builds its kernels, plus a few seconds
per new shape while torch.compile builds the FlexAttention kernels.

The real-layer test needs about 110 GB of GPU memory and about 60 GB of host
memory. Each fp32 copy of layers 0–5 is about 110 GB, so the test runs the
reference first, frees it, then builds the Megatron model.

| File | Tests | Needs the reference |
| --- | --- | --- |
| `test_inkling_config.py` | 11 | no |
| `test_inkling_converter.py` | 3 | no |
| `test_inkling_hf_mapping.py` | 9 | no |
| `test_inkling_model.py` | 8 | no |
| `test_inkling_attention.py` | 9 | no |
| `test_inkling_reference.py` | 10 | yes (skips without it) |

### Margin and sensitivity checks

| Check | Result |
| --- | --- |
| Tiny fp32 loss, reference vs Megatron | 6.206737 vs 6.206737 |
| Tiny fp32 logits max abs diff | 1.30e-07 (max abs ref 0.246) |
| Worst gradient relative error | 1.33e-06 (`decoder.layers.1.self_attention.rel_proj`) |
| Router top-k choice per token | identical |
| Inkling-Small layers 0–5 (1536 tokens), logits max abs diff / max abs ref | 1.49e-06 (eager and flex) |
| Inkling-Small layers 0–5, argmax agreement | 1.00000 (eager and flex) |

Zeroing or perturbing one feature must move the tiny model's logits by far more
than the parity error (1.3e-07). If a feature were dead code, the parity test
would still pass, so this rules that out:

| Change | Logits change |
| --- | --- |
| Zero `rel_proj` (both layers) | 1.511e-01 |
| Zero all ShortConv weights | 1.397e-01 |
| Disable log scaling on the global layer | 2.548e-03 |
| Router `global_scale` × 2 | 7.540e-02 |
| Perturb router `shared_weight` | 9.830e-02 |

On the real layers 0–5, the sequence (1536 tokens) runs past both the
512-token sliding window and the 1024-token global relative-bias extent.
Giving any one layer the other layer kind's geometry moves the logits by more
than 37,000 times the parity error (1.56e-06), so the test fails if the
sliding/global mix is wrong:

| Change on the Megatron model | Logits change |
| --- | --- |
| Global layer 5 given a 512 window | 1.870e-01 |
| Global layer 5 given the sliding rel extent (512) | 5.781e-02 |
| Sliding layer 0 given no window | 1.388e-01 |
| Sliding layer 4 given no window | 7.624e-02 |

Log scaling only starts past 128k tokens (`log_scaling_n_floor`), so the real
layers don't exercise it. The tiny model covers it, using a floor of 64 tokens.

### FlexAttention

`inkling_attention_backend: flex` adds the relative bias inside a FlexAttention
`score_mod` and skips masked tiles with a cached block mask, so no `[s, s]`
tensor is stored. On one `InklingCoreAttention` call it matches eager to 2e-5
in fp32 and 2e-2 in bf16, for the output and the gradients of q, k, v, the
relative states and `rel_proj`. That holds on sliding and global layers, with
log scaling active, with and without a padding mask. Flex has no attention
dropout.

Forward + backward of one Inkling-Small attention layer (32 heads, 8 KV heads,
head dim 128, bf16, batch 1), from `~/.inkling_run/flex_bench.py`:

| Tokens | Layer | Eager | Flex |
| --- | --- | --- | --- |
| 4096 | sliding | 47 ms, 8.4 GiB | 4.4 ms, 0.5 GiB |
| 4096 | global | 46 ms, 8.5 GiB | 17 ms, 1.3 GiB |
| 8192 | sliding | 194 ms, 33 GiB | 9.0 ms, 1.0 GiB |
| 8192 | global | 195 ms, 33 GiB | 61 ms, 2.6 GiB |
| 16384 | sliding | 856 ms, 131 GiB | 18 ms, 1.9 GiB |
| 16384 | global | 854 ms, 132 GiB | 299 ms, 5.1 GiB |

The global layers remain the slow part. For scale, plain causal SDPA with no
bias takes 5.7 ms at 8192 tokens.

### bf16 numerics

`inkling_fp32_activations: false` (the default) keeps the SwiGLUs, the
routing-weight products, the expert combine and `global_scale` in the model
dtype, as the transformers model does. `true` runs them in fp32, as Miles does.
`global_scale` is then also stored in fp32 when `PRIMUS_V4_KEEP_FP32=1` (see
`keep_in_fp32.py` for why that's opt-in). The flag needs TE grouped experts
(`use_turbo_grouped_gemm: false`).

The tiny model was run in bf16 from the reference's fp32 weights, for 3 seeds.
The errors are against the fp32 reference. Token error is the per-token max
logits difference over the max reference logit:

| Model | Median token error | Loss error |
| --- | --- | --- |
| transformers bf16 (ShortConvs fp32) | 7.0e-03 – 8.0e-03 | 7e-05 – 1.1e-04 |
| Megatron bf16, flag off | 6.2e-03 – 7.0e-03 | 3e-05 – 9e-05 |
| Megatron bf16, flag on | 6.2e-03 – 7.1e-03 | 3e-05 – 1.0e-04 |

Max errors aren't meaningful in bf16: a few near-tie tokens route to a
different expert, in the official model too, and dominate any max.
`test_tiny_bf16_tracks_reference_as_closely_as_transformers_bf16` requires
both Megatron modes to stay within 1.25× of transformers bf16. On this model
the flag makes no measurable difference, because the bf16 matmuls and
attention dominate the error.

## How to reproduce

### 1. Get a node and start the container

```bash
spur alloc --qos=amd-primus-cicd-qos --job-name=project-primus --nodes=1 \
    --gres=gpu:8 --exclusive --time=10:00:00
srun --pty /bin/bash        # now on the compute node

docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host \
    --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    --privileged -v $HOME:$HOME --shm-size 64G \
    --name primus_v26.8rc_rocm10.1 \
    unifiedtrainingdockers.azurecr.io/utd/ci:primus_ci_ac8b18c_20261005
```

Run from your checkout under `$HOME` (`/home/<user>/Primus`), not from the
image's own `/workspace/Primus`, which doesn't have the Inkling code. Run as
root inside the container: as a normal user, aiter's `rocminfo` probe fails.
Set `PYTHONDONTWRITEBYTECODE=1` so no root-owned `__pycache__` ends up in the
NFS checkout.

### 2. Install the reference and fetch the real layers (once per container)

Both go to the container's `/tmp`, so `/home` (NFS) stays untouched. The
container's transformers stays at 5.10; the pinned build is only added through
`PYTHONPATH`.

```bash
export PYTHONDONTWRITEBYTECODE=1
cd /home/<user>/Primus
pip install --no-deps --no-cache-dir -q --target /tmp/inkling_ref/site \
    "git+https://github.com/huggingface/transformers@263c5eb913849847dcff1885b7b5e387c483f844" \
    "tokenizers>=0.23.1,<0.24"
PYTHONPATH=/tmp/inkling_ref/site python -c \
    "import transformers; from transformers.models.inkling import modeling_inkling; print(transformers.__version__)"
# -> 5.19.0.dev0

# Ranged download of the layer 0-5, embedding and unembedding tensors from
# thinkingmachines/Inkling-Small: 122 tensors, 56.6 GB, about 3.5 min.
python tests/unit_tests/backends/megatron/inkling/fetch_release_layers.py \
    --layers 6 --out /tmp/inkling_ref/small_6L
```

`test_inkling_small_first_layers_match_reference` reads
`/tmp/inkling_ref/small_6L` by default; set `INKLING_SMALL_FIRST_LAYERS` to use
another directory. The test checks that the file holds exactly layers 0–5, so
an older 2-layer fetch fails with a clear message rather than skipping.

### 3. Run the suite

```bash
HIP_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/inkling_ref/site \
    python -m pytest -q -p no:cacheprovider -W ignore -rs --durations=8 \
    tests/unit_tests/backends/megatron/inkling
```

Without `PYTHONPATH=/tmp/inkling_ref/site`, 40 tests pass and the 10 reference
tests skip.

### 4. Margin and sensitivity scripts (optional)

These print the numbers in the margin and sensitivity tables above. They are
scratch scripts in `~/.inkling_run/` (not in the repo), and they set up a
single-rank process group themselves:

```bash
HIP_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/inkling_ref/site python ~/.inkling_run/p1_check.py
HIP_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/inkling_ref/site python ~/.inkling_run/p1_real.py
HIP_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/inkling_ref/site python ~/.inkling_run/p1_bf16.py
HIP_VISIBLE_DEVICES=0 python ~/.inkling_run/flex_bench.py
```

### From the login node

Every step above can also be run without attaching to the container by
wrapping it in `spur exec` and `docker exec`:

```bash
spur exec <jobid> bash -lc 'docker exec -e PYTHONDONTWRITEBYTECODE=1 primus_v26.8rc_rocm10.1 \
    bash -lc "cd /home/<user>/Primus && HIP_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/inkling_ref/site \
    python -m pytest -q -p no:cacheprovider -W ignore tests/unit_tests/backends/megatron/inkling"'
```

## 8 GPUs: parallel layouts

`parallel_reference_check.py` is a `torchrun` script, not a pytest file. It
checks DP, EP, PP and interleaved-PP layouts against the reference in fp64; see
the "Testing" section of
[the model README](../../../../../primus/backends/megatron/core/models/inkling/README.md):

```bash
PYTHONPATH=/tmp/inkling_ref/site torchrun --nproc-per-node 8 \
    tests/unit_tests/backends/megatron/inkling/parallel_reference_check.py
```

It takes about 5 minutes on 8 MI355X GPUs, most of it FlexAttention and
aiter compilation. Last run: Oct 14, 2026, `rocm/primus:v26.8`, 14 of 14
layouts passed.

## Not covered by P1 yet

- Faster global-layer FlexAttention. It's 3× faster than eager but about 10×
  slower than causal SDPA with no bias; this is P2/P3 performance work.
- `inkling_fp32_activations` with Primus turbo grouped GEMM.
- Whether the fp32 path helps over long training; that needs a loss-curve
  comparison in P2.
