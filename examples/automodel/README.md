# AutoModel Diffusion Examples

Text-to-image and text-to-video training through the `automodel` backend,
which runs [AutoModel](https://github.com/NVIDIA-NeMo/Automodel)'s diffusion
recipe with Primus patches applied at startup. No AutoModel or diffusers source is
modified; see the [backend README](../../primus/backends/automodel/README.md)
for how the patches are organised.

Experiments live in [`configs/MI355X/diffusion/`](./configs/MI355X/diffusion/):

| Model | Experiments |
| --- | --- |
| Wan 2.2 T2V A14B | `wan2_2_t2v_a14b-*.yaml` (pretrain, finetune, synthetic, TE FP8) |
| FLUX.1-dev / schnell | `flux_1_dev-pretrain.yaml`, `flux_1_dev-synthetic.yaml`, `flux_1_schnell-synthetic.yaml` |
| Ideogram-4 | `ideogram4-pretrain.yaml` (synthetic), `ideogram4-pretrain-cache.yaml` (encoded cache), `ideogram4-pretrain-ddp.yaml` (DDP + ZeRO-1), `ideogram4-pretrain-ddp-cache.yaml` (DDP + ZeRO-1, encoded cache) |
| MiniMax-H3 (text-to-audio-video) | `minimax_h3-sft-synthetic.yaml` (full fine-tuning), `minimax_h3-lora-synthetic.yaml` (LoRA); both on a random-weight proxy and synthetic data, see [MiniMax-H3 setup](#minimax-h3-setup) |

## Launch

Inside the Primus container, from the repository root:

```bash
USE_HIPBLASLT=1 TORCH_BLAS_PREFER_HIPBLASLT=1 HIP_FORCE_DEV_KERNARG=1 GPU_MAX_HW_QUEUES=6 \
GPUS_PER_NODE=8 \
bash ./runner/primus-cli direct -- train pretrain \
    --config examples/automodel/configs/MI355X/diffusion/<experiment>.yaml
```

Any config key can be overridden after `--config` as `key=value`, for example
`step_scheduler.max_steps=10`. Each experiment's header lists what it needs
before the first launch (model weights, a dataset, or a cache).

### The AutoModel pin

Primus pins AutoModel to the commit in `primus/_thirdparty.lock`, which is also the
`third_party/Automodel` submodule pointer. That commit requires
`transformers==5.15.1` and, through its `diffusion` extra, `diffusers>=0.39.0`.
The prepare hook also holds diffusers below 0.41: 0.41.0 imports a torchao API
that the image's torchao 0.15 lacks, so no diffusers model would load. These
examples are verified with diffusers 0.39.0 and 0.40.0.
Published Primus images may ship no AutoModel or an older one, so
initialise the submodule before the first launch:

```bash
git submodule update --init third_party/Automodel
```

On each launch the prepare hook compares the importable AutoModel with
the pinned commit. If they differ, or the installed copy's metadata or
dependencies do not match its code, it installs the submodule editable with the
image's ROCm packages (torch, triton, aiter, flash-attn) held at their installed
versions, and logs why and every package version the install moved. If the
container runs as root over a checkout owned by your host user, git refuses the
checkout, so the hook adds the AutoModel checkout to git's global
`safe.directory` and says so.

Environment variables change that (`PRIMUS_SKIP_PIP` is the switch the other
backends' install hooks honour too):

| Variable | Effect |
| --- | --- |
| `AUTOMODEL_REINSTALL=1` | Always install. An explicit `BACKEND_PATH` / `AUTOMODEL_PATH` does too, and takes precedence over `AUTOMODEL_REINSTALL=0`. |
| `AUTOMODEL_REINSTALL=0` | Keep whatever copy is importable, with a warning if it looks stale. |
| `PRIMUS_SKIP_PIP=1` | Install nothing; stop if AutoModel is not importable. |

On a copy other than the pinned commit, a Primus repair that cannot find the
AutoModel code it patches logs a warning and leaves AutoModel's stock behaviour in
place.

## Settings

Training behaviour is set in the YAML: batch sizes, `fsdp.activation_checkpointing`
(`true` or `selective`), `fsdp.reshard_after_forward`, `fsdp.enable_compile`, and
so on. The Primus repairs that make those keys take effect are always on.

Primus features that AutoModel has no setting for are configured in top-level
`primus_*` sections of the module config, either in the experiment's
`overrides:` or on the launch line (`primus_profiler.enabled=true`). Primus
removes these sections before the config reaches AutoModel. Every feature is
off by default, and the module that implements it documents its remaining keys.

| Setting | Effect |
| --- | --- |
| `primus_profiler.enabled` | torch profiler traces of a few steady-state steps, one per rank, written under `diffusion_profile/` in the experiment directory |
| `primus_turbo.fp8_linear`, `primus_turbo.mxfp4_linear`, `primus_te.mxfp4_linear` | FP8 / MXFP4 linear layers. At most one applies, and it turns on `model.transformer_engine_linear`, the AutoModel setting that performs the swap. MXFP4 is experimental. |
| `primus_turbo.fp8_attention`, `primus_turbo.nondeterministic_attention` | Primus-Turbo kernels behind `model.attention_backend: flash` or `aiter`. No effect with another backend, or with Ideogram-4's variable-length attention. |
| `primus_train.defer_host_sync` | Move the train loop's per-step loss and gradient-norm host syncs to logging steps, so the CPU can queue work ahead of the GPU. Logged values are unchanged; between logging steps the loss and gradient norm stay GPU tensors. If AutoModel's loop no longer matches the rewrite, the patch logs an error naming this setting and the loop runs unchanged. |
| `primus_ideogram4.varlen_attention` | Ideogram-4 variable-length packed attention. Its backward is non-deterministic: the deterministic kernel needs far more memory at image-sized sequences. |
| `primus_ideogram4.ac_every: n` | Ideogram-4: checkpoint every nth block instead of all |
| `primus_ideogram4.zero1` | ZeRO-1 optimizer sharding on the DDP path. A ZeRO-1 run cannot save or resume: model checkpoint saving (`checkpoint.enabled`, which defaults to true) must be set to false, or the optimizer build raises. Single-rank and FSDP2 runs keep the plain optimizer, with a warning. Tested on a single node only. |

## Ideogram-4 setup

The transformer is randomly initialised from a weightless config directory, so no
transformer weights are needed. Writing that directory needs the diffusers the
pinned AutoModel brings in, so on a fresh image install it first by running the
prepare hook once (a launch runs the same hook):

```bash
PYTHONPATH=. python3 runner/helpers/hooks/train/pretrain/automodel/prepare.py \
    --primus_path . --data_path ./data \
    --config examples/automodel/configs/MI355X/diffusion/ideogram4-pretrain.yaml
python tools/automodel/make_ideogram4_config_dir.py --out <config_dir>
```

Then launch with `model.pretrained_model_name_or_path=<config_dir>`. The cache
experiment also needs an encoded cache, built once on a single GPU. The text
encoder (Qwen3-VL-8B-Instruct) is public. The autoencoder is read from
`ideogram-ai/ideogram-4-nf4-diffusers`, which is gated on Hugging Face: accept
its terms and set `HF_TOKEN` for the build, or pass `--vae-source` a local copy.
Training reads only the cache and needs neither.

```bash
bash ./runner/primus-cli direct --single -- data automodel-cache --model ideogram4 \
    --image-dir /path/to/images --caption-dir /path/to/captions \
    --output-dir /path/to/cache --resolution 256 --max-text-tokens 256
```

A caption longer than `--max-text-tokens` is skipped, not truncated, and the
skip counts are printed at the end and recorded in the cache's `metadata.json`.
The budget counts the chat-templated tokens the text encoder sees, not only the
caption's own. The default is 128, which drops most long prompts; raise it
before treating a small cache as the whole dataset. A rebuild into the same
`--output-dir` replaces the cache there.

Then launch with `data.dataloader.cache_dir=/path/to/cache` as well.

Use `ideogram4-pretrain-cache.yaml` for FSDP2, or `ideogram4-pretrain-ddp-cache.yaml` for the DDP/ZeRO-1 path.

The presets train on the stock attention path. The variable-length attention
processor is opt-in, because its backward is non-deterministic and it needs
Primus-Turbo. To enable it, and the per-layer compilation it is designed for,
add to the experiment's `overrides:`

```yaml
primus_ideogram4:
  varlen_attention: true
  # precompute_cu_seqlens: true   (the default) the adapter builds the packing on
  #                               the host; false falls back to deriving it from
  #                               the mask in every layer, for A/B comparison only
fsdp:
  enable_compile: true
```

Context parallelism (`fsdp.cp_size` above 1) is available on the FSDP2 path. The
degree has to divide both the world size and the attention head count, and it
cannot be combined with `primus_ideogram4.varlen_attention`. The packed sequence
is left-padded so its length is divisible by the degree; a caption plus grid
that is not already a multiple still trains.

## MiniMax-H3 setup

MiniMax-H3 jointly generates video and stereo audio from text: one 33B-parameter
transformer (50 blocks) denoises a single packed sequence of text, audio and video
rows. Primus trains diffusers' `MiniMaxH3Transformer3DModel` (Apache 2.0) through
AutoModel's fine-tune path, for full fine-tuning and LoRA.

The released weights' license excludes the US, EU, UK and South Korea, so these
experiments train a **random-weight proxy on synthetic data**. Per step that costs
the same as fine-tuning the released weights, so throughput, memory and scaling
carry over; loss values and quality do not. Nothing is read from MiniMax's model
repository. Don't publish the proxy or numbers from it under the MiniMax-H3 name.

### Before the first launch

The transformer needs diffusers 0.40, which the prepare hook installs (the image
ships an older one), so run it once, then write the proxy checkpoint:

```bash
PYTHONPATH=. python3 runner/helpers/hooks/train/pretrain/automodel/prepare.py \
    --primus_path . --data_path ./data \
    --config examples/automodel/configs/MI355X/diffusion/minimax_h3-sft-synthetic.yaml
# Full size: about 132 GB in FP32 (full fine-tuning) or 66 GB with --dtype bfloat16 (LoRA).
python tools/automodel/make_minimax_h3_proxy_checkpoint.py --out /scratch/h3_proxy
# Reduced depth, same width and kernels: --num-layers 12 (8 GPUs) or 4 (one GPU).
export MINIMAX_H3_PROXY=/scratch/h3_proxy
```

Put the proxy on local NVMe: every rank reads it at startup. LoRA needs the
weights on disk, so don't use `--config-only` for it.

### What the presets do

The batch shares one packed layout and one timestep table, so H3 trains at
**micro-batch 1**: raise `step_scheduler.global_batch_size` for gradient
accumulation, not `local_batch_size`. The synthetic loader cycles clip shapes from
`data.dataloader.buckets` (`HEIGHTxWIDTHxFRAMES`; frames of the form 17n + 5, sides
multiples of 32): `768x1344x124` is 5.2 s (about 38K rows), `768x1344x243` is 10 s
(about 74K), `768x1344x362` is 15 s (about 109K).

On top of AutoModel's defaults, the trainer preset and the H3 patches set:

| What | Why |
| --- | --- |
| `primus_turbo.nondeterministic_attention: true`, with `PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32: "1"` in the experiments' `env:` | Attention is about 60% of the step. Primus-Turbo's non-deterministic backward is 17% faster per step than diffusers' default flash backend. FP32 atomics keep dq's error at 1.5e-4 against SDPA; Primus's MI355X default of BF16 atomics is 1.5 s faster per step but gives dq a 1.5e-2 error. Keep the `env:` block if you copy the preset. |
| `fsdp.enable_compile: true` | Compiles each block, fusing the element-wise work: 9% faster, slightly less memory. The first step pays about 13 s to compile. |
| Whole-block activation checkpointing (H3 sidecar) | AutoModel's own checkpointing leaves about 10 GB per block live; without this the full model runs out of memory at step 0. |
| FP32 compute for the keep-in-FP32 modules (H3 sidecar) | The patch projections, time embedder and rotary buffer run in FP32 under FSDP2, as diffusers declares. Full fine-tuning only: AutoModel's LoRA path stores the whole model in BF16. |
| AdaLN modulation gradients accumulated in FP32 (H3 patch) | The per-row modulation gather's backward otherwise sums every row into a few table entries in BF16: the AdaLN gradients (13B of the 33B parameters) were 9.7% off against FP32, and are 0.9% with it. |
| The load's cached memory released after sharding (H3 sidecar) | AutoModel puts the whole model on each GPU before sharding it; releasing the cache keeps that memory from staying reserved through training. |

The adapter logs the unweighted audio loss on its own every 10 micro-steps
(`[PrimusMiniMaxH3] audio_loss=...`), because AutoModel's recipe reports only the
total. Set `flow_matching.adapter_kwargs.audio_loss_log_interval` to change the
interval, or 0 to turn it off.

On **one GPU**, AutoModel skips FSDP2 and with it the cast of FP32 master weights,
and flash attention rejects FP32: use `model.torch_dtype=bfloat16
step_scheduler.global_batch_size=1` and a reduced-depth proxy.

### Measured throughput and memory

8× MI355X, one node, a Primus v26.8 image, the shipped experiments as they are,
768×1344, micro-batch 1, global batch 8. Step time is the median after the first
five steps. Training memory is PyTorch's peak allocation per GPU during training;
the device peak is `rocm-smi`'s per GPU over the whole run, loading included.

| Mode | Blocks | Clip (rows) | Step time | Samples/s/GPU | Training memory | Device peak |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Full fine-tuning | 50 (33B) | 5.2 s (~38K) | 14.9 s | 0.067 | 83 GiB | 127 GiB |
| Full fine-tuning | 50 (33B) | 10 s (~74K) | 45.3 s | 0.022 | 110 GiB | 139 GiB |
| Full fine-tuning | 12 (reduced depth) | 5.2 s (~38K) | 3.6 s | 0.277 | 34 GiB | 89 GiB |
| LoRA (attention + FFN, rank 64) | 50 (33B) | 5.2 s (~38K) | 15.2 s | 0.066 | 45 GiB | 116 GiB |
| LoRA (attention + FFN, rank 64) | 50 (33B) | 10 s (~74K) | 46.0 s | 0.022 | 74 GiB | 112 GiB |
| LoRA (attention + FFN, rank 64) | 12 (reduced depth) | 5.2 s (~38K) | 3.5 s | 0.287 | 24 GiB | 41 GiB |

[exited with code 0]

Most of the gap between the device peak and training memory is loading: AutoModel
puts the whole model on each GPU before sharding it (124 GiB for the full model in
FP32). Step time grows faster than the sequence, because attention's share grows with
length: doubling the clip triples the step. From the 5.2 s and 10 s numbers, a 15 s
clip should need about 133 GiB allocated for full fine-tuning (an estimate, not
measured), which fits on one node without context parallelism.

### Checkpoints, resume and LoRA export

`checkpoint.enabled: true` saves the sharded model, the optimizer, the RNG and the
data loader's position, and at the end a consolidated diffusers transformer
(`model/consolidated/`) that loads with `MiniMaxH3Transformer3DModel.from_pretrained`.
A run resumed from a checkpoint (`checkpoint.restore_from=epoch_0_step_5`)
continues the original run's loss curve.

A LoRA checkpoint is in PEFT's format, which diffusers' LoRA loaders don't read.
Export it to diffusers' `pytorch_lora_weights.safetensors`:

```bash
python tools/automodel/export_minimax_h3_lora.py \
    --checkpoint <checkpoint_dir>/epoch_1_step_11/model --out /scratch/h3_lora
```

Both the consolidated transformer and the exported LoRA load in the xDiT image
(`rocm/pytorch-xdit`).
