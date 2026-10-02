# NeMo AutoModel Diffusion Examples

Text-to-image and text-to-video training through the `nemo_automodel` backend,
which runs [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel)'s diffusion
recipe with Primus patches applied at startup. No AutoModel or diffusers source is
modified; see the [backend README](../../primus/backends/nemo_automodel/README.md)
for how the patches are organised.

Experiments live in [`configs/MI355X/diffusion/`](./configs/MI355X/diffusion/):

| Model | Experiments |
| --- | --- |
| Wan 2.2 T2V A14B | `wan2_2_t2v_a14b-*.yaml` (pretrain, finetune, synthetic, TE FP8) |
| FLUX.1-dev / schnell | `flux_1_dev-pretrain.yaml`, `flux_1_dev-synthetic.yaml`, `flux_1_schnell-synthetic.yaml` |
| Ideogram-4 | `ideogram4-pretrain.yaml` (synthetic), `ideogram4-pretrain-cache.yaml` (encoded cache), `ideogram4-pretrain-ddp.yaml` (DDP + ZeRO-1), `ideogram4-pretrain-ddp-cache.yaml` (DDP + ZeRO-1, encoded cache) |

## Launch

Inside the Primus container, from the repository root:

```bash
USE_HIPBLASLT=1 TORCH_BLAS_PREFER_HIPBLASLT=1 HIP_FORCE_DEV_KERNARG=1 GPU_MAX_HW_QUEUES=6 \
GPUS_PER_NODE=8 \
bash ./runner/primus-cli direct -- train pretrain \
    --config examples/nemo_automodel/configs/MI355X/diffusion/<experiment>.yaml
```

Any config key can be overridden after `--config` as `key=value`, for example
`step_scheduler.max_steps=10`. Each experiment's header lists what it needs
before the first launch (model weights, a dataset, or a cache).

### The AutoModel pin

Primus pins AutoModel to the commit in `primus/_thirdparty.lock`, which is also the
`third_party/Automodel` submodule pointer. That commit requires
`transformers==5.15.1` and, through its `diffusion` extra, `diffusers>=0.39.0`.
Published Primus images may ship no `nemo_automodel` or an older one, so
initialise the submodule before the first launch:

```bash
git submodule update --init third_party/Automodel
```

On each launch the prepare hook compares the importable `nemo_automodel` with
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
| `PRIMUS_SKIP_PIP=1` | Install nothing; stop if `nemo_automodel` is not importable. |

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
| `primus_profiler.enabled` | torch profiler traces of a few steady-state steps, one per rank |
| `primus_turbo.fp8_linear`, `primus_turbo.mxfp4_linear`, `primus_te.mxfp4_linear` | FP8 / MXFP4 linear layers. At most one applies, and it turns on `model.transformer_engine_linear`, the AutoModel setting that performs the swap. MXFP4 is experimental. |
| `primus_turbo.fp8_attention`, `primus_turbo.nondeterministic_attention` | Primus-Turbo kernels behind `model.attention_backend: flash` or `aiter`. No effect with another backend, or with Ideogram-4's variable-length attention. |
| `primus_ideogram4.varlen_attention` | Ideogram-4 variable-length packed attention. Its backward is non-deterministic: the deterministic kernel needs far more memory at image-sized sequences. |
| `primus_ideogram4.ac_every: n` | Ideogram-4: checkpoint every nth block instead of all |
| `primus_ideogram4.zero1` | ZeRO-1 optimizer sharding on the DDP path. Incompatible with `checkpoint.enabled`; tested on a single node only. |

## Ideogram-4 setup

The transformer is randomly initialised from a weightless config directory, so no
transformer weights are needed. Writing that directory needs the diffusers the
pinned AutoModel brings in, so on a fresh image install it first by running the
prepare hook once (a launch runs the same hook):

```bash
PYTHONPATH=. python3 runner/helpers/hooks/train/pretrain/nemo_automodel/prepare.py \
    --primus_path . --data_path ./data \
    --config examples/nemo_automodel/configs/MI355X/diffusion/ideogram4-pretrain.yaml
python tools/nemo_automodel/make_ideogram4_config_dir.py --out <config_dir>
```

Then launch with `model.pretrained_model_name_or_path=<config_dir>`. The cache
experiment also needs an encoded cache, built once on a single GPU. The default
encoders (Qwen3-VL-8B-Instruct and the Ideogram-4 autoencoder) are public.
FLUX.1-dev, used by the FLUX experiment, is the gated one.

```bash
bash ./runner/primus-cli direct --single -- data automodel-cache --model ideogram4 \
    --image-dir /path/to/images --caption-dir /path/to/captions \
    --output-dir /path/to/cache --resolution 256 --max-text-tokens 256
```

A caption longer than `--max-text-tokens` is skipped, not truncated, and the
skip counts are printed at the end. The default is 128, which drops most long
prompts; raise it before treating a small cache as the whole dataset.
Use `ideogram4-pretrain-cache.yaml` for FSDP2, or `ideogram4-pretrain-ddp-cache.yaml` for the DDP/ZeRO-1 path.

Context parallelism (`fsdp.cp_size` above 1) is available on the FSDP2 path. The
degree has to divide both the world size and the attention head count, and it
cannot be combined with `primus_ideogram4.varlen_attention`. The packed sequence
is left-padded so its length is divisible by the degree; a caption plus grid
that is not already a multiple still trains.
