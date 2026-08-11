# Primus NeMo AutoModel Backend

`nemo_automodel` runs diffusion training through the upstream
[NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) diffusion recipe,
vendored at `third_party/Automodel`. Primus owns config loading and launch;
AutoModel owns model construction, FSDP2, the dataloader, the optimizer and
checkpointing.

Everything Primus adds on top lives in this directory as **hooks**. The
AutoModel submodule and `diffusers` are never patched in place — see
[No-fork hooks](#no-fork-hooks).

## Layout

```text
nemo_automodel/
├── nemo_automodel_adapter.py           config -> AutoModel argument translation
├── nemo_automodel_pretrain_trainer.py  trainer + the hook registry
├── argument_builder.py                 params -> AutoModel YAML
├── quantization/                       SHARED, model-agnostic low precision
│   ├── primus_turbo_fp8.py             FP8/MXFP4 GEMM
│   ├── primus_turbo_fp8_attn.py        FP8 attention
│   └── aiter_bf16_attn.py              non-deterministic bf16 attention
└── models/                             one subpackage per model
    ├── flux/
    │   └── parallelize.py              real AC + FSDP2 strategy
    └── ideogram4/
        ├── adapter.py                  flow-matching adapter
        ├── attention.py                var-len (cu_seqlens) flash attention
        ├── packing_buffer.py           adapter -> attention packing metadata
        ├── parallelize.py              real AC + FSDP2 strategy
        ├── zero1.py                    DDP + ZeRO-1 sharded optimizer
        ├── profile.py                  torch.profiler train-loop wrapper
        ├── processor.py                offline VAE + text-encoder cache builder
        └── data/
            ├── synthetic.py            fixed synthetic batches (smoke)
            └── cache.py                the real pre-encoded cache
```

Two rules keep this navigable:

1. **A model subpackage never imports another's.** Anything two models need goes
   in `quantization/` (or a future `common/`). This is what lets one model be
   reviewed, or upstreamed, on its own.
2. **Model-agnostic code stays out of `models/`.** The low-precision hooks apply
   to any model the recipe can build, so they live at the top level.

## No-fork hooks

Each hook module exposes a single `install()` that monkey-patches an upstream
entry point and returns whether it did anything. `install()` must be:

- **env-gated** — a no-op unless its own flag is set, so a default run is
  unaffected;
- **idempotent** — safe to call twice (check for your own marker attribute
  before patching);
- **additive** — existing behaviour for other models must not change.

`NemoAutomodelPretrainTrainer._install_optional_hooks` imports every hook by
dotted path and calls `install()` before the recipe builds the transformer (that
is, before `set_attention_backend` and the first forward). Import failures are
caught and logged rather than raised, so a missing optional dependency degrades
to a skipped hook instead of a failed run.

That lazy-by-dotted-path import is also why the `__init__.py` files here do not
re-export their modules: importing one model's hooks must not drag in another's
optional dependencies.

> Because hooks patch upstream internals, they are the first thing to break when
> the `third_party/Automodel` pin moves. A hook whose patch target has been
> renamed or deleted upstream fails silently — the registry logs the skip and the
> run continues without it. After a pin bump, force each gate on and confirm
> `install()` still returns true.

## Adding a model

1. Create `models/<name>/` with an `__init__.py` that documents the modules but
   re-exports nothing.
2. Add the hook modules. If a strategy for the model's transformer class is not
   registered in AutoModel's `PARALLELIZATION_STRATEGIES`, note that
   `fsdp.activation_checkpointing` is a **silent no-op** until you register one —
   that is what `parallelize.py` does for FLUX and Ideogram-4.
3. Register the hooks in the trainer's `hooks` tuple, in a group of their own.
4. Add a module preset under `primus/configs/modules/nemo_automodel/` and an
   experiment config under `examples/nemo_automodel/configs/`.

## Naming

Upstream's importable Python package genuinely is `nemo_automodel`
(`import nemo_automodel.components...`), so those imports and any prose about
the upstream project keep that name. It is only the Primus-side namespace that
is ours to choose.
