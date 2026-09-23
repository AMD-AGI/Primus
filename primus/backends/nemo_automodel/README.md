# NeMo AutoModel backend

Primus backend for [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel)
diffusion pre-training.

## What this backend is

A thin wrapper, in the same shape as the MaxText and TorchTitan backends.
AutoModel owns FSDP2, the dataloader, the optimizer and checkpointing, so the
Primus side does only two things:

1. **Translate config.** `backend_args` (a Primus `SimpleNamespace`) becomes a
   cleaned dict, then a temporary YAML, which AutoModel's own
   `parse_args_and_load_config` loads into its `ConfigNode`. Going through
   AutoModel's loader rather than constructing its objects directly means we
   inherit its config semantics (`_target_`/`_fn` resolution, the
   `wandb.enable` toggle) and stay agnostic to its internals.
2. **Apply patches.** Everything Primus adds on top -- numerics, sharding
   repairs, profiling, per-model wiring -- is a patch, applied after the config
   is materialized and before the recipe builds the model.

Control then passes to `TrainDiffusionRecipe`.

## Adding a model or a feature

Add a `*_patches.py` module under `patches/`. Discovery is automatic: the
package walks its own tree on import, so **you do not edit the trainer, this
README, or any shared list**. That is the point of the mechanism -- a new
feature touches only its own files, so two features in flight do not conflict.

```python
from primus.core.patches import PatchContext, register_patch

@register_patch(
    "nemo_automodel.my_feature",
    backend="nemo_automodel",
    phase="before_train",
    description="one line, shown in the run log",
    condition=_enabled,   # a predicate; the patch is skipped when it is False
    priority=50,          # lower runs first
)
def _apply(ctx: PatchContext) -> None:
    ...
```

Three things are worth knowing before writing one:

- **Registration is global; application is conditional.** Importing the package
  registers every patch, including for jobs training a different model. The
  `condition` predicate is what keeps a patch from acting where it should not,
  so make it precise.
- **Order is `priority`, not import order.** Discovery walks the tree
  alphabetically, which is not a contract. If one patch must precede another,
  say so with `priority`.
- **A failing patch is logged and skipped**, not fatal, so an optional feature
  that breaks degrades the run instead of ending it. If yours must not fail
  quietly, validate loudly in its body.

Patches can read resolved config with
`get_param(ctx, "some.nested.key", default)`, and are conventionally gated on an
environment variable so a default run is unaffected.

Set `PRIMUS_PATCHES=<id>,<id>` to run only named patches, or `PRIMUS_PATCHES=none`
to disable all of them -- useful for bisecting which patch changed a result.

## Testing

`tests/unit_tests/backends/nemo_automodel/` covers backend registration and the
patch mechanism, and does not require the `nemo_automodel` package: the trainer
imports it lazily inside `init()`.
