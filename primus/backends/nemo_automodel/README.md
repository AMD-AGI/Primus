# NeMo AutoModel backend

Primus backend for [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel)
diffusion pre-training.

## What this backend is

A thin wrapper, in the same shape as the MaxText and TorchTitan backends.
AutoModel owns FSDP2, the dataloader, the optimizer and checkpointing, so the
Primus side does only two things:

1. **Apply patches.** Everything Primus adds on top -- numerics, sharding
   repairs, profiling, per-model wiring -- is a patch, applied before the
   config is handed to AutoModel, so a patch can still set an AutoModel key.
2. **Translate config.** `backend_args` (a Primus `SimpleNamespace`) becomes a
   cleaned dict, then a temporary YAML, which AutoModel's own
   `parse_args_and_load_config` loads into its `ConfigNode`. Going through
   AutoModel's loader rather than constructing its objects directly means we
   inherit its config semantics (`_target_`/`_fn` resolution, the
   `wandb.enable` toggle) and stay agnostic to its internals.

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
  so make it precise. `patches/_conditions.transformer_is(...)` gates on the
  configured transformer class.
- **Order is `priority`, not import order.** Discovery walks the tree
  alphabetically, which is not a contract. If one patch must precede another,
  say so with `priority`.
- **A failing patch is logged and skipped**, not fatal, so an optional feature
  that breaks degrades the run instead of ending it. If yours must not fail
  quietly, validate loudly in its body.

Patches can read resolved config with
`get_param(ctx, "some.nested.key", default)`. Two kinds exist:

- **Repairs** restore behaviour the config already asks for (a flag AutoModel
  parses and then drops, say). They are ungated.
- **Opt-in features** (numerics, profiling) are gated on a setting in a
  top-level `primus_<area>:` section of the module config, for example
  `primus_profiler.enabled`. Primus removes `primus_*` sections before the
  config reaches AutoModel. Code that runs outside a patch, such as a kernel
  wrapper, reads the same settings through `options.py`. Add a setting there
  rather than a new environment variable.

Set `PRIMUS_PATCHES=<id>,<id>` to run only named patches, or `PRIMUS_PATCHES=none`
to disable all of them -- useful for bisecting which patch changed a result.

## Staying compatible with AutoModel

AutoModel is pinned in `third_party/Automodel` and moves fast. Each repair is
written so that bumping the pin is safe:

- It hooks a documented extension point where one exists (the per-model
  `ModelParallelizer` sidecar, adapter registries) rather than a private helper.
- It only fills in what upstream left out, so once upstream fixes the same bug
  the repair finds nothing to do and can be deleted without a behaviour change.
- If its hook disappears it logs a warning naming the repair and declines,
  rather than raising.

Each repair's module docstring says what upstream behaviour it compensates for,
which is what to re-check on a bump.

## Testing

`tests/unit_tests/backends/nemo_automodel/` covers backend registration and the
patch mechanism, and does not require the `nemo_automodel` package: the trainer
imports it lazily inside `init()`. Tests marked `requires_automodel` exercise the
repairs against the real pinned AutoModel and are skipped when it is not
installed; run them after bumping the pin.
