# Environment and XLA flags

How to set environment variables — especially `XLA_FLAGS` — for a training run: which knob to reach for, where it goes, and how to confirm it took effect.

**Related documentation**

| Topic | Location |
| --- | --- |
| YAML structure, presets, merge order | [Configuration system](configuration-system.md) |
| Variable-by-variable reference | [Environment variables](../03-configuration-reference/environment-variables.md) |
| `--env` and `--env_file` launcher flags | [CLI reference](cli-reference.md) |
| Fabric and collective tuning | [Multi-node networking](../04-technical-guides/multi-node-networking.md) |

---

## Which knob do I need?

| I want to… | Use | Example |
| --- | --- | --- |
| Change an ordinary variable | top-level `env:` in the experiment YAML | `XLA_PYTHON_CLIENT_MEM_FRACTION: "0.96"` |
| Change one or two XLA flags | `XLA_FLAGS_APPEND` | `XLA_FLAGS_APPEND: "--xla_gpu_autotune_level=5"` |
| Supply a complete, verified XLA flag set | `XLA_FLAGS` (takes ownership — see below) | the MaxDiffusion example configs |
| Try something for one run only | `--env KEY=VALUE` on the CLI | `--env NCCL_DEBUG=INFO` |

**When in doubt, use `XLA_FLAGS_APPEND`.** It is additive, so you keep the defaults Primus maintains for your backend while still winning on the flag you care about.

---

## The `env:` block

Put `env:` at the **top level** of the experiment YAML, as a sibling of `modules:`:

```yaml
exp_name: my_experiment
workspace: ./output

env:
  XLA_PYTHON_CLIENT_MEM_FRACTION: "0.96"
  XLA_FLAGS_APPEND: "--xla_gpu_first_collective_call_terminate_timeout_seconds=2400"

modules:
  pre_trainer:
    framework: maxtext
    config: pre_trainer.yaml
    model: llama3.3_70B.yaml
```

The runtime applies it before distributed init and before `import jax`, so JAX, XLA, and RCCL all see it (`PrimusRuntime._apply_config_env`).

Three rules:

- **Top level only.** An `env:` block nested inside a module is treated as a training parameter and forwarded to the backend as a CLI argument, which is almost certainly not what you want.
- **Quote the values.** Write `"0.96"` and `"1"`, not `0.96` and `1`, so YAML does not hand the backend a float or an int where a string is expected.
- **Only `$VAR` and `${VAR}` expand, and only if already set.** There is no default syntax here: `${VAR:default}` and bash's `${VAR:-default}` are both passed through **literally**, and an unset `${VAR}` stays literal too.

---

## Why `XLA_FLAGS` needs its own rules

An ordinary variable holds one value, so setting it is unambiguous. `XLA_FLAGS` holds *many* independent settings in one string, and both the Docker image and Primus contribute to it. Two consequences follow:

1. **Setting `XLA_FLAGS` replaces every flag at once**, including defaults you probably want to keep — most importantly `--xla_gpu_autotune_level=4`, which prevents NaN loss on fp8 MoE models.
2. **XLA honors the *last* occurrence of a repeated flag.** This is what makes `XLA_FLAGS_APPEND` work: your flag lands after the defaults and wins, without disturbing the others.

### Layering

Later layers win:

| Layer | Comes from |
| --- | --- |
| Inherited | image `ENV`, or an `export` / `--env` outside the run |
| Backend defaults | `primus/backends/<backend>/env_spec.py` |
| Per-config `env:` | your experiment YAML |
| `XLA_FLAGS_APPEND` | your YAML or the shell |

A config that sets `XLA_FLAGS` **owns** the variable: the backend defaults step aside instead of overriding it. That is deliberate — it is how the MaxDiffusion configs carry a complete, validated flag set — but it also means you give up the managed defaults, including the fp8 autotune fix. Prefer `XLA_FLAGS_APPEND` unless owning the whole string is exactly your intent.

---

## Recipes

**Override one flag, keep the defaults**

```yaml
env:
  XLA_FLAGS_APPEND: "--xla_gpu_autotune_level=5"
```

**Raise the first-collective timeouts** (a slow first step on many nodes)

```yaml
env:
  XLA_FLAGS_APPEND: >-
    --xla_gpu_first_collective_call_warn_stuck_timeout_seconds=300
    --xla_gpu_first_collective_call_terminate_timeout_seconds=2400
```

**Dump HLO** — use the named variables rather than adding `--xla_dump_to` yourself; `env_spec.py` assembles the flag for you

```yaml
env:
  DUMP_HLO: "1"
  DUMP_HLO_DIR: "output/xla_dump_hlo"
```

**Test a flag without editing the config**

```bash
./runner/primus-cli container --env XLA_FLAGS_APPEND="--xla_gpu_autotune_level=5" \
  -- train pretrain --config examples/maxtext/configs/<your-config>.yaml
```

---

## Confirming what took effect

Read it from the log rather than inferring it from the YAML:

| Log line | Meaning |
| --- | --- |
| `[Primus:Runtime] env override: KEY=VALUE` | your `env:` entry was applied |
| `[Primus:<backend>] XLA_FLAGS appended (managed defaults override inherited)` | backend defaults were layered on top of the image's |
| `[Primus:<backend>] XLA_FLAGS comes from the config env: block; managed defaults skipped` | **your config took ownership** — the defaults were not applied |
| `[Primus] XLA_FLAGS_APPEND appended (wins): …` | your additive override was applied |

The third line is the one to watch. It is correct if you meant to supply a complete flag set, and a bug if you only meant to change one flag — in which case move that flag to `XLA_FLAGS_APPEND`.

If the final `XLA_FLAGS` string repeats a flag, the **last** occurrence is the effective one.

---

## Maintainers: changing the shared defaults

Backend defaults live in exactly one file per backend, `primus/backends/<backend>/env_spec.py`. Do not reintroduce environment into `examples/run_pretrain.sh` or the launcher prepare hooks; both were deliberately emptied in favor of that file.

Before adding a flag there, ask:

- **Is it universal?** The shared spec is for values every model on that backend needs. A flag that helps one architecture belongs in that model's config.
- **Is it arch-specific?** Use the `arch=` gate. Only two entries are gated today (`RCCL_WARP_SPEED_AUTO` on gfx950, `HSA_NO_SCRATCH_RECLAIM` on gfx942), and that scarcity is the point: every gate is a behavior difference someone has to debug later.
- **Will users want to tune it?** Give it a named variable the way `XLA_GPU_AUTOTUNE_LEVEL` parameterizes the autotune level, instead of burying the value in the flag string.

Document the **failure the flag prevents**, not just the flag name — a future reader needs the symptom to judge whether the flag is still needed. Managing a flag is also a commitment: config authors can no longer change it without taking ownership of the entire `XLA_FLAGS` string. When you change that contract, add a case to `tests/unit_tests/core/backend/test_env_registry.py`.

---

## Cross-references

- Layering and precedence mechanism: `primus/core/backend/env_registry.py`
- MaxText managed defaults: `primus/backends/maxtext/env_spec.py`
- MaxDiffusion (declares no defaults, so configs own `XLA_FLAGS`): `primus/backends/maxdiffusion/env_spec.py`
- Config `env:` application: `PrimusRuntime._apply_config_env` in `primus/core/runtime/train_runtime.py`

---

[← User guide](README.md)
