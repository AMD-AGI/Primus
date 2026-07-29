---
name: backend-patch-explorer
description: Inventory and explain the patch (monkey-patch) optimizations Primus layers over upstream training backends such as Megatron-LM, TorchTitan, and MaxText, including their version compatibility, dependencies, and Primus-Turbo integration details, by reading the current repository code only. Use when the user asks which patches a backend has, wants a customer-facing patch table, asks how a specific patch or Primus-Turbo feature works (for example deepep, turbo attention, or FP8), or wants a reference guide to port a Primus optimization into their own upstream framework. Read-only; no training or cluster commands.
---

# Backend Patch Explorer

Primus does not fork its upstream training backends (Megatron-LM, TorchTitan, MaxText, and others). Instead it layers **monkey-patches** on top of them: each patch is registered with `@register_patch`, applied at a training-lifecycle `phase` by `run_patches`, and replaces or wraps an upstream function / class / module to add an optimization, a ROCm fix, or an extra config switch.

Read the **current code only** (no training, no cluster commands) to serve two kinds of request:

- **Inventory** - list every patch of one backend as a customer-facing table.
- **Deep-dive + porting** - for one patch or Primus-Turbo feature, explain what it does and produce a reference guide (Primus-side plus Turbo-side integration detail) for porting it into the user's own framework version. This is reference guidance, not a guarantee of a successful port.

The set of patches changes over time, so derive every answer from the code as it exists now and re-scan on each run instead of trusting an earlier list.

## How patches work (enough to navigate)

The engine lives in `primus/core/patches/` (decorator, registry, phases, runner) and contains no patches itself. Each backend's patches live under `primus/backends/<backend>/patches/`.

- **Registration** - `@register_patch("<id>", backend=, phase=, description=, condition=, priority=, backend_versions=, tags=)` stores a `FunctionPatch`. The `id` is the first positional arg; `description` falls back to the handler docstring.
- **Phases** (`primus/core/patches/context.py`): `setup`, `build_args`, `before_train`, `after_train`.
- **Discovery differs per backend** - megatron and maxtext auto-import every `*_patches.py` / `*_patch.py` via `pkgutil`; torchtitan imports modules explicitly in its `patches/__init__.py`. Read that `__init__.py` to confirm what is active.
- **Application** - `run_patches(backend=, phase=)` filters by the `PRIMUS_PATCHES` env var (`all` / `none` / `id1,id2`), then by `applies_to(ctx)` (backend / phase / version / `condition`), sorts by `priority`, and applies each. Patches read config via `get_args(ctx)` / `get_param(ctx, "a.b.c", default)`.
- **Patch styles** (this matters for porting and for the "Patch logic" column):
  - *Binding replacement* - `module.Symbol = PrimusReplacement`.
  - *Function wrapping* - keep `orig = module.fn`, then install a wrapper that calls it.
  - *Config / arg rewrite* - mutate `backend_args` fields instead of any symbol (common for `build_args` patches).
  - *Env var / lifecycle hook* - set an env var, or only act in `setup` / `after_train`; no upstream symbol is replaced.
  - *Source-string rewrite* - `inspect.getsource()` + `exec()` to recompile a modified copy of an upstream function.
  - For the last three styles, the "Patch logic" column may read `N/A (rewrites args / sets env / lifecycle hook)`.

## Workflow

Commands below use `rg` (ripgrep). If `rg` is not installed, the `grep -rn` / `grep -rl` / `find` / Glob equivalents shown in the comments are interchangeable.

### Step 1 - Classify the request and resolve the backend

Decide between **inventory** and **deep-dive + porting**. Do not assume a fixed backend set; list what exists:

```bash
ls -d primus/backends/*/patches 2>/dev/null
```

If the backend is ambiguous, ask the user (megatron is usually the largest and most-asked).

### Step 2 - Discover the patch surface

```bash
# patch modules for one backend (naming convention: *_patches.py / *_patch.py)
rg --files primus/backends/<backend>/patches | rg '(_patches|_patch)\.py$'
# no rg: find primus/backends/<backend>/patches -name '*_patch*.py'

# torchtitan registers explicitly - read what it actually imports
cat primus/backends/torchtitan/patches/__init__.py
```

Subdir names group related patches. Treat these as stable regularities to verify per run, not a fixed list: `args/` holds mostly `build_args` arg-rewrite patches; `turbo/` patches funnel through one shared turbo condition helper; `parallelism/` patches gate on the pipeline flags (`patch_primus_pipeline` / `patch_zero_bubble`); `te_patches/` patches often carry Transformer Engine version gates. A re-export module (no `@register_patch` of its own) registers nothing - confirm by reading it.

### Step 3 - Extract each patch's metadata from source

Pull every registration point and its fields, then open each handler to confirm:

```bash
rg -n "@register_patch\(|backend=|phase=|description=|condition=|priority=" \
   primus/backends/<backend>/patches
# no rg: grep -rn "@register_patch(\|backend=\|phase=\|description=\|condition=\|priority=" \
#          primus/backends/<backend>/patches --include=*.py
```

The decorator gives id / phase / priority / version gates, but NOT the replaced symbol - that lives in the handler body. Grep the bodies for the binding patterns, then open the hits:

```bash
rg -n "= Primus|sys\.modules\[|functools\.partial\(|inspect\.getsource" \
   primus/backends/<backend>/patches
# no rg: grep -rn "= Primus\|sys.modules\[\|functools.partial(\|inspect.getsource" \
#          primus/backends/<backend>/patches --include=*.py
```

For every patch, capture the six inventory columns:

- **Name / id** - first positional arg of `@register_patch`.
- **What it does** - the `description=` text, else the handler docstring.
- **Patch logic** - the upstream symbol the body replaces or wraps; match it to one of the patch styles above (look for `module.Symbol = PrimusX`, `orig = module.fn` + wrapper, `sys.modules[...]= `, `functools.partial(...)`, or `inspect.getsource()` + `exec()`), and follow the handler's imports to the real implementation (often under `primus/backends/<backend>/core/extensions/`, `.../core/transformer/`, or `.../core/pipeline_parallel/`). A handler often rebinds the SAME symbol in more than one module (e.g. both the defining module and a `*_layer_specs` module); capture every `X.Symbol = PrimusX` line in the body, not just the first. If no symbol is replaced, say so per the styles above.
- **Code location** - the patch file, plus the real implementation file when the logic lives elsewhere.
- **How to enable** - the config key checked in `condition`, any version gate, and the `phase`. `condition=` is often a NAMED helper rather than an inline lambda, so follow it: e.g. megatron `turbo/` conditions funnel into `turbo/utils.py::is_primus_turbo_can_patch` (`enable_primus_turbo` AND `tp_size == 1` AND the `primus_turbo` package importable), so a single `use_turbo_*` switch is necessary but not sufficient. No `condition` means "always". Optionally cross-check `primus/configs/modules/<backend>/*.yaml`, but that schema lists only Primus-added keys; flags inherited from upstream Megatron argparse will not appear there.
- **Requires (version / deps)** - version compatibility from `backend_versions` / `primus_versions` and any version gate inside the condition (e.g. a Transformer Engine version check), plus the runtime packages the handler or its implementation imports (e.g. `primus_turbo`, a Transformer Engine version, ROCm-only paths). Follow the handler and implementation imports to confirm.
- **Implicit constraints** - real constraints often live in the implementation, not the `condition`. Grep the impl file and its helpers for `assert` / `raise` (e.g. `assert tp_size == 1`, unsupported-flag guards) and record them; they gate the port even though the `condition` never mentions them.

State only what the current code confirms; never invent a flag, patch, or upstream symbol.

### Step 4 - Locate a specific feature by keyword

For "how do I use feature X" requests (deepep is one such example), locate it live instead of recalling it:

```bash
rg -ni "<keyword>" primus/backends primus/configs   # e.g. deepep, fp8, zero_bubble
# no rg: grep -rni "<keyword>" primus/backends primus/configs --include=*.py
```

Follow the hits to the registering `*_patches.py`, the imported implementation class, the `condition` flag and its constraints, and the YAML schema, then apply Step 3 and the porting format below.

### Step 5 - Accuracy

- Assert only facts confirmed in the current code; re-scan every run.
- If a handler docstring names different flags or behavior than its `condition=` (docstrings drift), the code wins: report what the condition actually checks and flag the mismatch explicitly.
- Optional cross-check: if `docs/backends/<backend>/patch-notes.md` exists, read it as a hint only. It is usually partial (covers a subset of patches, keyed by config flag rather than patch id) and can drift from the code (stale symbol attribution, typos). The code always wins.

## Reaching into Primus-Turbo (dual-side detail)

Many optimizations (attention, FP8, grouped MLP, communication overlap, deepep) are Primus-Turbo integrations. The patch and its extension wrapper only *bind* a Turbo class or function; the real kernel lives in the external `primus_turbo` package, which is a dependency and is usually NOT vendored in this repo. Users mostly want to bring Turbo into their own framework, so cover both sides:

- **Primus side** (always in this repo): the patch installer plus the extension wrapper it imports (commonly under `primus/backends/<backend>/core/extensions/`). Record the adapted interface (constructor args, forward signature) and the call site where the Turbo class is instantiated.
- **Turbo side** (external): resolve the imported symbol to its origin. If the package is importable, locate and read it:

```bash
python3 -c "import primus_turbo, os; print(os.path.dirname(primus_turbo.__file__))" 2>/dev/null
# then read the module that defines the imported class / function
```

  If `primus_turbo` is not installed and its source is not checked out, do not guess its internals. Document the integration boundary instead (the imported Turbo symbol, its expected tensor / dtype contract, and the required package version) and point the user to the Primus-Turbo repository for the kernel implementation.

Read the extension / implementation files statically (open them with a file reader). Do NOT `import` such a module in a Python process that lacks `primus_turbo`: its top-level `import primus_turbo...` raises ImportError even if you only wanted to inspect a class definition.

## Output formats

Produce one of the two shapes below. Fill the placeholders from the scan and keep the structure fixed. Write the output in English by default; use another language only if the user explicitly asks.

### Inventory table

Group rows by the patch subdirectories that actually exist (as discovered in Step 2). Columns are fixed:

| Patch | What it does | Patch logic (upstream symbol replaced/wrapped) | Code location | How to enable | Requires (version / deps) |
|---|---|---|---|---|---|
| `id` | `description` | `UpstreamModule.Symbol` -> Primus impl | `patch_file` (+ `impl_file`) | `flag` / `phase=...` | `backend_versions`; `primus_turbo` / TE ver / ROCm |

Close with a note that `PRIMUS_PATCHES` (`all` / `none` / `id1,id2`) is a global allow-list applied on top of each patch's own flag.

### Deep-dive + porting

1. **What it does** - the optimization and when it applies.
2. **Patch logic** - the exact upstream symbol replaced/wrapped, the Primus replacement, and the `phase` + `condition` gate.
3. **Code location** - patch (installer) file + real implementation file + config schema file. For a Primus-Turbo feature, give both sides: the Primus-side wrapper / extension in this repo and the Turbo-side symbol it wraps (see Reaching into Primus-Turbo above).
4. **Version compatibility and dependencies** - the backend / Primus version ranges the patch supports (`backend_versions` / `primus_versions`), version gates inside the condition (e.g. Transformer Engine), and the exact packages and versions the port needs (e.g. `primus_turbo`, TE, ROCm). State these explicitly so the user can check their own framework version before porting.
5. **Porting into the user's own framework** - an ordered plan covering: which Primus files to copy (installer + implementation + helpers); which upstream symbol to rebind or wrap and where it is constructed / imported (so the patch lands before instantiation); required dependencies and new config keys; whether to monkey-patch directly or reuse Primus's `@register_patch` + `run_patches` engine; and the constraints copied verbatim from `condition` plus the implicit `assert` / `raise` constraints found in Step 3 (version range, parallelism limits, ROCm-only, and so on). Two things that are easy to miss and worth calling out per port:
   - **Complete the copy list by tracing transitive imports** (implementation -> local wrapper -> external package): copy every in-repo file the implementation pulls in, and mark the external package (e.g. `primus_turbo`) as a dependency to install rather than a file to copy.
   - **Check `from <module> import <Symbol>` aliases** in the target: rebinding `module.Symbol` does not affect a name already imported into another module (e.g. a `*_layer_specs` that did `from ... import Symbol`). Grep for such aliases and rebind each one.

## Saving the output

If the user did not ask for the result as a file, deliver it inline in the reply, then end with one short line offering to save it as a document (for example: "Want me to save this as a document?"). If they accept without naming a location, write it to `./` (the current working directory); otherwise use the path they give.
