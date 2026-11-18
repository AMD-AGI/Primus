# Release-docs reference

Supporting detail for the `release-docs-update` skill: page structure and rotation
rules, the Highlights format, the version-reference surface, and the extraction
facts that are easy to get wrong.

## Release notes: structure, rotation, and voice

### Page structure

[docs/01-getting-started/release-notes.md](../../docs/01-getting-started/release-notes.md)
is the single source of truth for image contents; other pages link to it rather
than restating version tables. Order:

1. Intro: the two image families and where each is documented.
2. `## Highlights — vX.Y` (new in this workflow).
3. `## vX.Y (current)` — detailed, with a subsection per image, a "Primus source"
   subsection, and a "Changes since" delta.
4. `## vX.Y-1`, `## vX.Y-2` — same shape, without `(current)`.
5. `## Earlier releases` — one headline row per image.
6. `## Verifying the stack in an image`.
7. `## Related documentation`.

### Rotation

`release_notes.py rotate --version vX.Y --body <file>` performs this. It is not a
manual edit:

- Inserts the new `## vX.Y (current)` section and drops `(current)` from the previous.
- Keeps exactly **three** detailed sections, evicting the rest.
- Is idempotent: a second run replaces the existing section rather than adding one.

With `--family` it merges one family into a release section that may already hold the
other family's (the two images can be weeks apart): that family's image block, its part
of `### Changes since`, and any other block the body carries (`### Primus source`) are
replaced or inserted; the other family's blocks are kept. `--highlights <file>` does the
same for `## Highlights for vX.Y`, whose subsections are `### Both families`,
``### `rocm/primus` — ...`` and ``### `rocm/jax-training:maxtext` — ...``. The golden
replay checks that v26.8 replayed a family at a time gives the page that shipped.

Two things it deliberately leaves to you, and reports as NEXT steps:

- Adding the evicted release's headline row under `## Earlier releases`. Never edit
  the rows already there; those releases are frozen history and some predate the
  in-image manifest.
- Re-pointing any link that referenced the removed section's anchors. Run
  `check_links.py` — rotating v26.4 out orphaned `#primus-source-for-v264` on the
  first run.

The outgoing release's `## Highlights for vX.Y` moves down into its own section as a
`### Highlights for vX.Y` subsection (every heading one level deeper), so the page keeps
its history. `rotate` does this as soon as a newer release is rotated in; it was a hand
edit until v26.8.

### What `check` verifies, and what it allows

For every `### \`rocm/...\`` block with an image-derived snapshot it verifies
`Image ID`, `Built`, `Size`, `Manifest`, the `Dockerfile` filename, and every
software-component row.

A cell only has to **contain** the extracted value, so prose annotations survive:

```
| ROCm | 7.15.0 (`rocm-sdk` 7.15.0a20260727) |
| RCCL | 2.30.4 (built from rocm-systems `9e5e4084`) |
| TensorFlow | 2.21.0 (CPU-only, rebuilt from the ROCm fork) |
| MaxText | `b47d74bf` (`release/v26.6`) |
```

A row whose label has no rule is reported as a warning, not silently passed. If a
release adds a component, add it to `components.py` rather than leaving the row
unverified.

### Extraction facts worth knowing

- **`Manifest` is `/workspace/.manifest/training_docker_version`**, a 40-hex sha1.
  Confirmed against all six v26.4-v26.6 images.
- **Size comes from `docker images`, not `docker image inspect`.** Inspect reports
  content size — 14.3 GB for `rocm/primus:v26.6` against the documented 54.0 GB.
- **The build-time manifest is not always what ships.**
  `rocm/jax-training:maxtext-v26.6` records `transformers 5.14.1` in its manifest
  but installs `4.57.3`, because the MaxDiffusion stage runs after the manifest is
  captured. `probe_image.py` treats the live `pip list` as authoritative and
  records `pip_manifest_divergences`. **Those divergences are exactly what the
  transformers callout in the notes is about** — check them when writing prose.
- **Distribution names move between releases.** Transformer Engine has shipped as
  five different names; the JAX plugin went `jax-rocm7-*` to `jax-rocm10-*` in
  v26.7, and the rendered row label follows whichever the image carries.

## Highlights

Placed at the top of the page, before the first release section. Written from
`output/release-docs/vX.Y/changelog.json`, which buckets commits already.

Buckets, in order, omitting any that are empty:

| Bucket | Contents |
| ------ | -------- |
| New features | `features` — new backends, models, capabilities |
| Performance | `performance` — tuning, kernel work, throughput |
| Bug fixes | `fixes` — correctness and crash fixes |
| Docs and tooling | `docs` + `maintenance` — only if user-visible |
| Dependencies | `dependencies` — usually one summary line, not a list |

Rules:

- Every bullet cites its PR as a link. `check` verifies the numbers exist in
  `changelog.json`, so an invented PR number fails.
- Lead with what changed for a user, not the commit subject. The commit body
  (`## Summary`) is usually the better source.
- The `other` bucket is unclassified — read those and place them by hand.
- Do not list every commit. Maintenance and dependency churn belong in one line
  or nowhere.

Split the highlights per image family whenever the two stacks diverge, with a shared
section for what reaches both. v26.7 needed this: the release was weighted towards the
PyTorch family, and a single merged list implied MaxText gained capability it did not.

### The management email

One email per image family: `output/release-docs/vX.Y/announcement-<family>.md`
(`primus` or `jax`), written to be pasted into an inbox rather than read next to the docs.
It restates the page highlights for a different reader, so it is a derived artifact and
never the source of a version number or of a highlight — take versions from
`data/vX.Y-*.json` like everything else, and put anything new on the page first.

It is not committed. Up to v26.8 the email lived in `tools/release_docs/announcements/` in
this public repository, which is the wrong home for internal mail; it now stays in the
gitignored `output/` folder, and whoever runs the release keeps it with the release's
records (the release agent stores it with its run). Write it as if it were public anyway:
the highlights only, nothing addressed to the next agent, no section recording what was
left out. That reasoning goes to the user and to
`output/release-docs/vX.Y/announcement-notes.md`; see the last editorial rule below.

#### Format

The v26.8 email (below, both families in one file as it was sent then) is the model; copy
its shape, one family per file. A reader who reads only the bold text should still come
away with the release.

```markdown
# vX.Y release highlights: JAX MaxText

Image: `rocm/jax-training:maxtext-vX.Y` (built YYYY-MM-DD).

## JAX MaxText — `rocm/jax-training:maxtext-vX.Y`

- **Upgraded to JAX A.B.C and TransformerEngine D.E** (dev build), on ROCm <version>
- **Pinned MaxText to vX.Y**, with about N upstream commits
  - including <the user-visible fix or capability it brings>
- **<Problem> fixed** with new <XLA / HIP / ...> settings in Primus defaults
  - <what changed, in one line>
- **<Packaging or behaviour change, stated plainly>**
  - <one-line consequence for the user>
```

```markdown
# vX.Y release highlights: PyTorch

Image: `rocm/primus:vX.Y` (built YYYY-MM-DD).

## PyTorch — `rocm/primus:vX.Y`

- **Upgraded to ROCm X and PyTorch Y**, with <build-provenance note, if it changed>
- **New models:** A and B
- **New features**
  - <one feature per line>
- **Performance:** <outcome in one line>; retuned <SKU> recipes
- **Fixes**
  - <bug>, now <resolved state>
```

- **One family per file**, headed with its image tag, under a line that names the image
  and its build date.
- **4-6 top-level bullets per family.** Each opens with a bold lead that makes the point
  on its own: the stack upgrade, the backend version, new models, new features,
  performance, fixes. Omit a lead that has nothing behind it rather than padding it.
- **One idea per line, about one line long.** When a bullet groups several items, the
  items go in sub-bullets, at most four, each a short phrase. A sub-bullet that needs a
  second sentence belongs on the release-notes page instead.
- **Plain language.** Say what the user gets, not how it was built: "faster gradient
  handling", not "skip overwritten grad clears and isolate the gradient reduce-scatter".
  When a technical term is unavoidable, gloss it once: "the GPU's copy engines (SDMA)".
- **Fixes say they are fixed.** "now correct", "detected and worked around" — a bare
  description of the bug reads as an open problem.
- **No per-model performance numbers.** Name the fix, not the gain: "MoE top-k routing
  no longer goes through a slow CUB radix sort", not "about 0.8 s/step on
  DeepSeek-V2-16B". Measured gains stay on the release-notes page.
- **No PR links, file paths or config keys**, except a single parenthetical name a reader
  would search for (`qk_clip`).

The v26.8 email:

````markdown
# v26.8 release highlights

Images: `rocm/primus:v26.8` (built 2026-10-08) and `rocm/jax-training:maxtext-v26.8`
(built 2026-10-01).

## JAX MaxText — `rocm/jax-training:maxtext-v26.8`

- **Upgraded to JAX 0.11.1 and TransformerEngine 2.18** (dev build), on a ROCm 10.2 nightly
- **Pinned MaxText to v26.8**, with about 290 upstream commits
  - including the fix for the FP8 correctness issue
- **Speed regressions from JAX 0.11.1 fixed** with new XLA and HIP settings in Primus defaults
  - MoE top-k routing no longer goes through a slow CUB radix sort
  - Collectives overlap compute again (stream priority restored)
  - Dynamic-slice fusion disabled, so attention's backward pass is no longer recomputed per layer
- **Primus source is no longer bundled** in the image
  - users run it with the matching Primus release branch, which the Primus launcher mounts automatically
- **Your XLA flags are respected** instead of being overridden by Primus defaults

## PyTorch — `rocm/primus:v26.8`

- **Upgraded to ROCm 10.1.0 and PyTorch 2.14**, with TransformerEngine and Flash Attention built from source
- **New models:** ZAYA1-8B and MiniMax-M3
- **New features**
  - Fused FP8/FP4 expert layers for MoE models
  - Distributed-optimizer parameter gather on the GPU's copy engines (SDMA)
  - MuonClip (`qk_clip`) attention-logit clipping for hybrid models
- **Performance:** faster gradient handling and FP4 training; retuned MI355X and MI325X recipes
- **Fixes**
  - A compiler bug that could silently corrupt gradients is now detected and worked around
  - DeepSeek-V4 throughput was over-reported 3.6×; now correct (training was unaffected)
````

#### Editorial rules

Each one a mistake avoided in an earlier release.

- **Every email bullet needs a counterpart on the release-notes page.** The email is a
  restatement; a highlight that exists only in the email has not been checked against
  the changelog and will be missing for users.
- **Check every adjective against the data.** v26.8's draft called the JAX image "leaner"
  because it stopped bundling Primus, but the image grew (43.7 → 43.9 GB). Size, speed and
  memory words need the delta table or a measured number behind them.
- **Name the mechanism correctly or not at all.** v26.8's JAX regression fixes were two
  XLA flags and one HIP environment variable; "new XLA flags" would have been wrong for
  one of the three.

- **"Upgraded to X" requires the version to have moved.** JAX 0.11.0 and TE 2.17.0 were
  rebuilt on ROCm 10.0.0 without changing version, so the upgrade phrasing the v26.6 and
  v26.4 emails used would have been false. Say "built against".
- **Drop deltas that are true but misleading.** v26.7 APEX reads as a downgrade only
  because v26.6 carried a ROCm 10.1 nightly on a 7.15 base.
- **Nothing marked NEEDS CONFIRMATION goes in.** An unverified known issue is worse in an
  exec summary than in a doc, where the marker is at least visible.
- **Report what was held back and why — outside the file.** Tell the user in the Gate B
  summary and write the same list to `output/release-docs/vX.Y/announcement-notes.md`,
  which is gitignored. A judgement that will recur (like "upgraded" needing a version
  change) becomes an editorial rule in this list instead, worded without naming
  anything unreleased. Exclusions made for disclosure reasons are told to the user only
  and never written to a tracked file, this one included.

## Version reference surface

Roughly 120 references across 30+ files, in classes that need different handling.

**Mechanically rewritten** (inside `rewrite_scope`: `README.md`, `docs/`,
`runner/`, `tools/installation*`, `primus/__init__.py`, `pyproject.toml`):

- image tags `rocm/primus:vX.Y`, `rocm/jax-training:maxtext-vX.Y`
- Dockerfile filenames `Dockerfile.{primus,jax}-vX.Y`
- `primus==X.Y.0`, `__version__`
- markdown headings naming the release

**Held for a decision:**

- historical statements ("v26.6 dropped the ROCm git fork", "As of v26.6 …")
- comparisons naming two releases
- `release/vX.Y` when the branch does not exist yet
- `BASE_IMAGE` in `ci.yaml` and `.github/workflows/docker/Dockerfile` — the CI dev
  image, coupled by `tools/ci/check_version_consistency.py`; bumping it changes
  what CI builds against and is a separate decision
- anything unclassified

**Never touched:**

- `.github/workflows/docker-release/**` — records what published images were built
  from
- `docs/01-getting-started/release-notes.md` — rotated, not substituted
- `examples/mlperf/`, `examples/megatron/models/`, `benchmark/`, `tools/docker/` — pin the
  image a result was validated against. The v26.6 release left every one of these
  on v26.5; the golden replay enforces that the tooling does the same.
- False positives that are not versions at all: `26.6GB` in a memory log,
  `rpds-py==2026.6.3`

## Per-backend recipe notes

Each of the three recipe pages opens with "Important notes for vX.Y" holding
required settings, architecture-specific tuning, and known issues. Sources:

- `changelog.json` `fixes` bucket, for issues fixed or newly known.
- The `examples/**` diff, where per-model workarounds live as YAML comments
  (`# fp8 MoE fix (v26.6)`).
- `new_recipes`, for models to add to the "pre-optimized models" list and to
  [model-support-matrix.md](../../docs/06-developer-guide/model-support-matrix.md).

"No issues are currently tracked for vX.Y" is a claim, not a template. Only write
it if the changelog supports it; otherwise ask.

## Regression testing

```bash
python tools/release_docs/golden_replay.py            # replay v26.5 -> v26.6 and the v26.8 rotation, plus invariants
python tools/release_docs/check_links.py              # anchors and relative links
python tools/release_docs/install_parity.py --check   # pins, package set, indexes
```

The golden replay runs today's tooling against the tree as it was before the v26.6
release (version bump, install parity) and before the v26.8 release notes (rotation, for
both families at once and one at a time), and compares the outcome with what shipped. Its gate is asymmetric on
purpose: **over-reach fails** (rewriting a reference the release deliberately kept
is silent corruption), while under-reach passes provided the item is surfaced for
review.

`golden_replay.py --only abbrev` also enforces an invariant the replay cannot see:

**Never read a SHA that git abbreviated for itself.** The width scales with the local
object count — 7 characters in a fresh clone, 8 in a long-lived one — so slicing one
makes the output depend on the machine that produced it. `submodule_bumps` shipped this
bug: `[:8]` of `git diff --raw` returned 7 characters on a clean checkout, where it read
as documentation drift against the full SHAs probed from the image. Ask for
`--abbrev=40` and truncate in the tool. The check scans the tools' string literals for
`%h`, `--short`, `--submodule=short` and any other `--abbrev=`.

**Nothing here runs automatically.** There are no unit tests for this tooling and no CI
step invokes it, both by choice, so these commands are the entire safety net and they
only fire when a person types them. That is why the phase gates say *both must pass*
rather than treating them as advisory: a release that skips them ships numbers nobody
checked, and drift then surfaces in a user's terminal instead of here.

Two consequences to respect when changing a checker. A checker that breaks
*permissively* still reports OK — dropping the fenced-block skip in `check_links.py`
makes headings inside code fences count as anchors, so dead links validate and the run
goes green — and nothing catches a parser that silently matches nothing. So verify a
changed checker still fails on a case you know is broken before trusting a clean run.
