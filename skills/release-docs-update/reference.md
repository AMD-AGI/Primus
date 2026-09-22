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

Two things it deliberately leaves to you, and reports as NEXT steps:

- Adding the evicted release's headline row under `## Earlier releases`. Never edit
  the rows already there; those releases are frozen history and some predate the
  in-image manifest.
- Re-pointing any link that referenced the removed section's anchors. Run
  `check_links.py` — rotating v26.4 out orphaned `#primus-source-for-v264` on the
  first run.

The outgoing release's `## Highlights` moves down into its own section as a
`### Highlights` subsection, so the page keeps its history.

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

`tools/release_docs/announcements/vX.Y.md`, 4-6 bullets per family, no PR links, written
to be pasted into an inbox rather than read next to the docs. It restates the page
highlights for a different reader, so it is a derived artifact and never the source of a
version number — take those from `data/vX.Y-*.json` like everything else.

Editorial rules, each one a mistake avoided in v26.7:

- **"Upgraded to X" requires the version to have moved.** JAX 0.11.0 and TE 2.17.0 were
  rebuilt on ROCm 10.0.0 without changing version, so the upgrade phrasing the v26.6 and
  v26.4 emails used would have been false. Say "built against".
- **Drop deltas that are true but misleading.** v26.7 APEX reads as a downgrade only
  because v26.6 carried a ROCm 10.1 nightly on a 7.15 base.
- **Nothing marked NEEDS CONFIRMATION goes in.** An unverified known issue is worse in an
  exec summary than in a doc, where the marker is at least visible.
- **Close with what was held back and why**, so the next release inherits the reasoning
  rather than re-deriving it — including anything excluded for disclosure reasons.

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
- `examples/mlperf/`, `examples/models/`, `benchmark/`, `tools/docker/` — pin the
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
python tools/release_docs/golden_replay.py            # replay v26.5 -> v26.6, plus invariants
python tools/release_docs/check_links.py              # anchors and relative links
python tools/release_docs/install_parity.py --check   # pins, package set, indexes
```

The golden replay runs today's tooling against the tree as it was before the v26.6
release and compares the outcome with what shipped. Its gate is asymmetric on
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
