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
python -m pytest tests/unit_tests/release_docs/       # parsers and rules
python tools/release_docs/golden_replay.py            # replay v26.5 -> v26.6
python tools/release_docs/check_links.py              # anchors and relative links
python tools/release_docs/install_parity.py --check   # pins, package set, indexes
```

The golden replay runs today's tooling against the tree as it was before the v26.6
release and compares the outcome with what shipped. Its gate is asymmetric on
purpose: **over-reach fails** (rewriting a reference the release deliberately kept
is silent corruption), while under-reach passes provided the item is surfaced for
review.
