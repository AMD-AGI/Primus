---
name: release-docs-update
description: Update the Primus documentation for a monthly training-image release — bump version references, realign the bare-metal install scripts with the release Dockerfiles, record the published image stacks and a Highlights section in the release notes, and refresh the per-backend recipe notes. Use when the user says they are releasing or have released a new `rocm/primus:vNN.N` / `rocm/jax-training:maxtext-vNN.N` pair, asks to update the docs for a release, bump the release version, refresh the release notes, or align `tools/installation*` with a new Dockerfile.
---

# Release docs update

Every monthly release needs the same documentation work: ~120 version references
across 30+ files, two per-image software-stack tables, a delta against the
previous release, the bare-metal install scripts re-pinned to the new
Dockerfiles, and the per-backend "Important notes" blocks rewritten.

**The facts are extracted, not transcribed.** `tools/release_docs/` reads the
published images and renders the tables; you write only prose, and
`release_notes.py check` refuses to let that prose contradict the images. Do not
hand-transcribe a version into a table — if a value is missing, fix the extractor
or the component spec.

## Prerequisites

1. Both images available to `docker` (locally or pullable).
2. Both release Dockerfiles committed under `.github/workflows/docker-release/`.
3. `GIT_SSH_COMMAND` set for `origin`, or `--ssh-key` passed to `preflight.py`.

No `gh` and no GitHub token: squash-merge commits already carry the full PR body,
so the changelog comes from `git log` alone. No release tag or `release/vX.Y`
branch is required either — see "Anchoring" below.

## Workflow

Six phases, three gates. Between gates, run unattended.

### Phase 1 — Snapshot

```bash
python tools/release_docs/preflight.py --version v26.7 --ssh-key <key>
python tools/release_docs/probe_image.py --version v26.7 --family primus
python tools/release_docs/probe_image.py --version v26.7 --family jax
python tools/release_docs/collect_changes.py --version v26.7
```

Create the working branch `dev/doc-update-v26.7` off `main` first. Read the
preflight output before continuing: it reports the release shape, the per-family
build commit and how it was resolved, whether the tag and release branch exist,
and which shape the checkout instructions must take.

`probe_image.py` also diffs the committed Dockerfile against the copy baked into
the image. **A `differs` result is a blocker** — it means the committed Dockerfile
is not what the image was built from, so every pin derived from it is suspect.

### Phase 2 — Version bump

```bash
python tools/release_docs/bump_version.py --from v26.6 --to v26.7            # dry run
python tools/release_docs/bump_version.py --from v26.6 --to v26.7 --apply
python tools/ci/check_version_consistency.py
```

Add `--branch-exists` only when preflight says `release/v26.7` exists on origin.
Without it, `release/vX.Y` references are held rather than rewritten, because
`git checkout release/v26.7` is a broken instruction until the branch is cut.

Collect the held items; do not decide them yourself. Most live in the install
scripts and are handled in Phase 3.

### Phase 3 — Install alignment

```bash
python tools/release_docs/install_parity.py --release v26.7
```

Update the pin blocks (`tools/installation/setup.sh` lines ~67-117,
`tools/installation-jax/setup.sh` lines ~33-74) and the scattered stage pins until
parity is clean. Also update the `(from Dockerfile.<family>-vX.Y)` header and the
`derived_from` manifest line, which is what makes the check self-anchoring.

If a divergence is deliberate, add it to `install_parity_rules.json` under
`documented_divergences` **with a reason**. Never silence drift by deleting a pin.

Expect `install_parity.py --check` to fail between Phase 2 and the end of Phase 3:
Phase 2 moves the `(from Dockerfile.<family>-vX.Y)` header to the new release
while the pins are still the old ones, which is exactly the drift it should
report. It must be clean before Gate A.

### GATE A

Present in one message: the parity table before and after, the bump diff summary,
and the held items as concrete questions. This is the decision point that matters
most — batch it rather than interrupting three times.

### Phase 4 — Hardware validation (not implemented; v2)

Bare-metal installs are still validated by hand. When built, this phase will run
the changed-pin stages, then a full install plus 8-GPU `--train_iters 10`, and
diff the resulting `.manifest/requirements.txt` against the image snapshot with
`install_parity.py --baremetal-manifest`. Phases 5 and 6 do not depend on it.

### Phase 5 — Release notes

```bash
python tools/release_docs/release_notes.py render --version v26.7 --previous v26.6
```

Splice the rendered blocks into [docs/01-getting-started/release-notes.md](../../docs/01-getting-started/release-notes.md),
rotate the sections, and write the `## Highlights` section from
`output/release-docs/v26.7/changelog.json`. See `reference.md` for the rotation
rules, the Highlights buckets, and the editorial voice.

```bash
python tools/release_docs/release_notes.py check
```

**This must pass.** It verifies every stated value against the image snapshots.

### Phase 6 — Recipe notes and new models

Rewrite the "Important notes for vX.Y" block in
[megatron-lm-training.md](../../docs/02-user-guide/megatron-lm-training.md),
[torchtitan-training.md](../../docs/02-user-guide/torchtitan-training.md) and
[jax-maxtext-training.md](../../docs/02-user-guide/jax-maxtext-training.md) from
the `fixes` bucket and the `examples/**` diff in `changelog.json`.

Then propagate `new_recipes` from `changelog.json` into the "pre-optimized
models" list in the Megatron page and the tables in
[model-support-matrix.md](../../docs/06-developer-guide/model-support-matrix.md).
A release that adds models changes no version string, so nothing else catches it.

Anything not traceable to a commit goes on a **"needs your input"** list. Never
invent a known issue or a workaround.

### GATE C

Prose review, then `docs/sphinx/_toc.yml.in`, relative links, `pre-commit run`,
and open the PR against `main`. If `release/v26.7` did not exist at preflight, the
PR description must carry the follow-up: cut the branch, then cherry-pick these
docs onto it (v26.6 did this in [#1160](https://github.com/AMD-AGI/Primus/pull/1160)),
and convert the commit-form checkout instructions to branch form.

## Anchoring: use the build commit

Every range anchors on **the commit each image was built from**, never HEAD, a
tag, or a branch:

- HEAD is wrong. For v26.6 there were 67 commits between the build commit and the
  doc-update commit, so HEAD would credit the release with work not in the image.
- Tags need not exist yet, and are **not on `main`** (`v26.6.0` is not an ancestor
  of main; release tags live on the release-branch lineage).
- `release/vX.Y` often does not exist yet. `release/v26.7` did not exist while
  `v26.7.0` was already tagged.

The two families have **different** build commits: the primus Dockerfile pins
`PRIMUS_BRANCH` to a commit, the JAX one pins `main`, so the JAX build commit can
only come from the image's own `/workspace/Primus`. `preflight.py` resolves both.

## Release shape

The two families are not built in lockstep, and MaxText ships patch releases with
no primus counterpart. `preflight.py` reports `full`, `single-family` or `patch`.
For `single-family`, touch only that family's subsection and keep the run additive
so the second image can be added weeks later without a rewrite.

## Git policy

- Branch `dev/doc-update-vX.Y` off `main`; one commit per phase.
- A dirty tree does not block the run, but **stage only the paths the release
  touches** — never `git add -A`. Submodule gitlinks in particular must not be
  swept in.
- The PR targets `main`. Docs land there first and are cherry-picked to the
  release branch afterwards.

## Never

- Never hand-write a version into a release-notes table. Extract it.
- Never rewrite `.github/workflows/docker-release/**` — those record what
  published images were built from.
- Never bump image tags under `examples/mlperf/`, `examples/models/`,
  `benchmark/` or `tools/docker/`. Those pin the image a result was validated
  against; the v26.6 release deliberately left all of them alone.
- Never resolve a held item by guessing. Ask.
