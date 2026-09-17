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
python tools/release_docs/release_notes.py render --version v26.7 --previous v26.6 > /tmp/v26.7-body.md
# add your callouts and the "Primus source for vX.Y" subsection to that file, then:
python tools/release_docs/release_notes.py rotate --version v26.7 --body /tmp/v26.7-body.md
```

`rotate` owns the structural edit: it inserts the section, demotes the previous
`(current)`, keeps three detailed releases and evicts the rest. **Do not splice by
hand.** The first run did, and put the previous release's image blocks under the new
heading; a stray search-and-replace also reached into the historical sections, which
are deliberately frozen. `rotate` is idempotent, so re-running after an interruption
is safe.

Then write `## Highlights for vX.Y` from `output/release-docs/v26.7/changelog.json`,
add the evicted release's headline row under `## Earlier releases`, and:

```bash
python tools/release_docs/release_notes.py check
python tools/release_docs/check_links.py
```

**Both must pass.** `check` verifies every stated value against the image
snapshots; `check_links.py` catches the anchors that rotation orphans and the
heading renames that break inbound links.

Then write the management email in `tools/release_docs/announcements/vX.Y.md`, using the
most recent release there as the model: 4-6 bullets per image family, no PR links, plain
enough to paste into an inbox. It goes under `tools/` rather than `docs/` because ROCm
documentation sources `docs/` from this repo and this copy is internal.

The page highlights and the email are not the same document. The page explains a change
to someone about to run the image; the email tells a manager what moved. Two rules the
first one earned: **do not write "upgraded to X" unless the version actually changed** —
v26.7 rebuilt JAX 0.11.0 and TE 2.17.0 on ROCm 10 without moving either version, so the
v26.6-style upgrade line would have been false — and **record what you held back** in a
closing section, so the next release inherits the judgement instead of re-deriving it.
Anything marked NEEDS CONFIRMATION in the recipe notes is not email material until it is
confirmed.

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

## Lessons from the first run (v26.7)

Read these before Phase 3; they are the things that actually went wrong.

- **A release can change the wheel indexes and the package set, not just versions.**
  v26.7 moved every index to `stable.repo.amd.com/rocm/*/whl-next`, grew
  Transformer Engine from two distributions to three, and renamed the JAX plugin
  pair. `install_parity.py` now reports `missing packages` and `stale indexes` for
  exactly this; treat both as blocking. Read the Dockerfile's `RUN` lines, not only
  its `ARG`s.
- **Validate the indexes before installing anything.** Fetching each pinned
  package's index page takes under a minute and catches a wrong index or version
  before a multi-hour build. Do this first in Phase 4.
- **Launch installs with a clean environment.** `VENV_DIR`, `WORKSPACE_DIR`,
  `UV_PYTHON_INSTALL_DIR` and `PRIMUS_PIP_CONSTRAINTS` are inherited, and sourcing
  one stack's `env.sh` then installing the other silently redirects the install. The
  scripts now refuse, but launch with `env -u VENV_DIR -u WORKSPACE_DIR
  -u UV_PYTHON_INSTALL_DIR -u PRIMUS_PIP_CONSTRAINTS` regardless.
- **Put `PRIMUS_BASE` on non-`/home` storage.** `/home` is commonly near quota;
  check `df` for a large local filesystem first.
- **Grep the install log for warnings, do not tail it.** A guard warned 24 times
  while an install went into the wrong venv, and tailing missed every one.
- **Verify a highlight's placement against the commit body.** The area
  classification is a file-path heuristic: a packaging fix that happened to touch a
  diffusion file was classified `maxdiffusion` and would have landed under MaxText.

## Never

- Never hand-write a version into a release-notes table. Extract it.
- Never splice or rotate the release notes by hand — use `release_notes.py rotate`.
- Never rewrite `.github/workflows/docker-release/**` — those record what
  published images were built from.
- Never bump image tags under `examples/mlperf/`, `examples/models/`,
  `benchmark/` or `tools/docker/`. Those pin the image a result was validated
  against; the v26.6 release deliberately left all of them alone.
- Never resolve a held item by guessing. Ask.
