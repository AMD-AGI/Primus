---
name: release-docs-update
description: Update the Primus documentation for a monthly training-image release, before the images are published — record each image's software stack (probed from the CI image that passed QA, under its release tag) and its Dockerfile, bump version references, write the release-notes Highlights and the per-family management email, refresh the per-backend recipe notes, then realign the bare-metal install scripts with the new Dockerfiles and validate them on a GPU node. One or both image families per run. Use when the user says they are releasing a new `rocm/primus:vNN.N` and/or `rocm/jax-training:maxtext-vNN.N`, asks to update the docs for a release, bump the release version, refresh the release notes, or align `tools/installation*` with a new Dockerfile.
---

# Release docs update

Every monthly release needs the same documentation work: ~120 version references
across 30+ files, two per-image software-stack tables, a delta against the
previous release, the bare-metal install scripts re-pinned to the new
Dockerfiles, and the per-backend "Important notes" blocks rewritten.

The docs are written **before the images are published**, from the CI images
that passed QA: each is probed under the tag it will be published as. A retag
keeps the image ID, build date and size, so the tables are those of the
published image.

**The facts are extracted, not transcribed.** `tools/release_docs/` reads the
published images and renders the tables; you write only prose, and
`release_notes.py check` refuses to let that prose contradict the images. Do not
hand-transcribe a version into a table — if a value is missing, fix the extractor
or the component spec.

## Prerequisites

1. **The CI images that passed QA**, one per family in this run, each with the
   tag it will be published as (`rocm/primus:v26.9`,
   `rocm/jax-training:maxtext-v26.9`) and, preferably, its digest.
2. **The Primus commit each image pairs with** — the primus Dockerfile pins one;
   the JAX image ships no Primus, so its commit is passed with `--build-commit`.
3. **A machine with docker for the probe** — a compute node, never a login node
   (`docker run` of a 40-60 GB image). The rest runs anywhere with git.
4. **A GPU node for the bare-metal check** (Phase 6), the same rules.
5. `GIT_SSH_COMMAND` set for `origin`, or `--ssh-key` passed to `preflight.py`.

No published image, no committed release Dockerfile (the probe writes it from the
image), no `gh` and no GitHub token: squash-merge commits already carry the full
PR body, so the changelog comes from `git log` alone. No release tag or
`release/vX.Y` branch is required either — see "Anchoring" below.

## One family or both

The two families are often released weeks apart. Every tool takes `--family`:
a run documents the families given and leaves the other family's tables,
Highlights subsection, recipe page and install script alone. When the second
family follows, its run adds to the same `## vX.Y` section (`rotate --family`
merges into it). When both images are released together, leave `--family` out
and do both in one run. The examples below show one family (`F` = `primus` or
`jax`).

## Workflow

Six phases, two gates. Between gates, run unattended. The install scripts and
the bare-metal check come last: they take hours and do not change the prose, so
they run while Gate A is being answered.

### Phase 1 — Snapshot

Create the working branch `dev/doc-update-v26.9` off `main` first. On a compute
node with docker, probe each image (copy `tools/release_docs/` there if needed;
it is stdlib-only, and bring back `data/` and the Dockerfile):

```bash
python tools/release_docs/probe_image.py --version v26.9 \
    --image <CI tag> --publish-as rocm/primus:v26.9 --expect-digest sha256:<digest> --write-dockerfile
```

`--write-dockerfile` saves the Dockerfile baked into the image
(`/workspace/.manifest/`) as `.github/workflows/docker-release/Dockerfile.primus-v26.9`.
If that file is already committed and **differs**, it is never overwritten and
the probe reports `differs` — **a blocker**: the committed Dockerfile is not what
the image was built from, so every pin derived from it is suspect.

Then, where the clone is:

```bash
python tools/release_docs/preflight.py --version v26.9 --family F --build-commit F=<sha> --ssh-key <key>
python tools/release_docs/collect_changes.py --version v26.9 --family F
```

Read the preflight output before continuing: the release shape, the build commit
per family and how it was resolved, the previous release's build commit (the
start of the changelog range), whether the tag and release branch exist, and
which shape the checkout instructions must take. Preflight needs no docker: it
reads the snapshots.

### Phase 2 — Version bump (docs)

```bash
python tools/release_docs/bump_version.py --from v26.8 --to v26.9 --family F --scope docs           # dry run
python tools/release_docs/bump_version.py --from v26.8 --to v26.9 --family F --scope docs --apply
python tools/ci/check_version_consistency.py
```

Add `--branch-exists` only when preflight says `release/v26.9` exists on origin.
Without it, `release/vX.Y` references are held rather than rewritten, because
`git checkout release/v26.9` is a broken instruction until the branch is cut.

`--scope docs` leaves the install scripts and the bare-metal guides for Phase 5;
`--family` leaves the other family's pages and tags for its own run (both are
counted as "left for ..."). Shared references — the release branch, `primus==`,
`__version__` — move with the first family released.

Collect the held items; do not decide them yourself.

### Phase 3 — Release notes

```bash
python tools/release_docs/release_notes.py render --version v26.9 --family F > /tmp/v26.9-body.md
# add your callouts (and, for the first family, the "Primus source for vX.Y" subsection);
# write the Highlights to /tmp/v26.9-highlights.md, then:
python tools/release_docs/release_notes.py rotate --version v26.9 --family F \
    --body /tmp/v26.9-body.md --highlights /tmp/v26.9-highlights.md
```

`rotate` owns the structural edit: it inserts the section (or merges this
family's blocks into it), demotes the previous `(current)`, moves the previous
release's Highlights into that release's own section, keeps three detailed
releases and evicts the rest. **Do not splice by hand.** The first run did, and
put the previous release's image blocks under the new heading; a stray
search-and-replace also reached into the historical sections, which are
deliberately frozen. `rotate` is idempotent, so re-running after an interruption
is safe.

Write the Highlights from `output/release-docs/v26.9/changelog-F.json`
(subsections `### Both families` and one per family — see reference.md), add the
evicted release's headline row under `## Earlier releases`, and:

```bash
python tools/release_docs/release_notes.py check
python tools/release_docs/check_links.py
```

**Both must pass.** `check` verifies every stated value against the image
snapshots; `check_links.py` catches the anchors that rotation orphans and the
heading renames that break inbound links.

If the other family is not released yet, say so in this section's intro (e.g.
"`rocm/jax-training:maxtext-v26.9` follows; until then the current JAX image is
`maxtext-v26.8`, below") — and remove the sentence in that family's run.

Then write the management email for each family in this run,
`output/release-docs/v26.9/announcement-F.md` (gitignored: it is internal mail
and is not committed), in the format set out in
[reference.md → The management email](reference.md#the-management-email):
4-6 top-level bullets, every bullet led by a bold phrase that makes the point on
its own, sub-bullets for grouped items, plain language, no PR links and no
per-model performance numbers. Write it as if it were public: the highlights
only. No preamble addressed to the next agent, and no "held back" section.

The page highlights and the email are not the same document. The page explains a change
to someone about to run the image; the email tells a manager what moved. Two rules the
first one earned: **do not write "upgraded to X" unless the version actually changed** —
v26.7 rebuilt JAX 0.11.0 and TE 2.17.0 on ROCm 10 without moving either version, so the
v26.6-style upgrade line would have been false — and **report what you held back to the
user, not in the file**. List the judgements in the Gate summaries and in
`output/release-docs/vX.Y/announcement-notes.md`, and turn any judgement that will recur
into an editorial rule in reference.md. Exclusions made for disclosure reasons go to the
user only, never into a tracked file. Anything marked NEEDS CONFIRMATION in the recipe
notes is not email material until it is confirmed.

**No performance numbers in the docs or the email.** Measured throughput belongs
to internal QA reports, not to public pages.

### Phase 4 — Recipe notes and new models

Rewrite the "Important notes for vX.Y" block of the family's recipe pages —
[megatron-lm-training.md](../../docs/02-user-guide/megatron-lm-training.md) and
[torchtitan-training.md](../../docs/02-user-guide/torchtitan-training.md) for
`primus`, [jax-maxtext-training.md](../../docs/02-user-guide/jax-maxtext-training.md)
for `jax` — from the `fixes` bucket and the `examples/**` diff in the changelog.

Then propagate `new_recipes` from the changelog into the "pre-optimized
models" list in the Megatron page and the tables in
[model-support-matrix.md](../../docs/06-developer-guide/model-support-matrix.md).
A release that adds models changes no version string, so nothing else catches it.

Anything not traceable to a commit goes on a **"needs your input"** list. Never
invent a known issue or a workaround. Known issues from the release's QA runs,
when the user provides them, are the input that list asks for.

### GATE A

Present in one message: the held items as concrete questions, the "needs your
input" list, and the prose (release notes, recipe notes, email) for review.
Batch it rather than interrupting three times. **Gate A does not block Phases 5
and 6**: start them, and apply the answers when they come by re-running only the
affected step and its checks. A Gate A without questions folds into Gate B.

### Phase 5 — Install alignment

```bash
python tools/release_docs/bump_version.py --from v26.8 --to v26.9 --family F --scope install --apply
python tools/release_docs/install_parity.py --family F --release v26.9
```

Update the pin blocks (`tools/installation/setup.sh` lines ~67-117,
`tools/installation-jax/setup.sh` lines ~33-74) and the scattered stage pins until
parity is clean, and the bare-metal guides with them. Also update the
`(from Dockerfile.<family>-vX.Y)` header and the `derived_from` manifest line,
which is what makes the check self-anchoring.

If a divergence is deliberate, add it to `install_parity_rules.json` under
`documented_divergences` **with a reason**. Never silence drift by deleting a pin.

`install_parity.py --check` must be clean before Phase 6. Held items from the
bump that live in the install scripts are decided here, in context; the ones you
cannot decide go to Gate B.

### Phase 6 — Bare-metal validation

On a GPU node (never a login node), from a checkout of the Primus commit the
release documents with this branch's install files on top — what a user gets
from `release/vX.Y` once the docs are cherry-picked:

```bash
bash tools/release_docs/validate_baremetal.sh --family F --base /tmp/primus-baremetal-v26.9 \
    --result-dir ~/baremetal-v26.9-F --config examples/<backend>/configs/<GPU>/<model>.yaml ...
```

It installs into a fresh base (clean environment, free space checked, warnings
grepped from the log), diffs the installed packages against the image snapshot
(`parity.json`), and trains each config for 10 steps with
`tools/perf/run_batch.sh` in direct mode. Pass: the install exits 0, nothing the
image has is missing for no documented reason, and every config trains. Suggested
configs: Megatron llama3.1_8B BF16 and FP8, mixtral_8x7B BF16 and TorchTitan
llama3.1_8B for `primus`; MaxText llama2_7B and mixtral_8x7B BF16 for `jax`.
Delete the base afterwards only when everything passed; keep it for debugging
otherwise.

### GATE B

The validation result, `check` / `check_links` / `install_parity --check`
output, and the full diff. After approval: one commit per phase, and the PR
against `main`. If `release/vX.Y` did not exist at preflight, the PR description
must carry the follow-up: cut the branch, then cherry-pick these docs onto it
(v26.6 did this in [#1160](https://github.com/AMD-AGI/Primus/pull/1160)), and
convert the commit-form checkout instructions to branch form. Otherwise the
cherry-pick to `release/vX.Y` follows the merge.

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
`PRIMUS_BRANCH` to a commit; the JAX image ships no Primus from v26.8 on, so its
commit is the Primus commit it is released with, passed as `--build-commit
jax=<sha>`. `preflight.py` resolves both, and the previous release's commit per
family (for a JAX release without one it falls back to the primus build commit
of that release, and says so). `collect_changes.py --family` uses that range.

## Release shape

The two families are not built in lockstep, and MaxText ships patch releases with
no primus counterpart. `preflight.py` reports `full`, `single-family` or `patch`
for the run's scope, and names a family whose snapshot for the version already
exists from an earlier run ("released earlier"); that family is not touched.

## Git policy

- Branch `dev/doc-update-vX.Y` off `main`; one commit per phase. A second family
  released later branches off `main` once the first PR is merged (or off the
  first branch while it is open).
- A dirty tree does not block the run, but **stage only the paths the release
  touches** — never `git add -A`. Submodule gitlinks in particular must not be
  swept in.
- The PR targets `main`. Docs land there first and are cherry-picked to the
  release branch afterwards.

## Lessons from the first run (v26.7)

Read these before Phase 5; they are the things that actually went wrong.

- **A release can change the wheel indexes and the package set, not just versions.**
  v26.7 moved every index to `stable.repo.amd.com/rocm/*/whl-next`, grew
  Transformer Engine from two distributions to three, and renamed the JAX plugin
  pair. `install_parity.py` now reports `missing packages` and `stale indexes` for
  exactly this; treat both as blocking. Read the Dockerfile's `RUN` lines, not only
  its `ARG`s.
- **Validate the indexes before installing anything.** Fetching each pinned
  package's index page takes under a minute and catches a wrong index or version
  before a multi-hour build. Do this first in Phase 6.
- **Launch installs with a clean environment.** `VENV_DIR`, `WORKSPACE_DIR`,
  `UV_PYTHON_INSTALL_DIR` and `PRIMUS_PIP_CONSTRAINTS` are inherited, and sourcing
  one stack's `env.sh` then installing the other silently redirects the install. The
  scripts now refuse, and `validate_baremetal.sh` re-launches itself with them
  unset.
- **Put `PRIMUS_BASE` on non-`/home` storage.** `/home` is commonly near quota;
  `validate_baremetal.sh` checks the free space first (`--min-free-gb`). Some
  nodes' `/tmp` is a small root disk too.
- **Grep the install log for warnings, do not tail it.** A guard warned 24 times
  while an install went into the wrong venv, and tailing missed every one.
  `validate_baremetal.sh` writes them to `install-warnings.txt`.
- **Verify a highlight's placement against the commit body.** The area
  classification is a file-path heuristic: a packaging fix that happened to touch a
  diffusion file was classified `maxdiffusion` and would have landed under MaxText.

## Never

- Never hand-write a version into a release-notes table. Extract it.
- Never splice or rotate the release notes by hand, Highlights included — use
  `release_notes.py rotate`.
- Never put performance numbers in the docs or the email.
- Never probe an image or install on a login node.
- Never rewrite `.github/workflows/docker-release/**` — those record what
  published images were built from. The probe adds the new release's file from
  the image and refuses to overwrite one that exists.
- Never bump image tags under `examples/mlperf/`, `examples/megatron/models/`,
  `benchmark/` or `tools/docker/`. Those pin the image a result was validated
  against; the v26.6 release deliberately left all of them alone.
- Never resolve a held item by guessing. Ask.
