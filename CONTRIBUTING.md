# Contributing to Primus

We welcome contributions to **Primus**! To make the process smoother and more organized, please follow the guidelines below.

## Table of Contents

* [Branch Naming Convention](#branch-naming-convention)
* [Commit Message Convention](#commit-message-convention)
* [Pull Request Process](#pull-request-process)
* [Roadmap and Issues](#roadmap-and-issues)

## Branch Naming Convention

Please follow this branch naming convention for all feature and bug fix branches:

```text
<type>/<scope>/<short-description>
```

### Types (type)

| Type       | Purpose                                     |
| ---------- | ------------------------------------------- |
| `feat`     | New feature or functionality                |
| `fix`      | Bug fix                                     |
| `docs`     | Documentation update                        |
| `refactor` | Code refactoring (no functionality change)  |
| `test`     | Tests and test-related changes              |
| `chore`    | Miscellaneous changes (e.g., build scripts) |
| `ci`       | Continuous integration-related changes      |

### Scope (optional)

The scope can be used to specify which part of the project is affected, for example: `engine`, `model`, `scheduler`, `docs`, `tests`, `config`.

### Examples

```text
feat/model/implement-moe-routing
fix/engine/init-error
docs/readme-add-moe-example
refactor/scheduler/remove-dead-code
```

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages. Example:

```text
feat(model): add MOE routing functionality
fix(engine): resolve initialization error
docs(readme): add MOE model example
```

This format helps us to automatically generate changelogs and provide more clarity in versioning.

## Pull Request Process

1. **Fork** the repository (if you don't have write access).
2. Create a **branch** for your work following the branch naming convention described above.
3. Make your changes and **commit** them following the commit message convention.
4. **Push** your changes to your fork or branch in the repository.
5. Create a **pull request** with a clear description of the changes, and reference any related issues. Lint CI runs on every PR; GPU unit tests run after a maintainer adds the `ci:gpu` label.
6. Add appropriate reviewers to the PR.
7. Wait for the review and make any requested changes.
8. Once approved, your PR will be merged.

## Roadmap and Issues

Primus and [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) plan and track work the same way,
with three kinds of issues:

| Kind | Title | Label | Opened by | Closed when |
| ---- | ----- | ----- | --------- | ----------- |
| Roadmap | `[Roadmap] Primus Roadmap H1 2027` | `roadmap` | maintainers, one per half-year, pinned | the next roadmap opens |
| Tracking | `[<Area> Tracking] <what it delivers>` | `tracking` | the owner of an effort with several PRs, or with work in both repositories | its plan is done |
| Feature / bug | `[Feature] …` / `[Bug] …` | `enhancement` / `bug` | anyone | the PR that fixes it merges |

* A roadmap item links the tracking issue or the feature issue that does the work.
* A tracking issue lists its steps in a Plan table and has them as
  [sub-issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/adding-sub-issues),
  in either repository. It lives where "done" is decided: here when done means a Primus recipe uses
  it, in Primus-Turbo when it is a kernel or an op. Whoever lands a PR of the plan updates the Plan
  table. [#1493](https://github.com/AMD-AGI/Primus/issues/1493) is an example.
* One feature issue is one deliverable in one repository. When it is a step of a tracking issue, its
  first line is `Part of #<n>` (`Part of AMD-AGI/Primus-Turbo#<n>` across repositories).
* A PR says `Fixes #<n>` to close an issue, or `Part of #<n>` when it is one step of it.

### Labels

Every issue gets one type label (`bug`, `enhancement`, `performance`, `documentation`, `design`), at
least one `area:` label, a `hw:` label when it concerns one GPU generation, and a priority (`P0`–`P2`)
once triaged. New issues from the templates carry `needs-triage`. The labels are defined in
[`.github/labels.json`](.github/labels.json) and created by the `Sync labels` workflow when that file
changes on `main`: add or change a label there, not in the GitHub UI.

### Milestones

One milestone per release, named after its tag (`v26.10.0`). An issue gets the milestone of the release
it is planned for; at a release, its open issues move to the next milestone or lose it.

### Cadence

* Every week: each `needs-triage` issue gets its labels, an owner and a milestone, or is closed; then
  `needs-triage` comes off.
* At every release: the roadmap issue is updated (release table, ticked items) and the milestone closed.
* Every half-year: a new roadmap issue that links the previous one, which is closed. The
  README's Roadmap section links the open issue labeled `roadmap`, so it needs no edit.

Thank you for contributing!
