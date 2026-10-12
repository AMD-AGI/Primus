# Contributing to Primus

We welcome contributions to **Primus**! To make the process smoother and more organized, please follow the guidelines below.

## Table of Contents

* [Branch Naming Convention](#branch-naming-convention)
* [Commit Message Convention](#commit-message-convention)
* [Pull Request Process](#pull-request-process)

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

Thank you for contributing!
