# Primus CLI CI/CD Guide

## Overview

Primus CLI includes automated CI/CD pipelines for both **GitHub Actions** and **GitLab CI**. These pipelines ensure code quality, run tests, and validate changes automatically.

---

## Table of Contents

1. [GitHub Actions](#github-actions)
2. [GitLab CI](#gitlab-ci)
3. [Pipeline Stages](#pipeline-stages)
4. [Local Testing](#local-testing)
5. [Troubleshooting](#troubleshooting)

---

## GitHub Actions

### Configuration

**File**: `.github/workflows/ci.yml`

### Triggers

The pipeline runs on:
- **Push** to `main` or `develop` branches
- **Pull Requests** to `main` or `develop`
- **Manual** workflow dispatch

### Jobs

| Job | Description | Status Required |
|-----|-------------|-----------------|
| **lint** | ShellCheck linting | ✅ Required |
| **test** | Individual test suites | ✅ Required |
| **test-all** | Run all tests | ✅ Required |
| **syntax-check** | Bash syntax validation | ✅ Required |
| **docs-check** | Documentation validation | ⚠️ Optional |
| **version-check** | Version format check | ✅ Required |
| **examples-check** | Validate examples | ✅ Required |
| **integration** | Integration tests | ✅ Required |
| **summary** | Overall status | ℹ️ Info |

### Running Locally

```bash
# Install act (GitHub Actions local runner)
brew install act  # macOS
# or
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run entire workflow
act

# Run specific job
act -j lint
act -j test
```

### Viewing Results

1. Go to **Actions** tab in GitHub repository
2. Click on workflow run to see details
3. Download artifacts for test results

---

## GitLab CI

### Configuration

**File**: `.gitlab-ci.yml`

### Stages

The pipeline has 4 stages:

1. **lint** - Code quality checks
2. **test** - Run test suites
3. **integration** - Integration tests
4. **deploy** - Manual deployment (for releases)

### Jobs

#### Lint Stage
- `shellcheck` - Run ShellCheck on all scripts
- `syntax-check` - Validate bash syntax

#### Test Stage
- `test-common` - Test common library
- `test-validation` - Test validation library
- `test-config` - Test configuration system
- `test-primus-cli` - Test main CLI
- `test-helpers` - Test helper scripts
- `test-all` - Run complete test suite

#### Integration Stage
- `integration-dry-run` - Test CLI commands
- `integration-config` - Test configuration loading
- `docs-check` - Validate documentation
- `version-check` - Check version format
- `coverage` - Generate test coverage

#### Deploy Stage (Manual)
- `deploy-docs` - Deploy documentation
- `deploy-release` - Create release

### Triggers

Runs on:
- **Merge Requests**
- **main** branch
- **develop** branch
- **Tags**

### Viewing Results

1. Go to **CI/CD > Pipelines** in GitLab
2. Click on pipeline to see job details
3. Download artifacts from job pages

### Running Locally

```bash
# Install gitlab-runner
# Ubuntu
curl -L https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh | sudo bash
sudo apt-get install gitlab-runner

# Register runner (optional)
sudo gitlab-runner register

# Execute pipeline locally
gitlab-runner exec docker shellcheck
gitlab-runner exec docker test-all
```

---

## Pipeline Stages

### Stage 1: Linting

**Purpose**: Ensure code quality and style

**Tools**:
- **ShellCheck**: Bash script linter
- **Bash -n**: Syntax validation

**Example Checks**:
```bash
# ShellCheck finds common issues
shellcheck runner/primus-cli

# Syntax check
bash -n runner/primus-cli
```

**Common Issues**:
- SC1091: Not following sources
- SC2164: Use cd ... || exit
- SC2086: Quote variables

**How to Fix**:
```bash
# Install ShellCheck locally
sudo apt-get install shellcheck

# Run before commit
find runner -name "*.sh" | xargs shellcheck
```

---

### Stage 2: Testing

**Purpose**: Validate functionality

**Test Suites**:
1. **common** - Common library functions
2. **validation** - Parameter validation
3. **config** - Configuration system
4. **primus-cli** - Main CLI
5. **helpers** - Helper scripts

**Run All Tests**:
```bash
bash tests/cli/run_all_tests.sh
```

**Run Individual Test**:
```bash
bash tests/cli/test_common.sh
bash tests/cli/test_validation.sh
bash tests/cli/test_config.sh
```

**Test Output**:
```
=========================================
  Primus CLI Common Library Tests
=========================================

[INFO] Test 1: Testing logging functions...
[SUCCESS] ✓ LOG_INFO function exists
[SUCCESS] ✓ LOG_ERROR function exists
[SUCCESS] ✓ LOG_SUCCESS function exists

...

=========================================
  Test Summary
=========================================
Passed: 18
Failed: 0
Total: 18
```

---

### Stage 3: Integration

**Purpose**: Test end-to-end workflows

**Tests**:
- CLI version and help
- Configuration loading
- Dry-run mode
- Documentation completeness
- Version format validation

**Example**:
```bash
# Test CLI works
primus-cli --version
primus-cli --help
primus-cli --show-config

# Test dry-run
primus-cli --dry-run direct -- train pretrain
```

---

### Stage 4: Deploy (Manual)

**Purpose**: Release and documentation deployment

**Jobs**:
- `deploy-docs` - Update documentation site
- `deploy-release` - Create GitHub/GitLab release

**Trigger**: Manual approval required

---

## Local Testing

### Quick Pre-Commit Check

```bash
#!/bin/bash
# scripts/pre-commit.sh

echo "=== Running pre-commit checks ==="

# 1. Syntax check
echo "Checking syntax..."
for script in $(find runner tests -name "*.sh" -o -name "primus-cli"); do
    bash -n "$script" || exit 1
done

# 2. ShellCheck (if available)
if command -v shellcheck &>/dev/null; then
    echo "Running ShellCheck..."
    find runner -name "*.sh" | xargs shellcheck -e SC1091 || exit 1
fi

# 3. Run tests
echo "Running tests..."
export NODE_RANK=0
bash tests/cli/run_all_tests.sh || exit 1

echo "✅ All pre-commit checks passed!"
```

### Install Git Hook

```bash
# Install pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
bash scripts/pre-commit.sh
EOF

chmod +x .git/hooks/pre-commit
```

---

## Troubleshooting

### CI Pipeline Fails on Lint

**Error**: ShellCheck warnings

**Solution**:
```bash
# Run locally
shellcheck runner/primus-cli

# Fix issues or add exceptions
# shellcheck disable=SC1091
source lib/common.sh
```

### CI Pipeline Fails on Test

**Error**: Test assertion fails

**Solution**:
```bash
# Run test locally
export NODE_RANK=0
bash tests/cli/test_common.sh

# Enable debug mode
bash -x tests/cli/test_common.sh
```

### Integration Test Timeout

**Error**: Job times out

**Solution**:
- Check if tests are hanging
- Add timeout to long-running commands
- Use `timeout` command

```bash
# Add timeout
timeout 300 bash tests/cli/run_all_tests.sh
```

### Artifacts Not Uploaded

**Error**: Test logs not available

**Solution**:

**GitHub Actions**:
```yaml
- name: Upload test results
  if: always()  # Upload even on failure
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: tests/cli/*.log
```

**GitLab CI**:
```yaml
artifacts:
  when: always  # Upload even on failure
  paths:
    - tests/cli/*.log
```

---

## Best Practices

### 1. Always Run Tests Locally Before Push

```bash
# Quick check
bash tests/cli/run_all_tests.sh

# Full check (with linting)
bash scripts/pre-commit.sh
```

### 2. Keep Pipelines Fast

- Run tests in parallel
- Cache dependencies
- Skip unnecessary jobs for small changes

### 3. Use Branch Protection

**GitHub**:
- Settings → Branches → Add rule
- Require status checks: `lint`, `test`, `test-all`

**GitLab**:
- Settings → Repository → Protected branches
- Require passing pipeline before merge

### 4. Monitor Pipeline Health

```bash
# GitHub: View workflow runs
https://github.com/your-org/Primus-CLI/actions

# GitLab: View pipelines
https://gitlab.com/your-org/Primus-CLI/-/pipelines
```

### 5. Add Tests for New Features

When adding features:
1. Write tests first (TDD)
2. Run tests locally
3. Ensure CI passes
4. Update documentation

---

## Continuous Deployment

### Automatic Releases

**GitHub Actions** (example):
```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Create Release
        uses: actions/create-release@v1
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
```

**GitLab CI**:
```yaml
release:
  stage: deploy
  script:
    - echo "Creating release..."
  only:
    - tags
  when: manual
```

### Documentation Deployment

**GitHub Pages**:
```yaml
deploy-docs:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs
```

**GitLab Pages**:
```yaml
pages:
  stage: deploy
  script:
    - mkdir -p public
    - cp -r docs/* public/
  artifacts:
    paths:
      - public
  only:
    - main
```

---

## Summary

### CI/CD Features

✅ **Automated Testing** - Run tests on every push
✅ **Code Quality** - Linting and syntax checks
✅ **Integration Tests** - End-to-end validation
✅ **Documentation Validation** - Ensure docs are up-to-date
✅ **Version Checking** - Validate version format
✅ **Manual Deployment** - Controlled releases

### Next Steps

1. **Enable CI/CD** in your repository
2. **Add branch protection** rules
3. **Configure badges** in README
4. **Set up notifications** for failures
5. **Monitor pipeline health** regularly

---

## CI Status Badges

### GitHub Actions

```markdown
![CI Status](https://github.com/your-org/Primus-CLI/workflows/Primus%20CLI%20CI/badge.svg)
```

### GitLab CI

```markdown
[![pipeline status](https://gitlab.com/your-org/Primus-CLI/badges/main/pipeline.svg)](https://gitlab.com/your-org/Primus-CLI/-/commits/main)
```

---

**Version**: 1.2.0
**Last Updated**: November 6, 2025
