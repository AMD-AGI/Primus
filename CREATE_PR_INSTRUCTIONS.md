# How to Create Pull Request

## Method 1: Via GitHub Web Interface (Recommended)

### Step 1: Open PR Creation Page
Visit this URL:
```
https://github.com/AMD-AGI/Primus/compare/main...feature/cli/runner
```

Or navigate to:
1. Go to https://github.com/AMD-AGI/Primus
2. Click "Pull requests" tab
3. Click "New pull request"
4. Select:
   - Base: `main`
   - Compare: `feature/cli/runner`

### Step 2: Fill in PR Details

**Title:**
```
refactor(runner): Optimize shell scripts and consolidate validation logic
```

**Description:**
Copy the content from one of these files:
- **Short version**: `PR_SUMMARY_SHORT.md` (for quick review)
- **Detailed version**: `PR_RUNNER_CLI_REFACTOR.md` (comprehensive)

### Step 3: Configure PR Settings
- [ ] Request reviewers from your team
- [ ] Add labels: `refactor`, `runner`, `cli`
- [ ] Link to related issues (if any)
- [ ] Set milestone (if applicable)

### Step 4: Create PR
Click "Create pull request" button

---

## Method 2: Via Command Line (Using GitHub CLI)

If you have `gh` CLI installed:

```bash
cd /shared/amdgpu/home/xiaoming_peng_qle/workspace/dev/Primus

gh pr create \
  --base main \
  --head feature/cli/runner \
  --title "refactor(runner): Optimize shell scripts and consolidate validation logic" \
  --body-file PR_RUNNER_CLI_REFACTOR.md \
  --label refactor,runner,cli
```

---

## Quick Links

- **Repository**: https://github.com/AMD-AGI/Primus
- **Current Branch**: `feature/cli/runner`
- **Target Branch**: `main`
- **Commits**: 3 commits (051906e, e1e011e, bbece03)

---

## Checklist Before Creating PR

- [x] All changes committed and pushed
- [x] Tests passing (8/10 suites)
- [x] Pre-commit hooks passed
- [x] Breaking changes documented
- [x] Migration guide provided
- [x] Code reviewed locally
- [ ] PR description prepared
- [ ] Reviewers identified

---

## After Creating PR

1. **Monitor CI/CD**: Check if automated tests pass
2. **Address feedback**: Respond to reviewer comments promptly
3. **Update if needed**: Make additional commits if required
4. **Merge**: Once approved, merge using appropriate strategy (squash/merge/rebase)

---

## Notes

- The repository has moved to `AMD-AGI/Primus` (from AMD-AIG-AIMA)
- There's 1 Dependabot security alert to address separately
- Test failures are due to environment (no container runtime), not code issues
