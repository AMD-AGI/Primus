# PR Description 自动归档使用说明

## ✅ 已实现功能

在 `.github/workflows/ci.yaml` 中添加了 **PR Description 自动归档** 功能。

### 功能说明

每当有 Pull Request 触发 CI 时，会自动：

1. **提取 PR 信息**：
   - PR 编号、标题、作者
   - PR URL、分支信息
   - PR 描述内容

2. **生成归档文件**：
   - 保存位置：`/apps/tas/0_public/primus_k8s_ci/pr_archive/PR_{NUMBER}.md`
   - 格式：Markdown
   - 包含完整的 PR 描述和 CI 测试结果

3. **仅在 PR 触发时执行**：
   - `if: github.event_name == 'pull_request'`
   - push 到 main 分支时不会执行

## 📁 归档文件格式

生成的文件示例：

```markdown
# Pull Request #123

**Title:** feat(runner): add flexible hook execution system

**Author:** [@username](https://github.com/username)

**URL:** https://github.com/AMD-AIG-AIMA/Primus/pull/123

**Branch:** `feature/hooks` → `main`

**CI Status:** ✅ All Tests Passed

**Archived:** 2025-11-13 10:30:45

---

## Description

Add a flexible hook execution system that allows injecting custom Bash and
Python scripts at various stages of command execution.

Features:
- Hook discovery and execution
- Support for .sh and .py files
- Comprehensive test suite

---

## CI Test Results

- ✅ **Code Lint**: Passed
- ✅ **Unit Tests**: Passed
- ✅ **Shell Tests**: Passed (126 tests)

---

_This description was automatically archived by CI workflow._
```

## 🔍 查看归档的 PR 描述

### 在 CI 服务器上：

```bash
# 查看所有归档的 PR
ls -lh /apps/tas/0_public/primus_k8s_ci/pr_archive/

# 查看特定 PR 的描述
cat /apps/tas/0_public/primus_k8s_ci/pr_archive/PR_123.md

# 搜索特定关键词的 PR
grep -r "hook execution" /apps/tas/0_public/primus_k8s_ci/pr_archive/

# 查看最近的 5 个 PR
ls -lt /apps/tas/0_public/primus_k8s_ci/pr_archive/ | head -6
```

### 在 GitHub Actions 日志中：

1. 进入 **Actions** 标签
2. 选择相应的 workflow run
3. 展开 **"Archive PR Description"** 步骤
4. 查看 PR 描述预览（显示前 25 行）

## 📊 CI 执行流程

```
PR 创建/更新
    ↓
触发 CI workflow
    ↓
code-lint job
    ↓
run-unittest job
    ├─ Install Primus
    ├─ Run Unit Tests
    ├─ Run Shell Tests
    ├─ Archive PR Description ← 新增步骤
    └─ Clean
```

## 🎯 高级用法

### 选项 1：添加到 Git Commit

如果需要将 PR description 作为 commit 提交到仓库，可以修改步骤：

```yaml
- name: Archive PR Description to Repo
  if: github.event_name == 'pull_request'
  run: |
    # ... (获取 PR 信息) ...

    # 保存到仓库目录
    mkdir -p docs/pr_archive
    cat > docs/pr_archive/PR_${PR_NUMBER}.md << EOFMARKER
    # ... (PR 信息) ...
    EOFMARKER

    # 配置 git
    git config user.name "GitHub Actions Bot"
    git config user.email "actions@github.com"

    # 提交
    git add docs/pr_archive/
    git commit -m "docs: archive PR #${PR_NUMBER} description"

    # 推送到特定分支
    git push origin HEAD:pr-descriptions
```

### 选项 2：发送到数据库

```yaml
- name: Save PR Description to Database
  if: github.event_name == 'pull_request'
  run: |
    PR_NUMBER="${{ github.event.pull_request.number }}"
    PR_TITLE="${{ github.event.pull_request.title }}"
    PR_BODY="${{ github.event.pull_request.body }}"

    # 使用 curl 发送到 API
    curl -X POST https://your-api.com/pr-archive \
      -H "Content-Type: application/json" \
      -d "{
        \"pr_number\": ${PR_NUMBER},
        \"title\": \"${PR_TITLE}\",
        \"body\": \"${PR_BODY}\",
        \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
      }"
```

### 选项 3：上传到 Artifact

```yaml
- name: Archive PR Description
  if: github.event_name == 'pull_request'
  run: |
    # ... (生成 PR 描述文件) ...
    mkdir -p pr_metadata
    cp ${{ env.PRIMUS_WORKDIR }}/pr_archive/PR_${PR_NUMBER}.md pr_metadata/

- name: Upload PR Description Artifact
  if: github.event_name == 'pull_request'
  uses: actions/upload-artifact@v4
  with:
    name: pr-description-${{ github.event.pull_request.number }}
    path: pr_metadata/
    retention-days: 90
```

### 选项 4：添加 PR 评论

```yaml
- name: Comment PR with Archive Link
  if: github.event_name == 'pull_request'
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: |
    PR_NUMBER="${{ github.event.pull_request.number }}"

    gh pr comment ${PR_NUMBER} --body "✅ **CI Tests Passed!**

Your PR description has been archived at:
\`/apps/tas/0_public/primus_k8s_ci/pr_archive/PR_${PR_NUMBER}.md\`

All tests completed successfully! 🎉"
```

## 🔧 自定义配置

### 修改归档目录

在 `ci.yaml` 中修改：

```yaml
# 修改前
mkdir -p ${{ env.PRIMUS_WORKDIR }}/pr_archive

# 修改后（使用自定义目录）
mkdir -p /custom/path/pr_archive
```

### 添加更多信息

在生成的 Markdown 中添加：

```yaml
cat > ... << EOFMARKER
# Pull Request #${PR_NUMBER}

**Title:** ${PR_TITLE}

**Commits:** ${{ github.event.pull_request.commits }}

**Changed Files:** ${{ github.event.pull_request.changed_files }}

**Additions:** +${{ github.event.pull_request.additions }}

**Deletions:** -${{ github.event.pull_request.deletions }}

**Labels:** ${{ join(github.event.pull_request.labels.*.name, ', ') }}

...
EOFMARKER
```

## 📈 查看统计

```bash
# 统计归档的 PR 数量
ls /apps/tas/0_public/primus_k8s_ci/pr_archive/ | wc -l

# 按月统计
ls -l /apps/tas/0_public/primus_k8s_ci/pr_archive/ | \
  awk '{print $6"-"$7}' | sort | uniq -c

# 查找特定作者的 PR
grep -l "@username" /apps/tas/0_public/primus_k8s_ci/pr_archive/*.md
```

## ⚠️ 注意事项

1. **磁盘空间**：长期运行会积累大量文件，建议定期清理
2. **权限**：确保 CI runner 有写入权限到目标目录
3. **敏感信息**：PR description 可能包含敏感信息，注意访问控制
4. **特殊字符**：PR description 中的特殊字符会被正确处理

## 🔄 清理旧归档

定期清理脚本：

```bash
#!/bin/bash
# 删除 90 天前的归档文件
find /apps/tas/0_public/primus_k8s_ci/pr_archive/ \
  -name "PR_*.md" \
  -mtime +90 \
  -delete

echo "Cleaned up PR archives older than 90 days"
```

## 📝 相关文件

- `CI_PR_DESCRIPTION_GUIDE.md` - 完整实现方案指南
- `ci_pr_description_patch.yaml` - 独立的步骤配置
- `.github/workflows/ci.yaml` - 主 CI 配置文件（已更新）

## 🎉 完成

现在每次 PR 触发 CI 时，PR description 都会自动归档！🚀
