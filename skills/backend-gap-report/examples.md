# Backend Gap Report Examples

## Example 1: TorchTitan vs Upstream Main

User request:

```text
Compare the current Primus TorchTitan with upstream pytorch/torchtitan main, generate the report and PDF, and refresh the dashboard.
```

Expected outputs:

```text
docs/backend-gap/reports/torchtitan/upstream-main/report.md
docs/backend-gap/reports/torchtitan/upstream-main/summary.md
docs/backend-gap/dashboard-data/reports/torchtitan-upstream-main-YYYY-MM-DD.json
docs/backend-gap/dashboard-data/index.json
```

Publish bundle output (generated, not tracked in repo) includes:

```text
<bundle-root>/reports/torchtitan/upstream-main/report.pdf
<bundle-root>/reports/torchtitan/upstream-main/summary.pdf
```

Standalone publish bundle for preview or deployment:

```bash
python3 tools/backend_gap_report/build_site_bundle.py --output-dir /tmp/backend-gap-site
```

## Example 2: Megatron Backend Comparison

User request:

```text
Compare Primus Megatron backend with upstream Megatron-LM main, keep the report factual, export PDFs, and publish it to the dashboard.
```

Suggested evidence sources:

- `third_party/Megatron-LM/README.md`
- `third_party/Megatron-LM/requirements*`
- `third_party/Megatron-LM/.github/workflows/*`
- `primus/backends/megatron/*`
- `primus/modules/trainer/megatron/*`
- `docs/backends/megatron/patch-notes.md`

Expected outputs:

```text
docs/backend-gap/reports/megatron/upstream-main/report.md
docs/backend-gap/reports/megatron/upstream-main/summary.md
docs/backend-gap/dashboard-data/reports/megatron-upstream-main-YYYY-MM-DD.json
docs/backend-gap/dashboard-data/index.json
```

## Example Metadata Artifact Set

Preferred artifact list inside a metadata JSON file:

```json
[
  {
    "label": "Detailed Report (PDF)",
    "path": "./reports/torchtitan/upstream-main/report.pdf",
    "format": "pdf",
    "language": "en",
    "kind": "detail",
    "primary": true
  },
  {
    "label": "One-Page Summary (PDF)",
    "path": "./reports/torchtitan/upstream-main/summary.pdf",
    "format": "pdf",
    "language": "en",
    "kind": "summary",
    "primary": false
  }
]
```
