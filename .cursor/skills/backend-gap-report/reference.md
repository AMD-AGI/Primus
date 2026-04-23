# Backend Gap Report Reference

## Preferred Artifact Layout

For new report series, prefer:

```text
docs/
  backend-gap/
    dashboard-data/
      reports/
        <backend>-<target>.json
      index.json
    reports/
      <backend>/
        <target>/
          report.md
          summary.md
```

For example:

```text
docs/
  backend-gap/
    reports/
      torchtitan/
        upstream-main/
          report.md
          summary.md
    dashboard-data/
      reports/
        torchtitan-upstream-main-2026-04-21.json
      index.json
```

PDF files are generated at publish time and are not tracked in the repository.

Use unsuffixed names for the default report set. Only add language suffixes when a non-default variant coexists or a legacy series already established them.

If the repository already has a legacy file pattern for the same comparison, update those files in place and register them in metadata instead of forcing a rename.

Shared generation templates belong under:

```text
tools/backend_gap_report/
  README.md
  build_dashboard_index.py
  build_site_bundle.py
  site/
    index.html
    assets/
      dashboard.css
      dashboard.js
  templates/
    pdf-report.css
```

## Metadata JSON Shape

Each file under `docs/backend-gap/dashboard-data/reports/` should follow this structure:

```json
{
  "id": "torchtitan-upstream-main-2026-04-21",
  "title": "Primus TorchTitan vs upstream main",
  "backend": {
    "key": "torchtitan",
    "label": "TorchTitan"
  },
  "generated_at": "2026-04-21",
  "status": "verified",
  "scope": "Current TorchTitan bundled in Primus vs upstream pytorch/torchtitan origin/main",
  "local": {
    "source_path": "third_party/torchtitan",
    "version": "0.1.0",
    "commit": "5fb7cc2e",
    "commit_date": "2025-10-15"
  },
  "upstream": {
    "repo": "https://github.com/pytorch/torchtitan",
    "ref": "origin/main",
    "version": "0.2.2",
    "commit": "fc54b897",
    "commit_date": "2026-04-20"
  },
  "stats": {
    "commit_gap": 493,
    "diff_files": 447,
    "insertions": 56071,
    "deletions": 17716
  },
  "integration": {
    "backend_files": 90,
    "tracked_files": 147,
    "integration_model": "third_party/torchtitan + Primus outer adapter / trainer / patches"
  },
  "highlights": [
    "Primus bundles TorchTitan 0.1.0 while upstream main is at 0.2.2.",
    "Upstream main tracks the latest nightly on CUDA cu130 / ROCm 7.1, while the current Primus baseline is centered on cu126.",
    "Primus has a non-trivial outer integration layer that directly depends on upstream internal paths."
  ],
  "artifacts": [
    {
      "label": "Detailed Report (PDF)",
      "path": "./reports/torchtitan/upstream-main/report.pdf",
      "format": "pdf",
      "language": "en",
      "kind": "detail",
      "primary": true
    }
  ]
}
```

## Required Metadata Fields

Top level:

- `id`
- `title`
- `backend.key`
- `backend.label`
- `generated_at`
- `status`
- `scope`
- `local.source_path`
- `local.commit`
- `upstream.repo`
- `upstream.ref`
- `upstream.commit`
- `stats.commit_gap`
- `highlights`
- `artifacts`

Each artifact requires:

- `label`
- `path`
- `format`
- `language`
- `kind`

## Metadata Conventions

- `generated_at` should use `YYYY-MM-DD`.
- `status` should normally be `verified`, `draft`, or `superseded`.
- Artifact `path` values are relative to `docs/backend-gap/`.
- Artifact `format` should be `pdf` (site bundle only includes PDFs).
- Each artifact `path` should map to an existing markdown source path with the same basename.
- Keep default artifact labels plain. Add language suffixes only for non-default variants when needed.
- Prefer marking the most user-facing PDF as `primary: true`.

## Recommended Report Sections

Detailed report:

1. High-level comparison
2. Dependency comparison
3. Directory and capability differences
4. Primus integration or coupling
5. Evidence sources

One-page summary:

1. High-level comparison
2. Key dependency differences
3. Representative upstream changes
4. Primus integration layer

## Evidence Priority

When two sources disagree, prefer:

1. actual repository files under the compared refs
2. workflow installation logic
3. release documentation
4. README examples

## Dashboard Refresh

After writing or updating metadata:

```bash
python3 tools/backend_gap_report/build_dashboard_index.py
```

This script validates required fields and checks that each PDF artifact has a corresponding markdown source under `docs/backend-gap/`.

One-click build and validation:

```bash
python3 tools/backend_gap_report/build_site_bundle.py --output-dir /tmp/backend-gap-site
```

## Overwrite Behavior

For the same `backend` + `target`:

- rewrite `report.md` and `summary.md`
- update the matching metadata JSON in place
- rewrite `docs/backend-gap/dashboard-data/index.json`
- regenerate PDFs when building the standalone site bundle

Create a new report subtree only when the backend key or target key changes.

## GitHub Pages Deployment Model

The dashboard is published from a generated standalone bundle:

- site templates come from `tools/backend_gap_report/site/`
- generated metadata and reports come from `docs/backend-gap/`
- `build_site_bundle.py` assembles the publishable site root
- `build_site_bundle.py` also verifies bundle integrity as a hard acceptance gate
- `index.html` is the dashboard entrypoint at the published site root
- `dashboard-data/index.json` is the runtime data source
- report PDFs are generated into the standalone bundle and directly linkable from the dashboard
