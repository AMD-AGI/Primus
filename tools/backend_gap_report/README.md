# Primus Engineering Dashboard Tooling

This directory contains the generation and publishing toolchain for the shared
Primus engineering dashboard. The same static dashboard surfaces two data
sources:

1. **Backend gap reports** — backend-to-upstream comparison reports and
   per-report artifacts under `docs/backend-gap/`.
2. **Weekly engineering reports** — the automated weekly Primus reports under
   `docs/weekly_reports/`.

## Responsibilities

- `build_dashboard_index.py`: validate backend-gap report metadata and rewrite
  `docs/backend-gap/dashboard-data/index.json`.
- `build_weekly_reports_index.py`: validate weekly-report metadata and rewrite
  `docs/weekly_reports/dashboard-data/index.json`.
- `build_site_bundle.py`: rebuild both dashboard indexes, assemble the
  standalone publishable bundle (site shell + both data roots), generate PDF
  artifacts for backend-gap reports, and run structural validation.
- `templates/pdf-report.css`: shared PDF export stylesheet (backend-gap only).
- `site/`: shared static dashboard source templates (one site, two sections).

## Current Model

- Markdown report sources and dashboard metadata remain in the Primus
  repository:
  - backend-gap: `docs/backend-gap/`
  - weekly reports: `docs/weekly_reports/`
- PDF artifacts are generated at bundle-build time (backend-gap only) and are
  not tracked in the repository.
- Weekly reports are **not** bundled into the site — the dashboard links the
  GitHub-rendered Markdown via `report_github_url` in each weekly metadata
  file.
- Dashboard source templates live under `tools/backend_gap_report/site/`.
- GitHub Pages publishes a generated bundle rather than the repository source
  directory directly.

## Bundle Layout

`build_site_bundle.py` produces the following structure under the output
directory:

- `index.html`, `assets/*` — shared dashboard shell
- `dashboard-data/` — backend-gap dashboard data (index + per-report metadata)
- `dashboard-data/reports/<backend>/<target>/*.pdf` — generated PDF artifacts
- `weekly-reports-data/` — weekly-report dashboard data (index + per-week
  metadata); copied verbatim from
  `docs/weekly_reports/dashboard-data/` when that directory exists

## Typical Flows

### Backend gap report
1. Generate or update:
   - `docs/backend-gap/reports/<backend>/<target>/report.md`
   - `docs/backend-gap/reports/<backend>/<target>/summary.md`
2. Update `docs/backend-gap/dashboard-data/reports/<backend>-<target>.json`
   with PDF artifact paths.
3. One-click build + validation:

```bash
python3 tools/backend_gap_report/build_site_bundle.py --output-dir /tmp/primus-dashboard-site
```

If you only want to refresh the backend-gap source index without building the
site bundle:

```bash
python3 tools/backend_gap_report/build_dashboard_index.py
```

### Weekly engineering report
1. Generate or update:
   - `docs/weekly_reports/{report_id}-primus-weekly.md`
   - `docs/weekly_reports/dashboard-data/reports/{report_id}.json`
2. Rebuild the aggregated weekly index (run directly, or as part of the
   bundle build):

```bash
python3 tools/backend_gap_report/build_weekly_reports_index.py
```

3. Full bundle rebuild validates both data planes together:

```bash
python3 tools/backend_gap_report/build_site_bundle.py --output-dir /tmp/primus-dashboard-site
```

## Bundle acceptance checks

`build_site_bundle.py` runs the following structural checks before finishing:

- bundle entrypoint and backend-gap dashboard data files exist
- `dashboard-data/index.json` and `dashboard-data/reports/*.json` are valid
  and consistent by report id
- every backend-gap artifact path declared in dashboard data resolves to an
  existing file in the built bundle
- if weekly-report data is present, the bundle includes
  `weekly-reports-data/index.json` and validates each weekly metadata file
  via `build_weekly_reports_index.py` before copy
