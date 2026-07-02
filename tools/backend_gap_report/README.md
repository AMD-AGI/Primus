# Primus Engineering Dashboard Tooling

This directory contains the generation and publishing toolchain for the shared
Primus engineering dashboard. The same static dashboard surfaces two data
sources:

1. **Backend gap reports** — backend-to-upstream comparison reports and
   per-report artifacts under `docs/backend-gap/`.
2. **Periodic engineering reports** — the automated Primus engineering reports,
   on a **weekly** cadence under `docs/weekly_reports/` and a **monthly**
   cadence under `docs/monthly_reports/`.

Weekly and monthly reports share one schema, one validation core, and one
combined dashboard index, so the dashboard renders both cadences through a
single generic path (cadence is just a data field).

## Responsibilities

- `periodic_reports.py`: shared core for the periodic-report data plane. Owns
  the cadence config (weekly / monthly), the schema validation rules, the
  per-cadence index, and the combined weekly + monthly index.
- `build_weekly_reports_index.py`: thin CLI that validates weekly metadata and
  rewrites `docs/weekly_reports/dashboard-data/index.json`.
- `build_monthly_reports_index.py`: thin CLI that validates monthly metadata and
  rewrites `docs/monthly_reports/dashboard-data/index.json`.
- `build_dashboard_index.py`: validate backend-gap report metadata and rewrite
  `docs/backend-gap/dashboard-data/index.json`.
- `build_site_bundle.py`: rebuild the backend-gap index, assemble the combined
  periodic-report index, build the standalone publishable bundle (site shell +
  data), generate PDF artifacts for backend-gap reports, and run structural
  validation.
- `templates/pdf-report.css`: shared PDF export stylesheet (backend-gap only).
- `site/`: shared static dashboard source templates (one site, three sections:
  latest snapshot, backend deep dives, report archive).

## Current Model

- Markdown report sources and dashboard metadata remain in the Primus
  repository:
  - backend-gap: `docs/backend-gap/`
  - weekly reports: `docs/weekly_reports/`
  - monthly reports: `docs/monthly_reports/`
- `report_id` uses ISO week format `YYYY-Www` (weekly) or `YYYY-MM` (monthly).
- PDF artifacts are generated at bundle-build time (backend-gap only) and are
  not tracked in the repository.
- Periodic reports are **not** bundled into the site — the dashboard links the
  GitHub-rendered Markdown via `report_github_url` in each metadata file.
- All aggregate `index.json` files are generated build artifacts and are not
  committed — the per-cadence `docs/<cadence>/dashboard-data/index.json`, the
  backend-gap `docs/backend-gap/dashboard-data/index.json`, and the combined
  `reports-data/index.json` (assembled into the bundle at build time). Only the
  per-report JSON + Markdown are tracked. These indexes are not in `.gitignore`,
  so leave any regenerated index unstaged.
- A cadence directory that does not exist yet is treated as zero reports, so the
  monthly plane activates automatically once the first monthly report lands.
- Dashboard source templates live under `tools/backend_gap_report/site/`.
- GitHub Pages publishes a generated bundle rather than the repository source
  directory directly.
- A cadence directory that does not exist yet is treated as zero reports, so the
  monthly plane activates automatically once the first monthly report lands.
- Dashboard source templates live under `tools/backend_gap_report/site/`.
- GitHub Pages publishes a generated bundle rather than the repository source
  directory directly.

## Bundle Layout

`build_site_bundle.py` produces the following structure under the output
directory:

- `index.html`, `assets/*` — shared dashboard shell
- `dashboard-data/` — backend-gap dashboard data (index + per-report metadata)
- `dashboard-data/reports/<backend>/<target>/*.pdf` — generated PDF artifacts
- `reports-data/index.json` — combined weekly + monthly periodic-report index,
  assembled from `docs/weekly_reports/dashboard-data/` and
  `docs/monthly_reports/dashboard-data/`

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

### Periodic engineering report (weekly or monthly)
1. Generate or update the report and its metadata:
   - weekly: `docs/weekly_reports/{YYYY-Www}-primus-weekly.md` +
     `docs/weekly_reports/dashboard-data/reports/{YYYY-Www}.json`
   - monthly: `docs/monthly_reports/{YYYY-MM}-primus-monthly.md` +
     `docs/monthly_reports/dashboard-data/reports/{YYYY-MM}.json`
2. Rebuild the aggregated per-cadence index (run directly, or as part of the
   bundle build):

```bash
python3 tools/backend_gap_report/build_weekly_reports_index.py
python3 tools/backend_gap_report/build_monthly_reports_index.py
```

3. Full bundle rebuild validates every data plane together and assembles the
   combined index:

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
- the combined `reports-data/index.json` exists and is well-formed; every
  periodic report is validated against the shared schema while it is loaded
