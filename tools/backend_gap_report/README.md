# Backend Gap Tooling

This directory contains the generation and publishing toolchain for backend-to-upstream comparison reports.

## Responsibilities

- `build_dashboard_index.py`: validate report metadata and rewrite `docs/backend-gap/dashboard-data/index.json`
- `build_site_bundle.py`: rebuild dashboard index, assemble the standalone publishable bundle, generate PDF artifacts, and run structural validation
- `templates/pdf-report.css`: shared PDF export stylesheet
- `site/`: dashboard site source templates

## Current Model

- markdown report artifacts and metadata remain in the Primus repository under `docs/backend-gap/`
- PDF artifacts are generated during site bundle build and are not tracked in the repository
- dashboard source templates live under `tools/backend_gap_report/site/`
- GitHub Pages publishes a generated site bundle rather than the repository source directory directly

## Typical Flow

1. Generate or update:
   - `docs/backend-gap/reports/<backend>/<target>/report.md`
   - `docs/backend-gap/reports/<backend>/<target>/summary.md`
2. Update `docs/backend-gap/dashboard-data/reports/<backend>-<target>.json` with PDF artifact paths
3. One-click build + validation:

```bash
python3 tools/backend_gap_report/build_site_bundle.py --output-dir /tmp/backend-gap-site
```

If you only want to refresh source index without building the site bundle:

```bash
python3 tools/backend_gap_report/build_dashboard_index.py
```

`build_site_bundle.py` includes structural acceptance checks:

- bundle entrypoint and dashboard data files exist
- `dashboard-data/index.json` and `dashboard-data/reports/*.json` are valid and consistent by report id
- every artifact path declared in dashboard data resolves to an existing file in the built bundle
