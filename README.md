# Primus Engineering Dashboard — data branch

This is an **orphan data branch** (no shared history with `main`). It holds only
the *content* consumed by the Primus Engineering Dashboard:

- `docs/weekly_reports/` — weekly report Markdown + per-report metadata JSON
- `docs/monthly_reports/` — monthly report Markdown + per-report metadata JSON
- `docs/backend-gap/` — backend-gap comparison reports + per-report metadata JSON

## Why a separate branch

The dashboard **tooling and site shell** live on `main`
(`tools/backend_gap_report/`). Keeping the *data* here decouples it from `main`
so that documentation-site restructures on `main` can never move or break the
dashboard's data source, and routine report runs never add PR noise to `main`.

## How it is published

The `Deploy Backend Gap Dashboard` workflow (defined on `main`) checks out the
tooling from `main` and this branch's `docs/` data, then builds and deploys the
static bundle to GitHub Pages. Aggregate `index.json` files and PDFs are
generated at build time and are not committed.

## What to commit here

Only per-report sources + metadata:

- `docs/<cadence>/{report_id}-primus-<cadence>.md`
- `docs/<cadence>/dashboard-data/reports/{report_id}.json`
- backend-gap `report.md` / `summary.md` + `dashboard-data/reports/*.json`

Do not hand-edit or commit any `dashboard-data/index.json` (generated).
