# 05 — Deployment (GitHub Pages)

The repository already publishes a single GitHub Pages site from `main` via
`.github/workflows/deploy-backend-gap-dashboard.yml`, which builds a bundle with
`tools/backend_gap_report/build_site_bundle.py` and deploys it with
`actions/deploy-pages`. A repo can only serve one Pages site, so the projection
site is published as a **subpath of that same bundle** rather than as a separate
deployment.

## How it is wired

`build_site_bundle.py` copies `examples/deepseek-v4/projection/site/` into the bundle at
`deepseek-v4-projection/` (after the backend-gap bundle is built, before
validation). The projection site uses only relative asset/data paths
(`./assets/...`, `./data/...`), so it works unchanged under a subpath.

Result URL:

```
https://<user>.github.io/<repo>/deepseek-v4-projection/?model=pro
https://<user>.github.io/<repo>/deepseek-v4-projection/?model=flash
```

## Triggering a deploy

The Pages workflow triggers on pushes to `main` touching its `paths:` list
(currently `docs/backend-gap/**`, `docs/weekly_reports/**`,
`docs/monthly_reports/**`, `tools/backend_gap_report/**`, and the workflow file).

- The change to `tools/backend_gap_report/build_site_bundle.py` in this work is
  itself under a watched path, so the **first** merge to `main` will rebuild and
  publish the projection site automatically.
- To make **projection-only** changes (new `site/data/*.json`, site tweaks) also
  auto-deploy, add the projection path to the workflow's `paths:` on `main`:

  ```yaml
  # .github/workflows/deploy-backend-gap-dashboard.yml  (on: push: paths:)
      - "examples/deepseek-v4/projection/site/**"
  ```

- Or trigger manually: the workflow has `workflow_dispatch` (Run workflow button).

No `.nojekyll` is needed: the bundle is uploaded as a Pages artifact and served
directly (no Jekyll processing), and no asset path starts with `_`.

## Local preview

```bash
python3 -m http.server -d examples/deepseek-v4/projection/site 8011
# http://localhost:8011/?model=pro
```

## Bundle-build smoke (optional, needs pandoc/weasyprint for the backend-gap PDFs)

```bash
python3 tools/backend_gap_report/build_site_bundle.py --output-dir /tmp/primus-site
ls /tmp/primus-site/deepseek-v4-projection/   # index.html, assets/, data/
```
