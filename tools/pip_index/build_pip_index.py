#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Generate a PEP 503 "simple" package index for Primus release artifacts.

The index HTML pages are hosted on GitHub Pages, while the wheels themselves
stay attached to GitHub Releases. Every file link points directly at the release
asset download URL, so the Pages site only hosts small HTML files and never
stores any binaries.

Layout produced under ``--output-dir`` (typically ``<site>/simple``)::

    simple/
      index.html              # lists every project
      primus/
        index.html            # lists every primus-*.whl / primus-*.tar.gz

Usage::

    python3 tools/pip_index/build_pip_index.py \
        --repo AMD-AIG-AIMA/Primus \
        --package primus \
        --requires-python ">=3.10" \
        --output-dir /tmp/site/simple

Install side (consumers)::

    pip install primus --extra-index-url https://<owner>.github.io/<repo>/simple/
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

GITHUB_API = "https://api.github.com"
DIST_SUFFIXES = (".whl", ".tar.gz")


def normalize(name: str) -> str:
    """PEP 503 project-name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def fetch_releases(repo: str, token: str | None, api_url: str) -> list[dict]:
    """Fetch all (non-paginated-away) releases for ``owner/repo`` via the API."""
    releases: list[dict] = []
    page = 1
    while True:
        url = f"{api_url}/repos/{repo}/releases?per_page=100&page={page}"
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        req.add_header("User-Agent", "primus-pip-index-builder")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req) as resp:
                batch = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise SystemExit(f"ERROR: GitHub API request failed ({exc.code}): {url}\n{detail}")
        except urllib.error.URLError as exc:
            raise SystemExit(f"ERROR: could not reach GitHub API ({url}): {exc}")
        if not batch:
            break
        releases.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return releases


def collect_files(releases: list[dict], package: str) -> list[tuple[str, str]]:
    """Return sorted ``(filename, download_url)`` pairs for the given package."""
    norm_pkg = normalize(package)
    files: list[tuple[str, str]] = []
    seen: set[str] = set()
    for rel in releases:
        if rel.get("draft"):
            continue
        for asset in rel.get("assets", []):
            name = asset.get("name", "")
            url = asset.get("browser_download_url", "")
            if not name or not url:
                continue
            if not name.endswith(DIST_SUFFIXES):
                continue
            # Only keep artifacts that belong to this project (e.g. primus-*).
            if not normalize(name).startswith(norm_pkg + "-"):
                continue
            if name in seen:
                continue
            seen.add(name)
            files.append((name, url))
    files.sort(key=lambda item: item[0])
    return files


def render_project_index(package: str, files: list[tuple[str, str]], requires_python: str) -> str:
    attr = ""
    if requires_python:
        attr = f' data-requires-python="{html.escape(requires_python, quote=True)}"'
    lines = [
        "<!DOCTYPE html>",
        '<html><head><meta charset="utf-8">',
        f"<title>Links for {html.escape(package)}</title></head>",
        "<body>",
        f"<h1>Links for {html.escape(package)}</h1>",
    ]
    for name, url in files:
        lines.append(f'<a href="{html.escape(url, quote=True)}"{attr}>{html.escape(name)}</a><br>')
    lines.append("</body></html>")
    return "\n".join(lines) + "\n"


def render_root_index(package: str) -> str:
    norm = normalize(package)
    return (
        "<!DOCTYPE html>\n"
        '<html><head><meta charset="utf-8"><title>Simple index</title></head>\n'
        "<body>\n"
        f'<a href="{html.escape(norm)}/">{html.escape(norm)}</a><br>\n'
        "</body></html>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a PEP 503 'simple' index for GitHub Release artifacts."
    )
    parser.add_argument("--repo", required=True, help="owner/repo, e.g. AMD-AIG-AIMA/Primus")
    parser.add_argument("--package", default="primus", help="Project name (default: primus)")
    parser.add_argument("--output-dir", required=True, help="Output dir for the simple/ index root")
    parser.add_argument(
        "--requires-python",
        default="",
        help="Optional data-requires-python value attached to each link (e.g. '>=3.10')",
    )
    parser.add_argument(
        "--token",
        default="",
        help="GitHub token (defaults to $GH_TOKEN or $GITHUB_TOKEN)",
    )
    parser.add_argument("--api-url", default=GITHUB_API, help="GitHub API base URL")
    parser.add_argument(
        "--releases-json",
        default="",
        help="Read releases from a local JSON file instead of the API (for testing).",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")

    if args.releases_json:
        releases = json.loads(Path(args.releases_json).read_text(encoding="utf-8"))
    else:
        releases = fetch_releases(args.repo, token, args.api_url)

    files = collect_files(releases, args.package)

    out_root = Path(args.output_dir).expanduser().resolve()
    norm = normalize(args.package)
    project_dir = out_root / norm
    project_dir.mkdir(parents=True, exist_ok=True)

    (out_root / "index.html").write_text(render_root_index(args.package), encoding="utf-8")
    (project_dir / "index.html").write_text(
        render_project_index(args.package, files, args.requires_python), encoding="utf-8"
    )

    print(f"[pip-index] {len(files)} artifact(s) for '{args.package}' -> {project_dir / 'index.html'}")
    for name, _ in files:
        print(f"  - {name}")
    if not files:
        print("[pip-index] WARNING: no matching release artifacts found; index is empty.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
