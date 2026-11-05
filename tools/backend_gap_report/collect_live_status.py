#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Collect the dashboard's deterministic, frequently-changing facts.

Two subcommands, both read-only against GitHub:

  live  Pinned upstream dependencies vs their latest stable releases, and CI
        health on ``main`` (recent runs, latest weekly full run, coverage).
        Written to ``live-data/index.json`` in the site bundle at deploy time,
        so these facts stay fresh between monthly reports.
  prs   Merged PRs of a window with the Primus areas each one changed. The
        monthly report agent uses it, so report snapshots and the dashboard
        share one area mapping.

Needs a token in GH_TOKEN or GITHUB_TOKEN (``actions: read`` for CI artifacts).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))
import coverage_summary  # noqa: E402

API = "https://api.github.com"
# v1.2 / v1.2.3 / core_v0.20.0. The "v" excludes date tags (Megatron's 23.04,
# 26.09-alpha.rc3). GitHub's pre-release flag is ignored: torchtitan marks its
# numbered v0.1.0-v0.2.2 releases as pre-releases.
STABLE_TAG = re.compile(r"^(core_)?v\d+\.\d+(\.\d+)?$")
PR_REF = re.compile(r"\(#(\d+)\)\s*$")


# ---------- GitHub API ----------


def _token() -> str | None:
    return os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")


def api(endpoint: str, **params) -> dict | list:
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    req = urllib.request.Request(f"{API}/{endpoint}{query}")
    req.add_header("Accept", "application/vnd.github+json")
    if _token():
        req.add_header("Authorization", f"Bearer {_token()}")
    # Retry transient 5xx and dropped connections.
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as err:
            if err.code < 500 or attempt == 3:
                raise
        except urllib.error.URLError:
            if attempt == 3:
                raise
        time.sleep(2**attempt)


def api_list(endpoint: str, key: str | None = None, limit: int = 300, **params) -> list:
    items, page = [], 1
    while len(items) < limit:
        data = api(endpoint, per_page=100, page=page, **params)
        batch = data.get(key, []) if key else data
        items.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return items[:limit]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def download_artifact(artifact_id: int) -> zipfile.ZipFile:
    # The redirect target (signed blob storage) rejects the bearer token, so
    # follow it by hand without the header.
    req = urllib.request.Request(f"{API}/repos/{args.repo}/actions/artifacts/{artifact_id}/zip")
    req.add_header("Authorization", f"Bearer {_token()}")
    try:
        urllib.request.build_opener(_NoRedirect).open(req, timeout=60)
        raise RuntimeError("expected a redirect for artifact download")
    except urllib.error.HTTPError as err:
        if err.code not in (301, 302, 303, 307, 308):
            raise
        location = err.headers["Location"]
    with urllib.request.urlopen(location, timeout=120) as resp:
        return zipfile.ZipFile(io.BytesIO(resp.read()))


def _date(value: str | None) -> str | None:
    return value[:10] if value else None


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# ---------- Pinned dependencies ----------


def load_pins() -> list[dict]:
    lock = json.loads((REPO_ROOT / "primus" / "_thirdparty.lock").read_text())
    pins = [
        {
            "name": entry["name"],
            "path": entry["path"],
            "upstream": re.sub(r"^https://github\.com/|\.git$", "", entry["url"]),
            "commit": entry["commit"],
        }
        for entry in lock["third_party"]
    ]
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yaml").read_text()
    turbo = re.search(r"^\s*PRIMUS_TURBO_COMMIT:\s*([0-9a-f]{40})", ci, re.M)
    if turbo:
        pins.append(
            {
                "name": "Primus-Turbo",
                "path": "CI: PRIMUS_TURBO_COMMIT",
                "upstream": "AMD-AGI/Primus-Turbo",
                "commit": turbo.group(1),
            }
        )
    return pins


def pin_history(path: str, as_of: datetime) -> tuple[dict | None, int | None]:
    """Last commit on main that moved a submodule pin, and moves in 12 months.

    Reads local git history, which the deploy fetches as a cheap blob-less
    partial clone, instead of one commits-API call per submodule.
    """
    try:
        log = subprocess.run(
            [
                "git",
                "log",
                "origin/main" if _has_ref("origin/main") else "HEAD",
                f"--until={as_of.isoformat()}",
                "--format=%cs|%s",
                "--",
                path,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None, None
    if not log:
        return None, None
    cutoff = (as_of - timedelta(days=365)).date().isoformat()
    day, subject = log[0].split("|", 1)
    pr = PR_REF.search(subject)
    last = {"date": day, "pr": int(pr.group(1)) if pr else None, "title": PR_REF.sub("", subject).strip()}
    return last, sum(1 for line in log if line[:10] >= cutoff)


def _has_ref(ref: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "-q", ref], cwd=REPO_ROOT, capture_output=True
        ).returncode
        == 0
    )


def _version_key(tag: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", tag))


def _compare(upstream: str, base: str, head: str) -> dict:
    try:
        return api(f"repos/{upstream}/compare/{base}...{head}")
    except urllib.error.HTTPError:
        return {}


def _tag_line(tag: str) -> str:
    return re.match(r"^\D*", tag).group(0)


def dependency_status(pin: dict, as_of: datetime) -> dict:
    upstream, commit = pin["upstream"], pin["commit"]
    pin_date = _date(api(f"repos/{upstream}/commits/{commit}")["commit"]["committer"]["date"])
    stable = [
        r
        for r in api_list(f"repos/{upstream}/releases", limit=200)
        if not r["draft"] and STABLE_TAG.match(r["tag_name"]) and _parse(r["published_at"]) <= as_of
    ]
    # Trust the maintainers' "Latest" and compare only tags of its line:
    # Megatron-LM's old v2.x tags would otherwise outrank core_v0.x.
    try:
        marked = api(f"repos/{upstream}/releases/latest")["tag_name"]
    except urllib.error.HTTPError:
        marked = None
    if marked and STABLE_TAG.match(marked):
        line = _tag_line(marked)
    else:
        line = Counter(_tag_line(r["tag_name"]) for r in stable).most_common(1)[0][0] if stable else None
    releases = sorted(
        (r for r in stable if _tag_line(r["tag_name"]) == line),
        key=lambda r: _version_key(r["tag_name"]),
        reverse=True,
    )
    latest = releases[0] if releases else None

    # The release the pin sits on: newest one the pin is identical to or ahead of.
    pin_release, base = None, None
    for rel in releases:
        if _date(rel["published_at"]) > pin_date:
            continue
        cmp = _compare(upstream, rel["tag_name"], commit)
        if cmp.get("status") in ("identical", "ahead"):
            base = rel["tag_name"]
            pin_release = base if cmp["status"] == "identical" else f"{base} + {cmp['ahead_by']}"
            break

    if base:
        behind = sum(1 for r in releases if _version_key(r["tag_name"]) > _version_key(base))
    else:
        behind = sum(1 for r in releases if _date(r["published_at"]) > pin_date)
    to_latest = _compare(upstream, commit, latest["tag_name"]).get("ahead_by") if latest and behind else 0

    last_bump, bumps = None, None
    if pin["path"].startswith("third_party/"):
        last_bump, bumps = pin_history(pin["path"], as_of)

    return {
        "name": pin["name"],
        "path": pin["path"],
        "upstream": upstream,
        "pin_commit": commit[:8],
        "pin_date": pin_date,
        "pin_release": pin_release,
        "latest_release": (
            {"tag": latest["tag_name"], "date": _date(latest["published_at"])} if latest else None
        ),
        "releases_behind": behind if latest else None,
        "commits_to_latest_release": to_latest if latest else None,
        "last_bump": last_bump,
        "bumps_last_12_months": bumps,
    }


# ---------- CI health ----------


def _run_row(run: dict) -> dict:
    start = _parse(run.get("run_started_at") or run["created_at"])
    return {
        "id": run["id"],
        "event": run["event"],
        "conclusion": run["conclusion"] or run["status"],
        "created_at": run["created_at"],
        "duration_min": round((_parse(run["updated_at"]) - start).total_seconds() / 60),
        "head_sha": run["head_sha"][:8],
        "title": run["display_title"],
        "url": run["html_url"],
    }


def _failed_steps(run_id: int) -> list[str]:
    jobs = api_list(f"repos/{args.repo}/actions/runs/{run_id}/jobs", key="jobs")
    return [
        f"{job['name']}: {step['name']}"
        for job in jobs
        if job["conclusion"] == "failure"
        for step in job.get("steps", [])
        if step["conclusion"] == "failure"
    ]


def _coverage(runs: list[dict]) -> dict | None:
    """Unit vs Unit+E2E coverage from the newest run that still has artifacts.

    Prefers the weekly full run (it exercises every model E2E); artifacts
    expire, so older runs are tried in turn.
    """
    ordered = [r for r in runs if r["event"] == "schedule"] + [r for r in runs if r["event"] != "schedule"]
    for run in ordered:
        artifacts = {
            a["name"]: a
            for a in api(f"repos/{args.repo}/actions/runs/{run['id']}/artifacts").get("artifacts", [])
            if not a["expired"]
        }
        if "coverage-torch" not in artifacts:
            continue
        files = {}
        for name in ("coverage-torch", "coverage-jax"):
            if name in artifacts:
                archive = download_artifact(artifacts[name]["id"])
                files.update(
                    {
                        Path(n).name: json.loads(archive.read(n))
                        for n in archive.namelist()
                        if n.endswith(".json")
                    }
                )
        if "coverage_unit.json" not in files:
            continue
        e2e = [files[n] for n in ("coverage_combined.json", "coverage_maxtext_e2e.json") if n in files]
        summary = coverage_summary.summarize(files["coverage_unit.json"], e2e)
        summary.update(
            {
                "run_id": run["id"],
                "run_url": run["url"],
                "event": run["event"],
                "created_at": run["created_at"],
            }
        )
        return summary
    return None


def ci_health(as_of: datetime, days: int) -> dict:
    since = (as_of - timedelta(days=days)).date().isoformat()
    raw = api_list(
        f"repos/{args.repo}/actions/workflows/{args.ci_workflow}/runs",
        key="workflow_runs",
        branch="main",
        created=f">={since}",
    )
    # `branch=main` also matches fork PRs from a branch named main; keep only
    # this repo's own runs so forks can't skew stats or feed coverage.
    runs = [
        _run_row(r)
        for r in raw
        if r["status"] == "completed"
        and r["event"] in ("push", "schedule", "workflow_dispatch")
        and (r.get("head_repository") or {}).get("full_name") == args.repo
    ]
    counts = Counter(r["conclusion"] for r in runs)
    decided = counts["success"] + counts["failure"]
    durations = sorted(r["duration_min"] for r in runs if r["conclusion"] == "success")

    def latest(pred):
        run = next((r for r in runs if pred(r)), None)
        if run and run["conclusion"] == "failure":
            run = {**run, "failed_steps": _failed_steps(run["id"])}
        return run

    return {
        "workflow": args.ci_workflow,
        "branch": "main",
        "window_days": days,
        "runs": runs,
        "summary": {
            "total": len(runs),
            "by_conclusion": dict(counts),
            "pass_rate": round(100 * counts["success"] / decided, 1) if decided else None,
            "median_success_min": statistics.median(durations) if durations else None,
            "p90_success_min": durations[int(len(durations) * 0.9)] if durations else None,
        },
        # Cancelled runs were superseded, so "latest" is the newest verdict.
        "latest_main": latest(lambda r: r["event"] == "push" and r["conclusion"] in ("success", "failure")),
        "latest_weekly": latest(lambda r: r["event"] == "schedule"),
        "coverage": _coverage(runs),
    }


# ---------- Merged PRs with areas ----------

_BACKENDS = {
    "megatron": "Megatron",
    "Megatron-LM": "Megatron",
    "transformer_engine": "Megatron",
    "Emerging-Optimizers": "Megatron",
    "mamba": "Megatron",
    "megatron_bridge": "Megatron-Bridge",
    "Megatron-Bridge": "Megatron-Bridge",
    "torchtitan": "TorchTitan",
    "maxtext": "MaxText",
    "maxdiffusion": "MaxText",
    "automodel": "AutoModel",
    "nemo_automodel": "AutoModel",
    "Automodel": "AutoModel",
    "specforge": "SpecForge",
    "worldmirror": "WorldMirror",
    "HunyuanWorld-Mirror-rocm": "WorldMirror",
    "hummingbirdxt": "HummingbirdXT",
    "HummingbirdXT": "HummingbirdXT",
    "diffusion": "Diffusion",
}
_META = {".github", "runner", "tests"}
_SKILLS = {"skills", ".cursor", ".claude", ".agents"}


def area_of(path: str) -> str:
    """Map a changed file to the Primus area shown on the dashboard (first match wins)."""
    parts = path.split("/")
    if path.startswith(("primus/configs/models", "primus/configs/modules")):
        return "Model configs"
    if path.startswith(("tools/perf", "tools/convergence_test")):
        return "Perf & convergence tools"
    if path.startswith("examples/mlperf"):
        return "MLPerf"
    if path in ("primus/_thirdparty.lock", ".gitmodules", "pyproject.toml", "requirements.txt"):
        return "Dependencies"
    if path.startswith("primus/core"):
        return "Core runtime"
    if len(parts) > 2 and parts[:2] == ["primus", "backends"]:
        return _BACKENDS.get(parts[2], "Other backends")
    if len(parts) > 1 and parts[0] in ("third_party", "examples"):
        return _BACKENDS.get(parts[1], "Other backends" if parts[0] == "third_party" else "Examples")
    if "projection" in path:
        return "Projection"
    if parts[0] in _META:
        return "CI & tests"
    if parts[0] in _SKILLS:
        return "Agent skills"
    if parts[0] == "docs" or path.endswith(".md"):
        return "Docs"
    if parts[0] == "benchmark":
        return "Benchmark"
    if parts[0] in ("primus", "tools"):
        return "Core & tools"
    return "Other"


def pr_areas(files: list[str]) -> list[str]:
    counts = Counter(area_of(f) for f in files)
    # Docs/CI riding along with a code change are not where the PR landed.
    code = [a for a in counts if a not in ("Docs", "CI & tests")]
    return sorted(code or counts, key=lambda a: -counts[a])


def merged_prs(since: str, until: str) -> list[dict]:
    query = f"repo:{args.repo} is:pr is:merged base:main merged:{since}..{until}"
    rows = []
    for item in api_list("search/issues", key="items", limit=1000, q=query):
        pr = api(f"repos/{args.repo}/pulls/{item['number']}")
        files = [
            f["filename"] for f in api_list(f"repos/{args.repo}/pulls/{item['number']}/files", limit=3000)
        ]
        rows.append(
            {
                "number": pr["number"],
                "title": pr["title"],
                "created_at": pr["created_at"],
                "merged_at": pr["merged_at"],
                "author": pr["user"]["login"],
                "additions": pr["additions"],
                "deletions": pr["deletions"],
                "areas": pr_areas(files),
            }
        )
    return sorted(rows, key=lambda r: r["merged_at"])


# ---------- CLI ----------


def main() -> int:
    if args.command == "prs":
        json.dump(merged_prs(args.since, args.until), sys.stdout, indent=2)
        print()
        return 0

    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00")) if args.as_of else datetime.now(timezone.utc)
    live: dict = {"generated_at": as_of.isoformat(timespec="seconds"), "repo": args.repo}
    # Sections fail independently; the dashboard hides a missing one.
    try:
        live["dependencies"] = [dependency_status(pin, as_of) for pin in load_pins()]
    except Exception as err:  # noqa: BLE001
        live["dependencies_error"] = str(err)
        print(f"[live] dependencies unavailable: {err}", file=sys.stderr)
    try:
        live["ci"] = ci_health(as_of, args.ci_days)
    except Exception as err:  # noqa: BLE001
        live["ci_error"] = str(err)
        print(f"[live] CI health unavailable: {err}", file=sys.stderr)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(live, indent=2) + "\n")
    print(
        f"[live] wrote {out} ({len(live.get('dependencies', []))} dependencies, {len((live.get('ci') or {}).get('runs', []))} CI runs)"
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="AMD-AGI/Primus")
    sub = parser.add_subparsers(dest="command", required=True)
    live = sub.add_parser("live", help="dependencies + CI health -> JSON file")
    live.add_argument("--output", required=True)
    live.add_argument("--ci-workflow", default="ci.yaml")
    live.add_argument("--ci-days", type=int, default=30)
    live.add_argument(
        "--as-of",
        help="ISO 8601 cutoff (default now). Run from a checkout of main at that time to snapshot a past report window.",
    )
    prs = sub.add_parser("prs", help="merged PRs of a window with their areas -> stdout")
    prs.add_argument("--since", required=True, help="ISO 8601, e.g. 2026-10-01T09:02:00Z")
    prs.add_argument("--until", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(main())
