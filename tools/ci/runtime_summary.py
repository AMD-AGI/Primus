###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Render the *current* job's own step wall-clock times (fetched from the
Actions API) as a Markdown table for the CI run summary.

Auto-discovers every step from the job's live metadata, so adding, removing,
or renaming a step in ci.yaml needs no matching edit here and no hand-rolled
per-step timing block -- unlike manually timing "just this stage" into a
file, which silently misses whatever nobody remembered to wrap.

Requires the calling step to export GITHUB_TOKEN with `actions: read` (see
ci.yaml); a missing token, network failure, or running outside Actions all
degrade to "print nothing, exit 0" so this can never fail the job:

    GITHUB_TOKEN=... python tools/ci/runtime_summary.py --title torch >> "$GITHUB_STEP_SUMMARY"

Steps shorter than --min-seconds (default 5s, almost always a banner `echo`
step) are hidden so the table stays focused.
"""

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

API_TIMEOUT_S = 30


def fmt(seconds):
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _parse_ts(s):
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def fetch_steps(repo, run_id, job_name, runner_name, token):
    """Return the step list for the run's job named `job_name`, preferring the
    one running on `runner_name` (in case the name is ever repeated, e.g. a
    future matrix), or [] if no such job is found."""
    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=API_TIMEOUT_S) as resp:  # noqa: S310 (fixed https:// API host)
        jobs = json.load(resp).get("jobs", [])
    matches = [j for j in jobs if j.get("name") == job_name]
    exact = [j for j in matches if j.get("runner_name") == runner_name]
    job = (exact or matches or [None])[0]
    return job.get("steps", []) if job else []


def step_durations(steps, min_seconds=5.0):
    """[(name, seconds), ...] in run order, for steps that actually completed
    (excludes the still-running caller itself, any not-yet-run step, and
    anything skipped) and cleared the min_seconds noise floor."""
    rows = []
    for step in steps:
        if step.get("conclusion") == "skipped":
            continue
        start, end = _parse_ts(step.get("started_at")), _parse_ts(step.get("completed_at"))
        if not start or not end:
            continue  # still running (incl. this very step) or never reached
        secs = (end - start).total_seconds()
        if secs >= min_seconds:
            rows.append((step.get("name", "?"), secs))
    return rows


def render(steps, title=None, min_seconds=5.0):
    rows = step_durations(steps, min_seconds)
    if not rows:
        return ""
    suffix = f" - {title}" if title else ""
    lines = [f"## CI runtime{suffix}\n", "| Stage | Time |", "|---|--:|"]
    lines += [f"| {name} | {fmt(secs)} |" for name, secs in rows]
    lines.append(f"| **TOTAL (shown stages)** | **{fmt(sum(s for _, s in rows))}** |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Render this job's own step timings as a Markdown table.")
    ap.add_argument("--title", default=None, help="Optional section title (e.g. torch).")
    ap.add_argument(
        "--min-seconds", type=float, default=5.0, help="Hide steps shorter than this (default 5s)."
    )
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    repo, run_id, job_name = (
        os.environ.get("GITHUB_REPOSITORY"),
        os.environ.get("GITHUB_RUN_ID"),
        os.environ.get("GITHUB_JOB"),
    )
    if not (token and repo and run_id and job_name):
        return 0  # not running in Actions (or token not wired) -- nothing to render
    try:
        steps = fetch_steps(repo, run_id, job_name, os.environ.get("RUNNER_NAME"), token)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return 0  # a reporting step must never fail the job
    out = render(steps, args.title, args.min_seconds)
    if out:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
