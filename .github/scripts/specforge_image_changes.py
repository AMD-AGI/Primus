#!/usr/bin/env python3
"""Skip nightly images when their tracked inputs match the last main publication."""

import json
import os
import subprocess
from pathlib import Path

WORKFLOW = "build-specforge-images.yml"
# Keep aligned with build.py's allowlisted tar context, plus CI build controls.
INPUTS = (
    ".gitattributes",
    ".github/workflows/build-specforge-images.yml",
    ".github/scripts/specforge_image_changes.py",
    "examples/specforge",
    "primus",
    "runner",
    "primus_cli.py",
    "pyproject.toml",
    "LICENSE",
    "README.md",
)


def github_api(endpoint):
    return json.loads(subprocess.check_output(["gh", "api", endpoint], text=True))


def published_both(jobs):
    """A green/skipped workflow is not evidence that images were published."""
    return all(
        any(
            job["name"] == f"build ({variant})"
            and job["conclusion"] == "success"
            and any(
                step["name"] == "Publish image" and step["conclusion"] == "success"
                for step in job.get("steps", [])
            )
            for job in jobs
        )
        for variant in ("mi30x", "mi35x")
    )


def last_published_revision(repository, api=github_api):
    page = 1
    while True:
        runs = api(
            f"repos/{repository}/actions/workflows/{WORKFLOW}/runs"
            f"?branch=main&status=success&per_page=100&page={page}"
        )["workflow_runs"]
        for run in runs:
            if run["head_branch"] != "main" or run["event"] not in ("push", "schedule", "workflow_dispatch"):
                continue
            jobs = api(f"repos/{repository}/actions/runs/{run['id']}/jobs?filter=latest&per_page=100")["jobs"]
            if published_both(jobs):
                return run["head_sha"]
        if len(runs) < 100:
            return None
        page += 1


def changed_inputs(revision):
    # Two-endpoint tree comparison catches additions/deletions/mode changes and
    # avoids rebuilding for changes subsequently reverted before the nightly.
    return subprocess.check_output(
        ["git", "diff", "--name-only", revision, "HEAD", "--", *INPUTS], text=True
    ).splitlines()


def decision(event, repository, find_revision=last_published_revision, diff=changed_inputs):
    if event != "schedule":
        return True, "PR, release-tag, and manual builds bypass the nightly change check."
    revision = find_revision(repository)
    if revision is None:
        return True, "No successful main publication found; bootstrap both image variants."
    paths = diff(revision)
    return bool(paths), f"Compared tracked image inputs with {revision}: {len(paths)} changed path(s)."


def main():
    build, reason = decision(os.environ["GITHUB_EVENT_NAME"], os.environ["GITHUB_REPOSITORY"])
    print(reason)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        output.write(f"build={str(build).lower()}\n")
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as summary:
        summary.write(f"### SpecForge image change check\n\n{reason}\n\nBuild images: **{build}**\n")


if __name__ == "__main__":
    main()
