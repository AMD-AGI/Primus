#!/usr/bin/env python3
"""Skip nightly images when their tracked inputs match the last main publication."""

import json
import os
import subprocess
from pathlib import Path

WORKFLOW = "build-specforge-images.yml"
PAGE_SIZE = 100
VARIANTS = tuple(
    json.loads((Path(__file__).resolve().parents[2] / "examples/specforge/docker/images.json").read_text())[
        "variants"
    ]
)
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
        for variant in VARIANTS
    )


def last_published_revision(repository, api=github_api):
    page = 1
    while True:
        runs = api(
            f"repos/{repository}/actions/workflows/{WORKFLOW}/runs"
            f"?branch=main&status=success&per_page={PAGE_SIZE}&page={page}"
        )["workflow_runs"]
        for run in runs:
            if run["head_branch"] != "main" or run["event"] not in ("push", "schedule", "workflow_dispatch"):
                continue
            jobs = api(f"repos/{repository}/actions/runs/{run['id']}/jobs?filter=latest&per_page=100")["jobs"]
            if published_both(jobs):
                return run["head_sha"]
        if len(runs) < PAGE_SIZE:
            return None
        page += 1


class MissingPublishedRevision(RuntimeError):
    pass


def changed_inputs(revision, cwd=None, env=None):
    available = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=cwd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if available.returncode:
        raise MissingPublishedRevision(revision)
    # Two-endpoint tree comparison catches additions/deletions/mode changes and
    # avoids rebuilding for changes subsequently reverted before the nightly.
    return subprocess.check_output(
        ["git", "diff", "--name-only", revision, "HEAD", "--", *INPUTS], text=True, cwd=cwd, env=env
    ).splitlines()


def decision(event, repository, find_revision=last_published_revision, diff=changed_inputs):
    if event != "schedule":
        return True, "PR, release-tag, and manual builds bypass the nightly change check."
    revision = find_revision(repository)
    if revision is None:
        return True, "No successful main publication found; bootstrap both image variants."
    try:
        paths = diff(revision)
    except MissingPublishedRevision:
        return (
            True,
            f"Published commit {revision} is unavailable locally; bootstrap both variants to recover.",
        )
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
