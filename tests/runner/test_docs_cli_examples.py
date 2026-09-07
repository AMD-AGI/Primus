###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HYBRID_GUIDE = ROOT / "docs" / "04-technical-guides" / "hybrid-models" / "README.md"

# Directories that are either deprecated, vendored, or gitignored build output.
SKIPPED_DIRS = {
    ".git",
    "docs_deprecated",
    "logs",
    "node_modules",
    "output",
    "third_party",
    "ut_out",
}

# Only executable invocations count: the script path has to follow `bash`, `sh`
# or `source` (optionally with short flags). Narrative mentions such as
# "run_slurm_pretrain.sh is avoided on purpose" therefore do not match. The
# trade-off is that a dead script named only in prose or in a Markdown link
# goes unnoticed; catching those would require distinguishing documentation
# about a script from instructions to run one, which is not worth the false
# positives.
SCRIPT_INVOCATION = re.compile(r"(?:^|[\s(&;|`])(?:bash|sh|source)\s+(?:-\w+\s+)*([A-Za-z0-9_./-]+\.sh)\b")

# Repo-relative references that are knowingly unresolvable.
ALLOWED_MISSING = {
    # Marked "(helper; not committed)" inline: a local calibration helper that
    # is intentionally kept out of the tree.
    "examples/deepseek-v4/projection/script/_calibrate_flash.sh",
    # Forward reference to the ODC rocSHMEM example, which lands with the PR
    # that the same README says the ops are still waiting on.
    "examples/llm_training/run.sh",
}


def _markdown_files():
    for path in sorted(ROOT.rglob("*.md")):
        if SKIPPED_DIRS.isdisjoint(path.relative_to(ROOT).parts):
            yield path


def _repo_relative_script_refs(line):
    """Yield repo-relative .sh paths invoked on a documentation line."""
    for match in SCRIPT_INVOCATION.finditer(line):
        ref = match.group(1)
        # Absolute paths point into a container or an unrelated checkout, and a
        # bare filename is relative to whatever directory the surrounding
        # snippet cd'd into. Neither can be resolved against the repo root.
        if ref.startswith(("/", "~")) or "/" not in ref.lstrip("./"):
            continue
        yield ref


def test_hybrid_data_path_env_stays_at_container_layer():
    guide = HYBRID_GUIDE.read_text(encoding="utf-8")
    assert "-- --env DATA_PATH" not in guide
    assert guide.count('--volume "$DATA_PATH:$DATA_PATH" \\\n  --env DATA_PATH \\\n  -- train pretrain') == 2


def test_documented_script_invocations_exist():
    dead = []
    for path in _markdown_files():
        doc = path.relative_to(ROOT)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for ref in _repo_relative_script_refs(line):
                target = ref[2:] if ref.startswith("./") else ref
                if target in ALLOWED_MISSING or (ROOT / target).is_file():
                    continue
                dead.append(f"{doc}:{lineno}: runs '{ref}', which does not exist")

    assert not dead, "Documentation invokes scripts that are not in the repository:\n" + "\n".join(dead)
