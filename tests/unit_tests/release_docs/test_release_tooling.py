###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the release-docs editing tools: version bumping, changelog
collection, and bare-metal install parity.

The classification rules are the risky part -- a wrong rewrite silently ships a
false statement, and a parity checker that cannot see real drift trains people
to ignore it. So the cases here are drawn from actual lines in the repository.
"""

import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_TOOLS = _ROOT / "tools/release_docs"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bump_version = _load("bump_version")
collect_changes = _load("collect_changes")
install_parity = _load("install_parity")
preflight = _load("preflight")


# --- bump_version: which references are candidates at all ------------------


def _classify(line, path_suffix=".md", branch_exists=False, in_fence=False):
    _, rules, _ = bump_version.load_rules("v26.6", "v26.7")
    suffix = "" if in_fence and path_suffix == ".md" else path_suffix
    rule = bump_version.classify(line, rules, suffix)
    action = rule["action"]
    if action == "conditional":
        action = "rewrite" if branch_exists else "hold"
    return rule["name"], action


def _tokens_match_version(line, version):
    import re

    return any(re.search(token, line) for token in bump_version.version_tokens(version))


def _tokens_match(line):
    return _tokens_match_version(line, "v26.6")


def test_memory_size_is_not_a_version_reference():
    # primus/backends/megatron/patches/ mentions "26.6GB" of memory.
    assert not _tokens_match("        # peak reserved 26.6GB after the resize")


def test_unrelated_package_version_is_not_a_reference():
    # docs/sphinx/requirements.txt pins rpds-py==2026.6.3.
    assert not _tokens_match("rpds-py==2026.6.3")


def test_image_tag_is_a_reference():
    assert _tokens_match("docker pull rocm/primus:v26.6")


def test_pip_spec_and_dunder_version_are_references():
    assert _tokens_match('pip install "primus==26.6.0"')
    assert _tokens_match('__version__ = "26.6"')


# --- bump_version: classification ------------------------------------------


def test_image_tags_and_dockerfile_names_are_mechanical():
    assert _classify("docker pull rocm/primus:v26.6") == ("docker-image-tag", "rewrite")
    assert _classify("- --image rocm/jax-training:maxtext-v26.6") == ("docker-image-tag", "rewrite")
    assert _classify("# from Dockerfile.primus-v26.6", ".sh") == ("dockerfile-filename", "rewrite")


def test_historical_statements_are_held():
    for line in [
        "v26.6 dropped the ROCm git fork in favour of the pip SDK.",
        "As of v26.6 the base config defaults changed.",
        "fp8 MoE workaround (v26.6)",
    ]:
        name, action = _classify(line)
        assert action == "hold", (line, name)


def test_dated_changelog_entries_are_held():
    # The README "What's New" list gains an entry per release and older ones keep
    # the tags they shipped with. Rewriting one produced the self-contradictory
    # 'Primus **v26.6** training images: `rocm/primus:v26.7`'.
    line = (
        "- **[2026/09/07]** Primus **v26.6** training images: `rocm/primus:v26.6` "
        "and `rocm/jax-training:maxtext-v26.6` (JAX 0.11.0)"
    )
    assert _classify(line) == ("dated-changelog-entry", "hold")


def test_an_undated_bullet_with_a_tag_is_still_rewritten():
    # The hold must key on the date stamp, not on being a list item.
    assert _classify("- pull `rocm/primus:v26.6` to get started") == ("docker-image-tag", "rewrite")


def test_comparisons_naming_two_releases_are_held():
    assert _classify("Compared with v26.6 and v26.5, the stack moved to ROCm 10.")[1] == "hold"


def test_release_branch_depends_on_whether_the_branch_exists():
    line = "git checkout release/v26.6"
    assert _classify(line, branch_exists=True) == ("release-branch", "rewrite")
    # Writing `git checkout release/v26.7` before that branch is cut would ship a
    # broken instruction, so it is held for the commit-form treatment instead.
    assert _classify(line, branch_exists=False) == ("release-branch", "hold")


def test_markdown_heading_moves_with_the_release():
    assert _classify("## Important notes for v26.6", ".md") == ("markdown-heading", "rewrite")


def test_heading_rule_does_not_apply_outside_markdown():
    # '#' is a comment in shell/YAML, where the text is as likely to be history.
    name, action = _classify("# v26.6: ROCm lives in the pip SDK", ".sh")
    assert (name, action) != ("markdown-heading", "rewrite")


def test_heading_rule_does_not_apply_inside_a_fenced_block():
    # bare-metal-installation-jax.md embeds this in a bash fence.
    name, action = _classify("# v26.6 image ships no librccl-net.so", ".md", in_fence=True)
    assert name != "markdown-heading"


def test_rewrites_only_touch_the_matched_construct():
    # A line mixing a live tag with a historical clause must only have the tag
    # rewritten, never the whole line.
    _, rules, _ = bump_version.load_rules("v26.6", "v26.7")
    rule = bump_version.classify("Since v26.6, use rocm/primus:v26.6.", rules, ".md")
    line = "Since v26.6, use rocm/primus:v26.6."
    for pattern, replacement in rule["replace"]:
        line = pattern.sub(replacement, line)
    assert line == "Since v26.6, use rocm/primus:v26.7."


def test_historical_artifacts_are_excluded_from_scanning():
    excluded, _, _ = bump_version.load_rules("v26.6", "v26.7")
    # Rewriting an old Dockerfile would falsify what a published image was built
    # from; the release notes are rotated by release_notes.py instead.
    assert any(path.startswith(".github/workflows/docker-release") for path in excluded)
    assert "docs/01-getting-started/release-notes.md" in excluded


def test_patch_release_references_are_not_candidates():
    # Regression: '\b' matches before a '.', so a v26.5 token used to match inside
    # 'v26.5.1' and rewrote Dockerfile.primus-v26.5.1 to a file that never existed.
    # v26.3.1, v26.3.2 and v26.5.1 all shipped, so this is not hypothetical.
    assert not _tokens_match_version("Dockerfile.primus-v26.5.1", "v26.5")
    assert not _tokens_match_version("see the v26.5.2 notes", "v26.5")
    assert _tokens_match_version("rocm/primus:v26.5", "v26.5")
    assert _tokens_match_version("on v26.5.", "v26.5")


def test_a_patch_reference_survives_a_rewrite_on_the_same_line():
    _, rules, _ = bump_version.load_rules("v26.5", "v26.6")
    line = "both rocm/primus:v26.5 and rocm/primus:v26.5.1 exist"
    rule = bump_version.classify(line, rules, ".md")
    for pattern, replacement in rule["replace"]:
        line = pattern.sub(replacement, line)
    assert line == "both rocm/primus:v26.6 and rocm/primus:v26.5.1 exist"


def test_pip_spec_still_bumps_its_patch_component():
    # The guard must not be applied to the bare form, whose rule owns the '.0'.
    _, rules, _ = bump_version.load_rules("v26.5", "v26.6")
    line = 'pip install "primus==26.5.0"'
    rule = bump_version.classify(line, rules, ".md")
    for pattern, replacement in rule["replace"]:
        line = pattern.sub(replacement, line)
    assert line == 'pip install "primus==26.6.0"'


def test_the_tooling_does_not_scan_its_own_source():
    # This tooling, its skill and its fixtures name releases as data and worked
    # examples. Scanning them added 82 self-referential entries to the review
    # list -- noise that can bury a real decision.
    excluded, _, _ = bump_version.load_rules("v26.6", "v26.7")
    for path in (
        "tools/release_docs/probe_image.py",
        "tools/release_docs/version_bump_rules.json",
        "skills/release-docs-update/SKILL.md",
        "tests/unit_tests/release_docs/test_release_tooling.py",
    ):
        assert any(path.startswith(prefix) for prefix in excluded), path


def test_pinned_reproduction_scripts_are_outside_the_rewrite_scope():
    # The v26.6 release left every examples/mlperf/, examples/models/ and
    # tools/docker/ image reference on v26.5: a benchmark submission records the
    # image it was validated against, so bumping it would falsify the result.
    _, _, scope = bump_version.load_rules("v26.6", "v26.7")
    for path in (
        "examples/mlperf/llama2_70b/README.md",
        "examples/models/kimi-k3/run_kimi_k3_curve_pretrain_mi355x.sh",
        "tools/docker/start_container.sh",
        "benchmark/kernel/rccl/run_slurm.sh",
    ):
        assert not bump_version.in_rewrite_scope(path, scope), path
    for path in ("README.md", "docs/01-getting-started/quickstart.md", "runner/.primus.yaml"):
        assert bump_version.in_rewrite_scope(path, scope), path


# --- collect_changes: bucketing --------------------------------------------


def test_conventional_prefixes_bucket_directly():
    cases = {
        "feat(gpt-oss): run the dense QKVO GEMMs on Turbo FlyDSL (#1127)": "features",
        "fix(lint): reformat the SDMA distributed-optimizer files (#1157)": "fixes",
        "perf: tune the MoE dispatch (#1)": "performance",
        "docs: drop the Instella-MoE blog (#1126)": "docs",
        "chore(core): remove the unused train_launcher CLI entry (#1068)": "maintenance",
        "ci: bump Primus-Turbo (#1048)": "maintenance",
        "test: pin FLAFlashAttention.forward layout contract (#1056)": "maintenance",
    }
    for subject, expected in cases.items():
        assert collect_changes.classify_bucket(subject, "Someone") == expected, subject


def test_bracket_prefixes_used_by_this_repo_are_recognised():
    assert collect_changes.classify_bucket("[Fix] add support for amd-smi (#1134)", "x") == "fixes"
    assert (
        collect_changes.classify_bucket("[OOB Release] add JAX v26.6 dockerfile (#1075)", "x")
        == "maintenance"
    )
    assert (
        collect_changes.classify_bucket("[Perf Optimization] Tune GDN/KDA 1B MI355X configs (#1080)", "x")
        == "performance"
    )


def test_unprefixed_tuning_commits_are_performance():
    # This repo lands a lot of un-prefixed config tuning.
    assert (
        collect_changes.classify_bucket("[maxtext] v26.6 mi300x batch size tuning (#1020)", "x")
        == "performance"
    )


def test_dependabot_is_bucketed_by_author_not_subject():
    assert (
        collect_changes.classify_bucket(
            "chore(deps): bump transformers from 4.50.0 to 5.10", "dependabot[bot]"
        )
        == "dependencies"
    )


def test_dependabot_changelog_walls_collapse_to_one_line():
    body = "Bumps transformers.\n" + "\n".join(f"* upstream change {n}" for n in range(200))
    assert collect_changes.collapse_body(body, "dependencies") == "Bumps transformers."


def test_pr_body_is_preserved_for_real_commits():
    body = "## Summary\n- Adds a thing.\n\n---------\n\nCo-authored-by: Cursor <x@y>"
    collapsed = collect_changes.collapse_body(body, "features")
    assert collapsed.startswith("## Summary")
    assert "Co-authored-by" not in collapsed


def test_areas_are_derived_from_touched_paths():
    areas = collect_changes.classify_areas(
        ["primus/backends/megatron/x.py", "examples/maxtext/configs/y.yaml", "docs/z.md"]
    )
    assert "megatron" in areas and "maxtext" in areas and "docs" in areas


def test_submodule_bumps_are_detected():
    # Regression: this used to parse `--submodule=short` output for a
    # "Submodule <path> a..b" summary line, which that format never emits, so it
    # silently reported no bumps ever. dc3f4ba18a moves third_party/maxtext.
    bumps = collect_changes.submodule_bumps("dc3f4ba18a^", "dc3f4ba18a")
    assert "third_party/maxtext" in bumps
    assert bumps["third_party/maxtext"]["from"].startswith("2ec83add")
    assert bumps["third_party/maxtext"]["to"].startswith("b47d74bf")


def test_no_submodule_bumps_reports_empty_rather_than_failing():
    bumps = collect_changes.submodule_bumps("2aa05ead", "2631e68d")
    assert bumps == {}


# --- install_parity --------------------------------------------------------


def test_shell_default_expansion_resolves_to_its_default():
    # setup.sh writes MAXTEXT_BRANCH="${MAXTEXT_BRANCH:-release/v26.6}".
    assert install_parity.SH_DEFAULT.match("${MAXTEXT_BRANCH:-release/v26.6}").group(1) == "release/v26.6"


def test_mirrored_release_is_read_from_the_script_itself():
    # Self-anchoring is what makes --check meaningful in CI.
    assert install_parity.mirrored_release("primus").startswith("v26.")
    assert install_parity.mirrored_release("jax").startswith("v26.")


def test_parity_is_clean_against_the_dockerfile_the_script_mirrors():
    rules = install_parity.load_rules()
    for family in ("primus", "jax"):
        release = install_parity.mirrored_release(family)
        result, error = install_parity.compare(family, release, rules)
        assert error is None
        assert result["drift"] == [], (family, result["drift"])
        assert result["matched"], family


def test_parity_detects_drift_against_a_different_release():
    # The negative test matters more than the positive one: a checker that cannot
    # see the drift it exists to catch is worse than nothing.
    rules = install_parity.load_rules()
    other = "v26.5" if install_parity.mirrored_release("primus") != "v26.5" else "v26.4"
    result, error = install_parity.compare("primus", other, rules)
    assert error is None
    assert result["drift"], "expected pin drift against a different release Dockerfile"


def test_documented_divergences_are_not_reported_as_drift():
    rules = install_parity.load_rules()
    release = install_parity.mirrored_release("primus")
    result, _ = install_parity.compare("primus", release, rules)
    names = {name for name, _, _ in result["documented_divergences"]}
    # torchvision/torchaudio float in the Dockerfile but are pinned bare-metal.
    assert "TORCHVISION_VERSION" in names
    assert names.isdisjoint({name for name, _, _, _ in result["drift"]})


def test_rules_files_are_valid_json_with_the_expected_shape():
    rules = json.loads((_TOOLS / "install_parity_rules.json").read_text())
    assert {"aliases", "not_pins", "documented_divergences", "inline_pins"} <= set(rules)
    bump = json.loads((_TOOLS / "version_bump_rules.json").read_text())
    assert {"excluded_paths", "rules"} <= set(bump)
    assert bump["rules"][-1]["action"] == "hold", "the fallback rule must hold, never rewrite"


# --- preflight -------------------------------------------------------------


def test_dockerfile_args_take_the_first_declaration():
    path = _ROOT / ".github/workflows/docker-release/Dockerfile.primus-v26.6"
    args = preflight.dockerfile_args(path)
    assert args["PRIMUS_BRANCH"] == "2aa05ead3401708cf1a1e2958c1def18d7aecf92"


def test_primus_dockerfile_pins_a_commit_and_jax_pins_main():
    # This asymmetry is why the build commit needs two resolution paths: the JAX
    # image can only be dated by its own /workspace/Primus checkout.
    primus = preflight.dockerfile_args(_ROOT / ".github/workflows/docker-release/Dockerfile.primus-v26.6")
    jax = preflight.dockerfile_args(_ROOT / ".github/workflows/docker-release/Dockerfile.jax-v26.6")
    assert preflight.SHA40.match(primus["PRIMUS_BRANCH"])
    assert jax["PRIMUS_BRANCH"] == "main"


def test_release_shape_detects_a_patch_release():
    families = {name: {"image_present": True} for name in ("primus", "jax")}
    assert preflight.release_shape(families, "v26.5.1")[0] == "patch"
    assert preflight.release_shape(families, "v26.7")[0] == "full"


def test_release_shape_detects_a_single_family_release():
    # The two families are not built in lockstep; one can land weeks later.
    families = {"primus": {"image_present": True}, "jax": {"image_present": False}}
    shape, present = preflight.release_shape(families, "v26.7")
    assert shape == "single-family" and present == ["primus"]
