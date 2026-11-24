"""CPU-only checks for the nightly image build gate."""

import ast
import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "image_changes", ROOT / ".github/scripts/specforge_image_changes.py"
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def successful_jobs():
    return [
        {
            "name": f"build ({variant})",
            "conclusion": "success",
            "steps": [{"name": "Publish image", "conclusion": "success"}],
        }
        for variant in ("mi30x", "mi35x")
    ]


def run(run_id, event="schedule", branch="main"):
    return {"id": run_id, "event": event, "head_branch": branch, "head_sha": f"sha-{run_id}"}


class ImageChangesTest(unittest.TestCase):
    def test_non_scheduled_builds_always_run(self):
        lookup = Mock(side_effect=AssertionError("Must not inspect history"))
        for event in ("pull_request", "push", "workflow_dispatch"):
            self.assertTrue(gate.decision(event, "owner/repo", lookup)[0])

    def test_nightly_bootstrap_unchanged_and_changed(self):
        diff = Mock(return_value=[])
        self.assertTrue(gate.decision("schedule", "owner/repo", lambda _: None, diff)[0])
        diff.assert_not_called()
        self.assertFalse(gate.decision("schedule", "owner/repo", lambda _: "base", diff)[0])
        diff.assert_called_once_with("base")
        diff.return_value = ["primus/new.py"]
        self.assertTrue(gate.decision("schedule", "owner/repo", lambda _: "base", diff)[0])

    def test_requires_both_successful_publications(self):
        self.assertTrue(gate.published_both(successful_jobs()))
        self.assertFalse(gate.published_both(successful_jobs()[:1]))
        for conclusion in ("failure", "skipped", "cancelled", None):
            jobs = successful_jobs()
            jobs[1]["steps"][0]["conclusion"] = conclusion
            self.assertFalse(gate.published_both(jobs))
            jobs = successful_jobs()
            jobs[1]["conclusion"] = conclusion
            self.assertFalse(gate.published_both(jobs))

    def test_skipped_nights_and_prs_do_not_advance_reference(self):
        responses = [
            {
                "workflow_runs": [
                    run(5, "pull_request"),
                    run(4, "push", "v26.9"),
                    run(3),
                    run(2),
                    run(1, "workflow_dispatch"),
                ]
            },
            {"jobs": []},
            {"jobs": successful_jobs()[:1]},
            {"jobs": successful_jobs()},
        ]
        api = Mock(side_effect=responses)
        self.assertEqual(gate.last_published_revision("owner/repo", api), "sha-1")
        self.assertEqual(api.call_count, 4)
        self.assertIn("status=success", api.call_args_list[0].args[0])

    def test_history_pagination(self):
        api = Mock(
            side_effect=[
                {"workflow_runs": [run(i, "pull_request") for i in range(100)]},
                {"workflow_runs": [run(101, "push")]},
                {"jobs": successful_jobs()},
            ]
        )
        self.assertEqual(gate.last_published_revision("owner/repo", api), "sha-101")
        self.assertIn("page=2", api.call_args_list[1].args[0])

    def test_history_empty_and_errors(self):
        self.assertIsNone(gate.last_published_revision("owner/repo", lambda _: {"workflow_runs": []}))
        with self.assertRaises(RuntimeError):
            gate.last_published_revision("owner/repo", Mock(side_effect=RuntimeError("API unavailable")))

    def test_all_tar_context_inputs_are_compared(self):
        tree = ast.parse((ROOT / "examples/specforge/docker/build.py").read_text(encoding="utf-8"))
        contexts = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.List)
            and node.elts
            and isinstance(node.elts[0], ast.Constant)
            and node.elts[0].value == "tar"
        ]
        self.assertEqual(len(contexts), 1)
        self.assertTrue(set(ast.literal_eval(contexts[0])[3:]).issubset(gate.INPUTS))

    def test_git_tree_comparison_ignores_docs_and_detects_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
            env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")

            def git(*args):
                return subprocess.check_output(
                    [
                        "git",
                        "-c",
                        "commit.gpgsign=false",
                        "-c",
                        "user.name=Test",
                        "-c",
                        "user.email=test@example.com",
                        *args,
                    ],
                    text=True,
                    cwd=root,
                    env=env,
                ).strip()

            git("init", "--quiet")
            (root / "README.md").write_text("original\n", encoding="utf-8")
            git("add", ".")
            git("commit", "--quiet", "-m", "base")
            base = git("rev-parse", "HEAD")
            (root / "docs").mkdir()
            (root / "docs/guide.md").write_text("unrelated\n", encoding="utf-8")
            git("add", ".")
            git("commit", "--quiet", "-m", "docs")
            self.assertEqual(gate.changed_inputs(base, cwd=root, env=env), [])
            git("rm", "README.md")
            git("commit", "--quiet", "-m", "delete")
            self.assertEqual(gate.changed_inputs(base, cwd=root, env=env), ["README.md"])
            git("checkout", base, "--", "README.md")
            git("commit", "--quiet", "-m", "restore")
            self.assertEqual(gate.changed_inputs(base, cwd=root, env=env), [])
            with self.assertRaises(gate.MissingPublishedRevision):
                gate.changed_inputs("0" * 40, cwd=root, env=env)

    def test_missing_revision_recovers_but_diff_errors_fail(self):
        missing = Mock(side_effect=gate.MissingPublishedRevision("gone"))
        build, reason = gate.decision("schedule", "owner/repo", lambda _: "gone", missing)
        self.assertTrue(build)
        self.assertIn("unavailable", reason)
        with self.assertRaises(subprocess.CalledProcessError):
            gate.decision(
                "schedule",
                "owner/repo",
                lambda _: "base",
                Mock(side_effect=subprocess.CalledProcessError(128, "git diff")),
            )

    def test_publication_contract_matches_workflow(self):
        import re

        workflow = (ROOT / ".github/workflows/build-specforge-images.yml").read_text()
        self.assertRegex(workflow, r"(?m)^  build:")
        self.assertIn("- name: Publish image", workflow)
        variants = re.search(r"variant: \[(.*?)\]", workflow).group(1)
        self.assertEqual({item.strip() for item in variants.split(",")}, set(gate.VARIANTS))


if __name__ == "__main__":
    unittest.main()
