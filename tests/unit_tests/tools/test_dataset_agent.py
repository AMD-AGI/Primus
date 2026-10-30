# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU checks for the dataset agent phase machine. No SDK and no network."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import os

from tools.dataset_agent.claude_backend import (  # noqa: E402
    ClaudeBackend,
    gateway_auth_failure,
)
from tools.dataset_agent.gate import (  # noqa: E402
    check_example_yaml,
    test_file_violations,
)
from tools.dataset_agent.hub import summarize_dataset  # noqa: E402
from tools.dataset_agent.integration import (  # noqa: E402
    binding_error,
    check_diffusion_rows,
    check_megatron_rows,
    is_infra_error,
    new_format_error,
    use_writable_hf_home,
)
from tools.dataset_agent.locate import (  # noqa: E402
    find_model_preset,
    layout_for,
    module_stem,
    slugify,
)
from tools.dataset_agent.orchestrator import render_report  # noqa: E402
from tools.dataset_agent.paths import bash_deny_reason, edit_deny_reason  # noqa: E402
from tools.dataset_agent.spec import PathPolicy  # noqa: E402


class DatasetAgentTests(unittest.TestCase):
    def test_gateway_401_names_the_subscription_header(self):
        message = gateway_auth_failure(
            401,
            '{ "statusCode": 401, "message": "Access denied due to missing subscription key." }',
        )
        self.assertIn("Ocp-Apim-Subscription-Key", message)
        self.assertEqual(gateway_auth_failure(200, "ok"), "")

    def test_probe_401_without_base_url_names_the_gateway(self):
        previous = os.environ.pop("ANTHROPIC_BASE_URL", None)
        try:
            message = ClaudeBackend._probe_failure(
                type("Probe", (), {"model": "claude-sonnet-4-5"})(),
                'api_error_origin":"first_party","result":"Invalid API key · Fix external API key"',
            )
        finally:
            if previous is not None:
                os.environ["ANTHROPIC_BASE_URL"] = previous
        self.assertIn("ANTHROPIC_BASE_URL", message)
        self.assertIn("Ocp-Apim-Subscription-Key", message)

    def test_report_names_the_yaml_and_the_gate(self):
        text = render_report(
            backend="diffusion",
            model="sample-model",
            dataset="org/sample-set",
            branch="dataset-agent/example",
            example_yaml="/tmp/example.yaml",
            test_file="/tmp/test_example.py",
            patch_path="/tmp/example.patch",
            hub_text="dataset: org/sample-set",
            plan_text="Use hf_repo.",
            passed=True,
            gate_log="1 passed",
            changed_files=["examples/diffusion/configs/dataset_agent/example.yaml"],
        )
        self.assertIn("The host gate **passed**.", text)
        self.assertIn("/tmp/example.yaml", text)
        self.assertIn("/tmp/example.patch", text)
        self.assertIn("org/sample-set", text)

    def test_slug_collapses_hub_ids(self):
        slug = slugify("diffusion", "sample-model", "org/sample-set")
        self.assertEqual(slug, "diffusion-sample-model-org-sample-set")

    def test_flux_model_preset_resolves(self):
        preset = find_model_preset(REPO, "diffusion", "flux.1_schnell_t2i")
        self.assertIsNotNone(preset)
        self.assertTrue(str(preset).endswith("primus/configs/models/diffusion/flux.1_schnell_t2i.yaml"))

    def test_layout_paths_stay_inside_the_repo(self):
        layout = layout_for(REPO, "diffusion", "sample-model", "org/sample-set", "posttrain")
        self.assertTrue(str(layout.example_yaml).startswith(str(REPO)))
        self.assertTrue(str(layout.test_file).startswith(str(REPO / "tests")))
        self.assertEqual(layout.branch, "dataset-agent/" + layout.slug)

    def test_hub_summary_counts_parquet_files(self):
        facts = summarize_dataset(
            {
                "id": "org/data",
                "siblings": [
                    {"rfilename": "data/train-00000.parquet"},
                    {"rfilename": "README.md"},
                ],
                "cardData": {"dataset_info": {"features": [{"name": "image"}, {"name": "text"}]}},
            },
            dataset="org/data",
        )
        self.assertEqual(facts["columns"], ["image", "text"])
        self.assertEqual(facts["suffix_counts"][".parquet"], 1)
        self.assertEqual(facts["error"], "")

    def test_integrator_cannot_write_tests(self):
        policy = PathPolicy(write_roots=("primus", "examples"), allow_bash=True)
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            reason = edit_deny_reason(repo, policy, "tests/unit_tests/tools/dataset_agent/test_x.py")
        self.assertIn("cannot write", reason)

    def test_tester_bash_cannot_commit(self):
        policy = PathPolicy(write_roots=("tests/unit_tests/tools/dataset_agent",), allow_bash=True)
        reason = bash_deny_reason(REPO, policy, "git commit -am 'update loader'")
        self.assertIn("Git history", reason)

    def test_plan_phase_cannot_run_bash(self):
        reason = bash_deny_reason(REPO, PathPolicy(), "ls primus")
        self.assertIn("cannot run shell", reason)

    def test_generated_test_must_name_the_dataset_and_not_mock_it(self):
        good = "from datasets import load_dataset\nload_dataset('org/data', split='train', streaming=True)\n"
        self.assertEqual(test_file_violations(good, dataset="org/data"), [])
        mocked = good + "\npatch('datasets.load_dataset')\n"
        self.assertTrue(test_file_violations(mocked, dataset="org/data"))
        skipped = good + "\nself.skipTest('torch missing')\n"
        self.assertTrue(test_file_violations(skipped, dataset="org/data"))
        torchy = good + "\nfrom primus.backends.megatron.sft.preprocessing import normalize_sft_sample\n"
        self.assertTrue(
            any("without torch" in reason for reason in test_file_violations(torchy, dataset="org/data"))
        )

    def test_diffusion_example_passes_the_argument_builder(self):
        preset = find_model_preset(REPO, "diffusion", "flux.1_schnell_t2i")
        text = """
modules:
  pre_trainer:
    framework: diffusion
    config: pre_trainer.yaml
    model: flux.1_schnell_t2i.yaml
    overrides:
      data:
        dataset_type: raw
        dataset_format: hf_repo
        dataset_path: org/sample-set
      training:
        local_batch_size: 1
        steps: 1
"""
        with tempfile.TemporaryDirectory() as tmp:
            example = Path(tmp) / "experiment.yaml"
            example.write_text(text, encoding="utf-8")
            message = check_example_yaml(
                REPO,
                example,
                backend="diffusion",
                dataset="org/sample-set",
                model_preset=preset,
            )
        self.assertEqual(message, "")

    def test_test_file_name_has_no_dots_for_python_import(self):
        layout = layout_for(REPO, "megatron", "llama3.1_8B", "tau/scrolls", "posttrain")
        self.assertNotIn(".", layout.test_file.name[len("test_") : -len(".py")])
        self.assertEqual(
            module_stem("megatron-posttrain-llama3.1_8b-tau-scrolls"),
            "megatron_posttrain_llama3_1_8b_tau_scrolls",
        )

    def test_a_directory_token_path_moves_the_hf_cache(self):
        previous = os.environ.pop("HF_HOME", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                token = root / "token"
                token.mkdir()
                home = root / "hf-home"
                note = use_writable_hf_home(token, home)
                self.assertIn(str(token), note)
                self.assertEqual(os.environ["HF_HOME"], str(home))
                self.assertTrue(home.is_dir())
        finally:
            os.environ.pop("HF_HOME", None)
            if previous is not None:
                os.environ["HF_HOME"] = previous

    def test_is_infra_error_flags_host_problems_not_code_problems(self):
        self.assertTrue(
            is_infra_error(
                "could not stream 'tau/scrolls' split 'train': IsADirectoryError: [Errno 21] Is a directory"
            )
        )
        self.assertFalse(is_infra_error("row 0 has no response after Megatron SFT field mapping"))

    def test_existing_format_is_accepted_and_a_missing_one_is_not(self):
        alpaca = {"modules": {"post_trainer": {"overrides": {"sft_conversation_format": "alpaca"}}}}
        self.assertEqual(new_format_error(alpaca, stage="posttrain", backend="megatron"), "")
        missing = {"modules": {"post_trainer": {"overrides": {}}}}
        self.assertIn("must set", new_format_error(missing, stage="posttrain", backend="megatron"))
        pretrain = {"modules": {"pre_trainer": {"overrides": {"dataset_format": "hf_repo"}}}}
        self.assertEqual(new_format_error(pretrain, stage="pretrain", backend="megatron"), "")

    def test_yaml_must_bind_the_dataset_field(self):
        document = {"modules": {"post_trainer": {"overrides": {"sft_dataset_name": "org/sample-set"}}}}
        self.assertEqual(binding_error(document, "org/sample-set"), "")
        self.assertIn("does not set", binding_error(document, "other/set"))

    def test_megatron_parser_accepts_instruction_context_response(self):
        rows = [{"instruction": "Say hi", "context": "to Ada", "response": "Hello Ada"}]
        self.assertEqual(check_megatron_rows(rows, "alpaca"), "")
        empty = [{"instruction": "Say hi", "context": "", "response": ""}]
        self.assertIn("no response", check_megatron_rows(empty, "alpaca"))

    def test_diffusion_parser_requires_caption_and_image(self):
        good = [{"caption": "a cat", "image": b"bytes"}]
        self.assertEqual(check_diffusion_rows(good), "")
        self.assertIn("no image", check_diffusion_rows([{"caption": "a cat"}]))

    def test_example_yaml_must_contain_the_dataset_literal(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            example = repo / "experiment.yaml"
            example.write_text(
                "modules:\n  pre_trainer:\n    framework: megatron\n    overrides:\n      data_path: other/set\n",
                encoding="utf-8",
            )
            message = check_example_yaml(
                repo,
                example,
                backend="megatron",
                dataset="org/data",
                model_preset=None,
            )
        self.assertIn("does not contain", message)


if __name__ == "__main__":
    unittest.main()
