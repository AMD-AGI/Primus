# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""System and user prompts for the campaign phases."""

from __future__ import annotations

from tools.dataset_agent.hub import format_hub_facts
from tools.dataset_agent.locate import Layout
from tools.dataset_agent.spec import Task


def _common(task: Task, layout: Layout, hub_text: str, knowledge: str) -> str:
    preset = str(layout.model_preset) if layout.model_preset else "(no model preset file found)"
    template = str(layout.template_example) if layout.template_example else "(no existing example yaml found)"
    return f"""Backend: {task.backend}
Model: {task.model}
Dataset: {task.dataset}
Stage: {task.stage}
Split: {task.split}
Records the test must read: {task.samples}

Model preset: {preset}
Template experiment YAML: {template}
Example YAML you must leave for the operator: {layout.example_yaml}
Pytest file: {layout.test_file}

Hugging Face lookup:
{hub_text}

Plug-in map:
{knowledge}
"""


def plan_prompts(task: Task, layout: Layout, hub_text: str, knowledge: str) -> tuple[str, str]:
    system = (
        "You are the planner for the Primus dataset agent. You only read the repo. "
        "Decide how this dataset plugs into this backend, model, and stage. "
        "The operator set the stage to pretrain or posttrain. "
        "If an existing format can read this dataset's columns, use that format. "
        "Create a new format only when no existing format can read the columns. "
        "The dataset is the Hugging Face repo or the published source named by the operator. "
        "Do not invent rows, images, or a stand-in corpus."
    )
    user = (
        _common(task, layout, hub_text, knowledge)
        + f"""
Read the model preset, the template experiment, and the backend loader. Then end your reply with one json fence:

```json
{{
  "summary": "one paragraph",
  "stage": "{task.stage}",
  "format_name": "existing or new format name",
  "loader_change_required": false,
  "files_to_edit": [],
  "yaml_keys": {{"sft_dataset_name": "{task.dataset}"}},
  "example_yaml": "{layout.example_yaml}",
  "test_file": "{layout.test_file}"
}}
```
"""
    )
    return system, user


def integrate_prompts(
    task: Task,
    layout: Layout,
    hub_text: str,
    knowledge: str,
    plan_text: str,
) -> tuple[str, str]:
    system = (
        "You are the integrator for the Primus dataset agent. "
        "Edit Primus so this model and backend load the operator's dataset. "
        "Write the example experiment YAML at the exact path given. "
        "If an existing format already reads this dataset, use that format and do not add another. "
        "If no existing format can read the columns, add one new format, register it, and use that name in the YAML. "
        "Post-train Megatron formats that already exist: alpaca, chatml, openai, messages, squad, bridge_squad. "
        "Pre-train and diffusion formats that already exist: jsonl, hf_dataset, hf_repo, webdataset. "
        "The example YAML module is post_trainer for post-train and pre_trainer for pre-train. "
        "The YAML must set sft_dataset_name, dataset_path, or dataset_name to the dataset id. "
        "Copy the template experiment and change the dataset keys. "
        "Keep the model preset filename. "
        "Do not edit tests/ or docs/. "
        "Do not commit. "
        "Do not invent a dataset."
    )
    module = "post_trainer" if task.stage == "posttrain" else "pre_trainer"
    user = (
        _common(task, layout, hub_text, knowledge)
        + f"""
Plan:
{plan_text}

Stage is {task.stage}. Write {layout.example_yaml} with modules.{module}. Set sft_dataset_name, dataset_path, or dataset_name to exactly {task.dataset}. Point the YAML at an existing format when one can read the rows. Otherwise register one new format and point the YAML at that name. For the diffusion backend the data section must be consumable by DiffusionArgBuilder.
"""
    )
    return system, user


def test_prompts(task: Task, layout: Layout, hub_text: str, knowledge: str) -> tuple[str, str]:
    system = (
        "You are the tester for the Primus dataset agent. "
        "Write one pytest file at the exact path given. "
        "The test streams the real dataset from the Hugging Face Hub, or from the dataset's published URL, "
        f"and reads {task.samples} records. "
        "Pass each record through the backend function that training uses to read fields. "
        "Do not reimplement that mapping inside the test. "
        "Do not construct a loader that downloads the full corpus when a streaming read can feed the same parser. "
        "Do not mock load_dataset, snapshot_download, or load_from_disk. "
        "This host has no torch and is not inside a container. "
        "Do not import torch, primus.backends.megatron.sft.preprocessing, or SFTDataset. "
        "Do not skip the test when an import fails. "
        "For Megatron post-train, stream the rows with datasets.load_dataset and pass them to "
        "tools.dataset_agent.integration.check_megatron_rows using the sft_conversation_format from the example YAML. "
        "That function loads the SFT schema and formatter without torch. "
        "Do not call torch.cuda or initialize a distributed process group. "
        "Do not edit primus/, examples/, or docs/. "
        "Do not commit."
    )
    user = (
        _common(task, layout, hub_text, knowledge)
        + f"""
Write {layout.test_file}. The dataset id string {task.dataset!r} must appear in the file. Use split {task.split!r}. Read {task.samples} records.
"""
    )
    return system, user


def document_prompts(task: Task, layout: Layout, hub_text: str, knowledge: str) -> tuple[str, str]:
    system = (
        "You are the documenter for the Primus dataset agent. "
        "The host integration check has already passed. "
        "Update the existing dataset documentation for this backend so an operator can use the new example YAML. "
        "You may edit docs/ and the example YAML. "
        "Do not edit primus/backends or tests/. "
        "Do not commit. "
        "State the backend, model, dataset name, YAML keys, and example path."
    )
    user = (
        _common(task, layout, hub_text, knowledge)
        + f"""
Example YAML: {layout.example_yaml}
The dataset name {task.dataset} must remain a literal in that file.
"""
    )
    return system, user


def format_hub_for_prompt(facts: dict) -> str:
    return format_hub_facts(facts)
