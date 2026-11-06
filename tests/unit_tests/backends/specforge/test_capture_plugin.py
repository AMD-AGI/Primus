# Copyright (c) 2026, Advanced Micro Devices, Inc.
# See LICENSE for license information.

"""Capture migration contracts; runnable without Torch, SGLang, or a GPU."""

import io
import sys
from types import SimpleNamespace

import pytest
import yaml

from primus.backends.specforge import capture_plugin, online_supervisor, stack_preflight
from primus.backends.specforge.argument_builder import (
    build_capture_argv,
    build_specforge_argv,
)
from primus.backends.specforge.online_launch import (
    build_online_overrides,
    build_role_argv,
    online_settings,
    sglang_server_argv,
)


@pytest.fixture
def recipe(tmp_path):
    root = tmp_path / "SpecForge"
    (root / "configs").mkdir(parents=True)
    (root / "configs/draft.json").write_text('{"dflash_config":{"target_layer_ids":[1,8,15,22,29]}}')
    config = root / "run.yaml"
    config.write_text(
        "model:\n  target_model_path: Qwen/Qwen3.5-4B\n  draft_model_config: configs/draft.json\n"
    )
    return SimpleNamespace(
        specforge_root=str(root),
        specforge_config=str(config),
        specforge_mode="train",
        specforge_train_mode="online",
        specforge_overrides={},
        specforge_online={
            "run_id": "plugin-test",
            "run_root": str(tmp_path / "run"),
            "consumer_state_dir": str(tmp_path / "consumer"),
            "server_count": 2,
            "server_tp": 2,
        },
    )


def test_plugin_launch_preserves_layers_and_server_settings(recipe):
    settings = online_settings(recipe, env={})
    argv = sglang_server_argv(1, settings)
    assert argv[argv.index("--speculative-algorithm") + 1] == "DFLASH"
    assert "--enable-return-aux-hidden-states" in argv
    start = argv.index("--return-aux-hidden-state-layer-ids") + 1
    assert argv[start : start + 5] == ["1", "8", "15", "22", "29"]
    assert argv[argv.index("--return-hidden-states-mode") + 1] == "full"
    assert argv[argv.index("--chunked-prefill-size") + 1] == "-1"
    assert argv[argv.index("--port") + 1] == "30001"
    assert argv[argv.index("--tp-size") + 1] == "2"
    assert "--skip-tokenizer-init" in argv
    assert "--disable-radix-cache" in argv
    assert argv[argv.index("--attention-backend") + 1] == "aiter"
    assert not any(arg.startswith("--spec-capture") or arg == "--enable-spec-capture" for arg in argv)


@pytest.mark.parametrize("architecture", ["LlamaForCausalLMEagle3", "LlamaForCausalLMEagle4"])
def test_eagle_drafts_select_eagle_capture_convention(recipe, architecture):
    import json
    from pathlib import Path

    path = Path(recipe.specforge_root) / "configs/draft.json"
    path.write_text(
        json.dumps(
            {
                "architectures": [architecture],
                "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 5, 9]},
            }
        )
    )
    settings = online_settings(recipe, env={})
    argv = sglang_server_argv(0, settings)
    assert argv[argv.index("--speculative-algorithm") + 1] == "EAGLE3"
    assert settings["capture_layer_ids"] == (1, 5, 9)


@pytest.mark.parametrize(
    "extra",
    [
        "--enable-spec-capture",
        "--spec-capture-method=dflash",
        "--return-hidden-states-mode=last",
        "--chunked-prefill-size=128",
    ],
)
def test_capture_flags_cannot_be_overridden(recipe, extra):
    recipe.specforge_online["sglang_extra_args"] = extra
    with pytest.raises(ValueError, match="Primus owns the capture plugin flags"):
        sglang_server_argv(0, online_settings(recipe, env={}))


@pytest.mark.parametrize("role", ["producer", "consumer"])
def test_roles_select_plugin(recipe, role):
    settings = online_settings(recipe, env={})
    overrides = build_online_overrides("192.0.2.1", settings)
    argv = build_role_argv(recipe, role, overrides)
    assert argv[argv.index("--role") + 1] == role
    assert f"{capture_plugin.CAPTURE_BACKEND_KEY}=plugin" in argv
    assert not any(item.endswith("server_capture=patch") for item in argv)


@pytest.mark.parametrize("source", ["overrides", "environment", "yaml", "extra_overrides"])
def test_patch_backend_is_rejected(recipe, source):
    env = {}
    if source == "overrides":
        recipe.specforge_overrides = {capture_plugin.CAPTURE_BACKEND_KEY: "patch"}
    elif source == "environment":
        env["DISAGG_SERVER_CAPTURE"] = "patch"
    elif source == "yaml":
        with open(recipe.specforge_config, "a") as stream:
            stream.write("deployment:\n  disaggregated:\n    server_capture: patch\n")
    with pytest.raises(ValueError, match="requires the SGLang plugin"):
        if source == "extra_overrides":
            build_role_argv(recipe, "producer", [f"{capture_plugin.CAPTURE_BACKEND_KEY}=patch"])
        else:
            online_settings(recipe, env=env)


@pytest.mark.parametrize("allowed", [None, "", "  ", "other", "other, specforge_spec_capture"])
def test_server_env_preserves_plugin_discovery(allowed):
    original = {"MOONCAKE_PROTOCOL": "tcp"}
    if allowed is not None:
        original["SGLANG_PLUGINS"] = allowed
    result = capture_plugin.capture_server_env(original)
    assert result["SPECFORGE_SPEC_CAPTURE"] == "1"
    assert "SPECFORGE_SPEC_CAPTURE" not in original
    assert result["MOONCAKE_PROTOCOL"] == "tcp"
    if allowed and allowed.strip():
        assert result["SGLANG_PLUGINS"].split(",") == ["other", "specforge_spec_capture"]
    elif allowed:
        assert result["SGLANG_PLUGINS"] == "specforge_spec_capture"
    else:
        assert result.get("SGLANG_PLUGINS") == allowed


def test_only_capture_children_enable_plugin(recipe, monkeypatch, tmp_path):
    monkeypatch.delenv("SPECFORGE_SPEC_CAPTURE", raising=False)
    monkeypatch.setenv("SGLANG_PLUGINS", "other")
    settings = online_settings(recipe, env={})
    children = []
    monkeypatch.setattr(online_supervisor, "_popen", lambda argv, env, path: children.append(env) or object())
    monkeypatch.setattr(online_supervisor, "_wait_sglang", lambda *args: None)
    monkeypatch.setattr(online_supervisor, "require_capture_plugin", lambda *args: None)
    monkeypatch.setattr(online_supervisor, "log_rank_0", lambda *args: None)
    online_supervisor._start_local_sglang(settings, "192.0.2.1", "192.0.2.2", tmp_path, 0)
    assert [env["HIP_VISIBLE_DEVICES"] for env in children] == ["0,1", "2,3"]
    assert all(env["SPECFORGE_SPEC_CAPTURE"] == "1" for env in children)
    assert all(env["SGLANG_PLUGINS"] == "other,specforge_spec_capture" for env in children)
    assert "SPECFORGE_SPEC_CAPTURE" not in online_supervisor._specforge_child_env({}, None)


@pytest.fixture
def plugin_stack(monkeypatch):
    modules = {
        "sglang.srt.model_executor.forward_observer": SimpleNamespace(),
        "sglang.srt.managers.deferred_output": SimpleNamespace(ResponseCompletion=object),
        "sglang.srt.plugins.hook_registry": SimpleNamespace(HookRegistry=object, HookType=object),
        "sglang.srt.server_args": SimpleNamespace(
            ServerArgs=SimpleNamespace(
                speculative_algorithm=None,
                enable_return_aux_hidden_states=False,
                return_aux_hidden_state_layer_ids=None,
                return_hidden_states_mode=None,
                chunked_prefill_size=None,
            )
        ),
    }
    monkeypatch.setattr(capture_plugin, "importlib", SimpleNamespace(import_module=modules.__getitem__))
    monkeypatch.setattr(
        capture_plugin,
        "entry_points",
        lambda **kwargs: [SimpleNamespace(name=capture_plugin.PLUGIN_NAME, load=lambda: lambda: None)],
    )
    return modules


def test_plugin_capabilities_pass_without_patch_api(plugin_stack):
    assert capture_plugin.capture_plugin_issues() == []


@pytest.mark.parametrize("missing", ["entry_point", "forward_observer", "ResponseCompletion", "server_arg"])
def test_incompatible_stack_has_actionable_errors(plugin_stack, monkeypatch, missing):
    if missing == "entry_point":
        monkeypatch.setattr(capture_plugin, "entry_points", lambda **kwargs: [])
    elif missing == "forward_observer":
        del plugin_stack["sglang.srt.model_executor.forward_observer"]
    elif missing == "ResponseCompletion":
        del plugin_stack["sglang.srt.managers.deferred_output"].ResponseCompletion
    else:
        del plugin_stack["sglang.srt.server_args"].ServerArgs.return_hidden_states_mode
    issues = capture_plugin.capture_plugin_issues()
    assert len(issues) == 1
    expected = {"entry_point": "install", "server_arg": "return_hidden_states_mode"}.get(missing, missing)
    assert expected in issues[0]


def test_online_preflight_uses_plugin_probe(recipe, monkeypatch):
    monkeypatch.setattr(stack_preflight, "torch_kind", lambda: "hip")
    monkeypatch.setattr(stack_preflight, "capture_plugin_issues", lambda: ["plugin is incompatible"])
    assert "plugin is incompatible" in stack_preflight.collect_issues(recipe, env={})
    recipe.specforge_train_mode = "offline"
    assert "plugin is incompatible" not in stack_preflight.collect_issues(recipe, env={})


@pytest.mark.parametrize("pin, expected", [(None, True), ("0.5.24", False), ("", False)])
def test_sglang_version_pin_is_overridable(recipe, monkeypatch, pin, expected):
    import sys

    monkeypatch.setattr(stack_preflight, "torch_kind", lambda: "hip")
    monkeypatch.setattr(stack_preflight, "capture_plugin_issues", lambda: [])
    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(__version__="0.5.24"))
    monkeypatch.setitem(sys.modules, "aiter", SimpleNamespace())
    env = {} if pin is None else {"PRIMUS_SPECFORGE_PIN_SGLANG": pin}
    issues = stack_preflight.collect_issues(recipe, env=env)
    assert any("does not match pin prefix" in issue for issue in issues) == expected


def test_offline_commands_do_not_gain_plugin_flags(recipe):
    recipe.specforge_train_mode = "offline"
    assert not any("server_capture" in arg for arg in build_specforge_argv(recipe))
    recipe.specforge_mode = "capture"
    recipe.specforge_capture = {
        "target_model_path": "Qwen/Qwen3.5-4B",
        "data_path": "input.jsonl",
        "output_path": "hidden_states",
    }
    argv = build_capture_argv(recipe)
    assert "--module" in argv
    assert "primus.backends.specforge.offline_capture" in argv
    assert not any("return-aux" in arg or "server_capture" in arg for arg in argv)


def test_example_recipe_paths():
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    for mode, directory in (
        ("online", "online/disaggregated/external"),
        ("offline", "offline/colocated"),
    ):
        config = yaml.safe_load(
            (root / f"examples/specforge/configs/qwen3.5-4b-dflash-{mode}.yaml").read_text()
        )
        path = config["modules"]["pre_trainer"]["overrides"]["specforge_config"]
        assert f"/{directory}/qwen3.5-4b-dflash-amd.yaml" in path


@pytest.mark.parametrize("value", [None, "null", "", "PLUGIN", "Auto", " plugin "])
@pytest.mark.parametrize("source", ["yaml", "overrides", "environment"])
def test_backend_normalization(recipe, value, source):
    env = {}
    if source == "yaml":
        with open(recipe.specforge_config, "a") as stream:
            stream.write(yaml.safe_dump({"deployment": {"disaggregated": {"server_capture": value}}}))
    elif source == "overrides":
        recipe.specforge_overrides = {capture_plugin.CAPTURE_BACKEND_KEY: value}
    else:
        if value is not None:
            env["DISAGG_SERVER_CAPTURE"] = value
    online_settings(recipe, env=env)
    argv = build_role_argv(recipe, "producer", [])
    assert [arg for arg in argv if "server_capture" in arg] == [
        f"{capture_plugin.CAPTURE_BACKEND_KEY}=plugin"
    ]


@pytest.mark.parametrize("value", ["patch", " PATCH ", "unsupported"])
def test_backend_normalization_still_rejects_conflicts(value):
    with pytest.raises(ValueError, match="requires the SGLang plugin"):
        capture_plugin.validate_capture_backend(value)


@pytest.mark.parametrize("prefix", ["", "+", "++", "~"])
@pytest.mark.parametrize("source", ["overrides", "extra"])
def test_owned_backend_replaces_hydra_assignments_and_deletions(recipe, prefix, source):
    key = prefix + capture_plugin.CAPTURE_BACKEND_KEY
    extra = []
    if source == "overrides":
        recipe.specforge_overrides = {key: "auto", "training.max_steps": 7}
    else:
        extra = [key if prefix == "~" else key + "=auto", "run_id=test"]
    original = dict(recipe.specforge_overrides)
    argv = build_role_argv(recipe, "consumer", extra)
    assert [arg for arg in argv if "server_capture" in arg] == [
        f"{capture_plugin.CAPTURE_BACKEND_KEY}=plugin"
    ]
    assert recipe.specforge_overrides == original
    assert ("training.max_steps=7" if source == "overrides" else "run_id=test") in argv


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("available", [False, True])
def test_capture_preflight_checks_transport_only_for_default_entrypoint(
    recipe, monkeypatch, custom, available
):
    recipe.specforge_mode = "capture"
    recipe.specforge_capture = {"script": "custom.py"} if custom else {}
    monkeypatch.setattr(stack_preflight, "torch_kind", lambda: "hip")
    probes = []
    monkeypatch.setattr(
        stack_preflight, "capture_plugin_issues", lambda: probes.append(True) or ["bad plugin"]
    )
    monkeypatch.setattr(
        stack_preflight.shutil, "which", lambda name: "/bin/mooncake_master" if available else None
    )
    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(__version__="0.5.23"))
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", SimpleNamespace(ServerArgs=object))
    monkeypatch.setitem(sys.modules, "aiter", SimpleNamespace())
    issues = stack_preflight.collect_issues(recipe, env={})
    assert bool(probes) is not custom
    assert ("bad plugin" in issues) is not custom
    assert any("mooncake_master is not on PATH" in issue for issue in issues) == (
        not custom and not available
    )


def test_custom_capture_preflight_requires_importable_sglang(recipe, monkeypatch):
    recipe.specforge_mode = "capture"
    recipe.specforge_capture = {"script": "custom.py"}
    monkeypatch.setattr(stack_preflight, "torch_kind", lambda: "hip")
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", None)
    issues = stack_preflight.collect_issues(recipe, env={})
    assert any("ServerArgs is not importable for custom capture" in issue for issue in issues)


@pytest.mark.parametrize("advertised", [False, True])
def test_server_plugin_advertisement_check(monkeypatch, advertised):
    body = {"internal_states": [{"spec_capture_backend": "plugin"}]} if advertised else {}
    urls = []

    def response(url, **kwargs):
        import json

        urls.append(url)
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(capture_plugin.urllib.request, "urlopen", response)
    if advertised:
        capture_plugin.require_capture_plugin("http://127.0.0.1:30000", "server.log")
    else:
        with pytest.raises(RuntimeError, match="did not enable the plugin"):
            capture_plugin.require_capture_plugin("http://127.0.0.1:30000", "server.log")
    assert urls == ["http://127.0.0.1:30000/server_info"]


def test_online_registration_failure_cleans_up_and_does_not_publish_urls(recipe, monkeypatch, tmp_path):
    settings = online_settings(recipe, env={})
    started, stopped = [], []

    def start(*args):
        proc = object()
        started.append(proc)
        return proc

    def check(*args):
        if len(started) == 2:
            raise RuntimeError("plugin missing")

    monkeypatch.setattr(online_supervisor, "_popen", start)
    monkeypatch.setattr(online_supervisor, "_wait_sglang", lambda *args: None)
    monkeypatch.setattr(online_supervisor, "require_capture_plugin", check)
    monkeypatch.setattr(online_supervisor, "_kill_group", stopped.append)
    with pytest.raises(RuntimeError, match="plugin missing"):
        online_supervisor._start_local_sglang(settings, "192.0.2.1", "192.0.2.2", tmp_path, 0)
    assert stopped == started[::-1]
    assert not online_supervisor.sglang_url_path(str(tmp_path), 0).exists()
