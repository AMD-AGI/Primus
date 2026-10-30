###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests of the posttrain checkpoint hook (01_convert_checkpoints.py) for Granite:
the conversion-path decision (``_route``), the install decision (``bridge_deps_needed``)
and main()'s refusals.

The hook imports neither torch nor Megatron, so this file needs neither: it runs on any
CPU-only CI job. The few tests that reach the native Granite converter's support check
(which imports torch through the shared Llama loader) skip without torch.
"""

import importlib.util
import json
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest

GRANITE_HF = {
    "architectures": ["GraniteForCausalLM"],
    "attention_bias": False,
    "attention_multiplier": 0.015625,
    "embedding_multiplier": 12.0,
    "hidden_size": 2560,
    "intermediate_size": 8192,
    "logits_scaling": 10.0,
    "max_position_embeddings": 131072,
    "mlp_bias": False,
    "model_type": "granite",
    "num_attention_heads": 40,
    "num_hidden_layers": 40,
    "num_key_value_heads": 8,
    "residual_multiplier": 0.22,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 10000000,
    "tie_word_embeddings": True,
    "vocab_size": 100352,
}


def _needs_torch():
    pytest.importorskip("torch")  # the native converter's support check imports torch


def _hook():
    path = (
        pathlib.Path(__file__).resolve().parents[5]
        / "runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py"
    )
    spec = importlib.util.spec_from_file_location("granite_convert_hook", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _BridgePath(Exception):
    pass


@pytest.fixture(autouse=True)
def _isolated_data_path(tmp_path, monkeypatch):
    """No test writes under the checkout's data/ (the hook's default DATA_PATH)."""
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))


def _run_hook_main(hook, monkeypatch, hf_dir, enabled, preset=None, snapshot_download=None):
    """Run the hook's main() with config reading stubbed; return which path it took.
    ``preset`` is the native family of the model preset's model_type (None: not native).
    ``snapshot_download`` replaces ``huggingface_hub.snapshot_download``; without it any
    download attempt fails the test (no network access from a unit test).
    The Bridge path is stopped at its first log line (nothing is converted)."""
    calls = {"info": []}

    def _no_download(*a, **_k):
        pytest.fail(f"unexpected Hub download {a}")

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download or _no_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(hook, "_preset_family", lambda _cfg: preset)
    monkeypatch.setattr(hook, "get_checkpoint_config", lambda _cfg: (str(hf_dir), None))
    monkeypatch.setattr(
        hook, "read_native_opts", lambda _cfg: {"enabled": enabled, "dtype": None, "out": None}
    )
    monkeypatch.setattr(
        hook,
        "_native_main",
        lambda path, opts, decision=None, hf_cfg=None, megatron_name=None: calls.setdefault(
            "native", dict(opts, family=decision[0])
        ),
    )

    def _info(msg):
        calls["info"].append(msg)
        if "Megatron-Bridge conversion path" in msg:
            raise _BridgePath()

    monkeypatch.setattr(hook, "log_info", _info)
    monkeypatch.setattr(hook.sys, "argv", ["hook", "--config", "/x.yaml"])
    try:
        hook.main()
    except _BridgePath:
        calls["bridge"] = True
    return calls


@pytest.mark.parametrize(
    "override, family",
    [
        ({}, "granite"),
        ({"architectures": None}, "granite"),  # model_type only
        ({"architectures": ["GraniteForSequenceClassification"]}, None),
        ({"rope_scaling": {"rope_type": "linear", "factor": 2.0}}, None),
        ({"mlp_bias": True}, None),
        ({"model_type": "granitemoe", "architectures": ["GraniteMoeForCausalLM"]}, None),
        ({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}, "llama"),
        ({"hidden_act": "gelu"}, None),
        ({"text_config": "not-a-dict"}, None),
        ({"num_key_value_heads": None}, "granite"),  # HF: MHA
        ({"attention_bias": "false"}, None),
        ({"rope_scaling": {"rope_type": "default", "type": "linear"}}, None),
        (
            {
                "architectures": ["LlavaNextForConditionalGeneration"],
                "model_type": "llava_next",
                "text_config": dict(GRANITE_HF),
            },
            None,
        ),
    ],
)
def test_hook_detects_native_granite_only_when_supported(tmp_path, override, family):
    _needs_torch()
    cfg = {**GRANITE_HF, **override}
    if override.get("architectures", 1) is None:
        del cfg["architectures"]
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    hook = _hook()
    supported, reason, named = hook._native_decision(str(tmp_path))
    assert supported == family and reason
    # The family named by config.json, also when the converter refuses it: main() uses it
    # to refuse a Granite config instead of falling back to the Bridge.
    expected_named = {"granite": "granite", "llama": "llama"}.get(cfg.get("model_type"))
    assert named == expected_named, (named, cfg.get("model_type"))
    if family is None and cfg.get("model_type") == "granite":
        assert "native Granite converter" in reason, reason  # the real reason, not "no family"


@pytest.mark.parametrize("enabled", [None, True])
def test_hook_granite_default_is_native(tmp_path, monkeypatch, enabled):
    _needs_torch()
    # The pinned Megatron-Bridge has no Granite bridge, so native is the default.

    (tmp_path / "config.json").write_text(json.dumps(GRANITE_HF))
    hook = _hook()
    calls = _run_hook_main(hook, monkeypatch, tmp_path, enabled, preset="granite")
    assert "native" in calls, calls
    assert calls["native"]["family"] == "granite"


def test_hook_granite_forced_bridge_fails_early(tmp_path, monkeypatch):
    # native_ckpt_convert: false for Granite must stop early with a clear message (no
    # Granite bridge exists) and must not run the native converter's checks.

    (tmp_path / "config.json").write_text(json.dumps(GRANITE_HF))
    hook = _hook()
    monkeypatch.setattr(
        hook, "_native_decision", lambda _p: pytest.fail("native decision on the forced-off path")
    )
    errors = []
    monkeypatch.setattr(hook, "log_error", errors.append)
    with pytest.raises(SystemExit):
        _run_hook_main(hook, monkeypatch, tmp_path, False)
    assert errors and "no granite bridge" in errors[0] and "native_ckpt_convert" in errors[0]


def test_hook_forced_bridge_skips_native_decision(tmp_path, monkeypatch):
    # native_ckpt_convert: false for a family with a bridge (Llama) takes the Bridge path
    # without importing or running the native converter.

    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama"})
    )
    hook = _hook()
    monkeypatch.setattr(
        hook, "_native_decision", lambda _p: pytest.fail("native decision on the Bridge path")
    )
    assert "bridge" in _run_hook_main(hook, monkeypatch, tmp_path, False)


def test_hook_refused_granite_default_fails_without_bridge_fallback(tmp_path, monkeypatch):
    _needs_torch()
    # A Granite config the native converter refuses has no Bridge fallback either: the
    # default path stops with the converter's reason instead of a Bridge error.

    cfg = {**GRANITE_HF, "rope_scaling": {"rope_type": "linear", "factor": 2.0}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    hook = _hook()
    errors = []
    monkeypatch.setattr(hook, "log_error", errors.append)
    with pytest.raises(SystemExit):
        _run_hook_main(hook, monkeypatch, tmp_path, None, preset="granite")
    assert errors and "native Granite converter" in errors[0] and "no granite bridge" in errors[0]


def test_hook_non_native_model_default_uses_bridge(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["MistralForCausalLM"], "model_type": "mistral"})
    )
    hook = _hook()
    assert "bridge" in _run_hook_main(hook, monkeypatch, tmp_path, None)


def test_hook_refused_granite_fails_when_native_forced(tmp_path, monkeypatch):
    _needs_torch()
    cfg = {**GRANITE_HF, "rope_scaling": {"rope_type": "linear", "factor": 2.0}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    hook = _hook()
    errors = []
    monkeypatch.setattr(hook, "log_error", errors.append)
    with pytest.raises(SystemExit):
        _run_hook_main(hook, monkeypatch, tmp_path, True, preset="granite")
    # No advice to use the Bridge path: it has no Granite bridge.
    assert errors and "no granite bridge" in errors[0]
    assert "Set native_ckpt_convert: false" not in errors[0]


def test_hook_native_main_rechecks_support(tmp_path, monkeypatch):
    _needs_torch()
    cfg = {**GRANITE_HF, "rope_scaling": {"rope_type": "linear", "factor": 2.0}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    hook = _hook()
    errors = []
    monkeypatch.setattr(hook, "log_error", errors.append)
    monkeypatch.setattr(
        hook, "native_convert_checkpoint", lambda *a: pytest.fail("converted a refused config")
    )
    with pytest.raises(SystemExit):
        hook._native_main(str(tmp_path), {"dtype": None, "out": str(tmp_path / "o")})
    assert errors and "native Granite converter" in errors[0]


@pytest.mark.parametrize("enabled", [None, True, False])
def test_hook_malformed_config_fails_early(tmp_path, monkeypatch, enabled):
    # A config.json that is not valid JSON stops main() with a clear error on every path.
    (tmp_path / "config.json").write_text("{bad")
    hook = _hook()
    errors = []
    monkeypatch.setattr(hook, "log_error", errors.append)
    with pytest.raises(SystemExit):
        _run_hook_main(hook, monkeypatch, tmp_path, enabled)
    assert errors and "config.json" in errors[0]


def test_hook_native_decision_malformed_config_returns_reason(tmp_path):
    (tmp_path / "config.json").write_text("{bad")
    family, reason, detected = _hook()._native_decision(str(tmp_path))
    assert family is None and detected is None and "config.json" in reason


def test_hook_native_decision_uses_given_config(tmp_path):
    _needs_torch()
    # main() passes the config it already read: config.json is not read a second time.
    hook = _hook()
    assert hook._native_decision(str(tmp_path / "nowhere"), dict(GRANITE_HF))[0] == "granite"
    assert hook._native_decision(str(tmp_path / "nowhere"), None)[1].startswith("no config.json")


def test_hook_native_dtype_from_local_config(tmp_path):
    hook = _hook()
    (tmp_path / "config.json").write_text(json.dumps({**GRANITE_HF, "torch_dtype": "float32"}))
    assert hook._native_dtype_from_config(str(tmp_path), None) == "fp32"
    assert hook._native_dtype_from_config(str(tmp_path), "bf16") == "bf16"


def test_hook_preset_family_reads_model_type(tmp_path):
    # The real experiment-config path: the granite preset sets model_type: granite.
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        "work_group: amd\nuser_name: root\nexp_name: granite_sft_probe\n"
        f"workspace: {tmp_path / 'out'}\n"
        "modules:\n  post_trainer:\n    framework: megatron\n    config: sft_trainer.yaml\n"
        "    model: granite_4_1_3b.yaml\n    overrides:\n"
        "      tokenizer_model: ibm-granite/granite-4.1-3b\n"
    )
    hook = _hook()
    assert hook._preset_family(str(exp)) == "granite"
    assert hook.bridge_deps_needed(str(exp), "ibm-granite/granite-4.1-3b") is False


def test_hook_reads_one_trainer_module(monkeypatch):
    # get_checkpoint_config, read_native_opts and _preset_family read the same module (the
    # first of sft_trainer / post_trainer), so their answers cannot come from different ones.
    hook = _hook()
    modules = {
        "sft_trainer": SimpleNamespace(params=SimpleNamespace(model_type="granite", tokenizer_model="a")),
        "post_trainer": SimpleNamespace(params=SimpleNamespace(model_type="llama", tokenizer_model="b")),
    }
    monkeypatch.setattr(hook, "load_primus_config", lambda *_a: None)
    monkeypatch.setattr(hook, "get_module_config", lambda _cfg, name: modules.get(name))
    assert hook._preset_family("/x.yaml") == "granite"
    assert hook.get_checkpoint_config("/x.yaml") == ("a", None)


# ---------------------------------------------------------------------------
# _route: one deterministic decision from the preset, the local config.json and
# native_ckpt_convert (never from the HF cache)
# ---------------------------------------------------------------------------
_LLAMA = {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}
_MISTRAL = {"architectures": ["MistralForCausalLM"], "model_type": "mistral"}
_HYBRID = {"architectures": ["GraniteMoeHybridForCausalLM"], "model_type": "granitemoehybrid"}
_GRANITE_MOE = {"architectures": ["GraniteMoeForCausalLM"], "model_type": "granitemoe"}


@pytest.mark.parametrize(
    "hf_cfg, preset, enabled, expected",
    [
        # A Granite preset never routes to the Bridge (it has no Granite bridge).
        (_HYBRID, "granite", None, ("refuse_mismatch", "granite")),  # not a Granite dense checkpoint
        (_GRANITE_MOE, "granite", None, ("refuse_mismatch", "granite")),
        ({}, "granite", None, ("refuse_mismatch", "granite")),
        (_MISTRAL, "granite", None, ("refuse_mismatch", "granite")),
        (dict(GRANITE_HF), "granite", None, ("native", "granite")),
        (dict(GRANITE_HF), "granite", True, ("native", "granite")),
        (None, "granite", None, ("download", "granite")),
        (None, "granite", True, ("download", "granite")),
        (None, "granite", False, ("refuse_forced_off", "granite")),
        (dict(GRANITE_HF), "granite", False, ("refuse_forced_off", "granite")),
        (_LLAMA, "granite", None, ("refuse_mismatch", "granite")),
        (_LLAMA, "granite", True, ("refuse_mismatch", "granite")),
        # A Granite checkpoint needs the Granite preset (else it builds without multipliers).
        (dict(GRANITE_HF), None, None, ("refuse_mismatch", "granite")),
        (dict(GRANITE_HF), "llama", None, ("refuse_mismatch", "granite")),
        (dict(GRANITE_HF), "llama", True, ("refuse_mismatch", "granite")),
        (dict(GRANITE_HF), None, False, ("refuse_forced_off", "granite")),
        (_LLAMA, None, None, ("native", "llama")),
        (_LLAMA, "llama", False, ("bridge", "llama")),
        (_MISTRAL, None, None, ("bridge", None)),
        (_MISTRAL, None, True, ("native", None)),
        (None, None, None, ("bridge", None)),
        (None, "llama", None, ("bridge", None)),
        (None, None, True, ("native", None)),
        # No family without a Bridge converter: the local config.json decides, as before.
        (_LLAMA, "qwen3", None, ("native", "llama")),
    ],
)
def test_route(hf_cfg, preset, enabled, expected):
    assert _hook()._route(hf_cfg, preset, enabled) == expected


def test_route_reads_nothing_but_its_arguments(monkeypatch):
    # The decision has no hidden input (HF cache, network, files): the same arguments give
    # the same route on every node.
    hook = _hook()
    monkeypatch.setattr(hook, "_load_hf_config", lambda *_a: pytest.fail("read a file"))
    monkeypatch.setattr(hook, "read_hf_config", lambda *_a: pytest.fail("read a file"))
    assert hook._route(None, None, None) == ("bridge", None)
    assert hook._route(None, "granite", None) == ("download", "granite")


def test_hook_granite_hub_id_forced_off_fails_early(tmp_path, monkeypatch):
    # native_ckpt_convert: false with the Granite preset stops before any download.
    hook = _hook()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hook, "_download_hf_checkpoint", lambda *a, **k: pytest.fail("downloaded"))
    errors = []
    monkeypatch.setattr(hook, "log_error", errors.append)
    with pytest.raises(SystemExit):
        _run_hook_main(hook, monkeypatch, "ibm-granite/granite-4.1-3b", False, preset="granite")
    assert errors and "no granite bridge" in errors[0]


def _download_setup(tmp_path, monkeypatch, enabled=None, node_rank="0", snapshot_cfg=GRANITE_HF):
    """main() on the Granite preset with a Hub id; returns (hook, fetched, converted, out)."""
    snap = tmp_path / "snapshots" / "c0650403"
    snap.mkdir(parents=True)
    if snapshot_cfg is not None:
        (snap / "config.json").write_text(json.dumps(snapshot_cfg))
    hook = _hook()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NODE_RANK", node_rank)
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    fetched, converted = [], []
    monkeypatch.setattr(
        hook, "_download_hf_checkpoint", lambda repo, family: fetched.append((repo, family)) or str(snap)
    )
    monkeypatch.setattr(
        hook, "_native_decision", lambda *_a: ("granite", "native granite converter", "granite")
    )
    monkeypatch.setattr(
        hook,
        "_native_main",
        lambda path, opts, decision=None, hf_cfg=None, megatron_name=None: converted.append(
            (path, megatron_name)
        ),
    )
    monkeypatch.setattr(hook, "_preset_family", lambda _cfg: "granite")
    monkeypatch.setattr(hook, "get_checkpoint_config", lambda _cfg: ("ibm-granite/granite-4.1-3b", None))
    monkeypatch.setattr(
        hook, "read_native_opts", lambda _cfg: {"enabled": enabled, "dtype": None, "out": None}
    )
    monkeypatch.setattr(hook.sys, "argv", ["hook", "--config", "/x.yaml"])
    out = tmp_path / "data" / "megatron_checkpoints" / "granite-4.1-3b_native"
    return hook, fetched, converted, out, snap


@pytest.mark.parametrize("enabled", [None, True])
def test_hook_granite_hub_id_is_downloaded_and_converted_natively(tmp_path, monkeypatch, enabled):
    # The Granite preset's tokenizer_model is a Hub id: node rank 0 downloads it and
    # converts natively; the output is named after the repo, not the snapshot hash.
    hook, fetched, converted, _out, snap = _download_setup(tmp_path, monkeypatch, enabled)
    hook.main()
    assert fetched == [("ibm-granite/granite-4.1-3b", "granite")]
    assert converted == [(str(snap), "granite-4.1-3b_native")]


def test_hook_hub_id_other_nodes_only_wait(tmp_path, monkeypatch, capsys):
    # Nodes other than rank 0 never download: they wait (the download timeout) for rank
    # 0's converted checkpoint and pass it on.
    hook, fetched, converted, out, _snap = _download_setup(tmp_path, monkeypatch, node_rank="1")
    monkeypatch.delenv("PRIMUS_CKPT_CONVERT_TIMEOUT", raising=False)
    waited = []
    monkeypatch.setattr(
        hook,
        "wait_for_conversion",
        lambda done, lock, timeout=None, failed_file=None, checkpoint_dir=None: waited.append(
            (done, timeout, failed_file, checkpoint_dir)
        ),
    )
    hook.main()
    assert fetched == [] and converted == []
    assert waited == [(pathlib.Path(f"{out}.done"), 7200, pathlib.Path(f"{out}.failed"), out)]
    assert f"extra.pretrained_checkpoint={out}" in capsys.readouterr().out


@pytest.mark.parametrize("node_rank", ["0", "1"])
def test_hook_hub_id_already_converted_needs_no_network(tmp_path, monkeypatch, capsys, node_rank):
    # A rerun with the converted checkpoint on disk uses it on every node, with no download
    # (so an offline node without HF cache works).
    hook, fetched, converted, out, _snap = _download_setup(tmp_path, monkeypatch, node_rank=node_rank)
    out.mkdir(parents=True)
    (out / "latest_checkpointed_iteration.txt").write_text("release")
    monkeypatch.setattr(hook, "wait_for_conversion", lambda *_a, **_k: pytest.fail("waited"))
    hook.main()
    assert fetched == [] and converted == []
    assert f"extra.pretrained_checkpoint={out}" in capsys.readouterr().out


def test_hook_downloaded_snapshot_without_config_is_refused(tmp_path, monkeypatch):
    # A repo without config.json (allow_patterns matched nothing) is refused with that
    # reason, never sent to the Bridge path; the waiting nodes are told at once.
    hook, fetched, converted, out, _snap = _download_setup(tmp_path, monkeypatch, snapshot_cfg=None)
    errors, info = [], []
    monkeypatch.setattr(hook, "log_error", errors.append)
    monkeypatch.setattr(hook, "log_info", info.append)
    with pytest.raises(SystemExit):
        hook.main()
    assert errors and "has no config.json" in errors[0], errors
    assert not any("Megatron-Bridge" in m for m in info) and converted == []
    assert pathlib.Path(f"{out}.failed").exists()


def test_hook_downloaded_non_granite_snapshot_is_refused(tmp_path, monkeypatch):
    hook, _fetched, converted, _out, _snap = _download_setup(
        tmp_path, monkeypatch, snapshot_cfg={"architectures": ["LlamaForCausalLM"], "model_type": "llama"}
    )
    errors = []
    monkeypatch.setattr(hook, "log_error", errors.append)
    with pytest.raises(SystemExit):
        hook.main()
    assert errors and "is not a Granite checkpoint" in errors[0] and converted == []


def _clock(hook, monkeypatch, on_sleep=None):
    """Fake monotonic clock; ``on_sleep(n)`` runs at the n-th sleep (n from 1)."""
    now, calls = [0.0], [0]

    def sleep(s):
        now[0] += s
        calls[0] += 1
        if on_sleep is not None:
            on_sleep(calls[0])

    monkeypatch.setattr(hook.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(hook.time, "sleep", sleep)
    return calls


def test_wait_for_conversion_stops_on_rank0_failure(tmp_path, monkeypatch):
    # rank 0 writes .failed while this node waits: stop at once with its reason.
    hook = _hook()
    failed = tmp_path / "x.failed"
    calls = _clock(hook, monkeypatch, lambda n: n == 1 and failed.write_text("OSError: offline\n"))
    with pytest.raises(RuntimeError, match="OSError: offline"):
        hook.wait_for_conversion(tmp_path / "x.done", tmp_path / "x.lock", 60, failed_file=failed)
    assert calls[0] == 1


@pytest.mark.parametrize("age_s", [0, 90, 10**6])
def test_wait_for_conversion_ignores_an_earlier_failure_marker(tmp_path, monkeypatch, age_s):
    # A .failed left by an earlier run (of any age, even seconds) does not stop this wait:
    # rank 0 may not have retried yet.
    import os
    import time as _time

    hook = _hook()
    failed = tmp_path / "x.failed"
    failed.write_text("old run\n")
    t = _time.time() - age_s
    os.utime(failed, (t, t))
    _clock(hook, monkeypatch)
    with pytest.raises(TimeoutError):
        hook.wait_for_conversion(tmp_path / "x.done", tmp_path / "x.lock", 10, failed_file=failed)


def test_wait_for_conversion_rank0_retry_after_failure(tmp_path, monkeypatch):
    # The earlier .failed is replaced (rank 0 deletes it, fails again): that one counts.
    hook = _hook()
    failed = tmp_path / "x.failed"
    failed.write_text("old run\n")

    def on_sleep(n):
        if n == 1:
            failed.unlink()
        if n == 2:
            failed.write_text("new failure, longer text\n")

    _clock(hook, monkeypatch, on_sleep)
    with pytest.raises(RuntimeError, match="new failure"):
        hook.wait_for_conversion(tmp_path / "x.done", tmp_path / "x.lock", 60, failed_file=failed)


def test_wait_for_conversion_ignores_a_done_without_checkpoint(tmp_path, monkeypatch):
    # .done is left but the checkpoint was deleted (rank 0 is converting again): keep waiting.
    hook = _hook()
    ckpt = tmp_path / "x"
    done = tmp_path / "x.done"
    done.touch()
    _clock(hook, monkeypatch)
    with pytest.raises(TimeoutError):
        hook.wait_for_conversion(done, tmp_path / "x.lock", 10, checkpoint_dir=ckpt)


def test_wait_for_conversion_ignores_a_done_older_than_the_checkpoint(tmp_path, monkeypatch):
    # rank 0 is rewriting the checkpoint (its latest file is newer than the old .done).
    import os

    hook = _hook()
    ckpt = tmp_path / "x"
    ckpt.mkdir()
    done = tmp_path / "x.done"
    done.touch()
    os.utime(done, (1, 1))
    (ckpt / "latest_checkpointed_iteration.txt").write_text("1")
    _clock(hook, monkeypatch)
    with pytest.raises(TimeoutError):
        hook.wait_for_conversion(done, tmp_path / "x.lock", 10, checkpoint_dir=ckpt)


def test_wait_for_conversion_accepts_done(tmp_path, monkeypatch):
    import os

    hook = _hook()
    ckpt = tmp_path / "x"
    ckpt.mkdir()
    latest = ckpt / "latest_checkpointed_iteration.txt"
    latest.write_text("1")
    os.utime(latest, (1, 1))
    done = tmp_path / "x.done"
    done.touch()  # complete checkpoint from an earlier run: accepted at once
    calls = _clock(hook, monkeypatch)
    hook.wait_for_conversion(done, tmp_path / "x.lock", 10, checkpoint_dir=ckpt)
    assert calls[0] == 0
    # written during the wait: accepted
    done2 = tmp_path / "y.done"
    calls = _clock(hook, monkeypatch, lambda n: n == 2 and done2.touch())
    hook.wait_for_conversion(done2, tmp_path / "y.lock", 60)
    assert calls[0] == 2


def test_rank0_retry_removes_earlier_markers(tmp_path):
    hook = _hook()
    out = tmp_path / "ckpt"
    for suffix in (".done", ".failed"):
        pathlib.Path(f"{out}{suffix}").write_text("old")
    hook._start_attempt(out)
    assert not pathlib.Path(f"{out}.done").exists() and not pathlib.Path(f"{out}.failed").exists()


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("", None),
        ("null", None),
        (True, True),
        ("true", True),
        ("1", True),
        (1, True),
        (False, False),
        ("false", False),
        ("False", False),
        ("0", False),
        (0, False),
        ("off", False),
    ],
)
def test_native_ckpt_convert_is_parsed_strictly(value, expected):
    assert _hook()._tri_state(value, "native_ckpt_convert") is expected


@pytest.mark.parametrize("value", ["maybe", 2, 0.5])
def test_native_ckpt_convert_refuses_other_values(value, monkeypatch):
    hook = _hook()
    errors = []
    monkeypatch.setattr(hook, "log_error", errors.append)
    with pytest.raises(SystemExit):
        hook._tri_state(value, "native_ckpt_convert")
    assert "must be true, false or unset" in errors[0]


def test_hook_failed_download_stops_with_the_reason(tmp_path, monkeypatch):
    hook = _hook()
    monkeypatch.chdir(tmp_path)

    def _fail(*_a, **_k):
        raise OSError("offline and not cached")

    errors = []
    monkeypatch.setattr(hook, "log_error", errors.append)
    with pytest.raises(SystemExit):
        _run_hook_main(
            hook, monkeypatch, "ibm-granite/granite-4.1-3b", None, preset="granite", snapshot_download=_fail
        )
    assert errors and "downloading it failed" in errors[0] and "offline and not cached" in errors[0]
    assert "huggingface-cli download" in errors[0]


def test_hook_granite_checkpoint_with_other_preset_is_refused(tmp_path, monkeypatch):
    # A Granite config.json with a non-Granite preset would build a plain GPT model and drop
    # the multipliers: refused.
    (tmp_path / "config.json").write_text(json.dumps(GRANITE_HF))
    hook = _hook()
    errors = []
    monkeypatch.setattr(hook, "log_error", errors.append)
    with pytest.raises(SystemExit):
        _run_hook_main(hook, monkeypatch, tmp_path, None, preset="llama")
    assert errors and "is a Granite checkpoint" in errors[0] and "model_type is llama" in errors[0]
    assert "model_type: granite" in errors[0] and "use a llama" not in errors[0].lower()


@pytest.mark.parametrize("value, expected", [(False, False), (True, True), (None, None)])
def test_read_native_opts_keeps_explicit_false(monkeypatch, value, expected):
    # native_ckpt_convert: false under overrides (params) is a value, not "unset".
    hook = _hook()
    module = SimpleNamespace(params=SimpleNamespace(native_ckpt_convert=value))
    monkeypatch.setattr(hook, "_trainer_module", lambda _cfg: module)
    assert hook.read_native_opts("/x.yaml")["enabled"] is expected


def test_hook_granite_preset_with_llama_checkpoint_is_refused(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(_LLAMA))
    hook = _hook()
    errors = []
    monkeypatch.setattr(hook, "log_error", errors.append)
    with pytest.raises(SystemExit):
        _run_hook_main(hook, monkeypatch, tmp_path, None, preset="granite")
    assert errors and "model_type is granite" in errors[0] and "is not a Granite checkpoint" in errors[0]
    assert "'llama'" in errors[0] and "model_type: granite" not in errors[0]  # one, consistent fix


@pytest.mark.parametrize("cfg", [_HYBRID, _GRANITE_MOE, _MISTRAL])
def test_hook_granite_preset_non_native_checkpoint_never_uses_bridge(tmp_path, monkeypatch, cfg):
    # A Granite preset with a granitemoehybrid (or other non-Granite) config.json is a
    # preset/checkpoint mismatch: refused early, like a Llama checkpoint, never Bridged.
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    hook = _hook()
    errors = []
    monkeypatch.setattr(hook, "log_error", errors.append)
    with pytest.raises(SystemExit):  # the Bridge path would raise _BridgePath instead
        _run_hook_main(hook, monkeypatch, tmp_path, None, preset="granite")
    assert errors and "is not a Granite checkpoint" in errors[0], errors
    assert repr(cfg["model_type"]) in errors[0]


@pytest.mark.parametrize(
    "preset, cfg, enabled, needed",
    [
        ("granite", None, None, False),  # Granite Hub id: downloaded, converted natively
        ("granite", None, False, False),  # native_ckpt_convert: false is refused for Granite
        ("granite", _HYBRID, None, False),
        ("granite", _LLAMA, None, False),  # refused: preset and checkpoint disagree
        ("granite", _LLAMA, False, False),
        (None, dict(GRANITE_HF), None, False),
        (None, {**GRANITE_HF, "rope_scaling": {"rope_type": "linear", "factor": 2.0}}, None, False),
        (None, _LLAMA, None, False),
        (None, _LLAMA, False, True),
        (None, _MISTRAL, None, True),
        (None, _MISTRAL, True, False),
        (None, None, None, True),  # a Hub id of a family with a bridge
    ],
)
def test_hook_bridge_deps_needed(tmp_path, monkeypatch, preset, cfg, enabled, needed):
    hook = _hook()
    monkeypatch.setattr(hook, "_preset_family", lambda _cfg: preset)
    monkeypatch.setattr(
        hook, "read_native_opts", lambda _cfg: {"enabled": enabled, "dtype": None, "out": None}
    )
    hf_path = str(tmp_path / "missing-hub-id")
    if cfg is not None:
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        hf_path = str(tmp_path)
    assert hook.bridge_deps_needed("/x.yaml", hf_path) is needed


@pytest.mark.parametrize(
    "preset, cfg, enabled, path",
    [
        ("granite", _LLAMA, False, "refused"),
        ("granite", _MISTRAL, None, "refused"),
        ("granite", _LLAMA, None, "refused"),
        ("granite", None, None, "native"),  # Hub id: downloaded (stubbed), then native
        (None, _LLAMA, False, "bridge"),
        (None, _MISTRAL, None, "bridge"),
        (None, _LLAMA, None, "native"),
        (None, None, None, "bridge"),
    ],
)
def test_hook_install_decision_matches_main(tmp_path, monkeypatch, preset, cfg, enabled, path):
    # The Bridge dependencies are installed iff main() takes the Bridge path. Each row
    # also pins the path main() takes, so an unrelated early exit cannot pass.
    hf_path = "org/model"
    monkeypatch.chdir(tmp_path)
    if cfg is not None:
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        hf_path = str(tmp_path)
    snap = tmp_path / "snapshot"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps(GRANITE_HF))
    hook = _hook()
    monkeypatch.setattr(hook, "_preset_family", lambda _cfg: preset)
    monkeypatch.setattr(
        hook, "read_native_opts", lambda _cfg: {"enabled": enabled, "dtype": None, "out": None}
    )
    monkeypatch.setattr(hook, "_native_decision", lambda *_a: ("x", "native x converter", "x"))
    needed = hook.bridge_deps_needed("/x.yaml", hf_path)
    errors = []
    monkeypatch.setattr(hook, "log_error", errors.append)
    try:
        calls = _run_hook_main(
            hook, monkeypatch, hf_path, enabled, preset=preset, snapshot_download=lambda *_a, **_k: str(snap)
        )
        took = "bridge" if calls.get("bridge") else "native" if "native" in calls else "none"
    except SystemExit:
        took = "refused"
        assert errors, "exited without a reason"
    assert took == path, (took, errors)
    assert needed is (path == "bridge")


def test_hook_reads_the_experiment_config_once(tmp_path, monkeypatch):
    # get_checkpoint_config, read_native_opts and _preset_family share one parsed config.
    hook = _hook()
    loads = []
    module = SimpleNamespace(
        params=SimpleNamespace(tokenizer_model="org/m", model_type="granite", native_ckpt_convert=True)
    )
    monkeypatch.setattr(hook, "load_primus_config", lambda *a: loads.append(a) or object())
    monkeypatch.setattr(
        hook, "get_module_config", lambda _cfg, name: module if name == "post_trainer" else None
    )
    path = str(tmp_path / "exp.yaml")
    assert hook.get_checkpoint_config(path) == ("org/m", None)
    assert hook.read_native_opts(path) == {"enabled": True, "dtype": None, "out": None}
    assert hook._preset_family(path) == "granite"
    assert len(loads) == 1


def test_wait_for_conversion_uses_wall_time(tmp_path, monkeypatch):
    # The deadline is wall time (the old loop counted 5 s per 2 s sleep), and the error
    # names the knob to wait longer.
    hook = _hook()
    now = [0.0]
    monkeypatch.setattr(hook.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(hook.time, "sleep", lambda s: now.__setitem__(0, now[0] + s))
    with pytest.raises(TimeoutError, match="PRIMUS_CKPT_CONVERT_TIMEOUT"):
        hook.wait_for_conversion(tmp_path / "x.done", tmp_path / "x.lock", timeout=60)
    assert 60 <= now[0] < 63


def test_wait_timeout_is_longer_for_a_download(monkeypatch):
    # Node rank 0 downloads the checkpoint before it converts, so the other nodes wait longer.
    hook = _hook()
    monkeypatch.delenv("PRIMUS_CKPT_CONVERT_TIMEOUT", raising=False)
    assert hook._wait_timeout(downloading=True) > hook._wait_timeout() == 600
    monkeypatch.setenv("PRIMUS_CKPT_CONVERT_TIMEOUT", "42")
    assert hook._wait_timeout(downloading=True) == hook._wait_timeout() == 42
