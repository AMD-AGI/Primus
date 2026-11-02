###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The posttrain conversion hook converts Muse Spark natively by default (no Megatron-Bridge model)."""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

HOOK = (
    Path(__file__).resolve().parents[3]
    / "runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py"
)


def _hook():
    spec = importlib.util.spec_from_file_location("convert_hook", HOOK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "enabled, family, expected",
    [
        # Muse Spark is native by default
        (None, "muse_spark", True),
        (False, "muse_spark", False),
        (True, "muse_spark", True),
        # other families keep the native default and honour the opt-out
        *[(None, f, True) for f in ("qwen3", "deepseek", "llama")],
        *[(False, f, False) for f in ("qwen3", "deepseek", "llama")],
        *[(True, f, True) for f in ("qwen3", "deepseek", "llama")],
        # no native converter: unset -> Megatron-Bridge
        (None, None, False),
    ],
)
def test_choose_native(enabled, family, expected):
    assert _hook().choose_native(enabled, family) is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        # numbers and number strings: only 0 / 1, the same rule for both
        (1.0, True),
        (0.0, False),
        ("1.0", True),
        ("0.0", False),
        *[(v, True) for v in ("true", "True", "1", "yes", "on", " ON ")],
        *[(v, False) for v in ("false", "False", "0", "no", "off")],
        ("", None),
        ("  ", None),  # an empty substitution means unset, not opt-out
    ],
)
def test_parse_native_enabled(value, expected):
    assert _hook().parse_native_enabled(value) is expected


@pytest.mark.parametrize(
    "value",
    # the same numbers are rejected as int, float and string; and the old bool() spellings
    [
        "maybe",
        "y",
        "enabled",
        2,
        "2",
        -1,
        "-1",
        1.5,
        "1.5",
        float("nan"),
        "nan",
        float("inf"),
        [1],
        # numbers outside the closed set of spellings ("1", "0", "1.0", "0.0")
        "1e0",
        "+1",
        "1.00",
        "-0",
        "0x1",
    ],
)
def test_parse_native_enabled_rejects_unknown_values(value):
    with pytest.raises(ValueError, match="native_ckpt_convert"):
        _hook().parse_native_enabled(value)


def test_main_reports_a_bad_flag_without_a_traceback(monkeypatch, tmp_path, capsys):
    hook = _hook()
    monkeypatch.setattr(sys, "argv", ["hook", "--config", str(tmp_path / "exp.yaml")])
    monkeypatch.setattr(hook, "get_checkpoint_config", lambda _c: (str(tmp_path), None))

    def bad(_c):
        raise ValueError("native_ckpt_convert: cannot parse 'maybe' as a boolean")

    monkeypatch.setattr(hook, "read_native_opts", bad)
    with pytest.raises(SystemExit) as exc:
        hook.main()
    assert exc.value.code == 1
    err = capsys.readouterr()
    assert "cannot parse 'maybe'" in err.out + err.err


@pytest.mark.parametrize("value, expected", [("false", False), ("true", True), (None, None), ("off", False)])
def test_read_native_opts_parses_strings(monkeypatch, value, expected):
    hook = _hook()
    params = types.SimpleNamespace(native_ckpt_convert=value)
    module = types.SimpleNamespace(params=params)
    monkeypatch.setattr(hook, "load_primus_config", lambda *_a: object())
    monkeypatch.setattr(
        hook, "get_module_config", lambda _cfg, name: module if name == "sft_trainer" else None
    )
    assert hook.read_native_opts("exp.yaml")["enabled"] is expected


def _cfg(tmp_path, cfg):
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return str(tmp_path)


@pytest.mark.parametrize(
    "cfg, family",
    [
        ({"model_type": "muse_spark"}, "muse_spark"),
        ({"architectures": ["MuseSparkForConditionalGeneration"]}, "muse_spark"),
        ({"architectures": ["MuseSparkForCausalLM"]}, "muse_spark"),
        ({"model_type": "muse_spark_text"}, "muse_spark"),
        ({"text_config": {"model_type": "muse_spark_text"}}, "muse_spark"),
        # other multimodal wrappers keep falling through to Megatron-Bridge
        ({"model_type": "llava", "text_config": {"model_type": "llama"}}, None),
        ({"model_type": "qwen3_vl", "text_config": {"model_type": "qwen3"}}, None),
        ({"architectures": ["LlamaForCausalLM"]}, "llama"),
        ({"model_type": "deepseek_v3"}, "deepseek"),
        ({"model_type": "qwen3_moe"}, "qwen3"),
        ({"model_type": "mistral"}, None),
    ],
)
def test_family_detection(tmp_path, cfg, family):
    assert _hook()._detect_native_family(_cfg(tmp_path, cfg)) == family


def _route(monkeypatch, tmp_path, enabled):
    """Run main() with mocked config / converters; return which path ran."""
    hook = _hook()
    hf = _cfg(tmp_path, {"text_config": {"model_type": "muse_spark_text"}})
    calls = []
    monkeypatch.setattr(sys, "argv", ["hook", "--config", str(tmp_path / "exp.yaml")])
    monkeypatch.setattr(hook, "get_checkpoint_config", lambda _c: (hf, None))

    def opts(_c):
        if isinstance(enabled, Exception):
            raise enabled
        return {"enabled": enabled, "dtype": None, "out": None}

    monkeypatch.setattr(hook, "read_native_opts", opts)
    monkeypatch.setattr(hook, "_native_main", lambda p, o, f: calls.append(("native", p)))
    monkeypatch.setattr(hook, "convert_checkpoint", lambda p, out: calls.append(("bridge", p)))
    monkeypatch.setattr(hook, "fix_common_pt_for_megatron_lm", lambda *_a: None)
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("NODE_RANK", "0")
    hook.main()
    return [c[0] for c in calls]


def test_main_fails_early_when_muse_spark_is_opted_out(monkeypatch, tmp_path, capsys):
    # explicit opt-out: Megatron-Bridge cannot convert muse_spark, so main()
    # exits before any converter runs instead of failing late inside the Bridge.
    with pytest.raises(SystemExit) as exc:
        _route(monkeypatch, tmp_path, False)
    assert exc.value.code == 1
    err = capsys.readouterr()
    assert "native_ckpt_convert" in err.out + err.err and "native converter" in err.out + err.err


@pytest.mark.parametrize("enabled", [None, True])
def test_main_routes_muse_spark_to_native_by_default(monkeypatch, tmp_path, enabled):
    assert _route(monkeypatch, tmp_path, enabled) == ["native"]


def test_native_convert_dispatches_to_the_muse_spark_text_loader(monkeypatch, tmp_path):
    hook = _hook()
    seen = {}
    common = types.ModuleType("primus.backends.megatron.checkpoint.native_convert_common")
    common.ensure_megatron_on_path = lambda: None
    common.apply_convert_patches = lambda module_name: 0
    conv = types.ModuleType("primus.backends.megatron.checkpoint.native_loader_muse_spark_text")
    conv.convert = lambda hf, out, **kw: seen.update(hf=hf, out=out, **kw)
    import primus.backends.megatron.checkpoint as ckpt_pkg

    monkeypatch.setitem(sys.modules, common.__name__, common)
    monkeypatch.setitem(sys.modules, conv.__name__, conv)
    monkeypatch.setattr(ckpt_pkg, "native_convert_common", common, raising=False)
    monkeypatch.setattr(ckpt_pkg, "native_loader_muse_spark_text", conv, raising=False)
    hf = _cfg(tmp_path, {"text_config": {"model_type": "muse_spark_text"}, "torch_dtype": "bfloat16"})
    hook.native_convert_checkpoint(hf, str(tmp_path / "out"), {"family": "muse_spark", "dtype": "bf16"})
    assert seen["hf"] == hf and seen["out"] == str(tmp_path / "out")
    assert seen["tensor_parallel_size"] == 1 and seen["expert_parallel_size"] == 1
    assert "max_params" not in seen  # unset: the converter default
    seen.clear()
    hook.native_convert_checkpoint(
        hf, str(tmp_path / "out"), {"family": "muse_spark", "dtype": "bf16", "max_params": "100e9"}
    )
    assert seen["max_params"] == "100e9"  # native_ckpt_max_params reaches the converter
    seen.clear()
    hook.native_convert_checkpoint(
        hf, str(tmp_path / "out"), {"family": "muse_spark", "dtype": "bf16", "max_params": 0}
    )
    assert seen["max_params"] == 0  # passed on as set: the converter rejects it


def test_read_native_opts_reads_the_max_params_key(monkeypatch):
    hook = _hook()
    monkeypatch.setattr(hook, "load_primus_config", lambda *_a: object())
    params = types.SimpleNamespace(native_ckpt_convert=True, native_ckpt_max_params=120_000_000_000)
    module = types.SimpleNamespace(params=params)
    monkeypatch.setattr(
        hook, "get_module_config", lambda _cfg, name: module if name == "sft_trainer" else None
    )
    assert hook.read_native_opts("exp.yaml")["max_params"] == 120_000_000_000
    module.params = types.SimpleNamespace(native_ckpt_convert=True)
    assert hook.read_native_opts("exp.yaml")["max_params"] is None


def test_read_native_opts_keeps_a_zero_max_params_for_the_converter_to_reject(monkeypatch):
    # 0 is a set (bad) value, not "unset": it must not fall back to the module level or the
    # 60B default, so parse_max_params rejects it with an error that names the key
    hook = _hook()
    monkeypatch.setattr(hook, "load_primus_config", lambda *_a: object())
    params = types.SimpleNamespace(native_ckpt_convert=True, native_ckpt_max_params=0)
    module = types.SimpleNamespace(params=params, native_ckpt_max_params=10**12)
    monkeypatch.setattr(
        hook, "get_module_config", lambda _cfg, name: module if name == "sft_trainer" else None
    )
    assert hook.read_native_opts("exp.yaml")["max_params"] == 0


def test_muse_spark_opt_out_fails_early_not_in_bridge():
    # Megatron-Bridge has no Muse Spark model: an opt-out stops with the fix
    hook = _hook()
    msg = hook.bridge_unsupported_reason("muse_spark", "/hf/muse")
    assert msg and "native_ckpt_convert is false" in msg and "native converter" in msg
    assert "native_ckpt_convert: None" not in msg
    for family in ("qwen3", "deepseek", "llama", None):
        assert hook.bridge_unsupported_reason(family) is None


def test_forced_native_on_an_unknown_family_names_the_supported_families(monkeypatch, tmp_path, capsys):
    hook = _hook()
    hf = _cfg(tmp_path, {"model_type": "mistral"})
    monkeypatch.setattr(sys, "argv", ["hook", "--config", str(tmp_path / "exp.yaml")])
    monkeypatch.setattr(hook, "get_checkpoint_config", lambda _c: (hf, None))
    monkeypatch.setattr(hook, "read_native_opts", lambda _c: {"enabled": True, "dtype": None, "out": None})
    with pytest.raises(SystemExit) as exc:
        hook.main()
    assert exc.value.code == 1
    out = capsys.readouterr()
    assert f"(supported: {hook.NATIVE_SUPPORTED_DESC})" in out.out + out.err
    assert "Muse Spark" in hook.NATIVE_SUPPORTED_DESC


def test_native_convert_without_a_family_names_the_supported_families(tmp_path):
    hook = _hook()
    hf = _cfg(tmp_path, {"model_type": "mistral"})
    with pytest.raises(ValueError) as err:
        hook.native_convert_checkpoint(hf, str(tmp_path / "out"), {})
    assert f"(supported: {hook.NATIVE_SUPPORTED_DESC})" in str(err.value)


def _install_script_decide(hook, monkeypatch, enabled, hf_path, pretrained=None):
    """Run the Python block embedded in 00_install_requirements.sh (decide() and its
    try / except, taken from the script, not rewritten here) against ``hook`` and return
    its ``result`` (a SKIP gets ``:<action>`` added) and its ``reason`` variable. The
    block's own output is swallowed (stdout redirected)."""
    import contextlib
    import io
    import pathlib
    import re as _re

    script = (
        pathlib.Path(__file__).resolve().parents[3]
        / "runner/helpers/hooks/train/posttrain/megatron/00_install_requirements.sh"
    ).read_text()
    m = _re.search(r"(SKIP_REASONS = .*?)\nprint\(", script, _re.S)
    block = m.group(1)
    assert "mod.plan_conversion(cfg)" in block  # the hook's own decision, not a copy
    _mock_config(hook, monkeypatch, enabled, hf_path, pretrained)
    ns = {"cfg": "exp.yaml", "hook_path": "unused", "os": __import__("os"), "sys": sys}
    fake_util = types.SimpleNamespace(
        spec_from_file_location=lambda *_a: None, module_from_spec=lambda _s: hook
    )
    ns["importlib"] = types.SimpleNamespace(util=fake_util)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(block.replace("spec.loader.exec_module(mod)", "pass"), ns)
    result = ns["result"]
    if result == "SKIP":  # the action, from the reason the script prints
        result += ":" + {v: k for k, v in ns["SKIP_REASONS"].items()}.get(ns["reason"], ns["reason"])
    return result, ns["reason"]


def _mock_config(hook, monkeypatch, enabled, hf_path, pretrained=None):
    monkeypatch.setattr(hook, "get_checkpoint_config", lambda _c: (hf_path, pretrained))

    def opts(_c):
        if isinstance(enabled, Exception):
            raise enabled
        return {"enabled": enabled, "dtype": None, "out": None}

    monkeypatch.setattr(hook, "read_native_opts", opts)


def _main_outcome(hook, monkeypatch, tmp_path, enabled, hf_path, pretrained=None):
    """What the hook's main() really does: native / bridge / error / none."""
    _mock_config(hook, monkeypatch, enabled, hf_path, pretrained)
    calls = []
    monkeypatch.setattr(sys, "argv", ["hook", "--config", str(tmp_path / "exp.yaml")])
    monkeypatch.setattr(hook, "_native_main", lambda p, o, f: calls.append("native"))
    monkeypatch.setattr(hook, "convert_checkpoint", lambda p, out: calls.append("bridge"))
    monkeypatch.setattr(hook, "fix_common_pt_for_megatron_lm", lambda *_a: None)
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("NODE_RANK", "0")
    try:
        hook.main()
    except SystemExit as exc:
        assert exc.code == 1
        return "error"
    return calls[0] if calls else "none"


# what the install script must do for each real outcome of the hook's main()
_EXPECTED_INSTALL = {"native": "SKIP:native", "error": "SKIP:error", "bridge": "INSTALL"}


@pytest.mark.parametrize(
    "enabled", [None, False, True, ValueError("native_ckpt_convert: cannot parse 'maybe' as a boolean")]
)
@pytest.mark.parametrize(
    "cfg",
    [
        {"text_config": {"model_type": "muse_spark_text"}},  # native family without a bridge model
        {"model_type": "qwen3_moe"},  # default-native family
        {"model_type": "mistral"},  # no native converter
    ],
)
def test_install_script_agrees_with_the_hook(monkeypatch, tmp_path, enabled, cfg):
    hook = _hook()
    hf = _cfg(tmp_path, cfg)
    outcome = _main_outcome(hook, monkeypatch, tmp_path, enabled, hf)
    result, printed = _install_script_decide(hook, monkeypatch, enabled, hf)
    assert result == _EXPECTED_INSTALL[outcome], (cfg, enabled, outcome)
    assert "could not compute" not in printed


def test_install_script_pretrained_and_no_hf_path(monkeypatch, tmp_path):
    hook = _hook()
    assert _install_script_decide(hook, monkeypatch, None, None, pretrained="/ckpt")[0] == "SKIP:pretrained"
    assert _main_outcome(hook, monkeypatch, tmp_path, None, None) == "none"
    assert _install_script_decide(hook, monkeypatch, None, None)[0] == "INSTALL"  # legacy behaviour


def test_install_script_reports_an_unexpected_error_and_installs(monkeypatch, tmp_path):
    hook = _hook()
    hf = _cfg(tmp_path, {"model_type": "mistral"})

    def broken(_c):
        raise OSError("config not readable")

    monkeypatch.setattr(hook, "plan_conversion", broken)
    result, printed = _install_script_decide(hook, monkeypatch, None, hf)
    assert result == "INSTALL"  # legacy behaviour ...
    assert (
        "could not compute the conversion decision (OSError: config not readable)" in printed
    )  # ... not silent


def test_install_script_reports_a_system_exit_and_installs(monkeypatch, tmp_path):
    hook = _hook()
    hf = _cfg(tmp_path, {"model_type": "mistral"})

    def leaves(_c):
        raise SystemExit(1)

    monkeypatch.setattr(hook, "plan_conversion", leaves)
    result, printed = _install_script_decide(hook, monkeypatch, None, hf)
    assert result == "INSTALL" and "SystemExit" in printed


def test_install_script_installs_for_any_other_action(monkeypatch, tmp_path):
    # only the known skip actions skip; a new plan_conversion action keeps the legacy install
    hook = _hook()
    hf = _cfg(tmp_path, {"model_type": "mistral"})
    monkeypatch.setattr(hook, "plan_conversion", lambda _c: ("later", "", hf, None, None))
    assert _install_script_decide(hook, monkeypatch, None, hf) == ("INSTALL", "conversion action later")


def test_install_script_stops_on_ctrl_c(monkeypatch, tmp_path):
    # KeyboardInterrupt is not turned into an install decision
    hook = _hook()
    hf = _cfg(tmp_path, {"model_type": "mistral"})

    def interrupted(_c):
        raise KeyboardInterrupt

    monkeypatch.setattr(hook, "plan_conversion", interrupted)
    with pytest.raises(SystemExit) as exc:
        _install_script_decide(hook, monkeypatch, None, hf)
    assert exc.value.code == 130


def _run_install_script(tmp_path, *args, python3="exit 1"):
    import pathlib
    import subprocess

    script = (
        pathlib.Path(__file__).resolve().parents[3]
        / "runner/helpers/hooks/train/posttrain/megatron/00_install_requirements.sh"
    )
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    (bin_ / "python3").write_text("#!/bin/bash\n" + python3 + "\n")
    (bin_ / "pip").write_text('#!/bin/bash\necho pip "$@"\n')
    for f in bin_.iterdir():
        f.chmod(0o755)
    env = {"PATH": f"{bin_}:/usr/bin:/bin", "DATA_PATH": str(tmp_path / "data"), "HOME": str(tmp_path)}
    return subprocess.run(["bash", str(script), *args], env=env, capture_output=True, text=True)


def test_install_script_reads_config_equals_form(tmp_path):
    # --config=<path> reaches the decision like --config <path> (python3 echoes its argument)
    out = _run_install_script(
        tmp_path, f"--config={tmp_path}/exp.yaml", python3='cat >/dev/null; echo "__DECISION__:SKIP:cfg=$2"'
    )
    assert out.returncode == 0, out.stderr
    assert f"cfg={tmp_path}/exp.yaml -> skipping" in out.stdout


def test_install_script_uses_the_last_config_like_the_hook(tmp_path):
    # the hook's argparse keeps the last --config; the install decision must read the same file
    out = _run_install_script(
        tmp_path,
        "--config",
        f"{tmp_path}/first.yaml",
        f"--config={tmp_path}/last.yaml",
        python3='cat >/dev/null; echo "__DECISION__:SKIP:cfg=$2"',
    )
    assert out.returncode == 0, out.stderr
    assert f"cfg={tmp_path}/last.yaml -> skipping" in out.stdout
    # ... and the hook's own parser (main) takes the last --config too
    hook = _hook()
    seen = []

    def plan(config_file):
        seen.append(config_file)
        return "pretrained", "stub", None, {}, None

    import unittest.mock as mock

    with mock.patch.object(hook, "plan_conversion", plan), mock.patch.object(
        sys, "argv", ["hook", "--config", "/a.yaml", "--config=/b.yaml", "--other", "x"]
    ):
        hook.main()  # "pretrained": returns before any conversion
    assert seen == ["/b.yaml"]


def test_install_script_shell_exits_on_an_interrupted_decision(tmp_path):
    out = _run_install_script(
        tmp_path, "--config", str(tmp_path / "exp.yaml"), python3="cat >/dev/null; exit 130"
    )
    assert out.returncode == 130
    assert "pip install" not in out.stdout


def test_install_script_shell_reports_a_python_crash(tmp_path):
    # bash level: python3 prints no decision (here it dies at once); the script says so and
    # installs. Fake python3 / pip on PATH: nothing is really installed.
    import pathlib
    import subprocess

    script = (
        pathlib.Path(__file__).resolve().parents[3]
        / "runner/helpers/hooks/train/posttrain/megatron/00_install_requirements.sh"
    )
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "python3").write_text("#!/bin/bash\necho 'Fatal Python error: boom' >&2\nexit 1\n")
    (bin_ / "pip").write_text('#!/bin/bash\necho pip "$@"\n')
    for f in bin_.iterdir():
        f.chmod(0o755)
    env = {"PATH": f"{bin_}:/usr/bin:/bin", "DATA_PATH": str(tmp_path / "data"), "HOME": str(tmp_path)}
    out = subprocess.run(
        ["bash", str(script), "--config", str(tmp_path / "exp.yaml")], env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert "python3 gave no decision: Fatal Python error: boom" in out.stdout
    assert "pip install" in out.stdout


def test_convert_timeout_defaults_and_validation(monkeypatch):
    hook = _hook()
    monkeypatch.delenv(hook.CONVERT_TIMEOUT_ENV, raising=False)
    # one bound for every family: a dead rank 0 is found by the heartbeat, not the timeout
    assert hook.convert_timeout() == 4 * 3600
    assert not hasattr(hook, "native_wait_timeout")
    monkeypatch.setenv(hook.CONVERT_TIMEOUT_ENV, "90.5")
    assert hook.convert_timeout() == 90.5
    for bad in ("0", "-1", "inf", "nan", "abc"):
        monkeypatch.setenv(hook.CONVERT_TIMEOUT_ENV, bad)
        with pytest.raises(ValueError, match="finite positive"):
            hook.convert_timeout()


def test_heartbeat_timeout_validation(monkeypatch):
    hook = _hook()
    monkeypatch.delenv(hook.HEARTBEAT_TIMEOUT_ENV, raising=False)
    assert hook.heartbeat_timeout() == 300.0
    monkeypatch.setenv(hook.HEARTBEAT_TIMEOUT_ENV, "600")
    assert hook.heartbeat_timeout() == 600.0
    for bad in ("0", "-1", "inf", "nan", "abc", "10"):  # 10: not above the 30 s heartbeat
        monkeypatch.setenv(hook.HEARTBEAT_TIMEOUT_ENV, bad)
        with pytest.raises(ValueError, match="HEARTBEAT_TIMEOUT"):
            hook.heartbeat_timeout()


def test_a_killed_rank0_is_found_by_its_stale_lock(tmp_path, monkeypatch):
    # SIGKILL / OOM: rank 0 writes no marker and leaves the lock; the heartbeat stops, so
    # the waiters stop after the heartbeat timeout, not after the 4 h wait
    hook = _hook()
    monkeypatch.setenv(hook.RUN_ID_ENV, "run-1")
    monkeypatch.delenv(hook.HEARTBEAT_TIMEOUT_ENV, raising=False)
    done, lock = tmp_path / "ck.done", tmp_path / "ck.converting.lock"
    lock.write_text("run-1\n")
    clock = _Clock(hook, monkeypatch)
    start = clock.now
    with pytest.raises(RuntimeError, match="has not changed"):
        hook.wait_for_conversion(done, lock, timeout=4 * 3600)
    assert hook.DEFAULT_HEARTBEAT_TIMEOUT <= clock.now - start < hook.DEFAULT_HEARTBEAT_TIMEOUT + 10


def test_a_live_rank0_keeps_the_waiters_waiting(tmp_path, monkeypatch):
    # rank 0's heartbeat changes the lock file: no stale-lock stop, only the overall timeout
    hook = _hook()
    monkeypatch.setenv(hook.RUN_ID_ENV, "run-1")
    done, lock = tmp_path / "ck.done", tmp_path / "ck.converting.lock"
    lock.write_text("run-1\n")
    clock = _Clock(hook, monkeypatch)
    beats = []

    def sleep(s):
        clock.now += s
        if int(clock.now) // 60 != len(beats):  # rank 0 touches the lock once a minute
            beats.append(1)
            hook._write_token(lock, f"run-1 {len(beats)}")

    monkeypatch.setattr(hook.time, "sleep", sleep)
    with pytest.raises(TimeoutError, match="Timeout"):
        hook.wait_for_conversion(done, lock, timeout=1000)
    assert clock.now >= 1000


def test_the_heartbeat_touches_the_lock(tmp_path):
    hook = _hook()
    lock = tmp_path / "ck.converting.lock"
    lock.write_text("")
    os_ = __import__("os")
    os_.utime(lock, (0, 0))
    import time as _t

    with hook._heartbeat(lock, interval=0.01):
        _t.sleep(0.2)
    assert lock.stat().st_mtime > 0


def test_no_rank0_start_stops_the_wait_early(tmp_path, monkeypatch):
    # no lock, no .done, no marker: rank 0 never started; stop after START_TIMEOUT, not 4 h
    hook = _hook()
    monkeypatch.setenv(hook.RUN_ID_ENV, "run-1")
    clock = _Clock(hook, monkeypatch)
    start = clock.now
    with pytest.raises(TimeoutError, match="did not start"):
        hook.wait_for_conversion(tmp_path / "ck.done", tmp_path / "ck.lock", timeout=4 * 3600)
    assert hook.START_TIMEOUT <= clock.now - start < hook.START_TIMEOUT + 10


def test_a_stale_marker_does_not_count_while_rank0_holds_the_lock_now(tmp_path, monkeypatch):
    # the marker grace uses the lock's current state: rank 0 started late (after the old
    # marker was seen with no lock), and its lock file must stop the marker from counting
    hook = _hook()
    _no_strong_token(hook, monkeypatch)
    monkeypatch.setenv(hook.MARKER_GRACE_ENV, "120")
    done, lock = tmp_path / "ck.done", tmp_path / "ck.converting.lock"
    hook.failed_marker(done).write_text("")  # an earlier launch's, not unlinked yet
    clock = _Clock(hook, monkeypatch)
    start = clock.now

    def sleep(s):
        clock.now += s
        if 100 <= clock.now - start < 103:
            lock.write_text("")  # rank 0 starts late, before the grace ends
        if clock.now - start >= 150:
            hook._write_token(done, "")  # ... and finishes

    monkeypatch.setattr(hook.time, "sleep", sleep)
    hook.wait_for_conversion(done, lock, timeout=1000)  # no RuntimeError from the old marker


def test_waiting_ranks_stop_when_rank0_fails(tmp_path, monkeypatch):
    hook = _hook()
    monkeypatch.setenv(hook.RUN_ID_ENV, "run-1")  # the launch's shared token
    # the marker is there before the wait: it counts after the grace with no lock file
    monkeypatch.setenv(hook.MARKER_GRACE_ENV, "0.05")
    done, lock = tmp_path / "ckpt.done", tmp_path / "ckpt.converting.lock"
    with pytest.raises(RuntimeError, match="boom"):
        with hook._rank0_conversion(done, lock):
            assert lock.exists()
            raise RuntimeError("boom")
    assert not lock.exists()
    assert hook.failed_marker(done).exists()
    with pytest.raises(RuntimeError, match="rank 0 failed"):
        hook.wait_for_conversion(done, lock, timeout=30)
    # a new attempt drops the old marker; success leaves none and writes .done (with the token)
    with hook._rank0_conversion(done, lock):
        assert not hook.failed_marker(done).exists()
        assert not done.exists()
    assert done.read_text().strip() == "run-1" and not hook.failed_marker(done).exists()
    hook.wait_for_conversion(done, lock, timeout=1)


def test_wait_for_conversion_times_out(tmp_path):
    hook = _hook()
    with pytest.raises(TimeoutError, match="PRIMUS_CKPT_CONVERT_TIMEOUT"):
        hook.wait_for_conversion(tmp_path / "x.done", tmp_path / "x.lock", timeout=0.01)


def test_a_stale_marker_of_an_earlier_run_does_not_stop_the_wait(tmp_path, monkeypatch):
    # rank 0 of this run has not unlinked the old marker yet (or is slow to start): the
    # waiting rank must not take it for this run's failure, however recent it is
    hook = _hook()
    monkeypatch.setattr(hook.time, "sleep", lambda _s: None)
    monkeypatch.setenv(hook.RUN_ID_ENV, "run-2")
    done, lock = tmp_path / "ckpt.done", tmp_path / "ckpt.converting.lock"
    hook.failed_marker(done).write_text("run-1\n")  # written just now by an earlier run
    with pytest.raises(TimeoutError):
        hook.wait_for_conversion(done, lock, timeout=0.05)
    # an old marker without a token (older hook) is stale too
    hook.failed_marker(done).write_text("")
    with pytest.raises(TimeoutError):
        hook.wait_for_conversion(done, lock, timeout=0.05)


def test_a_marker_with_this_runs_token_stops_the_wait_after_the_grace(tmp_path, monkeypatch):
    # rank 0 already failed before this rank started waiting: with no lock file for the
    # grace period, the marker with this run's token is ours
    hook = _hook()
    monkeypatch.setenv(hook.RUN_ID_ENV, "run-2")
    monkeypatch.setenv(hook.MARKER_GRACE_ENV, "0.05")
    done, lock = tmp_path / "ckpt.done", tmp_path / "ckpt.converting.lock"
    with pytest.raises(RuntimeError, match="boom"):
        with hook._rank0_conversion(done, lock):
            raise RuntimeError("boom")
    assert hook.failed_marker(done).read_text().strip() == "run-2"
    with pytest.raises(RuntimeError, match="rank 0 failed"):
        hook.wait_for_conversion(done, lock, timeout=30)


def test_a_relaunch_with_the_same_run_id_does_not_fail_on_the_old_marker(tmp_path, monkeypatch):
    # PRIMUS_CKPT_CONVERT_RUN_ID is often constant across relaunches: an earlier launch's
    # marker holds the same token. Before the grace period ends rank 0 unlinks it, takes
    # the lock and converts; the waiting rank must follow that, not fail on the first poll.
    hook = _hook()
    monkeypatch.setenv(hook.RUN_ID_ENV, "myjob")
    done, lock = tmp_path / "ckpt.done", tmp_path / "ckpt.converting.lock"
    hook.failed_marker(done).write_text("myjob\n")  # left by the earlier launch
    calls = []

    def rank0_succeeds(_s):
        calls.append(1)
        if len(calls) == 2:
            with hook._rank0_conversion(done, lock):
                pass

    monkeypatch.setattr(hook.time, "sleep", rank0_succeeds)
    hook.wait_for_conversion(done, lock, timeout=30)  # returns: no RuntimeError
    assert len(calls) == 2


def test_a_marker_rank0_rewrites_during_the_wait_stops_it(tmp_path, monkeypatch):
    # no shared run id: a stale marker is ignored until rank 0 replaces it (new inode)
    hook = _hook()
    for k in (hook.RUN_ID_ENV, "SLURM_JOB_ID", "SLURM_STEP_ID", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(k, raising=False)
    done, lock = tmp_path / "ckpt.done", tmp_path / "ckpt.converting.lock"
    failed = hook.failed_marker(done)
    failed.write_text("")  # stale
    calls = []

    def rank0_fails(_s):
        calls.append(1)
        if len(calls) == 2:  # rank 0 drops the stale marker, converts, fails
            with pytest.raises(RuntimeError):
                with hook._rank0_conversion(done, lock):
                    raise RuntimeError("boom")

    monkeypatch.setattr(hook.time, "sleep", rank0_fails)
    with pytest.raises(RuntimeError, match="rank 0 failed"):
        hook.wait_for_conversion(done, lock, timeout=30)
    assert len(calls) == 2  # not on the first poll (stale), on the first poll after the rewrite


def test_conversion_run_id(monkeypatch):
    hook = _hook()
    for k in (
        hook.RUN_ID_ENV,
        "SLURM_JOB_ID",
        "SLURM_STEP_ID",
        "SLURM_RESTART_COUNT",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        monkeypatch.delenv(k, raising=False)
    assert hook.conversion_run_id() is None
    monkeypatch.setenv("MASTER_ADDR", "node0")
    monkeypatch.setenv("MASTER_PORT", "1234")
    assert hook._run_token() == ("master:node0:1234", False)  # a weak token: may repeat
    monkeypatch.delenv("MASTER_ADDR")
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    assert hook._run_token() == ("slurm:42:batch:0", False)  # no step id: may repeat
    monkeypatch.setenv("SLURM_STEP_ID", "3")
    assert hook._run_token() == ("slurm:42:3:0", True)  # each srun step is a new run
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    assert hook.conversion_run_id() == "slurm:42:3:1"  # a requeued job is a new run
    monkeypatch.setenv(hook.RUN_ID_ENV, " mine ")
    assert hook.conversion_run_id() == "mine"


def test_the_bridge_wait_uses_the_default_timeout(monkeypatch, tmp_path):
    # main()'s Bridge branch passes no timeout: wait_for_conversion resolves convert_timeout()
    hook = _hook()
    hf = _cfg(tmp_path, {"model_type": "mistral"})
    _mock_config(hook, monkeypatch, None, hf)
    monkeypatch.setattr(sys, "argv", ["hook", "--config", str(tmp_path / "exp.yaml")])
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("NODE_RANK", "1")
    seen = []
    monkeypatch.setattr(hook, "wait_for_conversion", lambda d, l, *a, **k: seen.append((a, k)))
    hook.main()
    assert len(seen) == 1 and seen[0][0] == () and set(seen[0][1]) == {"ready"}  # no timeout


def _no_strong_token(hook, monkeypatch, master=True):
    for k in (
        hook.RUN_ID_ENV,
        "SLURM_JOB_ID",
        "SLURM_STEP_ID",
        "SLURM_RESTART_COUNT",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        monkeypatch.delenv(k, raising=False)
    if master:
        monkeypatch.setenv("MASTER_ADDR", "node0")
        monkeypatch.setenv("MASTER_PORT", "29500")


class _Clock:
    """A fake monotonic clock that advances on every sleep."""

    def __init__(self, hook, monkeypatch):
        self.now = 1000.0
        monkeypatch.setattr(hook.time, "monotonic", lambda: self.now)
        monkeypatch.setattr(hook.time, "sleep", self.sleep)

    def sleep(self, s):
        self.now += s


def test_a_stale_done_file_does_not_release_the_waiting_ranks(tmp_path, monkeypatch):
    # rank 0 re-converts (the checkpoint is gone) while <ckpt>.done of an earlier run is
    # still there: the waiters must keep waiting for this run's .done
    hook = _hook()
    monkeypatch.setenv(hook.RUN_ID_ENV, "run-2")
    done, lock = tmp_path / "ckpt.done", tmp_path / "ckpt.converting.lock"
    done.write_text("run-1\n")
    _Clock(hook, monkeypatch)
    with pytest.raises(TimeoutError):
        hook.wait_for_conversion(done, lock, timeout=60, ready=lambda: False)
    # rank 0 drops it before converting, and writes a new one when it is done
    with hook._rank0_conversion(done, lock):
        assert not done.exists() and lock.exists()
    assert done.read_text().strip() == "run-2"
    hook.wait_for_conversion(done, lock, timeout=60)


def test_a_failed_conversion_leaves_no_done_file(tmp_path, monkeypatch):
    hook = _hook()
    monkeypatch.setenv(hook.RUN_ID_ENV, "run-3")
    done, lock = tmp_path / "ckpt.done", tmp_path / "ckpt.converting.lock"
    done.write_text("run-2\n")
    with pytest.raises(RuntimeError):
        with hook._rank0_conversion(done, lock):
            raise RuntimeError("boom")
    assert not done.exists() and hook.failed_marker(done).read_text().strip() == "run-3"


def test_an_unwritable_failure_marker_is_logged_and_the_error_kept(tmp_path, monkeypatch):
    hook = _hook()
    monkeypatch.setenv(hook.RUN_ID_ENV, "run-3")
    done, lock = tmp_path / "ckpt.done", tmp_path / "ckpt.converting.lock"
    logged = []
    monkeypatch.setattr(hook, "log_error", logged.append)

    real_write = hook._write_token

    def no_space(path, token):
        if path.suffix == ".failed":
            raise OSError(28, "No space left on device")
        real_write(path, token)

    monkeypatch.setattr(hook, "_write_token", no_space)
    with pytest.raises(RuntimeError, match="boom"):
        with hook._rank0_conversion(done, lock):
            raise RuntimeError("boom")
    assert len(logged) == 1 and str(hook.failed_marker(done)) in logged[0]
    # the lock stays (its heartbeat has stopped): the waiters stop at the stale-lock check
    assert "is kept" in logged[0] and lock.exists()


def test_an_unwritable_done_file_is_a_failed_conversion_for_the_waiters(tmp_path, monkeypatch):
    # the body succeeded but .done cannot be written (ENOSPC): a .failed marker with this
    # run's token is written, so the waiting ranks stop, not after the whole timeout
    hook = _hook()
    monkeypatch.setenv(hook.RUN_ID_ENV, "run-4")
    done, lock = tmp_path / "ckpt.done", tmp_path / "ckpt.converting.lock"
    real_write = hook._write_token

    def no_space(path, token):
        if path == done:
            raise OSError(28, "No space left on device")
        real_write(path, token)

    monkeypatch.setattr(hook, "_write_token", no_space)
    with pytest.raises(OSError, match="No space"):
        with hook._rank0_conversion(done, lock):
            pass
    assert not done.exists() and not lock.exists()
    assert hook.failed_marker(done).read_text().strip() == "run-4"
    _Clock(hook, monkeypatch)
    with pytest.raises(RuntimeError, match="rank 0 failed"):
        hook.wait_for_conversion(done, lock, timeout=4 * 3600)


def test_unwritable_done_and_failure_files_keep_the_lock_for_the_staleness_check(tmp_path, monkeypatch):
    hook = _hook()
    monkeypatch.setenv(hook.RUN_ID_ENV, "run-4")
    done, lock = tmp_path / "ckpt.done", tmp_path / "ckpt.converting.lock"
    monkeypatch.setattr(hook, "log_error", lambda *_a: None)
    real_write = hook._write_token

    def read_only(path, token):
        if path != lock:
            raise OSError(30, "Read-only file system")
        real_write(path, token)

    monkeypatch.setattr(hook, "_write_token", read_only)
    with pytest.raises(OSError, match="Read-only"):
        with hook._rank0_conversion(done, lock):
            pass
    assert lock.exists() and not done.exists()  # the waiters stop when it goes stale


def test_a_stale_done_file_with_a_complete_checkpoint_releases_the_wait(tmp_path, monkeypatch):
    # no token at all: a waiter that arrives after rank 0 finished sees an unchanged .done;
    # the checkpoint itself (rank 0's skip-if-exists check) shows it is complete
    hook = _hook()
    _no_strong_token(hook, monkeypatch, master=False)
    done, lock = tmp_path / "ckpt.done", tmp_path / "ckpt.converting.lock"
    done.write_text("")
    _Clock(hook, monkeypatch)
    hook.wait_for_conversion(done, lock, timeout=60, ready=lambda: True)
    lock.touch()  # ... but not while rank 0 still converts
    with pytest.raises(TimeoutError):
        hook.wait_for_conversion(done, lock, timeout=60, ready=lambda: True)


def test_without_slurm_an_early_rank0_failure_stops_the_wait_after_the_grace(tmp_path, monkeypatch):
    # torchrun / k8s: rank 0 failed fast, before this node reached the hook. The marker is
    # in the start snapshot, but with no lock file it is this run's failure after the
    # grace period, not after the 4 h timeout
    hook = _hook()
    _no_strong_token(hook, monkeypatch)
    monkeypatch.delenv(hook.MARKER_GRACE_ENV, raising=False)
    done, lock = tmp_path / "ckpt.done", tmp_path / "ckpt.converting.lock"
    with pytest.raises(RuntimeError, match="boom"):
        with hook._rank0_conversion(done, lock):
            raise RuntimeError("boom")
    assert hook.failed_marker(done).read_text().strip() == "master:node0:29500"
    clock = _Clock(hook, monkeypatch)
    start = clock.now
    with pytest.raises(RuntimeError, match="rank 0 failed"):
        hook.wait_for_conversion(done, lock, timeout=4 * 3600)
    assert hook.DEFAULT_MARKER_GRACE <= clock.now - start < hook.DEFAULT_MARKER_GRACE + 10
    # the same holds with no token at all (marker without a token)
    _no_strong_token(hook, monkeypatch, master=False)
    with pytest.raises(RuntimeError, match="rank 0 failed"):
        hook.wait_for_conversion(done, lock, timeout=4 * 3600)


def test_without_slurm_a_stale_marker_is_ignored_while_rank0_converts(tmp_path, monkeypatch):
    hook = _hook()
    _no_strong_token(hook, monkeypatch)
    done, lock = tmp_path / "ckpt.done", tmp_path / "ckpt.converting.lock"
    hook.failed_marker(done).write_text("")  # stale, not unlinked yet
    clock = _Clock(hook, monkeypatch)
    lock.touch()  # rank 0 is converting (it drops the marker right before this, normally)
    with pytest.raises(TimeoutError):
        hook.wait_for_conversion(done, lock, timeout=200)  # below the heartbeat timeout
    # a marker with another launch's token is never this run's
    lock.unlink()
    hook.failed_marker(done).write_text("master:node9:1\n")
    with pytest.raises(TimeoutError):
        hook.wait_for_conversion(done, lock, timeout=1000)
    assert clock.now > 1000


def test_marker_grace_validation(monkeypatch):
    hook = _hook()
    monkeypatch.delenv(hook.MARKER_GRACE_ENV, raising=False)
    assert hook.marker_grace() == 120.0
    monkeypatch.setenv(hook.MARKER_GRACE_ENV, "30")
    assert hook.marker_grace() == 30.0
    for bad in ("0", "-1", "inf", "nan", "abc"):  # 0 would let any old marker stop the wait
        monkeypatch.setenv(hook.MARKER_GRACE_ENV, bad)
        with pytest.raises(ValueError, match="MARKER_GRACE"):
            hook.marker_grace()


def test_native_main_uses_the_planned_family(tmp_path, monkeypatch):
    hook = _hook()
    monkeypatch.setenv("NODE_RANK", "0")
    monkeypatch.setattr(hook, "_detect_native_family", lambda _p: pytest.fail("detected twice"))
    seen = {}
    monkeypatch.setattr(hook, "native_convert_checkpoint", lambda h, o, opts: seen.update(opts))
    hook._native_main(str(tmp_path / "hf"), {"out": str(tmp_path / "out")}, "muse_spark")
    assert seen["family"] == "muse_spark"
    assert (tmp_path / "out.done").exists()


def test_an_earlier_srun_steps_marker_does_not_stop_a_retry(tmp_path, monkeypatch):
    # one allocation, a second srun step: the first step's .failed is not this run's
    hook = _hook()
    _no_strong_token(hook, monkeypatch, master=False)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    done, lock = tmp_path / "ck.done", tmp_path / "ck.converting.lock"
    with pytest.raises(RuntimeError):
        with hook._rank0_conversion(done, lock):
            raise RuntimeError("boom")
    assert hook.failed_marker(done).read_text().strip() == "slurm:123:0:0"
    monkeypatch.setenv("SLURM_STEP_ID", "1")
    _Clock(hook, monkeypatch)
    lock.touch()  # the new rank 0 converts
    with pytest.raises(TimeoutError):
        hook.wait_for_conversion(done, lock, timeout=200)  # below the heartbeat timeout


def test_a_preexisting_done_with_this_token_waits_for_the_lock_and_the_checkpoint(tmp_path, monkeypatch):
    # an earlier step with the same explicit run id left .done; rank 0 re-converts
    hook = _hook()
    monkeypatch.setenv(hook.RUN_ID_ENV, "same")
    done, lock = tmp_path / "ck.done", tmp_path / "ck.converting.lock"
    done.write_text("same\n")
    _Clock(hook, monkeypatch)
    lock.touch()
    with pytest.raises(TimeoutError):
        hook.wait_for_conversion(done, lock, timeout=100, ready=lambda: True)
    lock.unlink()
    with pytest.raises(TimeoutError):  # incomplete checkpoint
        hook.wait_for_conversion(done, lock, timeout=100, ready=lambda: False)
    hook.wait_for_conversion(done, lock, timeout=100, ready=lambda: True)


def test_a_bad_marker_grace_stops_every_rank_before_conversion(tmp_path, monkeypatch):
    hook = _hook()
    hf = _cfg(tmp_path, {"model_type": "mistral"})
    _mock_config(hook, monkeypatch, None, hf)
    monkeypatch.delenv(hook.CONVERT_TIMEOUT_ENV, raising=False)
    monkeypatch.setenv(hook.MARKER_GRACE_ENV, "abc")
    action, detail, *_ = hook.plan_conversion(str(tmp_path / "exp.yaml"))
    assert action == "error" and "MARKER_GRACE" in detail


@pytest.mark.parametrize("bad", [0, -5, "abc", 1.5, True])
def test_a_bad_max_params_stops_every_rank_before_conversion(tmp_path, monkeypatch, bad):
    hook = _hook()
    hf = _cfg(tmp_path, {"text_config": {"model_type": "muse_spark_text"}})
    monkeypatch.setattr(hook, "get_checkpoint_config", lambda _c: (hf, None))
    monkeypatch.setattr(
        hook, "read_native_opts", lambda _c: {"enabled": None, "dtype": None, "out": None, "max_params": bad}
    )
    action, detail, *_ = hook.plan_conversion(str(tmp_path / "exp.yaml"))
    assert action == "error" and "native_ckpt_max_params" in detail


def test_a_good_max_params_is_parsed_up_front(tmp_path, monkeypatch):
    hook = _hook()
    hf = _cfg(tmp_path, {"text_config": {"model_type": "muse_spark_text"}})
    monkeypatch.setattr(hook, "get_checkpoint_config", lambda _c: (hf, None))
    monkeypatch.setattr(
        hook,
        "read_native_opts",
        lambda _c: {"enabled": None, "dtype": None, "out": None, "max_params": "100_000_000_000"},
    )
    action, _detail, _hf, opts, family = hook.plan_conversion(str(tmp_path / "exp.yaml"))
    assert (action, family, opts["max_params"]) == ("native", "muse_spark", 100_000_000_000)


def _hub_stub(monkeypatch, cached=None):
    """A huggingface_hub stub that records calls; ``cached`` maps ids to a cached config path."""
    calls = []
    hub = types.ModuleType("huggingface_hub")

    def try_to_load_from_cache(repo_id, filename, **kw):
        calls.append(("cache", repo_id))
        return (cached or {}).get(repo_id)

    def hf_hub_download(*a, **kw):  # the network: must never be called by the routing
        calls.append(("download", a, kw))
        raise AssertionError("network access")

    hub.try_to_load_from_cache = try_to_load_from_cache
    hub.hf_hub_download = hf_hub_download
    hub.snapshot_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return calls


def test_a_muse_hub_id_is_detected_without_a_local_config(monkeypatch, tmp_path):
    # the Muse preset's tokenizer_model is a Hub id: no local config.json, no cache, offline
    hook = _hook()
    monkeypatch.chdir(tmp_path)
    calls = _hub_stub(monkeypatch)
    assert hook._detect_native_family("meta-models/Muse-Spark-Open-816B-A42B") == "muse_spark"
    # other Hub ids keep the Megatron-Bridge path, as before, with no network access
    assert hook._detect_native_family("Qwen/Qwen3-8B") is None
    assert hook._detect_native_family("./local/dir") is None
    assert calls == []  # no cache, no network


def test_a_repo_name_alone_is_not_a_muse_spark_checkpoint(monkeypatch, tmp_path):
    # a fine-tune / adapter / GGUF repo named after Muse Spark is not matched by its name
    hook = _hook()
    monkeypatch.chdir(tmp_path)
    calls = _hub_stub(monkeypatch)
    for repo in ("org/my-muse-spark-finetune-llama", "org/Muse-Spark-Open-816B-A42B-GGUF", "other/musespark"):
        assert hook._detect_native_family(repo) is None
    assert calls == []


def test_no_network_for_any_hub_id_in_the_routing(monkeypatch, tmp_path):
    # plan_conversion runs on every node and in 00_install_requirements.sh: with
    # native_ckpt_convert unset it reads neither the HF cache nor the network
    hook = _hook()
    monkeypatch.chdir(tmp_path)
    calls = _hub_stub(monkeypatch)
    monkeypatch.setattr(hook, "get_checkpoint_config", lambda _c: ("Qwen/Qwen3-8B", None))
    monkeypatch.setattr(hook, "read_native_opts", lambda _c: {"enabled": None, "dtype": None, "out": None})
    assert hook.plan_conversion("exp.yaml")[0] == "bridge"
    assert calls == []


def test_a_muse_hub_id_with_native_off_is_an_error_not_bridge(monkeypatch, tmp_path):
    hook = _hook()
    monkeypatch.chdir(tmp_path)
    _hub_stub(monkeypatch)
    hub = "meta-models/Muse-Spark-Open-816B-A42B"
    monkeypatch.setattr(hook, "get_checkpoint_config", lambda _c: (hub, None))
    monkeypatch.setattr(hook, "read_native_opts", lambda _c: {"enabled": False, "dtype": None, "out": None})
    assert hook.plan_conversion("exp.yaml")[0] == "error"
    monkeypatch.setattr(hook, "read_native_opts", lambda _c: {"enabled": None, "dtype": None, "out": None})
    # the default (auto) route is chosen, then stops on the size cap (below), not on Bridge
    assert hook.plan_conversion("exp.yaml")[0] == "error"


def test_the_released_816b_hub_id_fails_up_front_above_the_converter_cap(monkeypatch, tmp_path):
    # review: the default cap (60B) made the auto-routed 816B conversion fail only after
    # rank 0 had downloaded its configs; now plan_conversion stops on every rank, offline
    hook = _hook()
    monkeypatch.chdir(tmp_path)
    calls = _hub_stub(monkeypatch)
    hub = "meta-models/Muse-Spark-Open-816B-A42B"
    monkeypatch.setattr(hook, "get_checkpoint_config", lambda _c: (hub, None))
    monkeypatch.setattr(hook, "read_native_opts", lambda _c: {"enabled": None, "dtype": None, "out": None})
    action, detail = hook.plan_conversion("exp.yaml")[:2]
    assert action == "error" and "816B" in detail and "pretrained_checkpoint" in detail
    assert "native_ckpt_max_params" in detail and calls == []
    # an explicit, large enough cap keeps the native route
    monkeypatch.setattr(
        hook,
        "read_native_opts",
        lambda _c: {"enabled": None, "dtype": None, "out": None, "max_params": "900e9"},
    )
    assert hook.plan_conversion("exp.yaml")[:2] == ("native", "muse_spark")


def test_a_cached_hub_config_decides_the_family_only_on_an_explicit_opt_in(monkeypatch, tmp_path):
    # a renamed / mirrored Muse repo: every node must route alike, so the node's own HF
    # cache does not decide when native_ckpt_convert is unset (review: cached on some
    # nodes only -> native here, Bridge there -> each waits on the other's markers)
    hook = _hook()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "m").mkdir()
    (tmp_path / "q").mkdir()
    muse = _cfg(tmp_path / "m", {"text_config": {"model_type": "muse_spark_text"}})
    qwen = _cfg(tmp_path / "q", {"model_type": "qwen3"})
    calls = _hub_stub(
        monkeypatch,
        {"someone/renamed-model": muse + "/config.json", "someone/other": qwen + "/config.json"},
    )
    assert hook._detect_native_family("someone/renamed-model") is None  # cached or not
    assert calls == []  # unset: not even the cache is read
    assert hook._detect_native_family("someone/renamed-model", fetch=True) == "muse_spark"
    assert hook._detect_native_family("someone/other", fetch=True) is None  # Bridge families: local dirs only


def test_an_opt_in_hub_id_downloads_its_config_when_not_cached(monkeypatch, tmp_path):
    hook = _hook()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "m").mkdir()
    muse = _cfg(tmp_path / "m", {"text_config": {"model_type": "muse_spark_text"}})
    calls = _hub_stub(monkeypatch)
    hub = sys.modules["huggingface_hub"]
    hub.hf_hub_download = lambda repo_id, filename, **kw: calls.append(("download", repo_id)) or (
        muse + "/" + filename
    )
    assert hook._detect_native_family("someorg/muse-spark-finetune", fetch=True) == "muse_spark"
    assert ("download", "someorg/muse-spark-finetune") in calls
    # offline: the exact list applies (an unknown id -> None: plan_conversion stops with an error)
    hub.hf_hub_download = lambda *a, **kw: (_ for _ in ()).throw(OSError("offline"))
    assert hook._detect_native_family("someorg/muse-spark-finetune", fetch=True) is None
    assert hook._detect_native_family("meta-models/Muse-Spark-Open-816B-A42B", fetch=True) == "muse_spark"


def test_another_launchs_marker_written_during_the_wait_is_not_this_runs(tmp_path, monkeypatch):
    # two launches share the checkpoint dir: launch B writes .failed / .done while A converts
    hook = _hook()
    monkeypatch.setenv(hook.RUN_ID_ENV, "runA")
    done, lock = tmp_path / "ckpt.done", tmp_path / "ckpt.converting.lock"
    failed = hook.failed_marker(done)
    hook._write_token(lock, "runA")  # A's rank 0 converts
    polls = []

    def other_launch_then_ours(_s):
        polls.append(1)
        if len(polls) == 1:
            hook._write_token(failed, "runB")  # B's conversion failed
        elif len(polls) == 2:
            hook._write_token(done, "runB")  # B's conversion ended
        elif len(polls) == 3:
            os.utime(lock)  # A's heartbeat
        elif len(polls) == 4:
            lock.unlink()
            hook._write_token(done, "runA")  # A's rank 0 is done

    monkeypatch.setattr(hook.time, "sleep", other_launch_then_ours)
    hook.wait_for_conversion(done, lock, timeout=30)
    assert len(polls) == 4  # returned only on A's own .done


def test_this_runs_failure_marker_written_during_the_wait_stops_it(tmp_path, monkeypatch):
    hook = _hook()
    monkeypatch.setenv(hook.RUN_ID_ENV, "runA")
    done, lock = tmp_path / "ckpt.done", tmp_path / "ckpt.converting.lock"
    hook._write_token(lock, "runA")
    monkeypatch.setattr(hook.time, "sleep", lambda _s: hook._write_token(hook.failed_marker(done), "runA"))
    with pytest.raises(RuntimeError, match="rank 0 failed"):
        hook.wait_for_conversion(done, lock, timeout=30)


def _bridge_main(hook, monkeypatch, tmp_path, node_rank):
    hf = _cfg(tmp_path, {"model_type": "mistral"})
    _mock_config(hook, monkeypatch, None, hf)
    monkeypatch.setattr(sys, "argv", ["hook", "--config", str(tmp_path / "exp.yaml")])
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("NODE_RANK", str(node_rank))
    return tmp_path / "data" / "megatron_checkpoints" / Path(hf).name


def test_a_partial_bridge_output_is_converted_again_not_used(monkeypatch, tmp_path, capsys):
    # review: the Bridge path skipped on megatron_path.exists() alone, so a dir left by a
    # killed conversion was handed to training; now it needs the tracker file, as native
    hook = _hook()
    out = _bridge_main(hook, monkeypatch, tmp_path, 0)
    out.mkdir(parents=True)  # partial: no latest_checkpointed_iteration.txt
    converted = []
    monkeypatch.setattr(hook, "convert_checkpoint", lambda hf, mp: converted.append(mp))
    monkeypatch.setattr(hook, "fix_common_pt_for_megatron_lm", lambda d: None)
    hook.main()
    assert converted == [str(out)]


def test_a_complete_bridge_output_is_skipped(monkeypatch, tmp_path, capsys):
    hook = _hook()
    out = _bridge_main(hook, monkeypatch, tmp_path, 0)
    out.mkdir(parents=True)
    (out / hook.TRACKER_FILE).write_text("release")
    monkeypatch.setattr(hook, "convert_checkpoint", lambda *a: pytest.fail("converted again"))
    hook.main()
    assert f"extra.pretrained_checkpoint={out}" in capsys.readouterr().out


def test_the_bridge_waiters_ready_check_needs_the_tracker_file(monkeypatch, tmp_path):
    hook = _hook()
    out = _bridge_main(hook, monkeypatch, tmp_path, 1)
    seen = []
    monkeypatch.setattr(hook, "wait_for_conversion", lambda d, l, ready=None: seen.append(ready))
    hook.main()
    (ready,) = seen
    assert not ready()
    out.mkdir(parents=True)
    assert not ready()  # the dir alone is not a checkpoint
    (out / hook.TRACKER_FILE).write_text("release")
    assert ready()
