"""Single-host online orchestration without GPUs or a SpecForge installation."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from primus.backends.specforge import online_supervisor as supervisor
from primus.backends.specforge.online_launch import (
    online_settings,
    validate_online_identity,
)


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / "specforge"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='specforge'\n")
    (root / "draft.json").write_text('{"dflash_config":{"target_layer_ids":[1,8,15]}}')
    (root / "recipe.yaml").write_text("model:\n  draft_model_config: draft.json\n")
    params = SimpleNamespace(
        specforge_mode="train",
        specforge_train_mode="online",
        specforge_root=str(root),
        specforge_config=str(root / "recipe.yaml"),
        specforge_overrides={},
        output_dir=None,
        specforge_online=dict(
            single_node=True,
            run_id="test",
            run_root=str(tmp_path / "run"),
            consumer_state_dir=str(tmp_path / "state"),
            server_gpus="0",
            trainer_gpus="1",
        ),
    )
    return params, online_settings(params, env={})


def test_single_node_identity_and_multinode_default(setup):
    params, settings = setup
    assert not validate_online_identity(settings, rank=0, nodes=1)
    assert validate_online_identity(settings, rank=1, nodes=1)
    assert validate_online_identity(settings, rank=0, nodes=2)
    params.specforge_online.pop("single_node")
    settings = online_settings(params, env={})
    assert not settings["single_node"]
    assert not validate_online_identity(settings, rank=0, nodes=2)
    assert validate_online_identity(settings, rank=0, nodes=1)
    assert online_settings(params, env={"PRIMUS_SPECFORGE_SINGLE_NODE": "true"})["single_node"]
    params.specforge_online["single_node"] = "false"
    assert not online_settings(params, env={})["single_node"]


@pytest.mark.parametrize(
    "change,match",
    [
        ({"trainer_gpus": "0"}, "disjoint"),
        ({"capture_nnodes": 2}, "capture_nnodes=trainer_nnodes=1"),
        ({"trainer_nnodes": 2}, "capture_nnodes=trainer_nnodes=1"),
        ({"trainer_gpus": "1,1", "trainer_nproc": 2}, "unique"),
        ({"server_gpus": "0,0", "server_count": 2}, "unique"),
    ],
)
def test_invalid_layout(setup, change, match):
    _, settings = setup
    settings.update(change)
    assert any(match in issue for issue in validate_online_identity(settings, rank=0, nodes=1))


class Process:
    def __init__(self, codes):
        self.codes = list(codes)
        self.returncode = None

    def poll(self):
        if len(self.codes) > 1:
            self.returncode = self.codes.pop(0)
        else:
            self.returncode = self.codes[0]
        return self.returncode


@pytest.fixture
def harness(monkeypatch):
    launched, killed = {}, []
    codes = {"producer": [None, 0], "consumer": [None, 0]}

    def popen(argv, env, log_path):
        name = "producer" if "producer" in argv else "consumer" if "consumer" in argv else "master"
        proc = Process(codes.get(name, [None]))
        launched[name] = (proc, env, argv)
        return proc

    monkeypatch.setattr(supervisor, "_popen", popen)
    monkeypatch.setattr(supervisor, "_kill_group", lambda proc: killed.append(proc) if proc else None)
    monkeypatch.setattr(supervisor, "resolve_head_ip", lambda **kw: "127.0.0.1")
    monkeypatch.setattr(supervisor.shutil, "which", lambda name: name)
    monkeypatch.setattr(supervisor, "_wait_mooncake", lambda *a: None)
    server = Process([None])
    monkeypatch.setattr(supervisor, "_start_local_sglang", lambda *a, **kw: [server])
    monkeypatch.setattr(supervisor, "_capture_stack_failed", lambda *a: None)
    monkeypatch.setattr(supervisor.time, "sleep", lambda *a: None)
    return launched, killed, codes, server


def test_both_roles_gpu_isolation_and_cleanup(setup, harness, monkeypatch):
    params, settings = setup
    launched, killed, _, server = harness

    def no_hostname_lookup(**kwargs):
        raise AssertionError("single-node mode must not resolve a routable hostname")

    monkeypatch.setattr(supervisor, "resolve_head_ip", no_hostname_lookup)
    monkeypatch.setenv("SLURM_PROCID", "4")
    monkeypatch.setenv("RANK", "4")
    supervisor._run_single_node(params, settings)
    for role, devices in (("producer", ""), ("consumer", "1")):
        proc, env, argv = launched[role]
        assert env["CUDA_VISIBLE_DEVICES"] == env["HIP_VISIBLE_DEVICES"] == devices
        assert env["NODE_RANK"] == "0" and env["NNODES"] == "1"
        assert "RANK" not in env and "SLURM_PROCID" not in env
        assert argv[argv.index("--role") + 1] == role
        assert proc in killed
    assert server in killed and launched["master"][0] in killed
    assert (Path(settings["run_root"]) / "inference.done").read_text().strip() == "0"
    assert (Path(settings["run_root"]) / "consumer.done").read_text().strip() == "0"


@pytest.mark.parametrize("role", ["producer", "consumer"])
def test_role_failure_stops_everything(setup, harness, role):
    params, settings = setup
    launched, killed, codes, server = harness
    codes[role] = [7]
    with pytest.raises(RuntimeError, match=role + " exited with status 7"):
        supervisor._run_single_node(params, settings)
    assert all(proc in killed for proc, _, _ in launched.values())
    assert server in killed
    assert (Path(settings["run_root"]) / "inference.done").read_text().strip() == "1"


@pytest.mark.parametrize("failure", ["stack", "timeout", "interrupt", "consumer_start"])
def test_abnormal_shutdown(setup, harness, monkeypatch, failure):
    params, settings = setup
    launched, killed, codes, server = harness
    if failure == "stack":
        monkeypatch.setattr(supervisor, "_capture_stack_failed", lambda *a: "Mooncake exited")
    elif failure == "timeout":
        codes.update(producer=[0], consumer=[None])
        settings["peer_timeout_s"] = 0
    elif failure == "interrupt":

        def interrupt(*args):
            raise KeyboardInterrupt()

        monkeypatch.setattr(supervisor.time, "sleep", interrupt)
    else:
        original = supervisor._popen

        def fail_consumer(argv, env, log_path):
            if "consumer" in argv:
                raise OSError("cannot start consumer")
            return original(argv, env, log_path)

        monkeypatch.setattr(supervisor, "_popen", fail_consumer)
    with pytest.raises((RuntimeError, KeyboardInterrupt, OSError)):
        supervisor._run_single_node(params, settings)
    assert all(proc in killed for proc, _, _ in launched.values())
    assert server in killed


def test_capture_startup_failure_cleans_partial_servers(setup, monkeypatch):
    _, settings = setup
    settings["server_count"] = 2
    settings["server_gpus"] = "0,1"
    started, killed = [], []

    def popen(*args):
        proc = Process([None])
        started.append(proc)
        return proc

    def wait(ip, index, *args):
        if index == 1:
            raise RuntimeError("capture readiness failed")

    monkeypatch.setattr(supervisor, "_popen", popen)
    monkeypatch.setattr(supervisor, "_wait_sglang", wait)
    monkeypatch.setattr(supervisor, "_kill_group", killed.append)
    with pytest.raises(RuntimeError, match="capture readiness failed"):
        supervisor._start_local_sglang(settings, "127.0.0.1", "127.0.0.1", Path(settings["run_root"]), 0)
    assert len(started) == 2 and killed == started


def test_dispatch_single_node_and_existing_multi_node(setup, monkeypatch):
    params, settings = setup
    calls = []
    monkeypatch.setattr(supervisor, "online_settings", lambda params: settings)
    monkeypatch.setattr(supervisor, "nnodes", lambda: 1)
    monkeypatch.setattr(supervisor, "node_rank", lambda: 0)
    monkeypatch.setattr(supervisor, "_run_single_node", lambda *a: calls.append("single"))
    monkeypatch.setattr(supervisor, "_run_capture_node", lambda *a: calls.append("capture"))
    monkeypatch.setattr(supervisor, "_run_trainer_node", lambda *a: calls.append("trainer"))
    supervisor.run_online(params)
    settings["single_node"] = False
    monkeypatch.setattr(supervisor, "nnodes", lambda: 2)
    supervisor.run_online(params)
    monkeypatch.setattr(supervisor, "node_rank", lambda: 1)
    supervisor.run_online(params)
    assert calls == ["single", "capture", "trainer"]


def test_sigterm_cleans_up_and_restores_handler(setup, harness, monkeypatch):
    params, settings = setup
    launched, killed, _, server = harness
    original = supervisor.signal.getsignal(supervisor.signal.SIGTERM)

    def terminate(*args):
        supervisor.signal.getsignal(supervisor.signal.SIGTERM)(supervisor.signal.SIGTERM, None)

    monkeypatch.setattr(supervisor.time, "sleep", terminate)
    with pytest.raises(SystemExit):
        supervisor._run_single_node(params, settings)
    assert all(proc in killed for proc, _, _ in launched.values())
    assert server in killed
    assert supervisor.signal.getsignal(supervisor.signal.SIGTERM) == original


def test_stale_run_is_rejected_before_launch(setup, harness):
    params, settings = setup
    root = Path(settings["run_root"])
    root.mkdir()
    (root / "old-state").write_text("preserve")
    with pytest.raises(RuntimeError, match="fresh RUN_ID"):
        supervisor._run_single_node(params, settings)
    assert not harness[0]
    assert (root / "old-state").read_text() == "preserve"


@pytest.mark.parametrize("configured,expected", [(None, str(32 << 30)), (str(64 << 30), str(64 << 30))])
def test_capture_storage_segment(setup, monkeypatch, configured, expected):
    _, settings = setup
    seen = []
    if configured is None:
        monkeypatch.delenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", raising=False)
    else:
        monkeypatch.setenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", configured)

    def popen(argv, env, path):
        seen.append(env)
        return Process([None])

    monkeypatch.setattr(supervisor, "_popen", popen)
    monkeypatch.setattr(supervisor, "_wait_sglang", lambda *a: None)
    supervisor._start_local_sglang(settings, "127.0.0.1", "127.0.0.1", Path(settings["run_root"]), 0)
    assert seen[0]["MOONCAKE_GLOBAL_SEGMENT_SIZE"] == expected
