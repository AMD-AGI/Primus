# Copyright (c) 2026, Advanced Micro Devices, Inc.
# See LICENSE for license information.

"""Plugin offline capture contracts without a running server or accelerator."""

import io
import signal
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest

from primus.backends.specforge import capture_plugin, offline_capture
from primus.backends.specforge.argument_builder import build_capture_argv


def test_default_entrypoint_and_explicit_script_override():
    params = SimpleNamespace(specforge_capture={})
    argv = build_capture_argv(params)
    assert argv[4:6] == ["--module", "primus.backends.specforge.offline_capture"]
    params.specforge_capture["script"] = "custom_prepare.py"
    argv = build_capture_argv(params)
    assert argv[4] == "custom_prepare.py"
    assert "--module" not in argv


@pytest.mark.parametrize("rank, expected", [(0, "1,3"), (1, "1,3"), (2, "5,7"), (3, "5,7")])
def test_tp_groups_preserve_visible_device_mapping(rank, expected):
    env = dict(
        LOCAL_WORLD_SIZE="4",
        LOCAL_RANK=str(rank),
        CUDA_VISIBLE_DEVICES="1,3,5,7",
        ROCR_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
    )
    assert offline_capture.server_devices(env, 2) == expected


@pytest.mark.parametrize("world,tp", [(1, 2), (3, 2), (2, 0)])
def test_cross_node_or_incomplete_tp_groups_are_rejected(world, tp):
    with pytest.raises(ValueError, match="TP stays within a node"):
        offline_capture.server_devices({"LOCAL_WORLD_SIZE": str(world)}, tp)


def test_rocr_mask_is_not_reindexed_twice():
    env = dict(LOCAL_WORLD_SIZE="2", LOCAL_RANK="1", ROCR_VISIBLE_DEVICES="4,5")
    assert offline_capture.server_devices(env, 1) == "1"


def test_capture_server_flags(monkeypatch):
    config = ModuleType("specforge.config")
    config.SGLANG_CAPTURE_CONTEXT_HEADROOM = 7
    monkeypatch.setitem(sys.modules, "specforge.config", config)
    args = SimpleNamespace(
        target_model_path="model",
        tp_size=2,
        max_length=512,
        sglang_context_length=None,
        sglang_attention_backend="aiter",
        sglang_mem_fraction_static=0.6,
        sglang_ep_size=1,
        trust_remote_code=True,
        sglang_disable_radix_cache=True,
    )
    argv = offline_capture.capture_server_argv(args, [1, 8, 15], "dflash", 32100)
    assert argv[argv.index("--speculative-algorithm") + 1] == "DFLASH"
    start = argv.index("--return-aux-hidden-state-layer-ids") + 1
    assert argv[start : start + 3] == ["1", "8", "15"]
    assert argv[argv.index("--context-length") + 1] == "519"
    assert "--disable-radix-cache" in argv and "--disable-cuda-graph" in argv
    assert not any(arg.startswith("--spec-capture") for arg in argv)


class FakeStore:
    def __init__(self, features):
        self.features = features
        self.released = []
        self.aborted = []
        self.discarded = 0
        self.drained = 0
        self.closed = False
        self.fail_get = False

    def get(self, ref, *, device):
        assert device == "cpu"
        if self.fail_get:
            raise OSError("read failed")
        return self.features[ref.sample_id], ref.sample_id

    def release(self, handle):
        self.released.append(handle)

    def abort(self, sid, **kwargs):
        self.aborted.append(sid)

    def discard_external_attempts(self, **kwargs):
        self.discarded += 1

    def drain_pending_removals(self):
        self.drained += 1

    def close(self):
        self.closed = True


def make_target(store, produce):
    return offline_capture.PluginOfflineTarget(
        SimpleNamespace(produce_refs=produce), store, object(), "test-run", SimpleNamespace
    )


def test_padded_rows_feature_order_and_masks_are_preserved():
    torch = pytest.importorskip("torch")
    rows = [[2, 3, 0], [4, 0, 0]]
    captured_tasks = []
    features = {}

    def produce(tasks, **kwargs):
        captured_tasks.extend(tasks)
        for task in tasks:
            ids = torch.tensor(task.payload["input_ids"], dtype=torch.bfloat16).reshape(1, 3, 1)
            features[task.task_id] = {
                "aux_hidden_states": ids.expand(1, 3, 4).clone(),
                "last_hidden_states": ids.expand(1, 3, 2).clone(),
            }
        return [SimpleNamespace(sample_id=task.task_id) for task in tasks]

    store = FakeStore(features)
    target = make_target(store, produce)
    ids = torch.tensor(rows)
    mask = (ids != 0).to(torch.long)
    loss_mask = torch.tensor([[0, 1, 0], [1, 0, 0]])
    before = loss_mask.clone()
    result = target.capture(input_ids=ids, attention_mask=mask, loss_mask=loss_mask)
    assert [task.payload["input_ids"] for task in captured_tasks] == rows
    assert result.hidden_states.shape == (2, 3, 4)
    assert result.last_hidden_states.shape == (2, 3, 2)
    assert result.hidden_states.dtype == torch.bfloat16
    assert torch.equal(result.hidden_states[:, :, 0], ids.to(torch.bfloat16))
    assert torch.equal(loss_mask, before)
    assert store.released == ["0", "1"] and store.aborted == ["0", "1"]
    assert store.discarded == 1 and store.drained == 1
    target.close()
    assert store.closed


@pytest.mark.parametrize("failure", ["response", "partial", "read"])
def test_failed_batches_release_known_and_provisional_objects(failure):
    store = FakeStore({})
    store.fail_get = True

    def produce(tasks, **kwargs):
        if failure == "response":
            raise OSError("response lost")
        if failure == "partial":
            return [SimpleNamespace(sample_id="0"), SimpleNamespace(reason="wrong layers")]
        return [SimpleNamespace(sample_id=task.task_id) for task in tasks]

    target = make_target(store, produce)
    with pytest.raises((OSError, RuntimeError)):
        target.capture_rows([[1, 2], [3, 4]])
    assert store.discarded == 1 and store.drained == 1
    assert store.aborted == ([] if failure == "response" else ["0"] if failure == "partial" else ["0", "1"])


@pytest.mark.parametrize("termination", [False, True])
def test_factory_restored_and_resources_closed_on_preparation_failure(monkeypatch, termination):
    events = []
    original = object()
    module = SimpleNamespace(
        build_target_model=original, destroy_distributed=lambda **kwargs: events.append(("destroy", kwargs))
    )

    @contextmanager
    def runtime(*args):
        events.append("start")
        try:
            yield "plugin-target"
        finally:
            events.append("close")

    def prepare():
        assert module.build_target_model(None, None, [1]) == "plugin-target"
        if termination:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        raise ValueError("writer failed")

    module.main = prepare
    monkeypatch.setattr(offline_capture, "load_preparation_script", lambda: module)
    monkeypatch.setattr(offline_capture, "local_capture_target", runtime)
    with pytest.raises(SystemExit if termination else ValueError):
        offline_capture.main()
    assert module.build_target_model is original
    assert events == ["start", ("destroy", {"abort": True}), "close"]


@pytest.mark.parametrize("failure", ["startup", "plugin_advertisement", "store"])
def test_started_sidecars_are_stopped_on_initialization_failure(monkeypatch, tmp_path, failure):
    torch = pytest.importorskip("torch")
    from primus.backends.specforge import online_supervisor

    def install(name, **attributes):
        parts = name.split(".")
        for count in range(1, len(parts) + 1):
            key = ".".join(parts[:count])
            if key not in sys.modules:
                package = ModuleType(key)
                package.__path__ = []
                monkeypatch.setitem(sys.modules, key, package)
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    def fail_store(**kwargs):
        raise ValueError("store failed")

    install("specforge.distributed", get_tp_group=lambda: None)
    install(
        "specforge.inference.adapters.server_capture",
        SGLangServerCaptureAdapter=object,
        ServerCaptureSchema=object,
    )
    install("specforge.inference.capture", CaptureConfig=object)
    install("specforge.runtime.data_plane.mooncake_store", MooncakeFeatureStore=fail_store)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *args: 0)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("TORCHELASTIC_USE_AGENT_STORE", "True")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "parent-agent")
    started, stopped = [], []

    def popen(*args):
        child_env = args[1]
        assert not any(key.startswith("TORCHELASTIC_") for key in child_env)
        assert "LOCAL_WORLD_SIZE" not in child_env
        proc = object()
        started.append(proc)
        return proc

    def wait_server(*args):
        if failure == "startup":
            raise TimeoutError("server failed")

    monkeypatch.setattr(online_supervisor, "_popen", popen)
    monkeypatch.setattr(online_supervisor, "_kill_group", stopped.append)
    monkeypatch.setattr(online_supervisor, "_wait_mooncake", lambda *args: None)
    monkeypatch.setattr(online_supervisor, "_wait_sglang", wait_server)
    monkeypatch.setattr(offline_capture, "capture_server_argv", lambda *args: ["python"])
    body = b'{"internal_states":[{"spec_capture_backend":"plugin"}]}' if failure == "store" else b"{}"
    monkeypatch.setattr(capture_plugin.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(body))
    args = SimpleNamespace(output_path=str(tmp_path), tp_size=1, batch_size=1, max_length=8)
    with pytest.raises((TimeoutError, RuntimeError, ValueError)):
        with offline_capture.local_capture_target(args, SimpleNamespace(hidden_size=2), [1], "dflash"):
            pytest.fail("initialization should fail")
    assert len(started) == 2
    assert stopped == started[::-1]
