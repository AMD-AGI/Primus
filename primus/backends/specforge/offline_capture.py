# Copyright (c) 2026, Advanced Micro Devices, Inc.
# See LICENSE for license information.

"""Temporary plugin transport for SpecForge's offline preparation script.

Remove this module and the default capture entry-point override once SpecForge
ships plugin-backed offline capture. The upstream preparation script owns data
preparation, capture layouts, shard writing, compression, and resume behavior.
Its target factory and error cleanup are wrapped only in this worker process;
the wrapper also handles SIGTERM so sidecars are closed before the worker exits.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import socket
import sys
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

from primus.backends.specforge.capture_plugin import (
    capture_server_env,
    require_capture_plugin,
)

LOCAL_BUFFER_BYTES = 256 << 20
MIN_SEGMENT_BYTES = 1 << 30


def server_devices(env: Mapping[str, str], tp_size: int) -> str:
    """Select this local TP group's devices within the allocation's ROCr mask."""
    local_world = int(env.get("LOCAL_WORLD_SIZE", "1"))
    local_rank = int(env.get("LOCAL_RANK", "0"))
    if tp_size < 1 or tp_size > local_world or local_world % tp_size:
        raise ValueError(
            "plugin offline capture requires tp_size to divide nproc_per_node (TP stays within a node)"
        )
    visible = env.get("CUDA_VISIBLE_DEVICES") or env.get("HIP_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else [str(i) for i in range(local_world)]
    if len(devices) < local_world:
        raise ValueError("not enough visible devices for offline capture workers")
    start = local_rank // tp_size * tp_size
    return ",".join(devices[start : start + tp_size])


def capture_server_argv(args, layers, method: str, port: int) -> list[str]:
    """Translate the existing preparation flags to a standalone plugin server."""
    from specforge.config import SGLANG_CAPTURE_CONTEXT_HEADROOM

    argv = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-path",
        args.target_model_path,
        "--tp-size",
        str(args.tp_size),
        "--skip-tokenizer-init",
        "--chunked-prefill-size",
        "-1",
        "--speculative-algorithm",
        method.upper(),
        "--enable-return-aux-hidden-states",
        "--return-aux-hidden-state-layer-ids",
        *map(str, layers),
        "--return-hidden-states-mode",
        "full",
        "--disable-cuda-graph",
        "--attention-backend",
        args.sglang_attention_backend,
        "--mem-fraction-static",
        str(args.sglang_mem_fraction_static),
        "--context-length",
        str(args.sglang_context_length or args.max_length + SGLANG_CAPTURE_CONTEXT_HEADROOM),
        "--ep-size",
        str(args.sglang_ep_size),
    ]
    for name in (
        "trust_remote_code",
        "sglang_disable_radix_cache",
        "sglang_enable_nccl_nvls",
        "sglang_enable_symm_mem",
        "sglang_enable_torch_compile",
        "sglang_enable_dp_attention",
        "sglang_enable_dp_lm_head",
    ):
        if getattr(args, name, False):
            argv.append("--" + name.removeprefix("sglang_").replace("_", "-"))
    return argv


class PluginOfflineTarget:
    """Adapt the validated server/store APIs to the upstream capture interface."""

    def __init__(self, adapter, store, capture, run_id: str, task_factory=None):
        self.adapter = adapter
        self.store = store
        self.capture_config = capture
        self.run_id = run_id
        self.sequence = 0
        if task_factory is None:
            from specforge.runtime.contracts import PromptTask

            task_factory = PromptTask
        self.task_factory = task_factory

    def capture_rows(self, rows):
        tasks = []
        for row in rows:
            task_id = str(self.sequence)
            self.sequence += 1
            tasks.append(
                self.task_factory(
                    task_id=task_id,
                    run_id=self.run_id,
                    source_id=task_id,
                    payload={"input_ids": list(row)},
                    max_length=len(row),
                )
            )
        refs = []
        aux, last = [], []
        try:
            refs = self.adapter.produce_refs(tasks, capture=self.capture_config)
            if len(refs) != len(tasks):
                raise RuntimeError("offline capture returned the wrong number of rows")
            for ref in refs:
                if not hasattr(ref, "sample_id"):
                    raise RuntimeError(f"offline plugin capture failed: {ref.reason}")
            for ref in refs:
                features, handle = self.store.get(ref, device="cpu")
                try:
                    aux.append(features["aux_hidden_states"].squeeze(0))
                    last.append(features["last_hidden_states"].squeeze(0))
                finally:
                    self.store.release(handle)
            return aux, last
        finally:
            # Also covers partially successful replies and failed reads. Unknown
            # objects from a lost/malformed response retain provisional ownership.
            for ref in refs:
                if hasattr(ref, "sample_id"):
                    self.store.abort(ref.sample_id, reason="offline-batch-finished")
            self.store.discard_external_attempts(reason="offline-batch-finished")
            self.store.drain_pending_removals()

    def capture(self, *, input_ids, attention_mask, loss_mask):
        import torch

        # Upstream captures padded rows verbatim, including padding positions.
        # Keep that convention so its loss masks and persisted shapes still agree.
        aux, last = self.capture_rows(input_ids.detach().cpu().tolist())
        return SimpleNamespace(hidden_states=torch.stack(aux), last_hidden_states=torch.stack(last))

    def close(self):
        try:
            self.store.discard_external_attempts(reason="offline-capture-finished")
            self.store.drain_pending_removals()
        finally:
            self.store.close()


class _NonWriterTarget:
    def capture(self, **kwargs):
        # Only TP rank zero writes. Other ranks retain the preparation script's
        # collective ordering; the separate server owns the actual TP forward.
        return SimpleNamespace(hidden_states=None, last_hidden_states=None)


@contextmanager
def local_capture_target(args, model_config, layers, method):
    import torch.distributed as dist
    from specforge.distributed import get_tp_group
    from specforge.inference.adapters.server_capture import (
        ServerCaptureSchema,
        SGLangServerCaptureAdapter,
    )
    from specforge.inference.capture import CaptureConfig
    from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore

    from primus.backends.specforge.online_launch import (
        mooncake_env,
        mooncake_master_argv,
    )
    from primus.backends.specforge.online_supervisor import (
        _kill_group,
        _popen,
        _specforge_child_env,
        _wait_mooncake,
        _wait_sglang,
    )

    devices = server_devices(os.environ, args.tp_size)
    if dist.get_rank(get_tp_group()) != 0:
        yield _NonWriterTarget()
        return
    run_id = uuid.uuid4().hex
    log_root = Path(args.output_path) / ".primus-plugin" / run_id
    log_root.mkdir(parents=True)
    text_config = getattr(model_config, "text_config", model_config)
    hidden_size = int(text_config.hidden_size)
    # All rows of a request batch must fit before its deferred response returns.
    # Use four bytes per element as an upper bound for the model's float dtype.
    batch_bytes = args.batch_size * args.max_length * hidden_size * (len(layers) + 1) * 4
    segment_bytes = max(MIN_SEGMENT_BYTES, batch_bytes + LOCAL_BUFFER_BYTES)
    with ExitStack() as stack:
        # Allocate distinct ports; release just before starting each service.
        # A competing bind in that short interval surfaces as a startup failure.
        sockets = []
        for _ in range(4):
            sock = stack.enter_context(socket.socket())
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        ports = [sock.getsockname()[1] for sock in sockets]
        settings = dict(
            mooncake_rpc_port=ports[0],
            mooncake_http_port=ports[1],
            mooncake_metrics_port=ports[2],
            mooncake_lease_ttl_ms=500,
            mooncake_protocol="tcp",
            server_port=ports[3],
            start_timeout_s=900,
        )
        env = _specforge_child_env(mooncake_env("127.0.0.1", settings, "127.0.0.1"), devices)
        # The server owns its rendezvous. Inheriting torchrun's agent-store
        # flag makes Torch wait for an agent at the server's fresh TCP port.
        for key in tuple(env):
            if key.startswith("TORCHELASTIC_") or key in {
                "LOCAL_WORLD_SIZE",
                "ROLE_WORLD_SIZE",
                "GROUP_WORLD_SIZE",
            }:
                env.pop(key)
        env.update(
            MOONCAKE_GLOBAL_SEGMENT_SIZE=str(segment_bytes),
            MOONCAKE_LOCAL_BUFFER_SIZE=str(LOCAL_BUFFER_BYTES),
        )
        for sock in sockets[:3]:
            sock.close()
        master_log = log_root / "mooncake.log"
        master = _popen(mooncake_master_argv(settings), env, master_log)
        stack.callback(_kill_group, master)
        _wait_mooncake("127.0.0.1", settings, master, master_log)
        sockets[3].close()
        server_log = log_root / "sglang.log"
        server_env = capture_server_env(env)
        # Offline readers consume CPU tensors over loopback TCP; use host puts.
        server_env["SGLANG_SPEC_CAPTURE_GPU_PUT"] = "0"
        server = _popen(capture_server_argv(args, layers, method, ports[3]), server_env, server_log)
        stack.callback(_kill_group, server)
        _wait_sglang("127.0.0.1", 0, settings, server, server_log)
        url = f"http://127.0.0.1:{ports[3]}"
        require_capture_plugin(url, server_log)
        store = MooncakeFeatureStore(
            store_id=run_id,
            setup_kwargs=dict(
                local_hostname="127.0.0.1",
                metadata_server=env["MOONCAKE_METADATA_SERVER"],
                master_server_addr=env["MOONCAKE_MASTER_SERVER_ADDR"],
                protocol="tcp",
                global_segment_size=0,
                local_buffer_size=LOCAL_BUFFER_BYTES,
            ),
        )
        # Register cleanup immediately, including adapter-construction failures.
        stack.callback(store.close)
        capture = CaptureConfig(
            feature_names=frozenset({"aux_hidden_states", "last_hidden_states"}),
            aux_hidden_state_layer_ids=tuple(layers),
            target_repr="hidden_state",
            target_hidden_size=hidden_size,
        )
        adapter = SGLangServerCaptureAdapter(
            url,
            store,
            run_id=run_id,
            algorithm=args.strategy,
            schema=ServerCaptureSchema(
                aux_feature="aux_hidden_states", last_hidden_feature="last_hidden_states", passthrough=()
            ),
            server_capture="plugin",
        )
        target = PluginOfflineTarget(adapter, store, capture, run_id)
        stack.callback(target.close)
        yield target


def load_preparation_script():
    import specforge

    root = Path(os.environ.get("SPECFORGE_ROOT") or Path(specforge.__file__).resolve().parent.parent)
    path = root / "scripts" / "prepare_hidden_states.py"
    name = "_primus_specforge_prepare_hidden_states"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load SpecForge preparation script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve the module during import.
    spec.loader.exec_module(module)
    return module


def main():
    module = load_preparation_script()
    original = module.build_target_model
    original_destroy = module.destroy_distributed
    previous_term = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(128 + signum)

    def destroy_distributed(**kwargs):
        # The preparation script tears down groups in its own finally block.
        # A rank-local failure must not wait for peers blocked in collectives.
        if sys.exc_info()[0] is not None:
            kwargs["abort"] = True
        return original_destroy(**kwargs)

    signal.signal(signal.SIGTERM, terminate)
    try:
        with ExitStack() as resources:

            def build_target(args, model_config, capture_layers, capture_method="eagle3"):
                return resources.enter_context(
                    local_capture_target(args, model_config, capture_layers, capture_method)
                )

            module.build_target_model = build_target
            module.destroy_distributed = destroy_distributed
            try:
                module.main()
            except BaseException:
                module.destroy_distributed(abort=True)
                raise
    finally:
        module.build_target_model = original
        module.destroy_distributed = original_destroy
        signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    main()
