# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe
"""CPU unit tests for train/eval_lookahead.py (no GPU, no torchrec, no process group).

A fake one-rank world stands in for the GPU and torchrec: streams log their
operations in host order, events remember where they were recorded, and a
stand-in for ship14's _StreamingTrainPipeline replays the eval progress()
order (fill, wait_for_batch, forward with the EC call and its first touch,
deferred next-batch input dist, dequeue). Every collective a rank issues is
appended to a per-world log, so the lookahead's per-rank collective order can be
compared with the stock pipeline's. The real eval_lookahead code runs on top.

Covered: issue order and idempotence (0/1/2/5-batch passes), no lookahead for
index 0, the num_eval_batches cap, the EVAL_INTERIM_LOG decline, stock fallback
on a miss, fence selection (ev_c(N-1), pass start, missing fence), drain at
detach/reset/attach, a failure-injection matrix around request.wait() and
compute_and_output_dist, record_stream bookkeeping, VERIFY (and its rank-
uniform extra collective), REPORT, ISSUE=top, PRIO=-1 polls, consecutive passes,
prepare_before_run_start (probe, pool, vote, fail closed, knob off touches
nothing), build_train_pipeline(role=...) selection (the real function, executed
with fakes), the train_ranker / _run_eval_window hooks, and the gin binding.

Fixes after the V1 review (iter64): the collective-free output dist guard (F1:
data_parallel / unknown sharding types, world size 1, torchrec sync
collectives; at the build and in phase A; the production TW+CW and RW plans arm
with no extra GPU op), the small-pool pre-grow (F2: alloc_pregrow's recipe,
fail closed, same _POOL_GIB switch), the TP_DEFER_INPUT_DIST build decline
(F3), TP_EVAL_PIPELINE=0 skipping the stream and pool (the vote still runs),
and the gin wording (VERIFY per sharding, F4 coverage note).
"""
import ast
import collections
import contextlib
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
import types
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import torch

from generative_recommenders.dlrm_v4.train import eval_lookahead as evla

try:
    import gin  # noqa: F401
except ImportError:  # pragma: no cover
    gin = None  # type: ignore[assignment]

KNOB = evla.ENV
_TRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.abspath(os.path.join(_TRAIN, "..", "..", ".."))
_GIN = os.path.join(_TRAIN, "gin", "yambda_5b.gin")
_BOOT = os.path.join(_TRAIN, "_env_bootstrap.py")
_RANKER = os.path.join(_TRAIN, "train_ranker.py")
_UTILS = os.path.join(_TRAIN, "utils.py")
_GIB = 1 << 30
_LENGTHS = torch.tensor([2, 4], dtype=torch.int64)


# ---------------------------------------------------------------------------
# Fake GPU
# ---------------------------------------------------------------------------
class World:
    """One rank: a host-order GPU op log, a collective log, failure plans."""

    current: "Optional[World]" = None

    def __init__(self, plan: Optional[Dict[str, Any]] = None) -> None:
        self.log: List[Tuple[str, str, str]] = []
        self.coll: List[Tuple[Any, ...]] = []
        self.plan: Dict[str, Any] = dict(plan or {})
        self.compute_calls: List[Tuple[int, str]] = []
        self.n_events = 0
        self.events_done = True
        self.cuda = FakeCuda(self)
        World.current = self

    def fail_at(self, point: str, idx: int) -> bool:
        return self.plan.get(point) == idx


class FakeStream:
    _next = [1000]

    def __init__(self, world: World, name: str, priority: int = 0, handle: Optional[int] = None) -> None:
        self.world = world
        self.name = name
        self.priority = priority
        if handle is None:
            FakeStream._next[0] += 1
            handle = FakeStream._next[0]
        self.cuda_stream = handle

    def _op(self, op: str, detail: str) -> None:
        self.world.log.append((self.name, op, detail))

    def kernel(self, what: str) -> None:
        self._op("kernel", what)

    def wait_event(self, ev: "FakeEvent") -> None:
        if self.world.plan.get("fence_raise") and self.name == "S":
            raise RuntimeError("injected: wait_event failed")
        self._op("wait_event", ev.label)

    def wait_stream(self, other: "FakeStream") -> None:
        self._op("wait_stream", other.name)


class FakeEvent:
    def __init__(self, world: World) -> None:
        self.world = world
        self.label: Optional[str] = None
        self.queries = 0
        self.done = world.events_done

    def record(self, stream: Optional[FakeStream] = None) -> None:
        stream = stream or self.world.cuda.current_stream()
        self.world.n_events += 1
        self.label = "ev%d@%s" % (self.world.n_events, stream.name)
        stream._op("record", self.label)

    def query(self) -> bool:
        self.queries += 1
        return self.done


class FakeCuda:
    def __init__(self, world: World) -> None:
        self.world = world
        self.main = FakeStream(world, "main", 0, handle=0)
        self._stack = [self.main]
        self.calls: List[Any] = []
        self.pool: List[FakeStream] = []
        self.n_streams = 0
        self.free = 60 * _GIB
        self.total = 256 * _GIB
        self.retries = 0
        self.retry_on_grow = False

    def current_stream(self) -> FakeStream:
        return self._stack[-1]

    @contextlib.contextmanager
    def stream(self, s: FakeStream):
        self._stack.append(s)
        try:
            yield
        finally:
            self._stack.pop()

    def Stream(self, priority: int = 0) -> FakeStream:  # noqa: N802
        self.calls.append(("Stream", priority))
        if self.pool:
            return self.pool.pop(0)
        self.n_streams += 1
        return FakeStream(self.world, "cand%d" % self.n_streams, priority)

    def Event(self) -> FakeEvent:  # noqa: N802
        return FakeEvent(self.world)

    def is_available(self) -> bool:
        self.calls.append("is_available")
        return True

    def mem_get_info(self) -> Tuple[int, int]:
        self.calls.append("mem_get_info")
        return self.free, self.total

    def memory_stats(self) -> Dict[str, int]:
        self.calls.append("memory_stats")
        return {"num_alloc_retries": self.retries}

    def synchronize(self) -> None:
        self.calls.append("synchronize")


class FakeTensor:
    is_cuda = True

    def __init__(self, world: World, name: str) -> None:
        self.world = world
        self.name = name
        self.recorded: List[str] = []

    def record_stream(self, s: FakeStream) -> None:
        if self.world.plan.get("raise_record_wire") and self.name.startswith("recv"):
            raise RuntimeError("injected: record_stream on the receive wire")
        self.recorded.append(s.name)


# ---------------------------------------------------------------------------
# Fake torchrec awaitables, module, forward and pipeline
# ---------------------------------------------------------------------------
class FakeAwaitable:
    def __init__(self) -> None:
        self._callbacks: List[Any] = []

    @property
    def callbacks(self) -> List[Any]:
        return self._callbacks

    def _wait_impl(self) -> Any:
        raise NotImplementedError

    def wait(self) -> Any:
        ret = self._wait_impl()
        for cb in self.callbacks:
            ret = cb(ret)
        return ret


class FakeLazy(FakeAwaitable):
    """LazyAwaitable: an unknown attribute resolves the awaitable (logged)."""

    def __init__(self) -> None:
        super().__init__()
        self._result = None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        World.current.log.append(("host", "lazy_getattr_wait", name))
        return getattr(resolve(self), name)


def resolve(x: Any) -> Any:
    """LazyAwaitable._wait_async: the model's first touch."""
    if isinstance(x, FakeLazy):
        if x.__dict__.get("_result") is None:
            x._result = x.wait()
        return x._result
    return x


class FakeJT:
    def __init__(self, values: torch.Tensor, lengths: torch.Tensor) -> None:
        self._v, self._l = values, lengths

    def values(self) -> torch.Tensor:
        return self._v

    def lengths(self) -> torch.Tensor:
        return self._l

    def offsets(self) -> torch.Tensor:
        return torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(self._l, 0)])

    def weights_or_none(self) -> None:
        return None


def emb_values(idx: int, shard: str, corrupt: bool = False) -> torch.Tensor:
    v = torch.arange(6, dtype=torch.float32) + 10.0 * idx + (100.0 if shard == "CW" else 0.0)
    if corrupt:
        v = v.clone()
        v[0] = v[0] + 0.5
    return v


class FakeRequest(FakeAwaitable):
    """comm_ops.Request of one sequence all_to_all (receive wire + dummy)."""

    def __init__(self, world: World, idx: int, shard: str, values: torch.Tensor) -> None:
        super().__init__()
        self.world, self.idx, self.shard, self._values = world, idx, shard, values
        s = world.cuda.current_stream()
        self.tensor = FakeTensor(world, "recv(%d,%s)@%s" % (idx, shard, s.name))
        self.dummy_tensor = FakeTensor(world, "dummy(%d,%s)@%s" % (idx, shard, s.name))
        self.req = "work"

    def _wait_impl(self) -> torch.Tensor:
        s = self.world.cuda.current_stream()
        s._op("work_wait", "sq(%d,%s)" % (self.idx, self.shard))
        s.kernel("K3'(%d,%s)" % (self.idx, self.shard))
        self.tensor = None
        self.req = None
        return self._values


class FakeSeqAwaitable(FakeAwaitable):
    def __init__(self, req: FakeRequest) -> None:
        super().__init__()
        self._tensor_awaitable = req

    def _wait_impl(self) -> Any:
        return self._tensor_awaitable.wait()


class FakeECAwaitable(FakeLazy):
    def __init__(self, world: World, idx: int, shards: List[FakeSeqAwaitable]) -> None:
        super().__init__()
        self.world, self.idx = world, idx
        self._awaitables_per_sharding = shards

    def _wait_impl(self) -> Dict[str, FakeJT]:
        out = {}
        for name, a in zip(("TW", "CW"), self._awaitables_per_sharding):
            out[name] = FakeJT(a.wait(), _LENGTHS)
        self.world.cuda.current_stream().kernel("cjt(%d)" % self.idx)
        return out


class FakeKJT:
    def __init__(self, world: World, idx: int, built_on: str) -> None:
        self.world, self.idx, self.built_on = world, idx, built_on
        self.recorded: List[str] = []

    def record_stream(self, s: FakeStream) -> None:
        if self.world.fail_at("kjt_record", self.idx):
            raise RuntimeError("injected: after request.wait()")
        self.recorded.append(s.name)


class FakeWork:
    def __init__(self) -> None:
        self.polls = 0

    def is_completed(self) -> bool:
        self.polls += 1
        return True


class FakeKJTA2A:
    def __init__(self) -> None:
        self._awaitables = [FakeWork(), FakeWork()]


class FakeKJTRequest(FakeAwaitable):
    """KJTListAwaitable of one batch (dist_init runs on the current stream)."""

    def __init__(self, world: World, idx: int) -> None:
        super().__init__()
        self.world, self.idx = world, idx
        self.awaitables = [FakeKJTA2A()]
        self.waits = 0

    def _wait_impl(self) -> FakeKJT:
        if self.world.fail_at("wait_inside", self.idx):
            raise RuntimeError("injected: inside request.wait()")
        self.waits += 1
        s = self.world.cuda.current_stream()
        s.kernel("dist_init(%d)" % self.idx)
        return FakeKJT(self.world, self.idx, s.name)


class FakeMctx:
    def __init__(self, world: World, idx: int) -> None:
        self.world, self.idx = world, idx
        self.recorded: List[str] = []

    def record_stream(self, s: FakeStream) -> None:
        self.recorded.append(s.name)


_PLAN_1N = ("table_wise", "column_wise")  # U3 section 1.3: 1n TW + CW; 2n/4n RW


class FakeEC:
    """ShardedEmbeddingCollection: lookup + K1 + one seq a2a per sharding.
    ``_sharding_type_to_sharding`` carries the plan's sharding types (keys
    only matter; the values are never read)."""

    def __init__(self, world: World, name: str = "ec",
                 shardings: Tuple[str, ...] = _PLAN_1N) -> None:
        self.world, self.name = world, name
        self.forward: Any = None
        self._sharding_type_to_sharding = {t: object() for t in shardings}

    def compute_and_output_dist(self, mctx: FakeMctx, kjt: FakeKJT) -> FakeECAwaitable:
        w = self.world
        s = w.cuda.current_stream()
        idx = kjt.idx
        n_before = sum(1 for i, _ in w.compute_calls if i == idx)
        w.compute_calls.append((idx, s.name))
        if w.fail_at("compute_before", idx):
            raise RuntimeError("injected: before the collective")
        s.kernel("lookup(%d)" % idx)
        s.kernel("K1(%d)" % idx)
        corrupt = w.plan.get("corrupt") == idx and n_before == 0
        shards = []
        for sh in ("TW", "CW"):
            w.coll.append(("sq", idx, sh))
            s._op("a2a", "sq(%d,%s)" % (idx, sh))
            shards.append(FakeSeqAwaitable(FakeRequest(w, idx, sh, emb_values(idx, sh, corrupt))))
            if w.fail_at("compute_after_collective", idx):
                raise RuntimeError("injected: after the collective")
        return FakeECAwaitable(w, idx, shards)


class FakePipelinedForward:
    """torchrec PipelinedForward with ship14's dist_init-on-main class flag."""

    _use_main_stream_for_dist_init = True

    def __init__(self, name: str, args: Any, module: FakeEC, context: Any, stream: Any = None) -> None:
        self._name, self._args, self._module = name, args, module
        self._context, self._stream = context, stream

    @property
    def name(self) -> str:
        return self._name

    def set_context(self, context: Any) -> None:
        self._context = context

    def __call__(self, *input, **kwargs):
        w = World.current
        if self._name not in self._context.input_dist_tensors_requests:
            raise AssertionError("Invalid PipelinedForward usage: %s" % self._name)
        request = self._context.input_dist_tensors_requests.pop(self._name)
        data = request.wait()  # on main: the class flag
        ctx = self._context.module_contexts.pop(self._name)
        cur = w.cuda.current_stream()
        if self._stream is not None:
            cur.wait_stream(self._stream)
            data.record_stream(cur)
            ctx.record_stream(cur)
        return self._module.compute_and_output_dist(ctx, data)


class FakeCtx:
    def __init__(self, index: int) -> None:
        self.index = index
        self.input_dist_tensors_requests: Dict[str, Any] = {}
        self.module_contexts: Dict[str, Any] = {}


class _ModList(list):
    """The model's ECs: a list (the fake pipeline iterates it) that is also
    callable like nn.Module.modules(), so the build-time walk sees them."""

    def __call__(self) -> Any:
        return iter(list(self))


class FakeModel:
    def __init__(self, world: World, n_modules: int = 1,
                 shardings: Tuple[str, ...] = _PLAN_1N) -> None:
        self.world = world
        self.training = False
        self.modules = _ModList(FakeEC(world, "ec%d" % i, shardings) for i in range(n_modules))

    def forward(self, idx: int) -> List[Any]:
        main = self.world.cuda.current_stream()
        outs = []
        for m in self.modules:
            aw = m.forward()  # the EC call (dlrm_hstu.py:1033)
            main.kernel("preproc(%d)" % idx)
            outs.append(resolve(aw))  # first touch (dlrm_hstu.py:1112)
        main.kernel("dense(%d)" % idx)
        return outs


class FakeBasePipeline:
    """Stand-in for ship14's _StreamingTrainPipeline in eval: the same queue
    shape and per-rank collective order (splits sp, tensors t, seq sq)."""

    _pipelined_forward_type = FakePipelinedForward

    def __init__(self, model: Any = None, optimizer: Any = None, device: Any = None,
                 execute_all_batches: bool = True, enqueue_batch_after_forward: bool = True,
                 custom_model_fwd: Any = None) -> None:
        w = World.current
        self.world = w
        self._model = model
        self._data_dist_stream = FakeStream(w, "dd", -1)
        self._memcpy_stream = FakeStream(w, "memcpy", -1)
        self.contexts: collections.deque = collections.deque()
        self.batches: collections.deque = collections.deque()
        self._pipelined_modules: List[Any] = []
        self._model_attached = True
        self._next_index = 0
        self._pending_t: Optional[FakeCtx] = None

    def _set_module_context(self, ctx: FakeCtx) -> None:
        for m in self._pipelined_modules:
            m.forward.set_context(ctx)

    def _enqueue(self, it: Any) -> bool:
        b = next(it, None)
        if b is None:
            return False
        self.batches.append(b)
        self.contexts.append(FakeCtx(self._next_index))
        self._next_index += 1
        return True

    def _issue_tensors(self, ctx: FakeCtx) -> None:
        w = self.world
        w.coll.append(("t", ctx.index))
        self._data_dist_stream._op("a2a", "t(%d)" % ctx.index)
        for m in self._model.modules:
            ctx.input_dist_tensors_requests[m.name] = FakeKJTRequest(w, ctx.index)
            ctx.module_contexts[m.name] = FakeMctx(w, ctx.index)

    def _issue_splits(self, ctx: FakeCtx) -> None:
        self.world.coll.append(("sp", ctx.index))
        self._data_dist_stream._op("a2a", "sp(%d)" % ctx.index)

    def fill_pipeline(self, it: Any) -> None:
        if len(self.batches) >= 2 or self.batches:
            return
        if not self._enqueue(it):
            return
        if not self._pipelined_modules:
            for m in self._model.modules:
                m.forward = self._pipelined_forward_type(
                    m.name, None, m, self.contexts[0], self._data_dist_stream
                )
                self._pipelined_modules.append(m)
        self._issue_splits(self.contexts[0])
        self._issue_tensors(self.contexts[0])
        self._enqueue(it)

    def progress(self, it: Any) -> Any:
        if not self._model_attached:
            self.attach(self._model)
        self.fill_pipeline(it)
        if not self.batches:
            raise StopIteration
        self._set_module_context(self.contexts[0])
        self.world.cuda.current_stream().wait_stream(self._data_dist_stream)  # _wait_for_batch
        out = self._model.forward(self.contexts[0].index)
        self._enqueue(it)  # enqueue_batch_after_forward
        if len(self.batches) >= 2:
            self._issue_splits(self.contexts[1])  # deferred start_sparse_data_dist
            self._pending_t = self.contexts[1]  # deferred split wait
        self.dequeue_batch()
        return out

    def dequeue_batch(self) -> None:
        if self._pending_t is not None:
            self._issue_tensors(self._pending_t)
            self._pending_t = None
        self.batches.popleft()
        self.contexts.popleft()
        if self.batches:
            self._set_module_context(self.contexts[0])

    def detach(self) -> Any:
        for m in self._pipelined_modules:
            m.forward = None
        self._model_attached = False
        return self._model

    def attach(self, model: Any = None, sparse_dist: bool = True) -> None:
        self._model_attached = True
        if self.contexts:
            # ship14 attach(): re-run the rewrite and re-issue batches[0]'s input dist
            ctx = self.contexts[0]
            ctx.input_dist_tensors_requests.clear()
            ctx.module_contexts.clear()
            self._pipelined_modules = []
            for m in self._model.modules:
                m.forward = self._pipelined_forward_type(m.name, None, m, ctx, self._data_dist_stream)
                self._pipelined_modules.append(m)
            self._issue_splits(ctx)
            self._issue_tensors(ctx)
        else:
            self._pipelined_modules = []

    def reset(self) -> None:
        self.contexts.clear()
        self.batches.clear()
        self._next_index = 0
        self._pending_t = None


class FakeHsf:
    def __init__(self) -> None:
        self.distinit = True
        self.polled: List[Any] = []

    def dist_init_on_main(self) -> bool:
        return self.distinit

    def _poll_until_done(self, work: Any) -> None:
        self.polled.append(work)
        while not work.is_completed():
            pass


def run_pass(world: World, p: Any, n_batches: int, cap: Optional[int] = None) -> List[Any]:
    """_run_eval_window's loop: progress() until StopIteration (or the cap),
    then detach() and reset()."""
    it = iter(range(n_batches))
    outs: List[Any] = []
    n = 0
    while True:
        world.log.append(("host", "progress", str(n)))
        try:
            out = p.progress(it)
        except StopIteration:
            break
        outs.append(out)
        n += 1
        if cap is not None and n >= cap:
            break
    p.detach()
    p.reset()
    return outs


def values_of(outs: List[Any]) -> List[List[Tuple[str, List[float]]]]:
    return [[(k, v.values().tolist()) for d in out for k, v in d.items()] for out in outs]


# ---------------------------------------------------------------------------
# Test base: env isolation, log capture
# ---------------------------------------------------------------------------
# The two torchrec readers, kept before setUp pins them (their own reader tests below).
_REAL_KJT_WAIT = evla._kjt_wait_records_current
_REAL_READY = evla._ready_value

_ENV_KEYS = (KNOB, evla.VERIFY_ENV, evla.ISSUE_ENV, evla.PRIO_ENV, evla.POOL_ENV,
             evla.REPORT_ENV, "RANK", "LOCAL_RANK", "TP_DEFER_INPUT_DIST",
             "TP_EVAL_PIPELINE")


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self._env = mock.patch.dict(
            os.environ, {k: v for k, v in os.environ.items() if k not in _ENV_KEYS}, clear=True
        )
        self._env.start()
        os.environ["RANK"] = "3"
        # These fakes stand in for torchrec, so pin the two helpers that read torchrec to their
        # torchrec-free behaviour. The suite then runs the same with or without torchrec.
        # In a bare unit-test process inside the GPU container (torchrec importable,
        # host_sync_free.p_kjtt_wait not installed), _kjt_wait_records_current() is False,
        # so every arming would decline. _ready_value would wrap values in torchrec's
        # LazyNoWait. A real run installs p_kjtt_wait before the eval pipeline is built.
        for name, new in (("_kjt_wait_records_current", lambda: True),
                          ("_ready_value", lambda value: value)):
            patcher = mock.patch.object(evla, name, new=new)
            patcher.start()
            self.addCleanup(patcher.stop)
        evla._reset_for_tests()
        self.records: List[logging.LogRecord] = []
        test = self

        class _Keep(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                test.records.append(record)

        self._h = _Keep(level=logging.DEBUG)
        evla.logger.addHandler(self._h)
        self._lvl = evla.logger.level
        evla.logger.setLevel(logging.DEBUG)

    def tearDown(self) -> None:
        evla.logger.removeHandler(self._h)
        evla.logger.setLevel(self._lvl)
        evla._reset_for_tests()
        self._env.stop()
        World.current = None

    def msgs(self, needle: str = "", level: Optional[int] = None) -> List[str]:
        return [r.getMessage() for r in self.records
                if needle in r.getMessage() and (level is None or r.levelno == level)]

    # an armed world: prepared side stream S
    def armed(self, world: World, prio: int = 0) -> FakeStream:
        s = FakeStream(world, "S", prio)
        evla._prepared.update(done=True, stream=s, prio=prio, tries=1, pool_gib=4.0,
                              ready=8, world=8, reason=None)
        return s

    def make(self, world: World, hsf: Optional[FakeHsf] = None, n_modules: int = 1,
             shardings: Tuple[str, ...] = _PLAN_1N, walkable: bool = True) -> Any:
        cls = evla.wrap(FakeBasePipeline, forward_base=FakePipelinedForward,
                        cuda=world.cuda, hsf=hsf or FakeHsf())
        model = FakeModel(world, n_modules, shardings)
        if not walkable:  # the build-time walk sees no module; phase A must catch it
            model.modules = list(model.modules)
        return cls(model=model)

    def stock_run(self, n: int, cap: Optional[int] = None, n_modules: int = 1) -> World:
        """The stock eval pipeline on its own world (World.current restored)."""
        prev = World.current
        w = World()
        p = FakeBasePipeline(model=FakeModel(w, n_modules))
        w.outs = run_pass(w, p, n, cap)  # type: ignore[attr-defined]
        World.current = prev
        return w


def _idx(log: List[Tuple[str, str, str]], entry: Tuple[str, str, str], start: int = 0) -> int:
    for i in range(start, len(log)):
        if log[i] == entry:
            return i
    raise AssertionError("%r not in the op log" % (entry,))


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
class IssueOrderTest(_Base):
    def test_collective_order_matches_stock_for_0_1_2_5_batch_passes(self) -> None:
        for n in (0, 1, 2, 5):
            with self.subTest(batches=n):
                stock = self.stock_run(n)
                w = World()
                self.armed(w)
                p = self.make(w)
                outs = run_pass(w, p, n)
                self.assertEqual(w.coll, stock.coll)
                self.assertEqual(values_of(outs), values_of(stock.outs))
                st = p._dlrm_evla
                # batch 0 stock on main, batches 1..n-1 looked ahead on S
                looked = sorted(i for i, s in w.compute_calls if s == "S")
                self.assertEqual(looked, list(range(1, n)))
                self.assertEqual(sorted(i for i, s in w.compute_calls if s == "main"),
                                 [0] if n else [])
                self.assertEqual([e for e in w.log if e[1] == "lazy_getattr_wait"], [])
                if n:
                    self.assertEqual(st.pass_no, 2)  # one pass counted
                # every TBE forward of the pass in batch order: 0 on main, then S
                self.assertEqual([i for i, _ in w.compute_calls], list(range(n)))

    def test_issue_is_idempotent_and_never_for_index_0(self) -> None:
        w = World()
        self.armed(w)
        p = self.make(w)
        it = iter(range(4))
        ctl = p._dlrm_evla
        p.fill_pipeline(it)  # what progress(0) does first
        self.assertEqual(p.contexts[0].index, 0)
        ctl.issue(p)  # index 0: never looked ahead
        self.assertEqual(w.compute_calls, [])
        p.progress(it)  # forward(0) stock; lookahead(1) at the end
        self.assertEqual(w.compute_calls, [(0, "main"), (1, "S")])
        n_coll = len(w.coll)
        ctl.issue(p)
        ctl.issue(p)
        self.assertEqual(len(w.coll), n_coll)
        self.assertEqual(w.compute_calls, [(0, "main"), (1, "S")])
        self.assertEqual(ctl.stats["issued"], 1)

    def test_stream_placement_of_each_step(self) -> None:
        w = World()
        S = self.armed(w)
        p = self.make(w)
        run_pass(w, p, 3)
        ops = {(s, d) for s, o, d in w.log if o == "kernel"}
        for d in ("dist_init(0)", "lookup(0)", "K1(0)", "K3'(0,TW)", "cjt(0)", "dense(0)"):
            self.assertIn(("main", d), ops)
        for k in (1, 2):
            for d in ("dist_init(%d)" % k, "lookup(%d)" % k, "K1(%d)" % k):
                self.assertIn(("S", d), ops)
                self.assertNotIn(("main", d), ops)
            for d in ("K3'(%d,TW)" % k, "K3'(%d,CW)" % k, "cjt(%d)" % k, "dense(%d)" % k):
                self.assertIn(("main", d), ops)  # decode and dense stay on main
        self.assertIn(("S", "a2a", "sq(1,TW)"), w.log)
        self.assertIn(("main", "work_wait", "sq(1,TW)"), w.log)
        self.assertIsNot(S, w.cuda.main)


class FenceTest(_Base):
    def _ev_after(self, w: World, kernel: str) -> str:
        i = _idx(w.log, ("main", "kernel", kernel))
        nxt = w.log[i + 1]
        self.assertEqual(nxt[:2], ("main", "record"), w.log[i:i + 3])
        return nxt[2]

    def test_each_lookahead_waits_on_the_previous_batch_consumed_event(self) -> None:
        w = World()
        self.armed(w)
        p = self.make(w)
        run_pass(w, p, 4)
        for k in (1, 2, 3):
            ev = self._ev_after(w, "cjt(%d)" % (k - 1))  # ev_c(k-1): right after the EC wait
            i = _idx(w.log, ("S", "wait_event", ev))
            self.assertEqual(w.log[i + 1], ("S", "wait_stream", "dd"))
            j = _idx(w.log, ("S", "kernel", "dist_init(%d)" % k), i)
            # nothing on S between the fence and dist_init(k)
            self.assertTrue(all(e[0] != "S" for e in w.log[i + 2:j]))
        self.assertEqual(self.msgs("fence missing"), [])
        # the first S op of every lookahead is its ev_c fence: S's op list splits
        # into one group per lookahead, each starting with its only wait_event
        s_ops = [e for e in w.log if e[0] == "S"]
        starts = [i for i, e in enumerate(s_ops) if e[1] == "wait_event"]
        self.assertEqual(starts[0], 0)
        self.assertEqual(len(starts), 3)
        for a, b in zip(starts, starts[1:] + [len(s_ops)]):
            self.assertEqual(s_ops[b - 1][1], "a2a")  # each group ends with its seq a2a

    def test_first_lookahead_of_a_pass_waits_on_batch0_of_that_pass(self) -> None:
        w = World()
        self.armed(w)
        p = self.make(w)
        run_pass(w, p, 3)
        mark = len(w.log)
        run_pass(w, p, 3)
        second = w.log[mark:]
        ev0 = None
        for i, e in enumerate(second):
            if e == ("main", "kernel", "cjt(0)"):
                ev0 = second[i + 1][2]
                break
        self.assertIsNotNone(ev0)
        i = _idx(second, ("S", "wait_event", ev0))
        _idx(second, ("S", "kernel", "dist_init(1)"), i)

    def test_missing_fence_waits_on_the_whole_main_stream_and_warns_once(self) -> None:
        w = World()
        self.armed(w)
        p = self.make(w)
        ctl = p._dlrm_evla
        orig = ctl._attach_ev
        ctl._attach_ev = lambda aw, idx: None  # no callback: ev_c never recorded
        try:
            run_pass(w, p, 4)
        finally:
            ctl._attach_ev = orig
        self.assertEqual(len(self.msgs("consumed-event fence missing")), 1)
        self.assertRegex(self.msgs("fence missing")[0],
                         r"consumed-event fence missing at index 1; side stream waited on the "
                         r"whole main stream rank=3$")
        for k in (1, 2, 3):
            j = _idx(w.log, ("S", "kernel", "dist_init(%d)" % k))
            prev = [e for e in w.log[:j] if e[0] == "S"][-2:]
            self.assertEqual(prev, [("S", "wait_stream", "main"), ("S", "wait_stream", "dd")])
        self.assertEqual(ctl.stats["fence_missing"], 0)  # per-pass stats reset at reset()


class RecordStreamTest(_Base):
    def test_records_at_issue_and_at_the_forward(self) -> None:
        w = World()
        self.armed(w)
        p = self.make(w)
        it = iter(range(3))
        p.progress(it)  # lookahead(1) parked on contexts[0]
        pend = getattr(p.contexts[0], evla.STORE_ATTR)["ec0"]
        self.assertEqual(pend.kjt.built_on, "S")
        self.assertEqual(pend.kjt.recorded, ["S"])
        self.assertEqual(pend.mctx.recorded, ["S"])
        for sa in pend.aw._awaitables_per_sharding:
            req = sa._tensor_awaitable
            self.assertTrue(req.tensor.name.endswith("@S"))
            self.assertEqual(req.tensor.recorded, ["main"])  # receive wire
            self.assertEqual(req.dummy_tensor.recorded, ["main"])  # Request dummy
        self.assertEqual(pend.shardings, 2)
        mark = len(w.log)
        p.progress(it)  # forward(1): hit
        self.assertEqual(pend.mctx.recorded, ["S", "main"])
        self.assertEqual(pend.kjt.recorded, ["S", "main"])
        # main waits on data_dist at the forward (stock rf:124-127), before the wire wait
        i = _idx(w.log, ("main", "wait_stream", "dd"), mark + 1)
        j = _idx(w.log, ("main", "work_wait", "sq(1,TW)"), mark)
        self.assertLess(i, j)

    def test_shared_arange_cache_recorded_on_both_streams(self) -> None:
        from generative_recommenders.dlrm_v4 import seq_a2a_rowperm as sarp

        w = World()
        self.armed(w)
        p = self.make(w)
        t = FakeTensor(w, "arange")
        key = ("fake-device", "int64")
        sarp._arange_cache[key] = t
        try:
            run_pass(w, p, 2)
        finally:
            sarp._arange_cache.pop(key, None)
        self.assertEqual(t.recorded, ["S", "main"])


class CapAndInterimTest(_Base):
    def test_cap_skips_the_lookahead_at_and_past_the_cap(self) -> None:
        stock = self.stock_run(6, cap=3)
        w = World()
        self.armed(w)
        p = self.make(w)
        p.dlrm_evla_configure(3, False)
        outs = run_pass(w, p, 6, cap=3)
        self.assertEqual(w.coll, stock.coll)
        self.assertEqual(values_of(outs), values_of(stock.outs))
        self.assertEqual(sorted(i for i, s in w.compute_calls if s == "S"), [1, 2])
        self.assertEqual(self.msgs("drained"), [])
        # None / 0 / negative: no cap
        for v in (None, 0, -1):
            p.dlrm_evla_configure(v, False)
            self.assertIsNone(p._dlrm_evla.cap)

    def test_interim_metrics_decline_to_the_stock_forward(self) -> None:
        stock = self.stock_run(4)
        w = World()
        self.armed(w)
        p = self.make(w)
        p.dlrm_evla_configure(None, True)
        self.assertIs(p._pipelined_forward_type, FakePipelinedForward)
        self.assertRegex(
            self.msgs("DECLINED")[0],
            r"^DLRMV4 eval emb lookahead DECLINED \(EVAL_INTERIM_LOG>0: compute_and_log "
            r"collectives sit between eval batches\): stock eval path rank=3$",
        )
        run_pass(w, p, 4)
        self.assertEqual(w.coll, stock.coll)
        self.assertEqual({s for _, s in w.compute_calls}, {"main"})
        self.assertNotIn("S", {e[0] for e in w.log})


class FallbackTest(_Base):
    def test_phase_a_decline_keeps_the_request_and_runs_stock(self) -> None:
        stock = self.stock_run(5)
        w = World()
        self.armed(w)
        hsf = FakeHsf()
        p = self.make(w, hsf)
        it = iter(range(5))
        outs = [p.progress(it)]  # lookahead(1) issued
        hsf.distinit = False  # phase A of lookahead(2) fails on this rank
        while True:
            try:
                outs.append(p.progress(it))
            except StopIteration:
                break
        p.detach()
        p.reset()
        self.assertEqual(w.coll, stock.coll)
        self.assertEqual(values_of(outs), values_of(stock.outs))
        self.assertEqual(w.compute_calls, [(0, "main"), (1, "S"), (2, "main"), (3, "main"),
                                           (4, "main")])
        dis = self.msgs("DISABLED")
        self.assertEqual(len(dis), 1)
        self.assertRegex(dis[0], r"^DLRMV4 eval emb lookahead DISABLED on rank 3: "
                                 r"dist_init_on_main\(\) is off; stock eval path for the rest "
                                 r"of the run$")

    def test_phase_a_matrix_declines_before_any_awaitable_call(self) -> None:
        def training(p: Any, w: World) -> None:
            p._model.training = True

        def wrong_forward(p: Any, w: World) -> None:
            m = p._pipelined_modules[0]
            m.forward = FakePipelinedForward(m.name, None, m, p.contexts[0], p._data_dist_stream)

        def no_request(p: Any, w: World) -> None:
            p.contexts[0].input_dist_tensors_requests.clear()

        def no_mctx(p: Any, w: World) -> None:
            p.contexts[0].module_contexts.clear()

        def raising(p: Any, w: World) -> None:
            type(p)._pipelined_modules = property(lambda self: 1 / 0)  # noqa: B023

        cases = [(training, "model in training mode"),
                 (wrong_forward, "not this pipeline's EvalLookaheadForward"),
                 (no_request, "no input-dist request for ec0 at index 1"),
                 (no_mctx, "no module context for ec0 at index 1"),
                 (raising, "phase A raised ZeroDivisionError")]
        for mutate, why in cases:
            with self.subTest(why=why):
                self.records.clear()
                evla._reset_for_tests()
                w = World()
                self.armed(w)
                p = self.make(w)
                it = iter(range(4))
                p.fill_pipeline(it)
                p.progress(it)  # stock forward(0); lookahead(1) issued at the end
                # undo lookahead(1) bookkeeping for the injected state: rebuild context 1
                ctx = p.contexts[0]
                delattr(ctx, evla.STORE_ATTR)
                p._issue_tensors(ctx)
                reqs_before = dict(ctx.input_dist_tensors_requests)
                n_coll = len(w.coll)
                n_calls = len(w.compute_calls)
                mutate(p, w)
                p._dlrm_evla.issue(p)
                if mutate is raising:
                    del type(p)._pipelined_modules
                self.assertEqual(len(w.coll), n_coll)
                self.assertEqual(len(w.compute_calls), n_calls)
                self.assertIsNone(getattr(ctx, evla.STORE_ATTR, None))
                if mutate not in (no_request,):
                    self.assertEqual(ctx.input_dist_tensors_requests, reqs_before)
                    self.assertTrue(all(r.waits == 0 for r in reqs_before.values()))
                self.assertIn(why, self.msgs("DISABLED")[0])
                self.assertFalse(p._dlrm_evla.on)

    def test_phase_a_output_guard_declines_before_any_awaitable_call(self) -> None:
        # V1 F1 in phase A: the build-time walk cannot see the module (or the
        # switch flips later); the guard on the pipelined modules stops the
        # first lookahead before request.wait(), so the stock forward runs every
        # batch with the stock collectives. The inputs (plan, WORLD size,
        # process-global switch) are the same on every rank: every rank stops
        # at the same index.
        def dp(p: Any) -> None:
            p._pipelined_modules[0]._sharding_type_to_sharding = {"data_parallel": None}

        def unknown(p: Any) -> None:
            del p._pipelined_modules[0]._sharding_type_to_sharding

        def world1(p: Any) -> None:
            evla._prepared["world"] = 1

        # (make() kwargs, mutation after the build, sync switch during the pass, reason)
        cases = [
            (dict(walkable=False, shardings=("table_wise", "data_parallel")), None, False,
             "module ec0 has a data_parallel sharding: its output dist issues no sequence "
             "all_to_all for the main stream to wait on"),
            ({}, dp, False, "module ec0 has a data_parallel sharding"),
            ({}, unknown, False, "module ec0: sharding types unknown (fail closed)"),
            ({}, world1, False, "world size 1: the sequence all_to_all is skipped"),
            ({}, None, True, "torchrec sync collectives on"),
            ({}, None, None, "torchrec sync-collectives switch unreadable (fail closed)"),
        ]
        for make_kw, mutate, sync, why in cases:
            with self.subTest(why=why):
                self.records.clear()
                evla._reset_for_tests()
                stock = self.stock_run(4)
                w = World()
                self.armed(w)
                p = self.make(w, **make_kw)
                self.assertTrue(p._dlrm_evla.installed)  # the build check passed
                it = iter(range(4))
                p.fill_pipeline(it)  # forwards installed, nothing looked ahead yet
                if mutate is not None:
                    mutate(p)
                outs = []
                with mock.patch.object(evla, "_sync_collectives_on", return_value=sync):
                    while True:
                        try:
                            outs.append(p.progress(it))
                        except StopIteration:
                            break
                    p.detach()
                    p.reset()
                self.assertEqual(w.coll, stock.coll)
                self.assertEqual(values_of(outs), values_of(stock.outs))
                self.assertEqual(w.compute_calls, [(k, "main") for k in range(4)])
                self.assertNotIn("S", {e[0] for e in w.log})
                dis = self.msgs("DISABLED")
                self.assertEqual(len(dis), 1)
                self.assertIn("DISABLED on rank 3: " + why, dis[0])
                self.assertFalse(p._dlrm_evla.on)

    def test_phase_a_guard_after_an_active_pass(self) -> None:
        # pass 1 looks ahead; the switch turns on before pass 2: pass 2 runs
        # stock from its batch 0, collectives identical to stock
        stock = self.stock_run(4)
        w = World()
        self.armed(w)
        p = self.make(w)
        run_pass(w, p, 4)
        self.assertEqual(sorted(i for i, s in w.compute_calls if s == "S"), [1, 2, 3])
        mark, calls = len(w.coll), len(w.compute_calls)
        with mock.patch.object(evla, "_sync_collectives_on", return_value=True):
            outs = run_pass(w, p, 4)
        self.assertEqual(w.coll[mark:], stock.coll)
        self.assertEqual(values_of(outs), values_of(stock.outs))
        self.assertEqual(w.compute_calls[calls:], [(k, "main") for k in range(4)])
        self.assertIn("torchrec sync collectives on", self.msgs("DISABLED")[0])

    def test_fence_failure_declines_before_any_awaitable_call(self) -> None:
        stock = self.stock_run(4)
        w = World(plan={"fence_raise": True})
        self.armed(w)
        p = self.make(w)
        outs = run_pass(w, p, 4)
        self.assertEqual(w.coll, stock.coll)
        self.assertEqual(values_of(outs), values_of(stock.outs))
        self.assertIn("fence raised RuntimeError", self.msgs("DISABLED")[0])
        self.assertEqual({s for _, s in w.compute_calls}, {"main"})

    def test_failures_from_request_wait_onwards_reraise_without_a_retry(self) -> None:
        for point in ("wait_inside", "kjt_record", "compute_before",
                      "compute_after_collective", "raise_record_wire"):
            with self.subTest(point=point):
                self.records.clear()
                evla._reset_for_tests()
                plan = {point: 2} if point != "raise_record_wire" else {}
                w = World(plan=plan)
                self.armed(w)
                p = self.make(w)
                it = iter(range(5))
                p.progress(it)  # forward(0), lookahead(1)
                if point == "raise_record_wire":
                    w.plan["raise_record_wire"] = True
                with self.assertRaises(RuntimeError) as cm:
                    p.progress(it)  # forward(1) hit; lookahead(2) fails
                self.assertIn("injected", str(cm.exception))
                errs = self.msgs("failure after collectives", logging.ERROR)
                self.assertEqual(len(errs), 1)
                self.assertRegex(errs[0], r"failure after collectives may have been issued "
                                          r"\(index 2\); re-raising rank=3$")
                sq2 = [c for c in w.coll if c[:2] == ("sq", 2)]
                expect = {"wait_inside": [], "kjt_record": [], "compute_before": [],
                          "compute_after_collective": [("sq", 2, "TW")],
                          "raise_record_wire": [("sq", 2, "TW"), ("sq", 2, "CW")]}[point]
                self.assertEqual(sq2, expect)  # issued at most once, never retried
                self.assertEqual(self.msgs("DISABLED"), [])
                self.assertIsNone(getattr(p.contexts[0], evla.STORE_ATTR, None))


class PassBoundaryTest(_Base):
    def test_detach_drains_unconsumed_lookahead_and_joins_side_stream(self) -> None:
        w = World()
        self.armed(w)
        p = self.make(w)
        it = iter(range(4))
        p.progress(it)  # lookahead(1) parked, never consumed
        mark = len(w.log)
        p.detach()
        tail = w.log[mark:]
        i = _idx(tail, ("main", "wait_stream", "dd"))
        j = _idx(tail, ("main", "work_wait", "sq(1,TW)"), i)
        _idx(tail, ("main", "kernel", "cjt(1)"), j)
        k = _idx(tail, ("main", "wait_stream", "S"), j)
        self.assertGreater(k, j)
        self.assertEqual(self.msgs("drained"),
                         ["DLRMV4 eval emb lookahead: drained 1 unconsumed lookahead(s) at "
                          "detach rank=3"])
        self.assertIsNone(getattr(p.contexts[0], evla.STORE_ATTR, None))
        p.reset()  # nothing left: no second drain; joins again
        self.assertEqual(len(self.msgs("drained")), 1)
        self.assertEqual(w.log[-1], ("main", "wait_stream", "S"))
        self.assertEqual(len(p.contexts), 0)

    def test_reset_drains_and_joins(self) -> None:
        w = World()
        self.armed(w)
        p = self.make(w)
        it = iter(range(4))
        p.progress(it)
        p.reset()
        self.assertEqual(len(self.msgs("drained 1 unconsumed lookahead(s) at reset")), 1)
        self.assertEqual(w.log[-1], ("main", "wait_stream", "S"))

    def test_attach_drains_and_never_resolves_across_detach_attach(self) -> None:
        w = World()
        self.armed(w)
        p = self.make(w)
        it = iter(range(4))
        p.progress(it)  # lookahead(1) parked
        p.attach()  # mid-pass re-attach (not in _run_eval_window): drain first
        self.assertEqual(len(self.msgs("drained 1 unconsumed lookahead(s) at attach")), 1)
        self.assertIsNone(getattr(p.contexts[0], evla.STORE_ATTR, None))
        self.assertEqual(p._dlrm_evla.ev_idx, -1)  # new generation: no stale fence
        out = p.progress(it)  # forward(1): re-issued input dist, stock path
        self.assertEqual(out[0]["TW"].values().tolist(), emb_values(1, "TW").tolist())
        self.assertEqual([c for c in w.compute_calls if c[0] == 1], [(1, "S"), (1, "main")])

    def test_three_consecutive_passes(self) -> None:
        w = World()
        self.armed(w)
        os.environ[evla.REPORT_ENV] = "1"
        p = self.make(w)
        stock = self.stock_run(4)
        for _ in range(3):
            mark = len(w.coll)
            outs = run_pass(w, p, 4)
            self.assertEqual(w.coll[mark:], stock.coll)
            self.assertEqual(values_of(outs), values_of(stock.outs))
        rep = self.msgs("[eval-lookahead]")
        self.assertEqual([re.search(r"pass=(\d+)", m).group(1) for m in rep], ["1", "2", "3"])
        self.assertEqual(len(self.msgs("ACTIVE")), 1)
        self.assertEqual(self.msgs("drained"), [])
        self.assertEqual(self.msgs("fence missing"), [])
        joins = [e for e in w.log if e == ("main", "wait_stream", "S")]
        self.assertEqual(len(joins), 6)  # detach + reset of every pass


class VerifyTest(_Base):
    def test_verify_compares_bytes_and_adds_one_recompute_per_verified_batch(self) -> None:
        os.environ[evla.VERIFY_ENV] = "2"
        stock = self.stock_run(5)
        w = World()
        self.armed(w)
        p = self.make(w)
        outs = run_pass(w, p, 5)
        self.assertEqual(values_of(outs), values_of(stock.outs))
        self.assertEqual(w.compute_calls, [(0, "main"), (1, "S"), (1, "main"), (2, "S"),
                                           (2, "main"), (3, "S"), (4, "S")])
        ver = self.msgs("VERIFY")
        self.assertEqual(ver, ["DLRMV4 eval emb lookahead VERIFY index=1 bitwise=True rank=3",
                               "DLRMV4 eval emb lookahead VERIFY index=2 bitwise=True rank=3"])
        # the fence of lookahead(k+1) is recorded after the recompute of k
        i = _idx(w.log, ("main", "kernel", "cjt(1)"))
        i = _idx(w.log, ("main", "kernel", "cjt(1)"), i + 1)  # the recompute's
        self.assertEqual(w.log[i + 1][1], "record")
        _idx(w.log, ("S", "wait_event", w.log[i + 1][2]))

    def test_mismatch_returns_stock_values_and_disables(self) -> None:
        os.environ[evla.VERIFY_ENV] = "3"
        stock = self.stock_run(5)
        w = World(plan={"corrupt": 2})
        self.armed(w)
        p = self.make(w)
        outs = run_pass(w, p, 5)
        self.assertEqual(values_of(outs), values_of(stock.outs))  # stock values returned
        self.assertIn("DLRMV4 eval emb lookahead VERIFY index=2 bitwise=False rank=3",
                      self.msgs("VERIFY", logging.ERROR))
        self.assertIn("VERIFY byte mismatch at index 2", self.msgs("DISABLED")[0])
        self.assertEqual([c for c in w.compute_calls if c[1] == "S"], [(1, "S"), (2, "S")])

    def test_verify_extra_collective_is_rank_uniform_across_paths(self) -> None:
        os.environ[evla.VERIFY_ENV] = "2"
        w_on = World()
        self.armed(w_on)
        run_pass(w_on, self.make(w_on), 5)
        evla._reset_for_tests()
        w_off = World()
        self.armed(w_off)
        hsf = FakeHsf()
        p = self.make(w_off, hsf)
        hsf.distinit = False  # this rank: DISABLED at the first issue, stock forwards
        run_pass(w_off, p, 5)
        self.assertTrue(self.msgs("DISABLED"))
        self.assertEqual(w_off.coll, w_on.coll)
        self.assertEqual([c for c in w_off.coll if c[:2] == ("sq", 1)],
                         [("sq", 1, "TW"), ("sq", 1, "CW")] * 2)


class DiagnosticsTest(_Base):
    def test_report_line(self) -> None:
        os.environ[evla.REPORT_ENV] = "1"
        w = World()
        w.events_done = False  # the host is ahead: ev_c(N) not complete at T_b(N)
        self.armed(w)
        p = self.make(w)
        run_pass(w, p, 5)
        self.assertEqual(self.msgs("[eval-lookahead]"),
                         ["[eval-lookahead] rank=3 pass=1 issued=4 hit=4 stock=1 host_ahead=4/4"])

    def test_issue_top_places_the_lookahead_in_progress_k(self) -> None:
        stock = self.stock_run(5)
        for mode in ("end", "top"):
            with self.subTest(mode=mode):
                os.environ[evla.ISSUE_ENV] = mode
                evla._reset_for_tests()
                w = World()
                self.armed(w)
                p = self.make(w)
                run_pass(w, p, 5)
                self.assertEqual(w.coll, stock.coll)
                for k in (1, 2, 3, 4):
                    top = _idx(w.log, ("host", "progress", str(k)))
                    d = _idx(w.log, ("S", "kernel", "dist_init(%d)" % k))
                    pre = _idx(w.log, ("main", "kernel", "preproc(%d)" % k))
                    self.assertLess(d, pre)
                    self.assertEqual(d > top, mode == "top")

    def test_priority_minus_one_polls_before_every_issue(self) -> None:
        w = World()
        self.armed(w, prio=-1)
        hsf = FakeHsf()
        p = self.make(w, hsf)
        run_pass(w, p, 4)
        self.assertEqual(len(hsf.polled), 3 * 2)  # 3 lookaheads x (lengths, values) Works
        self.assertEqual(p._dlrm_evla.stats["issued"], 0)  # reset at the pass end

    def test_installed_and_active_lines(self) -> None:
        w = World()
        self.armed(w)
        p = self.make(w)
        self.assertEqual(self.msgs("installed"),
                         ["DLRMV4 eval emb lookahead: eval pipeline subclass installed "
                          "(prio=0 issue=end rank=3)"])
        self.assertIs(p._pipelined_forward_type.__mro__[1], evla.forward_class(FakePipelinedForward))
        run_pass(w, p, 3)
        self.assertEqual(self.msgs("ACTIVE"),
                         ["DLRMV4 eval emb lookahead ACTIVE: first looked-ahead batch consumed "
                          "(pass=1 index=1 shardings=2 rank=3)"])
        self.assertEqual(self.msgs("DECLINED") + self.msgs("DISABLED"), [])

    def test_env_parsing(self) -> None:
        self.assertFalse(evla.enabled())
        for v, on in (("1", True), (" 1 ", True), ("0", False), ("true", False), ("", False)):
            os.environ[KNOB] = v
            self.assertEqual(evla.enabled(), on, v)
        os.environ[evla.VERIFY_ENV] = "x"
        self.assertEqual(evla.verify_count(), 0)
        os.environ[evla.ISSUE_ENV] = "middle"
        self.assertEqual(evla.issue_mode(), "end")
        os.environ[evla.PRIO_ENV] = "-2"
        self.assertEqual(evla.side_priority(), 0)
        os.environ[evla.PRIO_ENV] = "-1"
        self.assertEqual(evla.side_priority(), -1)
        for v in ("-1", "nan", "inf", "x"):
            os.environ[evla.POOL_ENV] = v
            self.assertEqual(evla.pool_gib(), 4.0, v)
        os.environ[evla.POOL_ENV] = "0"
        self.assertEqual(evla.pool_gib(), 0.0)


class BuildDeclineTest(_Base):
    def _declined(self, w: World, hsf: Optional[FakeHsf] = None,
                  shardings: Tuple[str, ...] = _PLAN_1N) -> Any:
        p = self.make(w, hsf, shardings=shardings)
        self.assertIs(p._pipelined_forward_type, FakePipelinedForward)
        self.assertFalse(p._dlrm_evla.on)
        stock = self.stock_run(4)
        run_pass(w, p, 4)
        self.assertEqual(w.coll, stock.coll)
        self.assertEqual({s for _, s in w.compute_calls}, {"main"})
        self.assertNotIn("S", {e[0] for e in w.log})  # S never used
        return p

    # ---- V1 F1: output dists that issue no collective ---------------------
    def test_collective_free_or_unknown_sharding_declines_at_build(self) -> None:
        # allowlist: anything but table_wise / column_wise / row_wise declines
        for plan, bad in ((("table_wise", "data_parallel"), "data_parallel"),
                          (("data_parallel",), "data_parallel"),
                          (("table_row_wise",), "table_row_wise"),
                          (("row_wise", "grid_shard"), "grid_shard")):
            with self.subTest(plan=plan):
                self.records.clear()
                evla._reset_for_tests()
                w = World()
                self.armed(w)
                self._declined(w, shardings=plan)
                self.assertEqual(
                    self.msgs("DECLINED", logging.WARNING),
                    ["DLRMV4 eval emb lookahead DECLINED (module FakeEC has a %s sharding: its "
                     "output dist issues no sequence all_to_all for the main stream to wait on): "
                     "stock eval path rank=3" % bad])

    def test_sync_collectives_world_size_one_or_unreadable_switch_decline_at_build(self) -> None:
        cases = (("sync", True, 8, "torchrec sync collectives on: the sequence all_to_all "
                                   "returns NoWait"),
                 ("unreadable", None, 8, "torchrec sync-collectives switch unreadable (fail "
                                         "closed)"),
                 ("world1", False, 1, "world size 1: the sequence all_to_all is skipped (NoWait)"))
        for name, sync, world, why in cases:
            with self.subTest(case=name):
                self.records.clear()
                evla._reset_for_tests()
                w = World()
                self.armed(w)
                evla._prepared["world"] = world
                with mock.patch.object(evla, "_sync_collectives_on", return_value=sync):
                    self._declined(w)
                msg = self.msgs("DECLINED", logging.WARNING)
                self.assertEqual(len(msg), 1)
                self.assertIn(why, msg[0])

    def test_sync_collectives_switch_reader(self) -> None:
        # no torchrec (this CPU venv): False; torchrec present: its switch; a
        # torchrec without the switch: None (the guard then fails closed)
        mods = {"torchrec": types.ModuleType("torchrec"),
                "torchrec.distributed": types.ModuleType("torchrec.distributed")}
        mods["torchrec"].__path__ = []  # type: ignore[attr-defined]
        mods["torchrec.distributed"].__path__ = []  # type: ignore[attr-defined]
        co = types.ModuleType("torchrec.distributed.comm_ops")
        mods["torchrec.distributed.comm_ops"] = co
        mods["torchrec.distributed"].comm_ops = co  # type: ignore[attr-defined]
        with mock.patch.dict(sys.modules, mods):
            self.assertIsNone(evla._sync_collectives_on())
            for v in (False, True):
                co.get_use_sync_collectives = lambda v=v: v  # type: ignore[attr-defined]
                self.assertIs(evla._sync_collectives_on(), v)
        with mock.patch.dict(sys.modules, {"torchrec": None}):
            self.assertIs(evla._sync_collectives_on(), False)

    def test_production_plans_arm_and_add_no_gpu_op(self) -> None:
        # 1n TW + CW, 2n/4n RW (U3 section 1.3): installed, looked ahead, and
        # the guard is host-only: main waits on no event, only on the Works
        # and the data-dist stream, exactly as before the guard
        for plan in (_PLAN_1N, ("row_wise",)):
            with self.subTest(plan=plan):
                self.records.clear()
                evla._reset_for_tests()
                stock = self.stock_run(4)
                w = World()
                self.armed(w)
                p = self.make(w, shardings=plan)
                self.assertTrue(p._dlrm_evla.installed)
                outs = run_pass(w, p, 4)
                self.assertEqual(w.coll, stock.coll)
                self.assertEqual(values_of(outs), values_of(stock.outs))
                self.assertEqual(sorted(i for i, s in w.compute_calls if s == "S"), [1, 2, 3])
                self.assertEqual(self.msgs("DECLINED") + self.msgs("DISABLED"), [])
                self.assertEqual([e for e in w.log if e[0] == "main" and e[1] == "wait_event"],
                                 [])
                self.assertEqual({d for s, o, d in w.log if s == "main" and o == "wait_stream"},
                                 {"dd", "S"})

    # ---- V1 F3: the deferred input dist -----------------------------------
    def test_non_deferred_input_dist_declines_at_build(self) -> None:
        # build_train_pipeline defers only on exactly "1" (tu:3639); any other
        # value declines on every rank, with the reason, before S is touched
        for v in ("0", "", "true", " 1", "1 "):
            with self.subTest(v=v):
                self.records.clear()
                evla._reset_for_tests()
                os.environ["TP_DEFER_INPUT_DIST"] = v
                w = World()
                self.armed(w)
                self._declined(w)
                self.assertEqual(
                    self.msgs("DECLINED", logging.WARNING),
                    ["DLRMV4 eval emb lookahead DECLINED (TP_DEFER_INPUT_DIST=%r: without the "
                     "deferred input dist a rank's stock fallback would pair different "
                     "collectives with its peers): stock eval path rank=3" % v])
        for v in (None, "1"):
            with self.subTest(v=v):
                self.records.clear()
                evla._reset_for_tests()
                if v is None:
                    os.environ.pop("TP_DEFER_INPUT_DIST", None)
                else:
                    os.environ["TP_DEFER_INPUT_DIST"] = v
                w = World()
                self.armed(w)
                p = self.make(w)
                self.assertTrue(p._dlrm_evla.installed)
                self.assertEqual(self.msgs("DECLINED"), [])

    def test_prepare_declined_for_the_input_dist_logs_one_warning(self) -> None:
        # prepare already warned (and made no stream): the build line is INFO
        os.environ[KNOB] = "1"
        os.environ["TP_DEFER_INPUT_DIST"] = "0"
        w = World()
        self.assertFalse(evla.prepare_before_run_start(cuda=w.cuda, dist=FakeDist(),
                                                       probe=mock.Mock(), grow=mock.Mock(),
                                                       vote=lambda r: 0,
                                                       grow_small=mock.Mock()))
        self._declined(w)
        self.assertEqual(len(self.msgs("DECLINED", logging.WARNING)), 1)
        info = self.msgs("DECLINED", logging.INFO)
        self.assertEqual(len(info), 1)
        self.assertIn("(TP_DEFER_INPUT_DIST='0': without the deferred input dist", info[0])

    def test_no_prepared_stream(self) -> None:
        w = World()
        self._declined(w)
        self.assertRegex(self.msgs("DECLINED", logging.WARNING)[0],
                         r"^DLRMV4 eval emb lookahead DECLINED \(no prepared side stream "
                         r"\(prepare_before_run_start did not run before RUN_START\)\): stock "
                         r"eval path rank=3$")

    def test_prepare_declined_warns_once(self) -> None:
        # the pre-RUN_START decline already logged its WARNING: the build line is INFO
        evla._prepared.update(done=True, stream=None, reason="hw-queue probe could not run")
        w = World()
        self._declined(w)
        self.assertEqual(self.msgs("DECLINED", logging.WARNING), [])
        self.assertEqual(self.msgs("DECLINED", logging.INFO),
                         ["DLRMV4 eval emb lookahead DECLINED (no prepared side stream (hw-queue "
                          "probe could not run)): stock eval path rank=3"])

    def test_syncfree_or_distinit_off(self) -> None:
        w = World()
        self.armed(w)
        hsf = FakeHsf()
        hsf.distinit = False
        self._declined(w, hsf)
        self.assertIn("DLRM_HSTU_FP8_SYNCFREE or DLRM_HSTU_FP8_SYNCFREE_DISTINIT_MAIN off",
                      self.msgs("DECLINED")[0])

    def test_kjt_wait_patch_missing(self) -> None:
        w = World()
        self.armed(w)
        with mock.patch.object(evla, "_kjt_wait_records_current", return_value=False):
            self._declined(w)
        self.assertIn("p_kjtt_wait", self.msgs("DECLINED")[0])

    def test_kjt_wait_guard_reader(self) -> None:
        # the real guard (setUp pins it): without torchrec True (unit tests inject their own
        # pipeline); with torchrec, True only once host_sync_free installed p_kjtt_wait
        dd = types.ModuleType("torchrec.distributed.dist_data")

        class KJTAllToAllTensorsAwaitable:
            def _wait_impl(self) -> None:  # torchrec's own wait
                return None

        dd.KJTAllToAllTensorsAwaitable = KJTAllToAllTensorsAwaitable  # type: ignore[attr-defined]
        tr, td = types.ModuleType("torchrec"), types.ModuleType("torchrec.distributed")
        tr.__path__ = []  # type: ignore[attr-defined]
        td.__path__ = []  # type: ignore[attr-defined]
        tr.distributed = td  # type: ignore[attr-defined]
        td.dist_data = dd  # type: ignore[attr-defined]
        mods = {"torchrec": tr, "torchrec.distributed": td, "torchrec.distributed.dist_data": dd}
        with mock.patch.dict(sys.modules, mods):
            self.assertFalse(_REAL_KJT_WAIT())

            def p_kjtt_wait(self: Any) -> None:
                return None

            KJTAllToAllTensorsAwaitable._wait_impl = p_kjtt_wait  # type: ignore[assignment]
            self.assertFalse(_REAL_KJT_WAIT())  # the module flag is still missing
            dd._dlrm_lpk_patched = True  # type: ignore[attr-defined]
            self.assertTrue(_REAL_KJT_WAIT())
        # absent: block every level (an already-imported submodule resolves from sys.modules)
        with mock.patch.dict(sys.modules, dict.fromkeys(mods)):
            self.assertTrue(_REAL_KJT_WAIT())

    def test_ready_value_reader(self) -> None:
        # the real lazy-touch wrapper (setUp pins it): torchrec's LazyNoWait when torchrec is
        # importable, the value itself without torchrec
        ty = types.ModuleType("torchrec.distributed.types")

        class LazyNoWait:
            def __init__(self, obj: Any) -> None:
                self.obj = obj

        ty.LazyNoWait = LazyNoWait  # type: ignore[attr-defined]
        tr, td = types.ModuleType("torchrec"), types.ModuleType("torchrec.distributed")
        tr.__path__ = []  # type: ignore[attr-defined]
        td.__path__ = []  # type: ignore[attr-defined]
        tr.distributed = td  # type: ignore[attr-defined]
        td.types = ty  # type: ignore[attr-defined]
        mods = {"torchrec": tr, "torchrec.distributed": td, "torchrec.distributed.types": ty}
        with mock.patch.dict(sys.modules, mods):
            r = _REAL_READY(5)
            self.assertIsInstance(r, LazyNoWait)
            self.assertEqual(r.obj, 5)
        # absent: block every level (an already-imported submodule resolves from sys.modules)
        with mock.patch.dict(sys.modules, dict.fromkeys(mods)):
            self.assertEqual(_REAL_READY(5), 5)


# ---------------------------------------------------------------------------
# prepare_before_run_start
# ---------------------------------------------------------------------------
class FakeDist:
    def __init__(self, world_size: int = 8, rank: int = 3) -> None:
        self._ws, self._rank = world_size, rank
        self.calls: List[str] = []

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return True

    def get_world_size(self) -> int:
        return self._ws

    def get_rank(self) -> int:
        return self._rank


class PrepareTest(_Base):
    def _prepare(self, w: World, probe_results: List[Optional[bool]], votes: Optional[int] = 8,
                 small_effect: Optional[str] = None, **env: str) -> Tuple[bool, Dict[str, Any]]:
        os.environ[KNOB] = "1"
        os.environ.update(env)
        seen: Dict[str, Any] = {"probe": [], "grow": [], "grow_small": [], "vote": []}
        results = list(probe_results)

        def probe(a: FakeStream, b: FakeStream) -> Optional[bool]:
            seen["probe"].append((a.name, b.name))
            return results.pop(0) if results else False

        def grow(nbytes: int) -> None:
            seen["grow"].append((nbytes, w.cuda.current_stream().name))
            w.cuda.calls.append("grow")
            if w.cuda.retry_on_grow:
                w.cuda.retries += 1

        def grow_small(nbytes: int) -> None:
            seen["grow_small"].append((nbytes, w.cuda.current_stream().name))
            w.cuda.calls.append("grow_small")
            if small_effect == "retry":  # the small grow alone hit the retry path
                w.cuda.retries += 1
            elif small_effect == "raise":
                raise RuntimeError("injected: small-pool grow out of memory")

        def vote(ready: int) -> int:
            seen["vote"].append(ready)
            if votes is None:
                raise RuntimeError("injected: all_reduce failed")
            return votes if ready else min(votes, 7)

        armed = evla.prepare_before_run_start(cuda=w.cuda, dist=FakeDist(), probe=probe,
                                              grow=grow, vote=vote, grow_small=grow_small)
        return armed, seen

    def test_knob_off_touches_nothing(self) -> None:
        for v in (None, "0", "true"):
            with self.subTest(v=v):
                evla._reset_for_tests()
                if v is None:
                    os.environ.pop(KNOB, None)
                else:
                    os.environ[KNOB] = v
                w = World()
                dist = mock.Mock()
                probe, grow, vote = mock.Mock(), mock.Mock(), mock.Mock()
                self.assertFalse(evla.prepare_before_run_start(
                    cuda=w.cuda, dist=dist, probe=probe, grow=grow, vote=vote))
                self.assertEqual(w.cuda.calls, [])
                self.assertEqual(dist.mock_calls, [])
                for f in (probe, grow, vote):
                    f.assert_not_called()
                self.assertFalse(evla._prepared["done"])
                self.assertEqual(self.records, [])

    def test_arms_the_first_candidate_off_mains_queue(self) -> None:
        w = World()
        armed, seen = self._prepare(w, [True, False])
        self.assertTrue(armed)
        s = evla.prepared_stream()
        self.assertEqual(s.name, "cand2")
        self.assertEqual(w.cuda.calls.count(("Stream", 0)), 2)
        self.assertEqual(seen["probe"], [("main", "cand1"), ("main", "cand2")])
        self.assertEqual(seen["grow"], [(4 * _GIB, "cand2")])  # pre-grown on S itself
        self.assertEqual(seen["grow_small"], [(64 << 20, "cand2")])  # V1 F2: S's small pool too
        # large then small, both before the synchronize and the retry re-read
        self.assertEqual(w.cuda.calls[-5:], ["memory_stats", "grow", "grow_small",
                                             "synchronize", "memory_stats"])
        self.assertEqual(seen["vote"], [1])
        self.assertEqual(self.msgs(evla.ARMED),
                         ["armed OK: eval emb lookahead stream ready (prio=0 tries=2 "
                          "pool=4.0GiB small=64MiB beside main; ranks ready 8/8) rank=3"])

    def test_rejects_handles_of_main_and_of_pre_created_side_streams(self) -> None:
        from generative_recommenders.dlrm_v4 import alloc_pregrow as ap

        w = World()
        side = FakeStream(w, "eval_dd", -1)
        w.cuda.pool = [FakeStream(w, "alias_main", 0, handle=0),
                       FakeStream(w, "alias_side", 0, handle=side.cuda_stream),
                       FakeStream(w, "good", 0)]
        with mock.patch.object(ap, "_side_streams", [{"memcpy": FakeStream(w, "m", -1),
                                                      "data_dist": side}]), \
                mock.patch.object(ap, "_pair_roles", ["eval"]):
            armed, seen = self._prepare(w, [False])
        self.assertTrue(armed)
        self.assertEqual(evla.prepared_stream().name, "good")
        self.assertEqual(seen["probe"], [("main", "good")])  # aliases rejected unprobed

    def test_probe_unavailable_fails_closed(self) -> None:
        w = World()
        armed, seen = self._prepare(w, [None])
        self.assertFalse(armed)
        self.assertIsNone(evla.prepared_stream())
        self.assertEqual(seen["grow"], [])
        self.assertEqual(seen["vote"], [0])
        self.assertEqual(self.msgs("DECLINED"),
                         ["DLRMV4 eval emb lookahead DECLINED (no usable side stream on 1/8 "
                          "ranks: hw-queue probe could not run (fail closed)): stock eval path "
                          "rank=3"])

    def test_at_most_eight_tries(self) -> None:
        w = World()
        armed, seen = self._prepare(w, [True] * 20)
        self.assertFalse(armed)
        self.assertEqual(len(seen["probe"]), evla.MAX_TRIES)
        self.assertIn("8/8 candidates rejected (shares main's hw queue)", self.msgs("DECLINED")[0])

    def test_peer_not_ready_declines_on_every_rank(self) -> None:
        w = World()
        armed, seen = self._prepare(w, [False], votes=7)
        self.assertFalse(armed)
        self.assertIsNone(evla.prepared_stream())
        self.assertEqual(self.msgs("DECLINED"),
                         ["DLRMV4 eval emb lookahead DECLINED (no usable side stream on 1/8 "
                          "ranks: this rank ready; peer ranks not): stock eval path rank=3"])

    def test_vote_failure_declines_locally(self) -> None:
        w = World()
        armed, _ = self._prepare(w, [False], votes=None)
        self.assertFalse(armed)
        self.assertTrue(self.msgs("readiness all_reduce failed", logging.ERROR))
        self.assertIn("vote failed", self.msgs("DECLINED")[0])

    def test_pool_must_fit_above_the_minimum_free(self) -> None:
        # (free above the floor, armed): 3.9 GiB short; 32 MiB short of the
        # small pool (V1 F2: the fit check counts both pools); exactly enough
        min_free = evla.POOL_MIN_FREE_AFTER_GIB * _GIB
        for free, ok in ((min_free + 3.9 * _GIB, False),
                         (min_free + 4 * _GIB + (32 << 20), False),
                         (min_free + 4 * _GIB + (64 << 20), True)):
            with self.subTest(free=free):
                self.records.clear()
                evla._reset_for_tests()
                w = World()
                w.cuda.free = int(free)
                armed, seen = self._prepare(w, [False])
                self.assertEqual(armed, ok)
                if ok:
                    self.assertEqual(len(seen["grow"]) + len(seen["grow_small"]), 2)
                    continue
                self.assertEqual(seen["grow"] + seen["grow_small"], [])
                self.assertEqual(seen["vote"], [0])
                self.assertIn("pool pre-grow 4.0 GiB + 64 MiB small does not fit",
                              self.msgs("DECLINED")[0])

    def test_pool_zero_and_custom(self) -> None:
        w = World()
        armed, seen = self._prepare(w, [False], **{evla.POOL_ENV: "0"})
        self.assertTrue(armed)
        self.assertEqual(seen["grow"], [])
        self.assertEqual(seen["grow_small"], [])  # the same switch: neither pool
        self.assertNotIn("mem_get_info", w.cuda.calls)
        self.assertIn("pool=0.0GiB small=0MiB beside main", self.msgs(evla.ARMED)[0])
        evla._reset_for_tests()
        w = World()
        armed, seen = self._prepare(w, [False], **{evla.POOL_ENV: "2.5"})
        self.assertTrue(armed)
        self.assertEqual(seen["grow"], [(int(2.5 * _GIB), "cand1")])
        self.assertEqual(seen["grow_small"], [(evla.SMALL_POOL_MIB << 20, "cand1")])

    def test_allocator_retry_fails_closed(self) -> None:
        w = World()
        w.cuda.retry_on_grow = True
        armed, _ = self._prepare(w, [False])
        self.assertFalse(armed)
        self.assertTrue(self.msgs("allocator retry path", logging.ERROR))

    def test_small_pool_retry_or_failure_fails_closed(self) -> None:
        # V1 F2: the small-pool grow is fail-closed exactly like the large one
        for effect, why in (("retry", "pool pre-grow hit the allocator retry path"),
                            ("raise", "prepare raised RuntimeError: injected: small-pool grow")):
            with self.subTest(effect=effect):
                self.records.clear()
                evla._reset_for_tests()
                w = World()
                armed, seen = self._prepare(w, [False], small_effect=effect)
                self.assertFalse(armed)
                self.assertIsNone(evla.prepared_stream())
                self.assertEqual(len(seen["grow"]), 1)
                self.assertEqual(seen["vote"], [0])  # not ready: the vote declines every rank
                self.assertIn(why, self.msgs("DECLINED", logging.WARNING)[0])
                self.assertEqual(self.msgs(evla.ARMED), [])

    def test_small_pool_recipe_holds_every_block_then_frees_them(self) -> None:
        # alloc_pregrow.py's small branch: 1 MiB uint8 blocks on the device, all
        # live at once (each maps fresh pages), then freed back to the pool
        made: List[Any] = []

        class _Blk:
            live = 0
            peak = 0

            def __init__(self) -> None:
                _Blk.live += 1
                _Blk.peak = max(_Blk.peak, _Blk.live)

            def __del__(self) -> None:
                _Blk.live -= 1

        def fake_empty(*size: Any, dtype: Any = None, device: Any = None) -> Any:
            made.append((size, dtype, device))
            return _Blk()

        with mock.patch.object(torch, "empty", side_effect=fake_empty):
            evla._grow_small_on_current(evla.SMALL_POOL_MIB << 20)
        self.assertEqual(len(made), evla.SMALL_POOL_MIB)
        self.assertEqual(set(made), {((1 << 20,), torch.uint8, "cuda")})
        self.assertEqual(_Blk.peak, evla.SMALL_POOL_MIB)
        self.assertEqual(_Blk.live, 0)

    def test_priority_minus_one_also_checks_the_eval_pair(self) -> None:
        from generative_recommenders.dlrm_v4 import alloc_pregrow as ap

        w = World()
        e_m, e_d = FakeStream(w, "eval_memcpy", -1), FakeStream(w, "eval_dd", -1)
        t_m, t_d = FakeStream(w, "train_memcpy", -1), FakeStream(w, "train_dd", -1)
        with mock.patch.object(ap, "_side_streams", [{"memcpy": t_m, "data_dist": t_d},
                                                     {"memcpy": e_m, "data_dist": e_d}]), \
                mock.patch.object(ap, "_pair_roles", ["train", "eval"]):
            # cand1: beside main, but on eval_dd's queue; cand2: clear
            armed, seen = self._prepare(w, [False, False, True, False, False, False],
                                        **{evla.PRIO_ENV: "-1"})
        self.assertTrue(armed)
        self.assertEqual(w.cuda.calls.count(("Stream", -1)), 2)
        self.assertEqual(seen["probe"], [("main", "cand1"), ("eval_memcpy", "cand1"),
                                         ("eval_dd", "cand1"), ("main", "cand2"),
                                         ("eval_memcpy", "cand2"), ("eval_dd", "cand2")])
        self.assertIn("prio=-1 tries=2", self.msgs(evla.ARMED)[0])

    def test_env_reasons_make_no_stream_or_pool_but_still_vote(self) -> None:
        # TP_EVAL_PIPELINE=0: no eval pipeline will exist (V1 INFO); F3: the
        # build would decline. No stream, probe, pool or device call; the rank
        # still votes 0, so a mixed env across ranks cannot strand a peer in
        # the all_reduce (every rank then declines).
        for env, val, why in (("TP_EVAL_PIPELINE", "0",
                               "TP_EVAL_PIPELINE='0': eval runs the serial loop, no eval "
                               "pipeline"),
                              ("TP_DEFER_INPUT_DIST", "0",
                               "TP_DEFER_INPUT_DIST='0': without the deferred input dist")):
            with self.subTest(env=env):
                self.records.clear()
                evla._reset_for_tests()
                w = World()
                armed, seen = self._prepare(w, [False], votes=0, **{env: val})
                os.environ.pop(env)
                self.assertFalse(armed)
                self.assertIsNone(evla.prepared_stream())
                self.assertEqual(w.cuda.calls, [])
                self.assertEqual((seen["probe"], seen["grow"], seen["grow_small"]), ([], [], []))
                self.assertEqual(seen["vote"], [0])
                self.assertEqual(self.msgs(evla.ARMED), [])
                dec = self.msgs("DECLINED", logging.WARNING)
                self.assertEqual(len(dec), 1)
                self.assertRegex(dec[0], r"^DLRMV4 eval emb lookahead DECLINED \(%s.*\): stock "
                                         r"eval path rank=3$" % re.escape(why))
                self.assertIn(why, evla._prepared["reason"])

    def test_idempotent(self) -> None:
        w = World()
        armed, seen = self._prepare(w, [False])
        n = len(w.cuda.calls)
        again = evla.prepare_before_run_start(cuda=w.cuda, dist=FakeDist(),
                                              probe=mock.Mock(), grow=mock.Mock(),
                                              vote=mock.Mock())
        self.assertEqual((armed, again), (True, True))
        self.assertEqual(len(w.cuda.calls), n)


# ---------------------------------------------------------------------------
# build_train_pipeline(role=...): the real function, executed with fakes
# ---------------------------------------------------------------------------
class _FakeTPSD:
    _pipelined_forward_type = FakePipelinedForward

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self._memcpy_stream = None
        self._data_dist_stream = None
        self._model = kwargs.get("model")
        self.contexts: collections.deque = collections.deque()
        self.batches: collections.deque = collections.deque()
        self._pipelined_modules: List[Any] = []


def _load_build_train_pipeline() -> Tuple[Any, Dict[str, Any]]:
    with open(_UTILS) as f:
        src = f.read()
    tree = ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "build_train_pipeline")
    fn.decorator_list = []
    hsf = types.SimpleNamespace(
        dist_init_on_main=lambda: False, split_wait_poll_on=lambda: False,
        train_pipe_stream_priority=lambda: None, pipe_stream_priority=lambda: None,
        log_train_pipe_priority_inactive=lambda role: False,
    )
    ap = types.SimpleNamespace(adopt_streams=lambda p, role: False,
                               make_reporter=lambda p: None,
                               guard_train_pair=lambda *a: False)
    from generative_recommenders.dlrm_v4.train import dd_own_h2d

    class _Wrapper:
        def __init__(self, model: Any) -> None:
            self.model = model

    ns: Dict[str, Any] = {
        "torch": torch, "Optimizer": object, "Any": Any, "os": os, "time": __import__("time"),
        "logger": logging.getLogger("eval_lookahead_test.utils"),
        "_IDIST_GATE_ENV": "DLRM_HSTU_FP8_IDIST_AFTER_PREV_FWD",
        "_idist_gate_resolve": lambda raw, defer, *topo: (0, False, "off"),
        "_idist_gate_topology": lambda: (1, 8, None),
        "_dd_own_h2d": dd_own_h2d, "_hsf": hsf, "_alloc_pregrow": ap,
        "_dlstart": types.SimpleNamespace(ACTIVE=False), "_clip_grad_norm_": None,
        "_PipelineModelWrapper": _Wrapper,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), _UTILS, "exec"), ns)
    return ns["build_train_pipeline"], ns


class BuildTrainPipelineTest(_Base):
    _MODEL = types.SimpleNamespace()
    _OPT = object()

    def _build(self, role: str, knob: Optional[str]) -> Any:
        if knob is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = knob
        fn, _ = _load_build_train_pipeline()
        tp = types.ModuleType("torchrec.distributed.train_pipeline")
        tp.TrainPipelineSparseDist = _FakeTPSD
        rf = types.ModuleType("torchrec.distributed.train_pipeline.runtime_forwards")
        rf.PipelinedForward = FakePipelinedForward
        mods = {"torchrec": types.ModuleType("torchrec"),
                "torchrec.distributed": types.ModuleType("torchrec.distributed"),
                "torchrec.distributed.train_pipeline": tp,
                "torchrec.distributed.train_pipeline.runtime_forwards": rf}
        with mock.patch.dict(sys.modules, mods):
            World()
            return fn(self._MODEL, self._OPT, torch.device("cpu"), role=role)

    def test_train_role_never_wraps(self) -> None:
        for knob in (None, "0", "1"):
            with self.subTest(knob=knob):
                p = self._build("train", knob)
                cls = type(p)
                self.assertEqual(cls.__name__, "_StreamingTrainPipeline")
                self.assertIs(cls.__mro__[1], _FakeTPSD)
                self.assertFalse(hasattr(p, "dlrm_evla_configure"))
                self.assertFalse(hasattr(p, "_dlrm_evla"))
        self.assertEqual(self.msgs(), [])  # eval_lookahead never logged for train

    def test_eval_role_wraps_only_with_the_knob(self) -> None:
        base_kwargs = None
        for knob, wrapped in ((None, False), ("0", False), ("1", True)):
            with self.subTest(knob=knob):
                p = self._build("eval", knob)
                cls = type(p)
                if wrapped:
                    self.assertEqual(cls.__mro__[1].__name__, "_StreamingTrainPipeline")
                    self.assertTrue(hasattr(p, "dlrm_evla_configure"))
                    # no prepared stream in this harness: declined to the stock forward
                    self.assertIs(p._pipelined_forward_type, FakePipelinedForward)
                else:
                    self.assertEqual(cls.__name__, "_StreamingTrainPipeline")
                    self.assertFalse(hasattr(p, "dlrm_evla_configure"))
                kw = {k: v for k, v in p.init_kwargs.items() if k != "custom_model_fwd"}
                if base_kwargs is None:
                    base_kwargs = kw
                self.assertEqual(kw, base_kwargs)  # same kwargs on every path
                self.assertEqual(
                    sorted(p.init_kwargs),
                    ["custom_model_fwd", "device", "enqueue_batch_after_forward",
                     "execute_all_batches", "model", "optimizer"])

    def test_selection_is_outside_the_shared_class_body(self) -> None:
        with open(_UTILS) as f:
            tree = ast.parse(f.read())
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "build_train_pipeline")
        cls = next(n for n in fn.body
                   if isinstance(n, ast.ClassDef) and n.name == "_StreamingTrainPipeline")
        names = {getattr(n, "id", None) for n in ast.walk(cls)} | {
            getattr(n, "attr", None) for n in ast.walk(cls)}
        self.assertFalse({"_evla", "eval_lookahead", "wrap"} & names)
        # the only eval_lookahead use is one import under `if role == "eval"`
        ifs = [n for n in fn.body if isinstance(n, ast.If)
               and ast.unparse(n.test) == "role == 'eval'"]
        self.assertEqual(len(ifs), 1)
        imports = [n for n in ast.walk(fn) if isinstance(n, ast.ImportFrom)
                   and any(a.name == "eval_lookahead" for a in n.names)]
        self.assertEqual(len(imports), 1)
        self.assertIn(imports[0], list(ast.walk(ifs[0])))
        ret = fn.body[-1]
        self.assertIsInstance(ret, ast.Return)
        self.assertEqual(ast.unparse(ret.value.func), "pipeline_cls")


class HookPlacementTest(unittest.TestCase):
    def test_train_ranker_prepares_right_after_the_pregrow(self) -> None:
        with open(_RANKER) as f:
            tree = ast.parse(f.read())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_main_func")

        def line(name: str) -> List[int]:
            return [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
                    and getattr(n.func, "attr", None) == name]

        pre, grow = line("prepare_before_run_start"), line("pregrow_before_run_start")
        run_start = line("log_run_start")
        self.assertEqual((len(pre), len(grow), len(run_start)), (1, 1, 1))
        self.assertLess(grow[0], pre[0])
        self.assertLess(pre[0], run_start[0])
        between = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                   and grow[0] < n.lineno < pre[0]]
        self.assertEqual(between, [])  # nothing runs between the two

    def test_run_eval_window_configures_right_after_the_lazy_build(self) -> None:
        with open(_UTILS) as f:
            src = f.read()
        tree = ast.parse(src)
        loop = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "streaming_train_eval_loop")
        win = next(n for n in ast.walk(loop)
                   if isinstance(n, ast.FunctionDef) and n.name == "_run_eval_window")
        build_if = next(n for n in ast.walk(win) if isinstance(n, ast.If)
                        and ast.unparse(n.test) == "use_eval_pipeline and eval_pipeline is None")
        body = [ast.unparse(s) for s in build_if.body]
        self.assertTrue(body[0].startswith("eval_pipeline = build_train_pipeline("))
        self.assertIn("role='eval'", body[0])
        self.assertIn("getattr(eval_pipeline, 'dlrm_evla_configure', None)", body[2])
        self.assertIn("_evla_configure(num_eval_batches, bool(eval_interim_metrics))", body[3])


class EnvMirrorTest(unittest.TestCase):
    """The module mirrors two train/utils.py env reads (F3, the TP_EVAL_PIPELINE
    skip). Both sides must keep the same expression: utils.py's source is
    checked, and the module's reads are checked value by value."""

    _EXPR = {"TP_DEFER_INPUT_DIST": "os.environ.get('TP_DEFER_INPUT_DIST', '1') == '1'",
             "TP_EVAL_PIPELINE": "os.environ.get('TP_EVAL_PIPELINE', '1') == '1'"}

    def test_utils_reads_the_same_expressions(self) -> None:
        with open(_UTILS) as f:
            tree = ast.parse(f.read())
        exprs = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Compare)]
        for env, expr in self._EXPR.items():
            self.assertEqual(sum(1 for e in exprs if env in e), 1, env)
            self.assertIn(expr, exprs)

    def test_module_reads_match_value_by_value(self) -> None:
        for env, fn in (("TP_DEFER_INPUT_DIST", evla.defer_input_dist),
                        ("TP_EVAL_PIPELINE", evla.eval_pipeline_on)):
            for v in (None, "1", "0", "", " 1", "1 ", "true", "01"):
                with self.subTest(env=env, v=v), mock.patch.dict(os.environ, {}, clear=False):
                    os.environ.pop(env, None)
                    if v is not None:
                        os.environ[env] = v
                    self.assertEqual(fn(), os.environ.get(env, "1") == "1")
                    self.assertEqual(fn(), v is None or v == "1")


# ---------------------------------------------------------------------------
# gin binding (the knob rule: gin binding + env kill switch, env wins)
# ---------------------------------------------------------------------------
_HEAD = "# (85) EVAL EMBEDDING LOOKAHEAD "


def _gin_lines() -> List[str]:
    with open(_GIN) as f:
        return f.read().splitlines()


def _binding_index(lines: List[str]) -> int:
    return next(i for i, ln in enumerate(lines) if ln.startswith("apply_env_bootstrap.%s " % KNOB))


def _item_block(lines: List[str]) -> List[str]:
    i_bind = _binding_index(lines)
    i_head = max(i for i in range(i_bind) if lines[i].startswith(_HEAD))
    return lines[i_head:i_bind]


def _flat(block: List[str]) -> str:
    return " ".join(ln[1:].strip() for ln in block)


class GinBindingTest(unittest.TestCase):
    def test_exactly_one_binding_and_it_is_true(self) -> None:
        lines = _gin_lines()
        pat = re.compile(r"^\s*apply_env_bootstrap\.%s\s*=\s*(.*?)\s*$" % KNOB)
        self.assertEqual([m.group(1) for m in map(pat.match, lines) if m], ["True"])
        for diag in (evla.VERIFY_ENV, evla.ISSUE_ENV, evla.PRIO_ENV, evla.POOL_ENV,
                     evla.REPORT_ENV):
            self.assertEqual([ln for ln in lines if diag in ln and not ln.lstrip().startswith("#")],
                             [])
        names = [m.group(1) for m in (re.match(r"^\s*apply_env_bootstrap\.(\w+)\s*=", ln)
                                      for ln in lines) if m]
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])
        # placed right after the eval-forward hygiene block
        i = _binding_index(lines)
        j = max(k for k in range(i) if lines[k].startswith(
            "apply_env_bootstrap.DLRM_HSTU_FP8_EVAL_FWD_HYGIENE_LOSS"))
        self.assertTrue(all(ln.startswith("#") or not ln.strip() for ln in lines[j + 1:i]))

    def test_item_block_documents_the_knob_and_quotes_the_real_lines(self) -> None:
        block = _item_block(_gin_lines())
        self.assertTrue(all(ln.startswith("#") for ln in block))
        self.assertEqual([ln for ln in block if len(ln) > 80], [])
        text = _flat(block)
        for marker in ("SHIPS ON, the binding below is True", "WHAT:", "NUMERICS: BITWISE",
                       "ARMED (every rank):", "Must be ABSENT:", "Batch 0 of every pass runs "
                       "the stock path", "EVAL_INTERIM_LOG>0", "role \"train\" never takes it",
                       "Opt out (kill switch): %s=0" % KNOB, "_ISSUE=end|top", "_PRIO=0|-1",
                       "_POOL_GIB=<GiB> (default 4; 0 = neither pool pre-grown)", "_REPORT=1",
                       "%s_VERIFY=N" % KNOB):
            self.assertIn(marker, text)
        # fixes after V1: pool sizes match the code, the F1/F3 declines, the
        # VERIFY cost per sharding (INFO) and VERIFY's coverage (F4)
        for marker in ("(_POOL_GIB, default %d GiB, plus %d MiB of 1 MiB blocks)" % (
                           evla.DEFAULT_POOL_GIB, evla.SMALL_POOL_MIB),
                       "TP_DEFER_INPUT_DIST other than 1 (a rank's stock fallback mid-pass "
                       "would then pair different collectives with its peers)",
                       "only table_wise, column_wise and row_wise arm; data_parallel",
                       "world size 1 and torchrec sync collectives decline",
                       "TP_EVAL_PIPELINE=0 (no eval pipeline) makes no side stream or pool",
                       "+1 TBE forward and +1 sequence all_to_all per sharding per verified "
                       "batch on every rank",
                       "the side-stream dist_init (KJT all_to_all wait + recat) is not "
                       "independently checked by VERIFY",
                       "The end-to-end bitwise check is eval metrics equal across knob 0/1 "
                       "arms run from identical weights"):
            self.assertIn(marker, text)
        self.assertNotIn("+1 all_to_all per verified batch", text)
        self.assertEqual(sorted(evla.ORDERED_SHARDINGS), ["column_wise", "row_wise", "table_wise"])
        quoted = re.findall(r'"([^"]+)"', text)
        for q in ('armed OK: eval emb lookahead stream ready',
                  'DLRMV4 eval emb lookahead: eval pipeline subclass installed',
                  'DLRMV4 eval emb lookahead ACTIVE'):
            self.assertIn(q, quoted)
        armed_texts = (evla.ARMED, evla.INSTALLED, evla.ACTIVE)
        self.assertTrue(evla.ARMED.startswith(quoted[quoted.index(
            'armed OK: eval emb lookahead stream ready')]))
        self.assertTrue(any(t.startswith('DLRMV4 eval emb lookahead ACTIVE') for t in armed_texts))
        absent = ["eval emb lookahead DECLINED", "eval emb lookahead DISABLED",
                  "consumed-event fence missing", "failure after collectives", "drained",
                  "bitwise=False"]
        for a in absent:
            self.assertIn(a, quoted)
        produced = (evla.DECLINED, evla.DISABLED, evla.FENCE_MISSING, evla.FAILURE,
                    evla.DRAINED)
        for a in absent[:-1]:
            self.assertTrue(any(a in p for p in produced), a)

    def test_bootstrap_declares_and_binds_the_parameter_once(self) -> None:
        with open(_BOOT) as f:
            tree = ast.parse(f.read())
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "apply_env_bootstrap")
        names = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        self.assertEqual(names.count(KNOB), 1)
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])
        arg = next(a for a in fn.args.args if a.arg == KNOB)
        self.assertEqual(ast.unparse(arg.annotation), "Optional[bool]")
        defaults = dict(zip(reversed([a.arg for a in fn.args.args]), reversed(fn.args.defaults)))
        self.assertTrue(isinstance(defaults[KNOB], ast.Constant) and defaults[KNOB].value is None)
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and n.args
                 and isinstance(n.args[0], ast.Constant) and n.args[0].value == KNOB]
        self.assertEqual(len(calls), 1)
        self.assertEqual(getattr(calls[0].func, "id", None), "_bind_bool")
        self.assertEqual([getattr(a, "id", None) for a in calls[0].args[1:]], [KNOB])
        self.assertEqual(names.index(KNOB), names.index("DLRM_HSTU_FP8_DD_OWN_H2D") + 1)


_CHILD = textwrap.dedent(
    """
    import json, os, sys
    root, gin_path = sys.argv[1], sys.argv[2]
    import gin
    from generative_recommenders.dlrm_v4.train import _env_bootstrap as boot
    from generative_recommenders.dlrm_v4.train import eval_lookahead as evla
    assert os.path.abspath(boot.__file__).startswith(root), boot.__file__
    gin.parse_config_file(gin_path, skip_unknown=True)  # train_ranker Phase 1
    boot.apply_env_bootstrap()
    on = evla.enabled()
    out = {"value": os.environ.get(evla.ENV), "enabled": on}
    if not on:  # knob off: prepare returns at once, without even importing torch
        out["prepared"] = evla.prepare_before_run_start()
        out["torch_imported"] = "torch" in sys.modules
    print("EVLA_GIN_CHILD " + json.dumps(out, sort_keys=True))
    """
)


def _run_child(preset: Dict[str, str]) -> Tuple[int, str, Optional[Dict[str, Any]]]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(KNOB)}
    env.update(preset)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_ROOT, env.get("PYTHONPATH", "")) if p)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-B", "-c", _CHILD, _ROOT, _GIN], env=env, cwd=_ROOT,
                       capture_output=True, text=True, timeout=300)
    out = [x for x in r.stdout.splitlines() if x.startswith("EVLA_GIN_CHILD ")]
    got = json.loads(out[0][len("EVLA_GIN_CHILD "):]) if r.returncode == 0 and len(out) == 1 else None
    return r.returncode, r.stdout[-1500:] + r.stderr[-3000:], got


@unittest.skipIf(gin is None, "gin-config not installed")
class GinRunTest(unittest.TestCase):
    """Clean child interpreters run train_ranker's Phase-1 sequence
    (gin.parse_config_file(yambda_5b.gin, skip_unknown=True), then
    apply_env_bootstrap()): the parameter exists (else gin rejects the binding),
    gin binds "1", a pre-set env var wins either way."""

    def _child(self, preset: Dict[str, str]) -> Dict[str, Any]:
        rc, log, got = _run_child(preset)
        self.assertEqual(rc, 0, log)
        self.assertIsNotNone(got, log)
        assert got is not None
        return got

    def test_gin_binds_on(self) -> None:
        got = self._child({})
        self.assertEqual(got, {"value": "1", "enabled": True})

    def test_preset_zero_is_the_kill_switch(self) -> None:
        got = self._child({KNOB: "0"})
        self.assertEqual(got, {"value": "0", "enabled": False, "prepared": False,
                               "torch_imported": False})

    def test_preset_one_wins_over_the_gin_binding(self) -> None:
        got = self._child({KNOB: "1"})
        self.assertEqual(got, {"value": "1", "enabled": True})


if __name__ == "__main__":
    unittest.main()
