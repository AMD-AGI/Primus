###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the deferred loss check and SIGTERM poll.

WHY THIS TEST EXISTS:
  The patch moves two per-step host syncs and promises that nothing else
  changes: a loss explosion raises the same error before any checkpoint, and a
  SIGTERM is acted on one step later. The stand-ins run the pinned AutoModel's
  own poll, and the pinned step's loss check is compared with the stand-in's,
  so an upstream change that the patch no longer matches fails here. The
  deferred paths need a GPU: the loss goes through pinned memory and the
  SIGTERM flag through an NCCL all-reduce, which two GPUs check reaches every
  rank on the same step.
"""

import ast
import importlib.util
import inspect
import math
import operator
import socket
import sys
import textwrap
from datetime import timedelta
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402

from primus.backends.automodel.training import step_syncs  # noqa: E402

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# StepScheduler as far as the patch uses it; the pinned sigterm_received is appended.
SCHEDULER = textwrap.dedent(
    """
    import torch


    class StepScheduler:
        def __init__(self, sig_handler=None, ckpt_every_steps=2):
            self.step = 0
            self.ckpt_every_steps = ckpt_every_steps
            self.sig_handler = sig_handler
            self.sigterm_flag = False
            self._sig_polled_step = None

        @property
        def is_ckpt_step(self):
            return self.step % self.ckpt_every_steps == self.ckpt_every_steps - 1

    """
)


def _pipeline_cls():
    class FlowMatchingPipeline:
        """Returns the given losses in turn and checks them as the pinned step does."""

        def __init__(self, losses):
            self.losses = list(losses)
            self.checked = []

        def step(
            self,
            model,
            batch,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            global_step=0,
            collect_metrics=True,
            check_loss=True,
        ):
            average_weighted_loss = self.losses.pop(0)
            self.checked.append(check_loss)
            if check_loss and (torch.isnan(average_weighted_loss) or average_weighted_loss > 100):
                raise ValueError(f"Loss exploded: {average_weighted_loss.item()}")
            return average_weighted_loss * 2, average_weighted_loss, None, {}

    return FlowMatchingPipeline


class _Handler:
    """DistributedSignalHandler, as far as the two polls use it."""

    def __init__(self):
        self.group = None
        self._signal_received = False
        self.gathers = 0

    def signals_received(self):
        self.gathers += 1
        return [self._signal_received]


def _pinned_sources(pinned_automodel):
    return {
        "sigterm_received": pinned_automodel(
            "components/training/step_scheduler.py", "StepScheduler", "sigterm_received"
        ),
        "get_device": pinned_automodel("components/training/signal_handler.py", None, "get_device"),
    }


def _stand_ins(directory, sources, setitem):
    """Stand-ins for the AutoModel modules the patch imports, put in ``sys.modules`` with ``setitem``."""

    def load(name, source):
        path = directory / f"{name.rsplit('.', 1)[-1]}.py"
        path.write_text("from __future__ import annotations\n" + source)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    training = "nemo_automodel.components.training"
    scheduler = load(f"{training}.step_scheduler", SCHEDULER + sources["sigterm_received"])
    signal_handler = load(f"{training}.signal_handler", "import torch\n\n\n" + sources["get_device"])
    pipeline = SimpleNamespace(FlowMatchingPipeline=_pipeline_cls())
    recipe = SimpleNamespace(TrainDiffusionRecipe=type("TrainDiffusionRecipe", (), {}))
    setitem(sys.modules, training, SimpleNamespace(signal_handler=signal_handler))
    setitem(sys.modules, "nemo_automodel.components.flow_matching.pipeline", pipeline)
    setitem(sys.modules, "nemo_automodel.recipes.diffusion.train", recipe)
    return SimpleNamespace(
        FlowMatchingPipeline=pipeline.FlowMatchingPipeline, StepScheduler=scheduler.StepScheduler
    )


@pytest.fixture
def automodel(tmp_path, monkeypatch, pinned_automodel):
    """Fresh stand-ins for the AutoModel modules the patch imports."""
    return _stand_ins(tmp_path, _pinned_sources(pinned_automodel), monkeypatch.setitem)


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def nccl_group():
    """A one-rank NCCL process group on the current GPU."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    if dist.is_initialized():
        pytest.skip("a process group is already initialized")
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{_free_port()}", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


@GPU
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 100.5, 1e4, 1.0, 100.0, -3.0])
def test_a_loss_explosion_raises_the_upstream_error_before_the_checkpoint(automodel, value, dtype):
    loss = torch.tensor(value, dtype=dtype, device="cuda")
    try:
        automodel.FlowMatchingPipeline([loss]).step(None, {})
        want = None
    except ValueError as exc:
        want = str(exc)
    assert (want is not None) == (math.isnan(value) or value > 100)

    step_syncs.install()
    pipeline = automodel.FlowMatchingPipeline([loss])
    assert pipeline.step(None, {})[1] is loss
    assert pipeline.checked == [False]
    scheduler = automodel.StepScheduler()
    if want is None:
        assert scheduler.is_ckpt_step is False
    else:
        with pytest.raises(ValueError) as raised:
            scheduler.is_ckpt_step
        assert str(raised.value) == want


@GPU
def test_the_next_step_raises_before_it_runs(automodel):
    step_syncs.install()
    pipeline = automodel.FlowMatchingPipeline([torch.tensor(v, device="cuda") for v in (float("nan"), 1.0)])
    pipeline.step(None, {})
    with pytest.raises(ValueError, match="Loss exploded: nan"):
        pipeline.step(None, {})
    assert pipeline.checked == [False]


@GPU
def test_the_deferred_loss_check_does_not_synchronize(automodel):
    upstream = automodel.FlowMatchingPipeline.step
    loss = torch.tensor(1.0, device="cuda")
    step_syncs.install()

    torch.cuda.set_sync_debug_mode("error")
    try:
        automodel.FlowMatchingPipeline([loss]).step(None, {})
        with pytest.raises(RuntimeError, match="synchroniz"):
            upstream(automodel.FlowMatchingPipeline([loss]), None, {})
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert automodel.StepScheduler().is_ckpt_step is False


@pytest.mark.parametrize(
    "call",
    [
        lambda pipeline: pipeline.step(None, {}, check_loss=False),
        lambda pipeline: pipeline.step(None, {}, None, None, 0, True, False),
    ],
    ids=["keyword", "positional"],
)
def test_with_check_loss_off_the_upstream_step_runs_as_is(automodel, call):
    step_syncs.install()
    loss = torch.tensor(float("nan"))
    pipeline = automodel.FlowMatchingPipeline([loss])
    assert call(pipeline)[1] is loss
    assert pipeline.checked == [False]
    assert automodel.StepScheduler().is_ckpt_step is False


def test_a_sigterm_is_acted_on_one_step_later(automodel, nccl_group, monkeypatch):
    upstream = automodel.StepScheduler.sigterm_received
    reductions = []
    all_reduce = dist.all_reduce

    def counting_all_reduce(tensor, *args, **kwargs):
        reductions.append(kwargs.get("op"))
        return all_reduce(tensor, *args, **kwargs)

    monkeypatch.setattr(dist, "all_reduce", counting_all_reduce)
    step_syncs.install()

    patched, reference = automodel.StepScheduler(_Handler()), automodel.StepScheduler(_Handler())
    seen = []
    for step in range(4):
        for scheduler in (patched, reference):
            scheduler.step = step
            scheduler.sig_handler._signal_received = step >= 1
        # Each is read twice: the second read in a step must not poll again.
        seen.append(
            (
                patched.sigterm_received,
                patched.sigterm_received,
                upstream.fget(reference),
                upstream.fget(reference),
            )
        )

    assert seen == [
        (False, False, False, False),
        (False, False, True, True),
        (True, True, True, True),
        (True, True, True, True),
    ]
    assert reductions == [dist.ReduceOp.MAX] * 3
    assert patched.sig_handler.gathers == 0


def _poll_on_one_of_two_ranks(rank, port, directory, sources, results):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        automodel = _stand_ins(directory / f"rank{rank}", sources, operator.setitem)
        step_syncs.install()
        scheduler = automodel.StepScheduler(_Handler())
        seen = []
        for step in range(4):
            scheduler.step = step
            scheduler.sig_handler._signal_received = rank == 1 and step >= 1
            seen.append(scheduler.sigterm_received)
        results.put((rank, seen))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_both_ranks_act_on_a_sigterm_at_the_same_step(tmp_path, pinned_automodel):
    """Only rank 1 is signalled; a rank that stopped first would leave the other in a collective."""
    for rank in range(2):
        (tmp_path / f"rank{rank}").mkdir()
    results = torch.multiprocessing.get_context("spawn").SimpleQueue()
    args = (_free_port(), tmp_path, _pinned_sources(pinned_automodel), results)
    torch.multiprocessing.spawn(_poll_on_one_of_two_ranks, args=args, nprocs=2)
    assert dict(results.get() for _ in range(2)) == {rank: [False, False, True, True] for rank in range(2)}


def test_without_a_process_group_the_upstream_poll_runs(automodel):
    step_syncs.install()
    scheduler = automodel.StepScheduler(_Handler())
    scheduler.sig_handler._signal_received = True
    assert scheduler.sigterm_received
    assert scheduler.sig_handler.gathers == 1


def test_the_stand_in_step_checks_the_loss_as_the_pinned_step_does(pinned_automodel):
    """Fails when the pinned step's loss check, its bound or its arguments drift."""
    source = pinned_automodel("components/flow_matching/pipeline.py", "FlowMatchingPipeline", "step")
    step = ast.parse(textwrap.dedent(source)).body[0]

    assert [a.arg for a in step.args.args] == list(inspect.signature(_pipeline_cls().step).parameters)
    assert ast.unparse(step.args.defaults[-1]) == "True"
    checks = [n for n in ast.walk(step) if isinstance(n, ast.If) and "check_loss" in ast.unparse(n.test)]
    assert [ast.unparse(n.test) for n in checks] == [
        "check_loss and (torch.isnan(average_weighted_loss) or "
        f"average_weighted_loss > {step_syncs._LOSS_LIMIT:g})"
    ]
    raised = [ast.unparse(n.exc) for n in ast.walk(checks[0]) if isinstance(n, ast.Raise)]
    assert raised == ["ValueError(f'Loss exploded: {average_weighted_loss.item()}')"]
    returned = [
        n.value for n in ast.walk(step) if isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple)
    ]
    assert returned and all(ast.unparse(value.elts[1]) == "average_weighted_loss" for value in returned)


def _assigned_attributes(source):
    targets = []
    for node in ast.walk(ast.parse(textwrap.dedent(source))):
        if isinstance(node, ast.Assign):
            targets += node.targets
        elif isinstance(node, ast.AnnAssign):
            targets.append(node.target)
    return {target.attr for target in targets if isinstance(target, ast.Attribute)}


def test_the_scheduler_and_the_handler_still_have_what_the_patch_reads(pinned_automodel):
    path = "components/training/step_scheduler.py"
    for name in ("is_ckpt_step", "sigterm_received"):
        method = ast.parse(textwrap.dedent(pinned_automodel(path, "StepScheduler", name))).body[0]
        assert [ast.unparse(d) for d in method.decorator_list] == ["property"], name

    scheduler = _assigned_attributes(pinned_automodel(path, "StepScheduler", "__init__"))
    assert {"step", "sig_handler", "sigterm_flag", "_sig_polled_step"} <= scheduler
    handler = pinned_automodel(
        "components/training/signal_handler.py", "DistributedSignalHandler", "__init__"
    )
    assert {"group", "_signal_received"} <= _assigned_attributes(handler)


def test_install_is_idempotent(automodel):
    def installed():
        scheduler = automodel.StepScheduler
        return automodel.FlowMatchingPipeline.step, scheduler.is_ckpt_step, scheduler.sigterm_received

    assert step_syncs.install()
    first = installed()
    assert step_syncs.install()
    assert installed() == first


def test_the_patch_is_registered_and_gated(set_option):
    from primus.core.patches import PatchContext
    from primus.core.patches.patch_registry import PatchRegistry

    patch = PatchRegistry.get("automodel.training.defer_step_syncs")
    assert patch is not None and (patch.backend, patch.phase) == ("automodel", "before_train")
    ctx = PatchContext(backend="automodel", phase="before_train")
    assert not patch.condition(ctx)
    set_option("primus_train.defer_step_syncs", False)
    assert not patch.condition(ctx)
    set_option("primus_train.defer_step_syncs", True)
    assert patch.condition(ctx)
