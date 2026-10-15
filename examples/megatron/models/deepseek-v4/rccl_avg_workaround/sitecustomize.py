"""gfx1250 single-GPU bring-up workarounds, auto-imported in every Python
worker via sitecustomize (this dir is on PYTHONPATH). Two independent fixes:

1. RCCL AVG hang: torch.distributed.all_reduce(op=AVG) HANGS on this build for
   (at least) single-rank process groups, while SUM works fine (verified by
   collective microbench). Megatron's MoE aux-loss metric reduction
   (moe_utils.reduce_aux_losses_tracker_across_ranks) uses op=AVG and
   deadlocks. Replace AVG with SUM + divide-by-world-size, which is
   mathematically identical for any world size.

2. primus_turbo import shim: the MI355X production containers bundle the
   `primus_turbo` package; the gfx1250 therock container does not. Most Primus
   call-sites guard the import (try/except -> HAVE_TURBO=False), but the V4
   model path imports it unconditionally:
     deepseek_v4_layer_specs.py -> transformer_engine_spec_provider.py
       (DeepSeekV4SpecProvider subclasses PrimusTurboSpecProvider)
       -> extensions/primus_turbo.py -> `import primus_turbo.pytorch`
   and backends/megatron/core/utils.py -> primus_turbo...attention_utils.
   With every use_turbo_* flag False the turbo classes are never SELECTED
   (the spec provider returns the TE classes), so a pure import-shim is safe:
   install a meta-path finder that fabricates stub modules for primus_turbo.*
   whose attributes are auto-generated dummy classes. Attribute chains
   evaluated at class-definition time (e.g. the ScalingGranularity.TENSORWISE
   default arg in PrimusTurboQuantConfig) resolve fine; actually CALLING or
   instantiating any stub raises RuntimeError, so a misrouted turbo path fails
   loudly instead of computing garbage. The shim only installs when the real
   package is absent, so it can never shadow a real primus_turbo install.
"""

import sys


def _install_rccl_avg_workaround():
    import torch.distributed as dist

    orig_all_reduce = dist.all_reduce
    avg_op = dist.ReduceOp.AVG

    def all_reduce_avg_safe(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        is_avg = False
        try:
            is_avg = op == avg_op
        except Exception:
            is_avg = str(op) == str(avg_op)
        if is_avg:
            work = orig_all_reduce(tensor, op=dist.ReduceOp.SUM, group=group, async_op=async_op)
            try:
                ws = dist.get_world_size(group)
            except Exception:
                ws = 1
            if ws and ws > 1:
                # Enqueued on the same stream after the all_reduce, so ordering holds.
                tensor.div_(ws)
            return work
        return orig_all_reduce(tensor, op=op, group=group, async_op=async_op)

    dist.all_reduce = all_reduce_avg_safe
    print(
        "[rccl_avg_workaround] patched torch.distributed.all_reduce (AVG -> SUM/ws)",
        file=sys.stderr,
        flush=True,
    )


def _install_primus_turbo_stub():
    import importlib.abc
    import importlib.machinery
    import importlib.util
    import types

    # Never shadow a real install.
    if importlib.util.find_spec("primus_turbo") is not None:
        return

    class _StubMeta(type):
        """Dummy-class metaclass: any attribute access mints another dummy
        class (covers enum-style chains like ScalingGranularity.TENSORWISE)."""

        def __getattr__(cls, name):
            if name.startswith("__"):
                raise AttributeError(name)
            dummy = _make_dummy(f"{cls._stub_qual}.{name}")
            setattr(cls, name, dummy)
            return dummy

    def _make_dummy(qual):
        def _raise(self, *args, **kwargs):
            raise RuntimeError(
                f"primus_turbo stub: {qual} was invoked, but primus_turbo is NOT "
                "installed in this container. A turbo code path ran despite all "
                "use_turbo_*/enable_primus_turbo flags being False — fix the flags "
                "instead of installing primus_turbo."
            )

        return _StubMeta(qual.rsplit(".", 1)[-1], (), {"__init__": _raise, "_stub_qual": qual})

    class _StubModule(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            dummy = _make_dummy(f"{self.__name__}.{name}")
            setattr(self, name, dummy)
            return dummy

    class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "primus_turbo" or fullname.startswith("primus_turbo."):
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            return _StubModule(spec.name)

        def exec_module(self, module):
            module.__path__ = []
            module._primus_turbo_stub = True

    sys.meta_path.append(_Finder())
    print(
        "[primus_turbo_stub] primus_turbo not installed -> import shim active "
        "(turbo classes import as raising stubs; all use_turbo_* must stay False)",
        file=sys.stderr,
        flush=True,
    )


# Only patch the actual training worker. Importing torch here is heavy, so we
# must NOT do it for pip/offload-arch/build helper invocations (they stall and
# block setup). Gate on the training entrypoint appearing in argv.
def _is_training_worker():
    argv = " ".join(sys.argv)
    return ("primus/cli/main.py" in argv) or ("run_pretrain" in argv) or ("pretrain" in argv)


if _is_training_worker():
    try:
        _install_rccl_avg_workaround()
    except Exception as e:  # noqa: BLE001
        print(f"[rccl_avg_workaround] install FAILED: {e}", file=sys.stderr, flush=True)
    try:
        _install_primus_turbo_stub()
    except Exception as e:  # noqa: BLE001
        print(f"[primus_turbo_stub] install FAILED: {e}", file=sys.stderr, flush=True)
