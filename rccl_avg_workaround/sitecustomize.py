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
            is_avg = (op == avg_op)
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
    print("[rccl_avg_workaround] patched torch.distributed.all_reduce (AVG -> SUM/ws)",
          file=sys.stderr, flush=True)


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
    print("[primus_turbo_stub] primus_turbo not installed -> import shim active "
          "(turbo classes import as raising stubs; all use_turbo_* must stay False)",
          file=sys.stderr, flush=True)


def _install_flydsl_repairs():
    """FlyDSL cold-compile repairs (from mi450/gfx1250_smoke/flydsl_env.py),
    needed when the real primus_turbo is installed (it imports flydsl kernels at
    import time). No-op when flydsl is not importable (non-turbo runs)."""
    import importlib.util
    if importlib.util.find_spec("flydsl") is None:
        return
    import inspect

    def _clean_getfile(object):
        if inspect.ismodule(object):
            if getattr(object, "__file__", None):
                return object.__file__
            raise TypeError(f"{object!r} is a built-in module")
        if inspect.isclass(object):
            if hasattr(object, "__module__"):
                module = sys.modules.get(object.__module__)
                if getattr(module, "__file__", None):
                    return module.__file__
                if object.__module__ == "builtins":
                    raise OSError("source not available")
            raise TypeError(f"{object!r} is a built-in class")
        if inspect.ismethod(object):
            object = object.__func__
        if inspect.isfunction(object):
            object = object.__code__
        if inspect.istraceback(object):
            object = object.tb_frame
        if inspect.isframe(object):
            object = object.f_code
        if inspect.iscode(object):
            return object.co_filename
        raise TypeError(f"unsupported object {type(object).__name__}")

    inspect.getfile = _clean_getfile
    sys.setrecursionlimit(20000)
    try:
        import flydsl.compiler.jit_function as jf
        jf._collect_dependency_sources = lambda *a, **k: []
    except Exception as e:  # noqa: BLE001
        print(f"[flydsl_repairs] dep-walk stub skipped: {e}", file=sys.stderr, flush=True)
    print("[flydsl_repairs] applied (clean inspect.getfile + dep-walk stub + recursionlimit)",
          file=sys.stderr, flush=True)


# Only patch the actual training worker. Importing torch here is heavy, so we
# must NOT do it for pip/offload-arch/build helper invocations (they stall and
# block setup). Gate on the training entrypoint appearing in argv.
def _is_training_worker():
    argv = " ".join(sys.argv)
    return ("primus/cli/main.py" in argv) or ("run_pretrain" in argv) or ("pretrain" in argv)


if _is_training_worker():
    # RCCL AVG->SUM rewrite: needed on builds where all_reduce(op=AVG) hangs.
    # Believed fixed on rocm7.14; set PRIMUS_RCCL_AVG_WORKAROUND=0 to disable there.
    import os as _os
    if _os.environ.get("PRIMUS_RCCL_AVG_WORKAROUND", "1") != "0":
        try:
            _install_rccl_avg_workaround()
        except Exception as e:  # noqa: BLE001
            print(f"[rccl_avg_workaround] install FAILED: {e}", file=sys.stderr, flush=True)
    else:
        print("[rccl_avg_workaround] disabled via PRIMUS_RCCL_AVG_WORKAROUND=0", file=sys.stderr, flush=True)
    try:
        _install_primus_turbo_stub()
    except Exception as e:  # noqa: BLE001
        print(f"[primus_turbo_stub] install FAILED: {e}", file=sys.stderr, flush=True)
    try:
        _install_flydsl_repairs()
    except Exception as e:  # noqa: BLE001
        print(f"[flydsl_repairs] install FAILED: {e}", file=sys.stderr, flush=True)
