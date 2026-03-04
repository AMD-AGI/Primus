from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from primus.moe_umco.types import MoEStepPlan

logger = logging.getLogger("primus.moe_umco")


@dataclass(frozen=True)
class DispatchResult:
    hidden_states: Any
    tokens_per_expert: Any


class MoEDispatcher(Protocol):
    def dispatch(self, hidden_states: Any, routing: Any, probs: Any, **kwargs: Any) -> DispatchResult: ...

    def gather(self, expert_out: Any, routing: Any, **kwargs: Any) -> Any: ...


class BaselineMegatronDispatcher:
    def __init__(
        self,
        dispatch_impl: Callable[..., Any],
        gather_impl: Callable[..., Any],
        dispatch_fn_ref: Callable[..., Any] | None = None,
        gather_fn_ref: Callable[..., Any] | None = None,
    ) -> None:
        self._dispatch_impl = dispatch_impl
        self._gather_impl = gather_impl
        self._dispatch_fn_ref = dispatch_fn_ref
        self._gather_fn_ref = gather_fn_ref

    def dispatch(self, hidden_states: Any, routing: Any, probs: Any, **kwargs: Any) -> DispatchResult:
        dispatched, tokens_per_expert = self._dispatch_impl(hidden_states, probs, routing, **kwargs)
        return DispatchResult(hidden_states=dispatched, tokens_per_expert=tokens_per_expert)

    def gather(self, expert_out: Any, routing: Any, **kwargs: Any) -> Any:
        output, _ = self._gather_impl(expert_out, **kwargs)
        return output


class UmcoDispatcher:
    def __init__(self, baseline: BaselineMegatronDispatcher, plan: MoEStepPlan) -> None:
        self._baseline = baseline
        self._plan = plan
        self._verify = _env_enabled("PRIMUS_UMCO_VERIFY")

    def dispatch(self, hidden_states: Any, routing: Any, probs: Any, **kwargs: Any) -> DispatchResult:
        expert_compute_fn = kwargs.pop("expert_compute_fn", None)
        baseline_ref: DispatchResult | None = None
        if self._verify and _small_tensor(hidden_states):
            baseline_ref = self._baseline.dispatch(
                hidden_states=hidden_states, routing=routing, probs=probs, **kwargs
            )

        run_chunked = self._baseline._dispatch_fn_ref is not None
        if not run_chunked:
            return self._baseline.dispatch(
                hidden_states=hidden_states, routing=routing, probs=probs, **kwargs
            )

        with _patched_all_to_all(
            fn_ref=self._baseline._dispatch_fn_ref,
            chunk_tokens=max(1, self._plan.chunk_tokens),
            max_inflight=max(1, self._plan.max_inflight),
            phase="dispatch",
            expert_compute_fn=expert_compute_fn,
        ):
            result = self._baseline.dispatch(
                hidden_states=hidden_states, routing=routing, probs=probs, **kwargs
            )

        logger.debug(
            "UMCO dispatch using plan: chunk_tokens=%s num_chunks=%s max_inflight=%s",
            self._plan.chunk_tokens,
            self._plan.num_chunks,
            self._plan.max_inflight,
        )
        if baseline_ref is not None:
            _compare_dispatch_results(baseline_ref, result)
        return result

    def gather(self, expert_out: Any, routing: Any, **kwargs: Any) -> Any:
        expert_compute_fn = kwargs.pop("expert_compute_fn", None)
        baseline_ref: Any = None
        if self._verify and _small_tensor(expert_out):
            baseline_ref = self._baseline.gather(expert_out=expert_out, routing=routing, **kwargs)

        run_chunked = self._baseline._gather_fn_ref is not None
        if not run_chunked:
            return self._baseline.gather(expert_out=expert_out, routing=routing, **kwargs)
        with _patched_all_to_all(
            fn_ref=self._baseline._gather_fn_ref,
            chunk_tokens=max(1, self._plan.chunk_tokens),
            max_inflight=max(1, self._plan.max_inflight),
            phase="gather",
            expert_compute_fn=expert_compute_fn,
        ):
            output = self._baseline.gather(expert_out=expert_out, routing=routing, **kwargs)
        if baseline_ref is not None:
            _compare_tensors("gather.output", baseline_ref, output)
        return output


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _small_tensor(value: Any, limit: int = 8192) -> bool:
    return hasattr(value, "numel") and int(value.numel()) <= limit


def _compare_dispatch_results(ref: DispatchResult, got: DispatchResult) -> None:
    _compare_tensors("dispatch.hidden_states", ref.hidden_states, got.hidden_states)
    if hasattr(ref.tokens_per_expert, "__iter__") and hasattr(got.tokens_per_expert, "__iter__"):
        if list(ref.tokens_per_expert) != list(got.tokens_per_expert):
            logger.warning("UMCO verify mismatch: tokens_per_expert differs.")


def _compare_tensors(name: str, ref: Any, got: Any, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    if ref is None or got is None:
        return
    try:
        import torch
    except Exception:
        return
    if not (torch.is_tensor(ref) and torch.is_tensor(got)):
        return
    if ref.shape != got.shape:
        logger.warning("UMCO verify mismatch: %s shape %s != %s", name, tuple(ref.shape), tuple(got.shape))
        return
    atol, rtol = _dtype_tolerance(ref, atol=atol, rtol=rtol)
    close = torch.allclose(ref, got, atol=atol, rtol=rtol)
    if not close:
        logger.warning("UMCO verify mismatch: %s values differ", name)


def _dtype_tolerance(tensor: Any, atol: float, rtol: float) -> tuple[float, float]:
    dtype = getattr(tensor, "dtype", None)
    name = str(dtype)
    if "bfloat16" in name:
        return 5e-2, 5e-2
    if "float16" in name or "half" in name:
        return 1e-2, 1e-2
    return atol, rtol


@contextlib.contextmanager
def _patched_all_to_all(
    fn_ref: Callable[..., Any],
    chunk_tokens: int,
    max_inflight: int,
    phase: str,
    expert_compute_fn: Callable[[Any], Any] | None,
):
    original = fn_ref.__globals__.get("all_to_all")
    if original is None:
        yield
        return

    def _chunked_all_to_all(
        group: Any, input_tensor: Any, output_splits: Any = None, input_splits: Any = None
    ):
        return _all_to_all_single_chunked(
            group=group,
            input_tensor=input_tensor,
            output_splits=output_splits,
            input_splits=input_splits,
            chunk_tokens=chunk_tokens,
            max_inflight=max_inflight,
            phase=phase,
            expert_compute_fn=expert_compute_fn,
        )

    fn_ref.__globals__["all_to_all"] = _chunked_all_to_all
    try:
        yield
    finally:
        fn_ref.__globals__["all_to_all"] = original


def _all_to_all_single_chunked(
    group: Any,
    input_tensor: Any,
    output_splits: Any,
    input_splits: Any,
    chunk_tokens: int,
    max_inflight: int,
    phase: str,
    expert_compute_fn: Callable[[Any], Any] | None,
) -> Any:
    try:
        import torch
        import torch.distributed as dist
    except Exception:
        return input_tensor

    if not (dist.is_available() and dist.is_initialized()):
        return input_tensor
    world_size = int(dist.get_world_size(group=group))
    if world_size <= 1:
        return input_tensor

    if input_splits is None or output_splits is None:
        out = torch.empty_like(input_tensor)
        dist.all_to_all_single(out, input_tensor, group=group)
        return out

    in_splits = [int(v) for v in input_splits]
    out_splits = [int(v) for v in output_splits]
    if len(in_splits) != world_size or len(out_splits) != world_size:
        out = torch.empty_like(input_tensor)
        dist.all_to_all_single(
            out,
            input_tensor,
            output_split_sizes=out_splits,
            input_split_sizes=in_splits,
            group=group,
        )
        return out

    out_tokens = int(sum(out_splits))
    out_shape = [out_tokens] + list(input_tensor.shape[1:])
    output_tensor = torch.empty(out_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    in_offsets = _prefix_sums(in_splits)
    out_offsets = _prefix_sums(out_splits)
    in_done = [0 for _ in in_splits]
    out_done = [0 for _ in out_splits]

    use_cuda_streams = input_tensor.is_cuda and torch.cuda.is_available()
    comm_stream = torch.cuda.Stream(device=input_tensor.device) if use_cuda_streams else None
    compute_stream = torch.cuda.Stream(device=input_tensor.device) if use_cuda_streams else None
    comm_done_event = torch.cuda.Event(enable_timing=False) if use_cuda_streams else None
    inflight: list[Any] = []

    while any(in_done[idx] < in_splits[idx] for idx in range(world_size)):
        for _ in range(max(1, max_inflight)):
            if not any(in_done[idx] < in_splits[idx] for idx in range(world_size)):
                break
            chunk_in, chunk_out = _next_chunk_sizes(in_splits, out_splits, in_done, out_done, chunk_tokens)
            if sum(chunk_in) == 0 and sum(chunk_out) == 0:
                break
            in_chunk = _pack_chunk(input_tensor, in_offsets, in_done, chunk_in)
            out_chunk_shape = [int(sum(chunk_out))] + list(input_tensor.shape[1:])
            out_chunk = torch.empty(out_chunk_shape, dtype=input_tensor.dtype, device=input_tensor.device)

            if use_cuda_streams and comm_stream is not None:
                with torch.cuda.stream(comm_stream):
                    work = dist.all_to_all_single(
                        out_chunk,
                        in_chunk,
                        output_split_sizes=chunk_out,
                        input_split_sizes=chunk_in,
                        group=group,
                        async_op=True,
                    )
            else:
                work = dist.all_to_all_single(
                    out_chunk,
                    in_chunk,
                    output_split_sizes=chunk_out,
                    input_split_sizes=chunk_in,
                    group=group,
                    async_op=False,
                )
            inflight.append((work, out_chunk, chunk_out, list(out_done)))

            for idx in range(world_size):
                in_done[idx] += chunk_in[idx]
                out_done[idx] += chunk_out[idx]

        # Drain one op to keep inflight bounded and allow overlap on CUDA streams.
        work, out_chunk, chunk_out, out_cursor = inflight.pop(0)
        if work is not None:
            work.wait()
        if use_cuda_streams and compute_stream is not None:
            assert comm_stream is not None
            assert comm_done_event is not None
            with torch.cuda.stream(comm_stream):
                comm_done_event.record(comm_stream)
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(comm_done_event)
                if expert_compute_fn is not None:
                    expert_compute_fn(out_chunk)
                _unpack_chunk(output_tensor, out_chunk, out_offsets, out_cursor, chunk_out)
        else:
            if expert_compute_fn is not None:
                expert_compute_fn(out_chunk)
            _unpack_chunk(output_tensor, out_chunk, out_offsets, out_cursor, chunk_out)

    for work, out_chunk, chunk_out, out_cursor in inflight:
        if work is not None:
            work.wait()
        if expert_compute_fn is not None:
            expert_compute_fn(out_chunk)
        _unpack_chunk(output_tensor, out_chunk, out_offsets, out_cursor, chunk_out)

    if use_cuda_streams and comm_stream is not None:
        torch.cuda.current_stream(device=input_tensor.device).wait_stream(comm_stream)
    if use_cuda_streams and compute_stream is not None:
        torch.cuda.current_stream(device=input_tensor.device).wait_stream(compute_stream)

    logger.debug("UMCO %s chunked all_to_all_single complete with chunk_tokens=%s", phase, chunk_tokens)
    return output_tensor


def _prefix_sums(values: list[int]) -> list[int]:
    out = [0]
    for value in values[:-1]:
        out.append(out[-1] + int(value))
    return out


def _next_chunk_sizes(
    in_splits: list[int],
    out_splits: list[int],
    in_done: list[int],
    out_done: list[int],
    chunk_tokens: int,
) -> tuple[list[int], list[int]]:
    chunk_in: list[int] = []
    chunk_out: list[int] = []
    for idx in range(len(in_splits)):
        chunk_in.append(min(max(0, in_splits[idx] - in_done[idx]), chunk_tokens))
        chunk_out.append(min(max(0, out_splits[idx] - out_done[idx]), chunk_tokens))
    return chunk_in, chunk_out


def _pack_chunk(input_tensor: Any, offsets: list[int], done: list[int], sizes: list[int]) -> Any:
    try:
        import torch
    except Exception:
        return input_tensor
    parts = []
    for idx, size in enumerate(sizes):
        if size <= 0:
            continue
        begin = offsets[idx] + done[idx]
        end = begin + size
        parts.append(input_tensor[begin:end])
    if not parts:
        shape = [0] + list(input_tensor.shape[1:])
        return torch.empty(shape, dtype=input_tensor.dtype, device=input_tensor.device)
    return torch.cat(parts, dim=0)


def _unpack_chunk(
    output_tensor: Any,
    chunk_tensor: Any,
    offsets: list[int],
    done_before_chunk: list[int],
    sizes: list[int],
) -> None:
    cursor = 0
    for idx, size in enumerate(sizes):
        if size <= 0:
            continue
        out_begin = offsets[idx] + done_before_chunk[idx]
        out_end = out_begin + size
        output_tensor[out_begin:out_end].copy_(chunk_tensor[cursor : cursor + size])
        cursor += size
