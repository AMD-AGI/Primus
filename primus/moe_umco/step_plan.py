from __future__ import annotations

from primus.moe_umco.types import MoEBufferLayout, MoEChunkSpec


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def compute_chunking(max_tokens: int, chunk_tokens: int) -> list[MoEChunkSpec]:
    if max_tokens <= 0:
        return [MoEChunkSpec(chunk_id=0, token_begin=0, token_end=0, tokens=0)]
    effective_chunk = max(1, min(chunk_tokens, max_tokens))
    chunks: list[MoEChunkSpec] = []
    begin = 0
    idx = 0
    while begin < max_tokens:
        end = min(begin + effective_chunk, max_tokens)
        chunks.append(
            MoEChunkSpec(
                chunk_id=idx,
                token_begin=begin,
                token_end=end,
                tokens=end - begin,
            )
        )
        begin = end
        idx += 1
    return chunks


def compute_buffer_layout(
    max_tokens: int,
    dtype_bytes: int,
    ep_size: int,
    alignment: int = 128,
    chunk_tokens: int = 0,
) -> MoEBufferLayout:
    tokens = max(max_tokens, 1)
    dbytes = max(dtype_bytes, 1)
    eps = max(ep_size, 1)

    dispatch_in = _align_up(tokens * dbytes, alignment)
    dispatch_out = _align_up(tokens * eps * dbytes, alignment)
    gather_in = _align_up(tokens * eps * dbytes, alignment)
    gather_out = _align_up(tokens * dbytes, alignment)
    return MoEBufferLayout(
        dispatch_in_bytes=dispatch_in,
        dispatch_out_bytes=dispatch_out,
        gather_in_bytes=gather_in,
        gather_out_bytes=gather_out,
        alignment_bytes=alignment,
        chunk_tokens=max(chunk_tokens, 0),
    )
