def _make_triton_wrapper(original_fn):
    from triton import knobs as _triton_knobs

    def _chunk_state_bwd_db_no_buffer_ops(x, dt, dA_cumsum, dstates, seq_idx=None, B=None, ngroups=1):
        with _triton_knobs.amd.scope():
            _triton_knobs.amd.use_buffer_ops = False
            return original_fn(x, dt, dA_cumsum, dstates, seq_idx=seq_idx, B=B, ngroups=ngroups)

    return _chunk_state_bwd_db_no_buffer_ops
