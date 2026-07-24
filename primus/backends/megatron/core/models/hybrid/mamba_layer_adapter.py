"""Adapter that lets upstream Megatron `MambaLayer` plug into Primus's
`HybridStack` (the one used by the GDN/KDA hybrids).

Why this exists
---------------
`HybridStack.__init__` builds each mamba layer with:

    build_module(
        submodules.mamba_layer,
        config=...,
        residual_in_fp32=...,
        layer_number=...,
        pg_collection=...,
    )

`GatedDeltaNetLayer.__init__` and `KimiDeltaAttentionLayer.__init__` both
accept the `residual_in_fp32` kwarg, so they slot in fine.  Upstream
`MambaLayer.__init__`, however, does NOT accept it — it has
`pp_layer_offset` in its place — so plugging `MambaLayer` directly into
`HybridStack` raises:

    TypeError: MambaLayer.__init__() got an unexpected keyword argument
    'residual_in_fp32'

We avoid touching upstream Megatron by wrapping `MambaLayer` in a thin
adapter that accepts `residual_in_fp32` (stored on `self` for parity with
the other hybrid layers), and never forwards it to `MambaLayer.__init__`.
`pp_layer_offset` defaults to 0 (HybridStack doesn't have pipeline parallel
plumbing for the mamba leg anyway — pp_offset is applied separately to the
TransformerLayer attention/MLP branches).
"""
from __future__ import annotations

from megatron.core.ssm.mamba_layer import MambaLayer


class Mamba2HybridLayer(MambaLayer):
    """`MambaLayer` that silently accepts `residual_in_fp32`.

    Inherits the full forward/init path of upstream `MambaLayer`; only the
    constructor is shimmed to filter the extra kwarg coming from `HybridStack`.
    """

    def __init__(
        self,
        *args,
        residual_in_fp32: bool = False,
        pp_layer_offset: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, pp_layer_offset=pp_layer_offset, **kwargs)
        # Persist for parity with GatedDeltaNetLayer/KimiDeltaAttentionLayer;
        # MambaLayer's own forward path manages residual dtype internally, so
        # nothing else needs to consume this flag at runtime.
        self.residual_in_fp32 = residual_in_fp32
