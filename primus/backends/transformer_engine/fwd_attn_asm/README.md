# `fwd_attn_asm` — hand-tuned gfx950 attention kernels

Vendored integration for the hand-tuned attention assembly kernels from
[`mawad-amd/fwd-attn-asm`](https://github.com/mawad-amd/fwd-attn-asm) and
[`mawad-amd/bwd-attn-asm`](https://github.com/mawad-amd/bwd-attn-asm).

This package was previously shipped inline in the MLPerf-training
Dockerfile and is now owned by Primus.

## Layout

```
fwd_attn_asm/
├── install.sh                       # build helper: clone + amdclang + .pth install
├── scripts/
│   ├── aiter_hd64_asm_override.py   # TE fused_attn_fwd runtime override
│   └── aiter_hd64_asm_override.pth  # site-packages activator (1-liner)
└── README.md                        # this file
```

## Use

From a Dockerfile:

```dockerfile
RUN bash /workspace/Primus/primus/backends/transformer_engine/fwd_attn_asm/install.sh
```

The script:

1. Clones `mawad-amd/fwd-attn-asm` and `mawad-amd/bwd-attn-asm` at pinned
   SHAs into `${DEPS_DIR}` (default `/workspace/deps`).
2. Assembles both `.co` binaries with `amdclang -mcpu=gfx950`.
3. (Optional, default on) Overwrites the aiter bwd `.co` shipped inside
   TransformerEngine's QoLA submodule
   (`3rdparty/QoLA/3rdparty/aiter/hsa/gfx950/fmha_v3_bwd/bwd_hd64_bf16_causal_a16_rtz.co`)
   with the hand-tuned binary, so QoLA bakes the hand-tuned kernel into
   `te_libmha_bwd.so` at TE pip-install time. Disable with
   `BWD_ATTN_ASM_ENABLE=0`. The original aiter binary is preserved as
   `*.aiter_orig` so a subsequent `cp .aiter_orig …` reverts without a
   rebuild. **Run this script before `pip install -e .` on TE.**
4. Installs the `.pth` shim and override module into `site-packages` so
   that the FMHA-fwd runtime hook auto-activates at every Python startup.

## Runtime gates

| Env var                          | Default | Effect |
|----------------------------------|---------|--------|
| `MLPERF_ENABLE_FWD_ATTN_ASM`     | `0`     | `1` activates the runtime FMHA-fwd hand-tuned dispatch path |
| `FMHA_HD64_ASM_CO`               | `/workspace/deps/fwd-attn-asm/build/fwd_d64_opt128.co` | path to the `.co` binary loaded via `hipModuleLoadData` |
| `FMHA_HD64_ASM_LOG`              | `0`     | `1` logs every dispatch decision (one line per call) |

## Eligibility (forward path)

The runtime hook routes a `fused_attn_fwd` call to the hand-tuned kernel
only when **all** of the following hold (otherwise CK runs unchanged):

* `q.dtype == k.dtype == v.dtype == bf16`
* `q.shape[-1] == 64` (head dim)
* `qkv_layout in ("bshd_bshd_bshd", "sbhd_sbhd_sbhd")`
* causal or SWA-causal mask, no dropout, no attn-bias / softmax-offset
* device GCN arch starts with `gfx950`

See `scripts/aiter_hd64_asm_override.py` for the full eligibility gate
and `_launch` for the kernel-arg packing.
