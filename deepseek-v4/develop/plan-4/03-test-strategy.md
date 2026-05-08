# 03 — Plan-4 Test Strategy

> Plan-4 reuses the test conventions from `../plan-3/03-test-strategy.md`.
> Each gate below maps to one phase from `01-roadmap.md` /
> `02-phase-details.md`. The release gate for plan-4 is the matrix of
> forward + backward equivalence tests at every V4-Flash + V4-Pro
> attention shape (G23 / G24 / G26 / G27); the smoke (P27) is the
> end-to-end sanity check.

## Gate matrix

| Gate | Phase | Type | What it checks | Where it lives |
|---|---|---|---|---|
| **G22** | P24 | runtime (CPU) | Eager-Python ops extracted into `v4_attention_kernels/reference.py` produce **bit-identical** output (or fp32-tolerance equivalent) to the pre-extraction `_attention_forward` / `_csa_forward` on the existing 1L V4 toy. Refactor-only safety net. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p24_reference_op.py` |
| **G23** | P25 | runtime (GPU) | Forward equivalence: `v4_attention(q, k, v, sink, swa_window, mask)` matches `eager_v4_attention(...)` at every (variant ∈ {flash, pro}) × (compress_ratio ∈ {0, 128}) × (S ∈ {small, medium}) × (dtype ∈ {fp32, bf16}) × (sink ∈ {on, off}) shape, within the dtype tolerance budget (`fp32 → atol=1e-5`, `bf16 → atol=2e-2`). MQA layout (`K.shape[1] == 1`) covered explicitly. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p25_v4_attention_fwd.py` |
| **G24** | P25 | runtime (GPU) | Backward equivalence: `dq, dk, dv, dsink` from the kernel match autograd-on-eager within the dtype tolerance budget (`fp32 → atol=1e-5`, `bf16 → atol=5e-2`) at every G23 shape. Sink gradient explicitly asserted on a per-head basis. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p25_v4_attention_bwd.py` |
| **G25** | P25 | runtime (GPU) | Determinism: with `attn_dropout == 0.0`, two back-to-back forward calls with the same inputs return bit-identical outputs (fp32) / `atol=1e-5` outputs (bf16, since MQA atomic-add may reorder). | `test_v4_p25_v4_attention_fwd.py::test_determinism` |
| **G26** | P26 | runtime (GPU) | Forward equivalence: `v4_csa_attention(q, k_local, v_local, gathered, sink, swa_window, sparse_mask)` matches `eager_v4_csa_attention(...)` at every (variant ∈ {flash, pro}) × (compress_ratio == 4) × (S ∈ {small, medium}) × (dtype ∈ {fp32, bf16}) × (sink ∈ {on, off}) shape. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p26_v4_csa_attention_fwd.py` |
| **G27** | P26 | runtime (GPU) | Backward equivalence: `dq, dk_local, dv_local, dgathered, dsink` from the kernel match autograd-on-eager within the dtype tolerance budget at every G26 shape. The `dgathered → dpool` scatter-add (done in the wrapper / caller, not the kernel) is included end-to-end so the test exercises the full CSA backward path. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p26_v4_csa_attention_bwd.py` |
| **G28** | P27 | static + runtime | Dispatch precedence: with `use_turbo_attention=True use_v4_triton_attention=True` the dense path uses Turbo (precedence honored); with `use_turbo_attention=False use_v4_triton_attention=True` it uses `v4_attention`; with both off it uses eager. CSA path: `use_v4_triton_csa_attention=True` uses `v4_csa_attention`; off → eager. The startup log line is asserted to fire exactly once per layer kind. | `tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p27_dispatch_precedence.py` |
| **G29** | P27 | smoke | TP=1 PP=1 EP=8 10-iter smoke with `USE_V4_TRITON_ATTENTION=True USE_V4_TRITON_CSA_ATTENTION=True`. Loss curve stable; no NaN / Inf; no banned warnings. | `deepseek-v4/develop/progress/p27/run_smoke_v4_kernels_ep8_pp1.sh` |

## Banned-warning ratchet

Plan-4 inherits the plan-3 ratchet (no `"submodule init failed"` /
`"fallback to nn.Linear"` / `"unsupported dispatcher module"`) and adds:

1. `"v4_attention SMEM exceeded"` — Triton kernels MUST size their tiles
   to fit the MI355 160 KiB SMEM budget. If the kernel ever raises an
   `out of resource: shared memory` error during the smoke, the run
   fails. Mitigation: P25 / P26 land with conservative
   `BLOCK_M = BLOCK_N = 64` tiles validated under SMEM at compile time.
2. `"v4_attention NaN observed"` / `"v4_csa_attention NaN observed"` —
   the wrapper asserts no NaN in the forward output before returning;
   any NaN fails the run loud.

## GPU-toy harness

G23 / G24 / G25 / G26 / G27 require GPU (Triton kernels). The harness:

- Uses small / medium tier shapes from `v4_attention_shapes.py` so each
  gate runs in <60 s on a single MI355.
- Marks large-tier shapes (`S=4096, full V4-Flash dims`) with
  `pytest.mark.slow` and `pytest.mark.gpu` so they only run when
  explicitly requested (e.g., release-prep).
- Skips on machines without CUDA / Triton (`pytest.importorskip("triton")`)
  so plan-4 unit tests do not break CPU-only CI.

## Reporting hand-off

Plan-4 closes with the P27 entry in `../progress/status.md` plus the
local smoke log captured under `../progress/p27/` (gitignored, per the
plan-3 directive). The plan-4 hand-off note records:

- Commit SHAs for P24 / P25 / P26 / P27.
- The smoke's iter / TFLOP/s / ms-per-iter numbers vs. the eager-Python
  baseline at the same TP / PP / EP / SEQ.
- Any follow-up perf work surfaced (e.g., HCA LSE-merge variant,
  CSA in-kernel `topk_idxs` gather, FP8 quantisation hooks).
