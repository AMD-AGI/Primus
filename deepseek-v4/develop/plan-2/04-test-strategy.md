# 04 — Plan-2 Test Strategy

> Test-pyramid for the rewritten DeepSeek-V4. Every phase from P13 onward
> contributes tests at the right level; P19 / P20 are the integration
> and release gates.

## Test Levels

| Level | Where it lives | Triggers |
|---|---|---|
| **L0 — unit** | `tests/backends/megatron/...` (CPU-only) | per-PR |
| **L1 — module integration** | `tests/integration/deepseek_v4/...` (CPU and 1-GPU) | per-PR (CPU); nightly (GPU) |
| **L2 — distributed smoke** | `examples/megatron/configs/MI355X/deepseek_v4_*.yaml` + `run_deepseek_v4*.sh` | per-merge |
| **L3 — release gate** | `tests/release_gates/deepseek_v4/...` | release branches only |

## Regression Matrix

| Gate | Description | Phase | Level | Pass criteria |
|---|---|---|---|---|
| **G1** | YAML schema parses for Base / Flash / Pro | P18 | L0 | All three yamls construct a `DeepSeekV4TransformerConfig` without runtime errors |
| **G2** | Attention forward agrees with HF reference | P13 | L1 | 1L toy, fp32 CPU; token-0 hidden ≤1e-3 abs; identical gradients to ≤1e-3 |
| **G3** | Activation pre-mul clamp matches HF | P14 | L0 | randomized inputs; max-abs error ≤1e-6 |
| **G4** | Routers (hash + learned) match HF | P14 | L0 | identical weights; (probs, indices) match exactly; gradient flows on `gate_linear` |
| **G5** | MoE forward agrees with HF | P14 | L1 | 1L toy, fp32 CPU; token-0 hidden ≤1e-3 abs |
| **G6** | TransformerLayer / TransformerBlock equivalence under PP | P15 | L1 | 4L toy; PP=1 vs PP=2 vs PP=4 token-0 hidden ≤1e-4 abs; loss curve over 50 iters ≤1e-4 |
| **G7** | MTP loss path | P16 | L1 | 4L toy + `mtp_num_layers=1`; both LM and MTP loss appear; `mtp_num_layers=0` ablation matches LM loss to 1e-6 |
| **G8** | State-dict round-trip | P17 | L0 | Random init Primus → adapter → HF dict → adapter → Primus; bit-exact |
| **G9** | V4-Flash safetensors load + numerical | P17 | L1 | Token-0 logits ≤1e-2 abs vs HF reference; max-abs ≤1e-1 in top-100 |
| **G10** | Distributed smoke matrix | P19 | L2 | 5 configurations reach `iteration 50` without hang; loss decreases monotonically |
| **G11** | Routing determinism across PP / EP | P19 | L2 | snapshot diff = 0 |
| **G12** | Short-run convergence | P20 | L3 | 200-step loss curve within ±0.05 of HF reference baseline |
| **G13** | TE on/off perf comparison | P20 | L3 | TFLOPS ratio + HBM delta reported |

## CPU-only Toy Configurations

Plan-2 commits to a CPU toy config for fast PR-time validation:

```yaml
# tests/integration/deepseek_v4/toy_4l.yaml
num_layers: 4
hidden_size: 128
num_attention_heads: 4
num_query_groups: 1
kv_channels: 32
qk_pos_emb_head_dim: 8
ffn_hidden_size: 256
moe_ffn_hidden_size: 256
moe_shared_expert_intermediate_size: 256
q_lora_rank: 64
o_lora_rank: 64
o_groups: 2
num_experts: 8
moe_router_topk: 2
num_hash_layers: 1
hc_mult: 2
hc_sinkhorn_iters: 5
compress_ratios: [0, 0, 4, 0]
attn_sliding_window: 8
attn_sink: true
swiglu_limit: 7.0
mtp_num_layers: 1
vocab_size: 1024
```

## Numerical-Alignment Harness

```python
# tests/integration/deepseek_v4/test_attention_alignment.py
def test_v4_attention_matches_hf_reference():
    config = build_toy_config(layers=1)
    hf_attn = ReferenceAttention(config)
    primus_attn = DeepseekV4Attention(config, ...)

    state = hf_attn.state_dict()
    adapter.apply_to_primus(primus_attn, state)

    h = torch.randn(2, 16, config.hidden_size, dtype=torch.float32)
    out_hf, _ = hf_attn(h, freqs_cis=hf_attn.build_freqs(16))
    out_primus = primus_attn(h, position_ids=torch.arange(16))

    assert (out_hf - out_primus).abs().max() < 1e-3
```

The same pattern applies to MoE / activation / hybrid layer / model.

## Distributed Smoke Configurations

```bash
# 1x8 BF16 baseline
TP=1 PP=1 EP=1 ./run_deepseek_v4.sh

# PP=2 EP=4
TP=1 PP=2 EP=4 ./run_deepseek_v4.sh

# TP=2 PP=2 EP=2
TP=2 PP=2 EP=2 ./run_deepseek_v4.sh

# PP=4 EP=2
TP=1 PP=4 EP=2 ./run_deepseek_v4.sh

# 2x8 multi-node
TP=2 PP=2 EP=2 DP=2 ./run_deepseek_v4_multinode.sh
```

Each config dumps a routing snapshot (after iter 1) for G11.

## Release Gates Workflow

1. P19 must be green (G10, G11) before kicking off P20.
2. P20's three reports (numerical / convergence / perf) attach to the
   release ticket.
3. Final gate review checklist:
   - G1–G9 passing on CI (last 7 days).
   - G10–G13 passing on the release branch.
   - No CRIT findings in `00-review-findings.md` left untracked.
   - State-dict adapter signed off by a second engineer.
   - HF reference fork pinned at a known commit; documented in
     `state_dict_adapter.py`.

## Ownership

| Test | Primary owner |
|---|---|
| G1 / YAML schema | configs maintainer |
| G2–G5 / module alignment | core attention + MoE owners |
| G6 / PP equivalence | distributed lead |
| G7 / MTP | MTP engineer |
| G8–G9 / checkpoint | adapter author |
| G10–G11 / smoke matrix | infra / training-ops |
| G12 / convergence | research / training-quality |
| G13 / perf | TE / Turbo lead |

Failures roll up to a single release-board issue; no merge-on-red.
