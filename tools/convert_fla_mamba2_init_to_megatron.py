#!/usr/bin/env python3
"""
Initialize a Megatron (Primus) checkpoint from FLA's Mamba2 *hybrid* model weights.

This is the mamba2 analogue of tools/convert_fla_gdn_init_to_megatron.py and
tools/convert_fla_kda_init_to_megatron.py.  It exists to answer ONE question:

    Is the ~+8% loss offset of the 300M mamba_hybrid vs FLA's mamba2 reference
    caused by initialization (no FLA-init converter existed for mamba) or by a
    real bug in Megatron's Mamba2 mixer?

If, after loading these FLA-exact weights + FLA token order, the Megatron loss
lands on FLA's curve (like the pure GDN/KDA models do) -> the offset was 100%
initialization, mixer is fine.  If it still diverges with identical weights ->
there is a genuine numerical/forward bug to chase.

IMPORTANT: this targets the FLA-EXACT architecture replica
(zebra_llama_300M_mamba_hybrid_flaexact.yaml), NOT the production GDN-matched
mamba_hybrid (whose dims differ from FLA, so weights cannot map 1:1).

Usage (inside the rocm/primus container, where `fla` is importable):
    python tools/convert_fla_mamba2_init_to_megatron.py \
        --fla-config /home/<user>/flash-linear-attention/legacy/training/configs/mamba2_300M_hybrid.json \
        --output-dir output/fla_init_mamba2_flaexact \
        --seed 42

Then point the training config at it:
    load: output/fla_init_mamba2_flaexact
    finetune: true
    no_load_optim: true
    no_load_rng: true

--------------------------------------------------------------------------------
Weight mapping (FLA hybrid -> Megatron no-TE HybridStack, pattern *-MMM*-MMM*-MMM)
--------------------------------------------------------------------------------
FLA block layout (12 blocks; attn at config.attn.layers = [0,4,8]):
  attn block i : backbone.layers.{i}.mixer_norm.weight
                 backbone.layers.{i}.mixer.q_proj.{0,1,2}.weight   (down, RMSNorm, up)
                 backbone.layers.{i}.mixer.k_rope.weight
                 backbone.layers.{i}.mixer.kv_proj.{0,1,2}.weight  (down, RMSNorm, up)
                 backbone.layers.{i}.mixer.o_proj.weight
                 backbone.layers.{i}.mlp_norm.weight
                 backbone.layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight
  mamba block i: backbone.layers.{i}.mixer_norm.weight
                 backbone.layers.{i}.mixer.{in_proj,conv1d,A_log,D,dt_bias,norm,out_proj}

Megatron sublayer types follow the hybrid_override_pattern; an attn FLA block
expands into TWO Megatron sublayers (MLA layer + MLP layer), a mamba FLA block
into ONE Megatron mamba sublayer.

MLA fusion notes (verified against megatron/core/transformer/multi_latent_attention.py):
  * linear_kv_down_proj fuses [kv_down ; k_rope]; Megatron splits it as
    [kv_lora_rank, qk_pos_emb_head_dim] -> cat FLA kv_proj.0 then k_rope.
  * in_proj/q_up/kv_up per-head layout is [nope, rope] / [nope_k, v] contiguous,
    matching FLA's rearrange('(h d)') + split -> direct copy.
  * Mamba2 in_proj order is [z, x, B, C, dt] in BOTH frameworks -> direct copy.

CAVEAT: weights map 1:1, but RoPE *application* convention in Megatron's MLA vs
FLA's RotaryEmbedding could still differ.  Validate with the iter-1 loss: if it
matches FLA's iter-1 loss closely, the whole mapping (incl. MLA) is correct.
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch

_primus_root = Path(__file__).resolve().parents[1]
_fla_root = _primus_root.parent / "flash-linear-attention"
if str(_fla_root) not in sys.path:
    sys.path.insert(0, str(_fla_root))


def init_fla_model(config_path, seed=42):
    """Initialize an FLA Mamba2 (hybrid) model and return its state_dict + config."""
    torch.manual_seed(seed)

    from fla.models.mamba2 import Mamba2Config
    from fla.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM

    with open(config_path) as f:
        cfg_dict = json.load(f)

    config = Mamba2Config(**cfg_dict)
    # Raw weights only — fusions are toggled at train time via PRIMUS_FLA_* knobs.
    config.fuse_cross_entropy = False
    config.fuse_norm = False
    config.fuse_swiglu = False
    config.fuse_linear_cross_entropy = False

    model = Mamba2ForCausalLM(config)
    model.eval()

    n = sum(p.numel() for p in model.parameters())
    print(f"FLA mamba2 hybrid initialized: {n:,} params (seed={seed})")
    return model.state_dict(), config


def _attn_layer_indices(config):
    attn = getattr(config, "attn", None) or {}
    return set(attn.get("layers", []))


def convert_fla_to_megatron(fla, config):
    """Map FLA hybrid state dict -> Megatron no-TE HybridStack state dict.

    The Megatron sublayer index advances by 1 per mamba block and by 2 per attn
    block (MLA sublayer + MLP sublayer), reproducing the *-MMM*-MMM*-MMM layout.
    """
    n_blocks = config.num_hidden_layers
    attn_layers = _attn_layer_indices(config)
    kv_lora_rank = config.attn["kv_lora_rank"]
    qk_rope = config.attn["qk_rope_head_dim"]

    print(f"\nConverting FLA mamba2 hybrid -> Megatron:")
    print(f"  blocks={n_blocks}  attn_layers={sorted(attn_layers)}")
    print(f"  hidden={config.hidden_size}  state={config.state_size}  "
          f"n_groups={config.n_groups}  heads(mamba)={config.num_heads if hasattr(config,'num_heads') else 'n/a'}")
    print(f"  kv_lora_rank={kv_lora_rank}  qk_rope_head_dim={qk_rope}")

    mg = OrderedDict()
    mg["embedding.word_embeddings.weight"] = fla["backbone.embeddings.weight"].clone()

    mg_idx = 0
    for i in range(n_blocks):
        bp = f"backbone.layers.{i}"
        if i in attn_layers:
            aidx, pidx = mg_idx, mg_idx + 1
            mg_idx += 2
            # ---- MLA sublayer ----
            mg[f"decoder.layers.{aidx}.input_layernorm.weight"] = fla[f"{bp}.mixer_norm.weight"].clone()
            mg[f"decoder.layers.{aidx}.self_attention.linear_q_down_proj.weight"] = fla[f"{bp}.mixer.q_proj.0.weight"].clone()
            mg[f"decoder.layers.{aidx}.self_attention.q_layernorm.weight"] = fla[f"{bp}.mixer.q_proj.1.weight"].clone()
            mg[f"decoder.layers.{aidx}.self_attention.linear_q_up_proj.weight"] = fla[f"{bp}.mixer.q_proj.2.weight"].clone()
            # Megatron fuses [kv_down ; k_rope] -> split [kv_lora_rank, qk_pos_emb_head_dim]
            kv_down = fla[f"{bp}.mixer.kv_proj.0.weight"]
            k_rope = fla[f"{bp}.mixer.k_rope.weight"]
            mg[f"decoder.layers.{aidx}.self_attention.linear_kv_down_proj.weight"] = torch.cat([kv_down, k_rope], dim=0).clone()
            mg[f"decoder.layers.{aidx}.self_attention.kv_layernorm.weight"] = fla[f"{bp}.mixer.kv_proj.1.weight"].clone()
            mg[f"decoder.layers.{aidx}.self_attention.linear_kv_up_proj.weight"] = fla[f"{bp}.mixer.kv_proj.2.weight"].clone()
            mg[f"decoder.layers.{aidx}.self_attention.linear_proj.weight"] = fla[f"{bp}.mixer.o_proj.weight"].clone()
            # ---- MLP sublayer (attn blocks only) ----
            mg[f"decoder.layers.{pidx}.pre_mlp_layernorm.weight"] = fla[f"{bp}.mlp_norm.weight"].clone()
            gate = fla[f"{bp}.mlp.gate_proj.weight"]
            up = fla[f"{bp}.mlp.up_proj.weight"]
            mg[f"decoder.layers.{pidx}.mlp.linear_fc1.weight"] = torch.cat([gate, up], dim=0).clone()
            mg[f"decoder.layers.{pidx}.mlp.linear_fc2.weight"] = fla[f"{bp}.mlp.down_proj.weight"].clone()
        else:
            midx = mg_idx
            mg_idx += 1
            # ---- Mamba2 sublayer (direct copy; in_proj order [z,x,B,C,dt] matches) ----
            mg[f"decoder.layers.{midx}.norm.weight"] = fla[f"{bp}.mixer_norm.weight"].clone()
            mg[f"decoder.layers.{midx}.mixer.in_proj.weight"] = fla[f"{bp}.mixer.in_proj.weight"].clone()
            mg[f"decoder.layers.{midx}.mixer.conv1d.weight"] = fla[f"{bp}.mixer.conv1d.weight"].clone()
            if f"{bp}.mixer.conv1d.bias" in fla:
                mg[f"decoder.layers.{midx}.mixer.conv1d.bias"] = fla[f"{bp}.mixer.conv1d.bias"].clone()
            mg[f"decoder.layers.{midx}.mixer.A_log"] = fla[f"{bp}.mixer.A_log"].clone()
            mg[f"decoder.layers.{midx}.mixer.D"] = fla[f"{bp}.mixer.D"].clone()
            mg[f"decoder.layers.{midx}.mixer.dt_bias"] = fla[f"{bp}.mixer.dt_bias"].clone()
            mg[f"decoder.layers.{midx}.mixer.norm.weight"] = fla[f"{bp}.mixer.norm.weight"].clone()
            mg[f"decoder.layers.{midx}.mixer.out_proj.weight"] = fla[f"{bp}.mixer.out_proj.weight"].clone()

    mg["decoder.final_norm.weight"] = fla["backbone.norm_f.weight"].clone()

    print(f"  total Megatron sublayers: {mg_idx}")
    print(f"  converted {len(mg)} parameter tensors")
    return mg


def verify(fla, mg, config):
    fla_total = sum(v.numel() for v in fla.values())
    # FLA ties lm_head to embeddings; Megatron stores embedding once.
    dup = fla["backbone.embeddings.weight"].numel()
    fla_unique = fla_total - (dup if "lm_head.weight" in fla else 0)
    mg_total = sum(v.numel() for v in mg.values())
    print("\nVerification:")
    print(f"  FLA params (unique): {fla_unique:,}")
    print(f"  Megatron params:     {mg_total:,}")
    print(f"  {'OK' if fla_unique == mg_total else 'WARNING diff=' + format(abs(fla_unique - mg_total), ',')}")
    emb_ok = torch.equal(fla["backbone.embeddings.weight"], mg["embedding.word_embeddings.weight"])
    print(f"  embedding match: {emb_ok}")


def save_megatron_checkpoint(mg, output_dir, iteration=0):
    ckpt_dir = Path(output_dir) / f"iter_{iteration:07d}" / "mp_rank_00"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"iteration": iteration, "model": mg, "checkpoint_version": 3.0},
        ckpt_dir / "model_optim_rng.pt",
    )
    (Path(output_dir) / "latest_checkpointed_iteration.txt").write_text(str(iteration))
    print(f"\nSaved Megatron checkpoint to: {ckpt_dir}")
    return str(Path(output_dir))


def main():
    ap = argparse.ArgumentParser(description="Init Primus checkpoint from FLA mamba2 hybrid weights")
    ap.add_argument("--fla-config", required=True, help="Path to FLA mamba2_300M_hybrid.json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 70)
    print("FLA mamba2 hybrid -> Primus (Megatron) Weight Initialization")
    print("=" * 70)

    fla, config = init_fla_model(args.fla_config, args.seed)
    mg = convert_fla_to_megatron(fla, config)
    verify(fla, mg, config)
    out = save_megatron_checkpoint(mg, args.output_dir)

    print(f"\n{'=' * 70}")
    print("Done. Point the FLA-exact training config at it:")
    print(f"  load: {out}")
    print("  finetune: true")
    print("  no_load_optim: true")
    print("  no_load_rng: true")
    print("Then verify iter-1 loss matches FLA's iter-1 (~11.86) — confirms the")
    print("mapping (incl. MLA) is correct before trusting the loss curve.")
    print("=" * 70)


if __name__ == "__main__":
    main()
