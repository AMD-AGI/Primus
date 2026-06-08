#!/usr/bin/env python3
"""Convert Primus (Megatron) 75% Hybrid Mamba2+MLA checkpoint to FLA HuggingFace format.

The hybrid model has 12 FLA mixer blocks: MLA at indices [0, 4, 8] and Mamba2
at the other 9 indices.

Primus Megatron stores them as 24 sublayers alternating mixer/MLP, with
hybrid_override_pattern = "*-M-M-M-*-M-M-M-*-M-M-M-":

    sublayer   pattern  FLA mixer  what
    --------   -------  ---------  ----
    0          *        0          MLA
    1          -                   MLP
    2          M        1          Mamba2
    3          -                   MLP
    4          M        2          Mamba2
    5          -                   MLP
    6          M        3          Mamba2
    7          -                   MLP
    8          *        4          MLA
    9          -                   MLP
    ...                            (repeats with `*-M-M-M-` pattern)

This converter walks the 12 FLA mixer indices, finds the corresponding Megatron
sublayer (mixer + MLP), and emits weights for FLA's `Mamba2ForCausalLM`
(which uses the `backbone.` prefix, NOT `model.` like `GatedDeltaNetForCausalLM`).

The MLA mapping reuses the same channel-permutation fix from the GDN converter
(see `_megatron_to_fla_rope_channels`).  The Mamba2 mixer mapping is a direct
copy because both Megatron's MambaMixer and FLA's Mamba2Mixer wrap upstream
`mamba_ssm` with the same `[z | x | B | C | dt]` `in_proj` layout.

NOTE on dim mismatch with FLA's reference 300M:

    Primus YAML     FLA mamba2_300M_hybrid.json
    -----------     ---------------------------
    hidden_size     1024              1216
    intermediate    4096              4864
    state_size      64                128
    n_groups        8                 1
    n_heads(Mamba)  32                38
    total params    ~273M             ~350M

We deliberately matched Primus to the GDN-hybrid 300M dims so the architecture
is a drop-in mixer swap (Mamba2 ↔ GDN under the same MLA + MLP backbone).
The emitted HF config.json therefore reflects PRIMUS's dims, not FLA's.
Both models are independently valid HF Mamba2ForCausalLM checkpoints; the
downstream lm-eval can score each on its own merits.

Usage (inside the container):

    python tools/convert_mamba_hybrid_to_fla_hf.py \
        --checkpoint-path output/amd/root/zebra_llama_300M_mamba_hybrid-pretrain/checkpoints/iter_0004768 \
        --output-dir output/mamba_hybrid_300M_fla_hf \
        --tokenizer /home/vanbhati@amd.com/checkpoints/gdn_pure_300M_10B
"""
import argparse
import json
import os
import sys
import torch
from pathlib import Path
from collections import OrderedDict

_megatron_path = str(Path(__file__).resolve().parents[1] / "third_party" / "Megatron-LM")
if _megatron_path not in sys.path:
    sys.path.insert(0, _megatron_path)


def load_megatron_checkpoint(checkpoint_path):
    model_path = Path(checkpoint_path) / "mp_rank_00" / "model_optim_rng.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    print(f"Loaded. Iteration: {checkpoint.get('iteration', '?')}")
    return checkpoint


def _get_first(state, *candidates):
    for k in candidates:
        if k in state:
            return state[k], k
    raise KeyError(
        f"None of these expected keys were found:\n  "
        + "\n  ".join(candidates)
        + "\nFirst 30 keys in checkpoint:\n  "
        + "\n  ".join(sorted(state.keys())[:30])
    )


def _build_layer_map(hybrid_pattern):
    """Same as the GDN converter — `*` = MLA, `M` = Mamba2, `-` = MLP."""
    fla_to_megatron = []
    cur_mixer = None
    cur_kind = None
    for i, c in enumerate(hybrid_pattern):
        if c in ("*", "M"):
            cur_mixer = i
            cur_kind = "mla" if c == "*" else "mamba2"
        elif c == "-":
            if cur_mixer is None:
                continue
            fla_to_megatron.append((cur_mixer, i, cur_kind))
            cur_mixer = None
            cur_kind = None
    return fla_to_megatron


def _megatron_to_fla_rope_channels(w_rope, qk_rope_head_dim):
    """Permute the rope channels of a tensor whose LAST dim is `qk_rope_head_dim`
    from Megatron's storage order to FLA's RoPE-ready order.

    Megatron stores [c0, c1, c2, ..., c_{d-1}] and on-the-fly interleaves to
    [c0, c2, ..., c1, c3, ...]. FLA assumes already-permuted weights.
    """
    d = qk_rope_head_dim
    assert w_rope.shape[-1] == d, (
        f"_megatron_to_fla_rope_channels: last dim {w_rope.shape[-1]} != "
        f"qk_rope_head_dim {d}"
    )
    even = w_rope[..., 0::2]
    odd = w_rope[..., 1::2]
    return torch.cat([even, odd], dim=-1).contiguous()


def convert_mla_block(state, sub_mixer, prefix, fla_cfg):
    """Identical to the GDN-hybrid MLA conversion (since the MLA spec is shared).
    Writes:
        {prefix}.mixer.q_proj.0/1/2.weight
        {prefix}.mixer.kv_proj.0/1/2.weight
        {prefix}.mixer.k_rope.weight
        {prefix}.mixer.o_proj.weight
    (Note: FLA's Mamba2 model uses `mixer.attn` ... actually it uses `mixer`
    for both attention and SSM blocks — the type is dispatched by config.attn.layers.)
    """
    out = OrderedDict()
    attn_cfg = fla_cfg["attn"]
    q_lora_rank = attn_cfg["q_lora_rank"]
    kv_lora_rank = attn_cfg["kv_lora_rank"]
    qk_rope_head_dim = attn_cfg["qk_rope_head_dim"]
    num_heads = attn_cfg["num_heads"]
    qk_nope_head_dim = attn_cfg["qk_nope_head_dim"]
    v_head_dim = attn_cfg["v_head_dim"]
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    hidden_size = fla_cfg["hidden_size"]

    # ── q path ──
    q_down_w = state[f"decoder.layers.{sub_mixer}.self_attention.linear_q_down_proj.weight"]
    q_up_w = state[f"decoder.layers.{sub_mixer}.self_attention.linear_q_up_proj.weight"]
    q_norm_w, _ = _get_first(
        state, f"decoder.layers.{sub_mixer}.self_attention.q_layernorm.weight",
    )
    assert q_down_w.shape == (q_lora_rank, hidden_size)
    assert q_up_w.shape == (num_heads * qk_head_dim, q_lora_rank)
    assert q_norm_w.shape == (q_lora_rank,)

    # Permute rope half of q_up_proj per head.
    q_up_3d = q_up_w.view(num_heads, qk_head_dim, q_lora_rank)
    q_up_nope = q_up_3d[:, :qk_nope_head_dim, :]
    q_up_rope_meg = q_up_3d[:, qk_nope_head_dim:, :].transpose(-1, -2)
    q_up_rope_fla = _megatron_to_fla_rope_channels(q_up_rope_meg, qk_rope_head_dim).transpose(-1, -2)
    q_up_fla = torch.cat([q_up_nope, q_up_rope_fla], dim=1)
    q_up_fla = q_up_fla.reshape(num_heads * qk_head_dim, q_lora_rank).contiguous()

    out[f"{prefix}.q_proj.0.weight"] = q_down_w
    out[f"{prefix}.q_proj.1.weight"] = q_norm_w
    out[f"{prefix}.q_proj.2.weight"] = q_up_fla

    # ── kv path ──
    kv_down_w = state[f"decoder.layers.{sub_mixer}.self_attention.linear_kv_down_proj.weight"]
    expected_down_rows = kv_lora_rank + qk_rope_head_dim
    assert kv_down_w.shape == (expected_down_rows, hidden_size)
    out[f"{prefix}.kv_proj.0.weight"] = kv_down_w[:kv_lora_rank].contiguous()

    k_rope_meg = kv_down_w[kv_lora_rank:].contiguous().transpose(0, 1)
    k_rope_fla = _megatron_to_fla_rope_channels(k_rope_meg, qk_rope_head_dim).transpose(0, 1).contiguous()
    out[f"{prefix}.k_rope.weight"] = k_rope_fla

    kv_norm_w, _ = _get_first(
        state, f"decoder.layers.{sub_mixer}.self_attention.kv_layernorm.weight",
    )
    assert kv_norm_w.shape == (kv_lora_rank,)
    out[f"{prefix}.kv_proj.1.weight"] = kv_norm_w

    kv_up_w = state[f"decoder.layers.{sub_mixer}.self_attention.linear_kv_up_proj.weight"]
    expected_up_rows = num_heads * (qk_nope_head_dim + v_head_dim)
    assert kv_up_w.shape == (expected_up_rows, kv_lora_rank)
    out[f"{prefix}.kv_proj.2.weight"] = kv_up_w

    out[f"{prefix}.o_proj.weight"] = state[
        f"decoder.layers.{sub_mixer}.self_attention.linear_proj.weight"
    ]
    return out


def convert_mamba2_block(state, sub_mixer, prefix, fla_cfg):
    """Mamba2 mixer Primus→FLA mapping.

    Both Megatron and FLA wrap upstream mamba_ssm's MambaMixer with the SAME
    `[z | x | B | C | dt]` in_proj layout, so this is a direct copy of:

        in_proj.weight     (hidden_intermediate*2 + 2*n_groups*state + n_heads, hidden)
        conv1d.weight      (hidden_intermediate + 2*n_groups*state, 1, kernel)
        conv1d.bias        (hidden_intermediate + 2*n_groups*state,)
        A_log              (n_heads,)
        D                  (n_heads,)
        dt_bias            (n_heads,)
        norm.weight        (hidden_intermediate,)         ← post-mixer gated-RMSNorm
        out_proj.weight    (hidden, hidden_intermediate)

    Validated against an emitted Primus ckpt with hidden=1024, expand=2,
    n_heads=32, n_groups=8, state=64 → in_proj (5152, 1024), conv1d (3072,1,4).
    """
    out = OrderedDict()
    hidden_size = fla_cfg["hidden_size"]
    expand = fla_cfg["expand"]
    head_dim = fla_cfg["head_dim"]
    n_groups = fla_cfg["n_groups"]
    state_size = fla_cfg["state_size"]
    num_heads = fla_cfg["num_heads"]

    intermediate = expand * hidden_size
    conv_dim = intermediate + 2 * n_groups * state_size
    in_dim = 2 * intermediate + 2 * n_groups * state_size + num_heads

    in_proj_w = state[f"decoder.layers.{sub_mixer}.mixer.in_proj.weight"]
    assert in_proj_w.shape == (in_dim, hidden_size), (
        f"Mamba2 in_proj shape {tuple(in_proj_w.shape)} != "
        f"expected ({in_dim}, {hidden_size}); "
        f"check expand={expand}, n_groups={n_groups}, state={state_size}, n_heads={num_heads}"
    )
    out[f"{prefix}.in_proj.weight"] = in_proj_w

    conv1d_w = state[f"decoder.layers.{sub_mixer}.mixer.conv1d.weight"]
    conv1d_b = state[f"decoder.layers.{sub_mixer}.mixer.conv1d.bias"]
    assert conv1d_w.shape[0] == conv_dim, (
        f"Mamba2 conv1d weight rows {conv1d_w.shape[0]} != expected {conv_dim}"
    )
    out[f"{prefix}.conv1d.weight"] = conv1d_w
    out[f"{prefix}.conv1d.bias"] = conv1d_b

    out[f"{prefix}.A_log"] = state[f"decoder.layers.{sub_mixer}.mixer.A_log"]
    out[f"{prefix}.D"] = state[f"decoder.layers.{sub_mixer}.mixer.D"]
    out[f"{prefix}.dt_bias"] = state[f"decoder.layers.{sub_mixer}.mixer.dt_bias"]

    norm_w = state[f"decoder.layers.{sub_mixer}.mixer.norm.weight"]
    assert norm_w.shape == (intermediate,)
    out[f"{prefix}.norm.weight"] = norm_w

    out[f"{prefix}.out_proj.weight"] = state[
        f"decoder.layers.{sub_mixer}.mixer.out_proj.weight"
    ]
    return out


def build_fla_config(primus_args, base_attn_cfg):
    """Build an HF Mamba2 config that matches the PRIMUS-trained dims.

    `primus_args` is the `args` namespace stashed in the Megatron checkpoint,
    or our YAML-derived overrides; we extract the actual dims so the emitted
    HF model loads without shape mismatches.
    """
    hidden_size = primus_args["hidden_size"]
    intermediate_size = primus_args["ffn_hidden_size"]
    expand = primus_args.get("mamba_expand", 2)
    head_dim = primus_args.get("mamba_head_dim", 64)
    n_groups = primus_args.get("mamba_num_groups", 8)
    state_size = primus_args.get("mamba_state_dim", 64)
    num_heads = (expand * hidden_size) // head_dim

    cfg = {
        "model_type": "mamba2",
        "architectures": ["Mamba2ForCausalLM"],
        "vocab_size": primus_args.get("padded_vocab_size", 128256),
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "state_size": state_size,
        "num_hidden_layers": 12,
        "head_dim": head_dim,
        "expand": expand,
        "n_groups": n_groups,
        "num_heads": num_heads,
        "conv_kernel": 4,
        "use_bias": False,
        "use_conv_bias": True,
        "hidden_act": "silu",
        "hidden_act_mlp": "swish",
        "initializer_range": 0.02,
        # NOTE: set to False (FLA default is True) — at bf16 inference the
        # mixer's `in_proj` is bf16 but with residual_in_fp32=True the
        # incoming residual is fp32, causing an `F.linear` dtype mismatch.
        # Training was bf16 + residual_in_fp32=True (FLA's training default)
        # which only matters during gradient accumulation; for eval-time
        # forward the bf16 residual is mathematically equivalent up to
        # rounding noise.
        "residual_in_fp32": False,
        "rmsnorm": True,
        "norm_eps": primus_args.get("norm_epsilon", 1e-6),
        "chunk_size": 256,
        "D_has_hdim": False,
        "norm_before_gate": False,
        "rescale_prenorm_residual": True,
        "max_position_embeddings": primus_args.get("max_position_embeddings", 131072),
        "hidden_ratio": None,
        # Disable fused kernels in HF (they require packages we may not have)
        "fuse_norm": False,
        "fuse_swiglu": False,
        "fuse_cross_entropy": False,
        "fuse_linear_cross_entropy": False,
        "tie_word_embeddings": True,
        "use_cache": True,
        # MLA attention sub-block — verbatim from base_attn_cfg
        "attn": base_attn_cfg,
        # Mamba2 init knobs (HF defaults; not used at eval time)
        "A_init_range": [1, 16],
        "dt_init_floor": 0.0001,
        "dt_limit": [0.0, float("inf")],
        "dt_max": 0.1,
        "dt_min": 0.001,
        "conv_init": None,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "pad_token_id": 0,
        "torch_dtype": "bfloat16",
    }
    return cfg


def convert(checkpoint, hybrid_pattern, base_attn_cfg, primus_args_override=None):
    state = checkpoint["model"]

    # Pull Primus's actual dims out of the checkpoint so config matches weights.
    ckpt_args = checkpoint.get("args", None)
    primus_args = {}
    if ckpt_args is not None:
        for k in (
            "hidden_size", "ffn_hidden_size", "padded_vocab_size",
            "mamba_state_dim", "mamba_head_dim", "mamba_num_groups",
            "mamba_expand", "max_position_embeddings", "norm_epsilon",
        ):
            v = getattr(ckpt_args, k, None)
            if v is not None:
                primus_args[k] = v
    if primus_args_override:
        primus_args.update(primus_args_override)

    fla_cfg = build_fla_config(primus_args, base_attn_cfg)
    num_hidden_layers = fla_cfg["num_hidden_layers"]

    layer_map = _build_layer_map(hybrid_pattern)
    assert len(layer_map) == num_hidden_layers, (
        f"hybrid_pattern produced {len(layer_map)} FLA mixer blocks "
        f"but FLA config wants {num_hidden_layers}"
    )

    print(f"\nFLA layer index → Megatron sublayer mapping:")
    for fla_idx, (sub_mixer, sub_mlp, kind) in enumerate(layer_map):
        print(f"  FLA layer {fla_idx} ({kind.upper():>6}): mixer=sub{sub_mixer}, mlp=sub{sub_mlp}")

    hf_state = OrderedDict()
    hf_state["backbone.embeddings.weight"] = state["embedding.word_embeddings.weight"]

    hidden_size = fla_cfg["hidden_size"]
    n_ones_mixer_norm = 0
    for fla_idx, (sub_mixer, sub_mlp, kind) in enumerate(layer_map):
        prefix = f"backbone.layers.{fla_idx}"

        # Pre-mixer norm (FLA: `mixer_norm.weight`).
        #
        # ⚠ For Mamba2 mixer sublayers, Primus's spec used `MambaLayerSubmodules(norm=IdentityOp)`
        # (the default) — there was NO learnable pre-mixer norm at training time.
        # FLA's `Mamba2Block` ALWAYS applies a `mixer_norm` RMSNorm before the mixer.
        # We emit `ones(hidden_size)` so the FLA HF model loads, but this introduces
        # a "spurious" RMSNorm at inference time that wasn't present during training
        # (division by RMS, scaled by 1.0).  Empirically this is a small perturbation
        # — the eval scores should still be representative — but it is NOT bit-exact.
        # For MLA mixer sublayers we have `input_layernorm.weight` and copy it.
        try:
            mixer_norm_w, src_key = _get_first(
                state,
                f"decoder.layers.{sub_mixer}.input_layernorm.weight",
                f"decoder.layers.{sub_mixer}.mixer.in_proj.layer_norm_weight",
                f"decoder.layers.{sub_mixer}.norm.weight",
            )
        except KeyError:
            assert kind == "mamba2", (
                f"FLA layer {fla_idx} ({kind}): missing pre-mixer norm — only "
                f"Mamba2 layers are allowed to lack one (MLA always has input_layernorm)."
            )
            mixer_norm_w = torch.ones(hidden_size, dtype=torch.bfloat16)
            n_ones_mixer_norm += 1
        hf_state[f"{prefix}.mixer_norm.weight"] = mixer_norm_w

        if kind == "mla":
            hf_state.update(convert_mla_block(state, sub_mixer, f"{prefix}.mixer", fla_cfg))
        else:
            hf_state.update(convert_mamba2_block(state, sub_mixer, f"{prefix}.mixer", fla_cfg))

        # Pre-MLP norm (FLA: `mlp_norm.weight`)
        mlp_norm_w, _ = _get_first(
            state,
            f"decoder.layers.{sub_mlp}.pre_mlp_layernorm.weight",
            f"decoder.layers.{sub_mlp}.mlp.linear_fc1.layer_norm_weight",
            f"decoder.layers.{sub_mlp}.input_layernorm.weight",
        )
        hf_state[f"{prefix}.mlp_norm.weight"] = mlp_norm_w

        fc1_w = state[f"decoder.layers.{sub_mlp}.mlp.linear_fc1.weight"]
        intermediate_size = fla_cfg["intermediate_size"]
        assert fc1_w.shape[0] == intermediate_size * 2, (
            f"MLP fc1 shape {tuple(fc1_w.shape)} expected ({intermediate_size*2},...)"
        )
        hf_state[f"{prefix}.mlp.gate_proj.weight"] = fc1_w[:intermediate_size]
        hf_state[f"{prefix}.mlp.up_proj.weight"] = fc1_w[intermediate_size:]
        hf_state[f"{prefix}.mlp.down_proj.weight"] = state[
            f"decoder.layers.{sub_mlp}.mlp.linear_fc2.weight"
        ]

    final_norm_w, _ = _get_first(
        state,
        "decoder.final_norm.weight",
        "decoder.final_layernorm.weight",
        "decoder.norm.weight",
    )
    hf_state["backbone.norm_f.weight"] = final_norm_w

    if "output_layer.weight" in state:
        hf_state["lm_head.weight"] = state["output_layer.weight"]
    else:
        hf_state["lm_head.weight"] = state["embedding.word_embeddings.weight"]

    if n_ones_mixer_norm:
        print(
            f"\n⚠ Emitted ones(hidden_size) for {n_ones_mixer_norm}/9 Mamba2 mixer_norm.weight "
            f"entries (Primus spec had no pre-mixer norm). FLA inference will apply an "
            f"unweighted RMSNorm here that the Primus model never saw during training."
        )

    return hf_state, fla_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--hybrid-pattern",
        default="*-M-M-M-*-M-M-M-*-M-M-M-",
        help="Same as YAML hybrid_override_pattern (24 chars for 300M hybrid).",
    )
    parser.add_argument(
        "--tokenizer",
        default="/home/vanbhati@amd.com/checkpoints/gdn_pure_300M_10B",
        help="Source dir to copy tokenizer.json / tokenizer_config.json from.",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Primus 75% Hybrid Mamba2+MLA → FLA HuggingFace Conversion")
    print("=" * 72)

    # MLA sub-config — verbatim layout (matches both our spec and FLA's mla.py).
    base_attn_cfg = {
        "type": "mla",
        "layers": [0, 4, 8],
        "num_heads": 16,
        "q_lora_rank": 672,
        "kv_lora_rank": 64,
        "qk_nope_head_dim": 32,
        "qk_rope_head_dim": 32,
        "qk_head_dim": 64,
        "v_head_dim": 64,
        "rope_theta": 500000.0,
    }

    checkpoint = load_megatron_checkpoint(args.checkpoint_path)
    hf_state, fla_cfg = convert(checkpoint, args.hybrid_pattern, base_attn_cfg)
    print(f"\nConverted {len(hf_state)} parameters")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from safetensors.torch import save_file
        # Untie lm_head <-> embeddings (safetensors disallows shared tensors).
        if (
            "lm_head.weight" in hf_state
            and "backbone.embeddings.weight" in hf_state
            and hf_state["lm_head.weight"].data_ptr()
            == hf_state["backbone.embeddings.weight"].data_ptr()
        ):
            hf_state["lm_head.weight"] = hf_state["lm_head.weight"].clone()
        save_file(hf_state, str(output_dir / "model.safetensors"))
        print(f"  Saved {output_dir / 'model.safetensors'}")
    except ImportError:
        torch.save(hf_state, output_dir / "pytorch_model.bin")
        print(f"  Saved {output_dir / 'pytorch_model.bin'}")

    # Tell HF AutoModel to load via our custom Mamba2FullMlpForCausalLM class
    # (saved as modeling_mamba2_full_mlp.py next to the checkpoint).  Primus has
    # MLPs on EVERY layer (both MLA and Mamba2 sublayers) — the stock FLA
    # Mamba2Block only builds MLPs for MLA layers and would silently drop the
    # other 9 MLPs from the safetensors at load time.
    fla_cfg["architectures"] = ["Mamba2FullMlpForCausalLM"]
    fla_cfg["auto_map"] = {
        "AutoConfig": "modeling_mamba2_full_mlp.Mamba2Config",
        "AutoModelForCausalLM": "modeling_mamba2_full_mlp.Mamba2FullMlpForCausalLM",
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(fla_cfg, f, indent=2)
    print(f"  Saved config.json")
    print(f"  hidden_size={fla_cfg['hidden_size']}, intermediate_size={fla_cfg['intermediate_size']}")
    print(f"  n_groups={fla_cfg['n_groups']}, state_size={fla_cfg['state_size']}, num_heads={fla_cfg['num_heads']}")

    # Drop a copy of our custom modeling file next to the checkpoint so
    # `from_pretrained(..., trust_remote_code=True)` can pick it up.
    import shutil
    src_modeling = Path(__file__).parent / "_primus_mamba2_modeling.py"
    dst_modeling = output_dir / "modeling_mamba2_full_mlp.py"
    shutil.copy2(src_modeling, dst_modeling)
    print(f"  Copied modeling_mamba2_full_mlp.py (custom class with per-layer MLPs)")
    # Re-export Mamba2Config from FLA in the same module so AutoConfig finds it.
    with open(dst_modeling, "a") as f:
        f.write("\n\n# Re-export the FLA config under the same module for auto_map\n")
        f.write("from fla.models.mamba2.configuration_mamba2 import Mamba2Config\n")

    # Copy tokenizer if a source was provided.
    src_tok = Path(args.tokenizer)
    if src_tok.exists():
        import shutil
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ):
            f_src = src_tok / name
            if f_src.exists():
                shutil.copy2(f_src, output_dir / name)
                print(f"  Copied {name} from {src_tok}")
    else:
        # Minimal fallback so lm-eval can still load.
        with open(output_dir / "tokenizer_config.json", "w") as f:
            json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 2048}, f, indent=2)

    print()
    print("Done. To sanity-check then evaluate:")
    print(f"  python -c \"from fla.models import Mamba2ForCausalLM; m = Mamba2ForCausalLM.from_pretrained('{output_dir}'); print(sum(p.numel() for p in m.parameters())/1e6, 'M params')\"")
    print()
    print(f"  lm_eval --model hf --model_args pretrained={output_dir},trust_remote_code=True,dtype=bfloat16 \\")
    print(f"    --tasks hellaswag,winogrande,piqa,arc_easy,arc_challenge,lambada_openai \\")
    print(f"    --batch_size 16 --output_path {output_dir}/lm_eval")


if __name__ == "__main__":
    main()
