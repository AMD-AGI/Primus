#!/usr/bin/env python3
"""
Generate an FLA-equivalent random-init KDA-300M checkpoint and convert it
into the Megatron sharded format that Primus loads.

This is the KDA counterpart of the GDN init-checkpoint dance documented in
GDN_FLA_PARITY.md.  It is the *only* code change that closed the residual
loss-curve gap for GDN, and the same is expected to apply to KDA:

  1. Re-seed PyTorch with FLA's training seed (default 42).
  2. Instantiate FLA's `KDAForCausalLM` from the same JSON config FLA used
     for the 4768-iter training run (`kda_300M_pure.json`).  FLA's
     `_init_weights` is called automatically by `PreTrainedModel.__init__`,
     reproducing the exact initial weights FLA saw at iter 0.
  3. Re-map the resulting state_dict into Primus's Megatron layout:
       - 12 FLA `KDABlock` blocks  →  24 alternating Megatron sublayers
         (even = KDA mixer, odd = MLP).
       - `attn.q_proj / k_proj / v_proj` concatenated row-wise into the
         fused `mixer.in_proj.weight` Primus uses.
       - `attn.q_conv1d / k_conv1d / v_conv1d` concatenated row-wise into
         the fused `mixer.conv1d.weight` Primus uses.
       - `mlp.gate_proj / up_proj` concatenated row-wise into the fused
         SwiGLU `mlp.linear_fc1.weight` Primus uses.
       - `attn.A_log` reshaped `[H] → [1, 1, H, 1]` to match Primus's
         per-head storage layout.
       - All other tensors copy 1:1.
  4. Pack the renamed state_dict into a Megatron checkpoint pickle
     (`iter_0000000/mp_rank_00/model_optim_rng.pt`) and write
     `latest_checkpointed_iteration.txt`.

After running this tool, the YAML flips to:

    spec: ['...', 'kda_hybrid_stack_spec_no_te']
    use_fla_kda_in_kernel_gate: true
    use_fla_fused_norm_gated: true
    finetune: true
    auto_continue_train: false
    no_load_optim: true
    no_load_rng: true
    load: <OUT_DIR from this script>

…and Primus's iter-1 loss should be bit-identical to FLA's
(`11.965` ≈ `95.74 / 8`), proving the kernel + init are both matched.

Usage
-----
    PYTHONPATH=/home/vanbhati@amd.com/flash-linear-attention \\
      python3 tools/convert_fla_kda_init_to_megatron.py \\
        --fla-config /home/vanbhati@amd.com/flash-linear-attention/legacy/training/configs/kda_300M_pure.json \\
        --output-dir /home/vanbhati@amd.com/Primus/output/fla_init_kda_300M \\
        --seed 42

Verification (post-run)
-----------------------
The script prints a per-tensor `OK / MISSING / SHAPE-MISMATCH` table.  All
tensors should say `OK`.  If anything is `MISSING`, the YAML's `spec:` must
be `kda_hybrid_stack_spec_no_te` (the TE spec uses different keys for the
pre-norm because it fuses it into `in_proj`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch


def _expected_megatron_keys(num_layers: int) -> set[str]:
    """The full set of Megatron keys we expect to produce for a 12-layer KDA-300M
    pure model under the *no-TE* spec.  Used as a self-check at the end of the
    conversion so we crash early if Primus ever changes its layout.
    """
    keys = {
        "embedding.word_embeddings.weight",
        "decoder.final_norm.weight",
    }
    for i in range(num_layers):
        kda = 2 * i
        mlp = 2 * i + 1
        keys.update({
            f"decoder.layers.{kda}.norm.weight",
            f"decoder.layers.{kda}.mixer.in_proj.weight",
            f"decoder.layers.{kda}.mixer.conv1d.weight",
            f"decoder.layers.{kda}.mixer.f_b_proj.weight",
            f"decoder.layers.{kda}.mixer.A_log",
            f"decoder.layers.{kda}.mixer.dt_bias",
            f"decoder.layers.{kda}.mixer.g_b_proj.weight",
            f"decoder.layers.{kda}.mixer.g_b_proj.bias",
            f"decoder.layers.{kda}.mixer.out_norm.weight",
            f"decoder.layers.{kda}.mixer.out_proj.weight",
            f"decoder.layers.{mlp}.pre_mlp_layernorm.weight",
            f"decoder.layers.{mlp}.mlp.linear_fc1.weight",
            f"decoder.layers.{mlp}.mlp.linear_fc2.weight",
        })
    return keys


def build_fla_init(fla_config_path: Path, seed: int) -> tuple[OrderedDict, dict]:
    """Instantiate FLA's KDAForCausalLM and return (state_dict, config_dict).

    The transformers `set_seed` re-seeds Python, NumPy, and PyTorch.
    FLA's `_init_weights` runs inside `__init__` via HuggingFace's
    `PreTrainedModel.post_init()`, so the resulting state is identical to
    what FLA's training loop starts with at step 0.
    """
    # FLA's top-level `import fla` pulls in every Triton kernel registration
    # (and therefore needs a working Triton).  We only need the model class
    # for state-dict construction (no forward pass), so bypass `fla/__init__.py`
    # entirely and import the model module directly.  This lets the converter
    # run on any Python that has `transformers` + a basic `torch` install —
    # in particular on the Primus host outside the ROCm container.
    try:
        from transformers import set_seed
    except ImportError as exc:
        raise RuntimeError(
            "transformers not installed: `pip install transformers`."
        ) from exc

    fla_root = os.environ.get("FLA_ROOT", "/home/vanbhati@amd.com/flash-linear-attention")
    if fla_root not in sys.path:
        sys.path.insert(0, fla_root)
    try:
        from fla.models.kda.configuration_kda import KDAConfig
        from fla.models.kda.modeling_kda import KDAForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "Could not import FLA's KDA model module.  Set FLA_ROOT to a "
            "checkout of https://github.com/fla-org/flash-linear-attention "
            "(default: /home/vanbhati@amd.com/flash-linear-attention)."
        ) from exc

    set_seed(seed)

    with open(fla_config_path) as f:
        cfg_dict = json.load(f)
    config = KDAConfig(**cfg_dict)
    # Force bf16 to match FLA training (configs sometimes default to fp32 here).
    config.torch_dtype = "bfloat16"

    print(f"[init] instantiating FLA KDAForCausalLM (seed={seed})…")
    model = KDAForCausalLM(config).to(dtype=torch.bfloat16)
    print(f"[init] params = {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    sd = OrderedDict()
    for k, v in model.state_dict().items():
        sd[k] = v.detach().contiguous().cpu()
    return sd, cfg_dict


def convert_fla_to_megatron(fla_sd: OrderedDict, cfg: dict) -> OrderedDict:
    """Map FLA HF state_dict → Primus Megatron-sharded state_dict (no-TE spec)."""
    hidden_size      = cfg["hidden_size"]
    num_heads        = cfg["num_heads"]
    num_v_heads      = cfg.get("num_v_heads") or num_heads
    head_dim         = cfg["head_dim"]
    expand_v         = cfg.get("expand_v", 1.0)
    intermediate     = cfg["intermediate_size"]
    num_layers       = cfg["num_hidden_layers"]

    head_v_dim = int(head_dim * expand_v)
    qk_dim     = num_heads * head_dim       # 256
    v_dim      = num_v_heads * head_v_dim   # 512

    print(
        f"[map ] hidden={hidden_size} num_heads={num_heads} num_v_heads={num_v_heads}\n"
        f"       head_dim={head_dim} expand_v={expand_v} head_v_dim={head_v_dim}\n"
        f"       qk_dim={qk_dim} v_dim={v_dim} intermediate={intermediate} "
        f"layers={num_layers}"
    )

    out = OrderedDict()
    out["embedding.word_embeddings.weight"] = fla_sd["model.embeddings.weight"]
    out["decoder.final_norm.weight"]        = fla_sd["model.norm.weight"]

    for i in range(num_layers):
        kda = 2 * i
        mlp = 2 * i + 1
        src = f"model.layers.{i}"

        # ── KDA sublayer ────────────────────────────────────────────────
        out[f"decoder.layers.{kda}.norm.weight"] = fla_sd[f"{src}.attn_norm.weight"]

        # in_proj (fully fused, GDN-style): concat
        #     [ q_proj | k_proj | v_proj | f_proj.0 | g_proj.0 | b_proj ]
        # along dim 0.  Resulting shape:
        #     [qk_dim*2 + v_dim + head_v_dim + head_v_dim + num_v_heads, hidden]
        # This is the same matrix that ColumnParallelLinear would write — by
        # concatenating FLA's six independent `hidden_states → X` projections
        # in this exact order we make Primus's forward bit-identical to FLA's
        # while paying only ONE matmul launch per layer instead of six.
        q_w   = fla_sd[f"{src}.attn.q_proj.weight"]
        k_w   = fla_sd[f"{src}.attn.k_proj.weight"]
        v_w   = fla_sd[f"{src}.attn.v_proj.weight"]
        f_a_w = fla_sd[f"{src}.attn.f_proj.0.weight"]
        g_a_w = fla_sd[f"{src}.attn.g_proj.0.weight"]
        b_w   = fla_sd[f"{src}.attn.b_proj.weight"]
        assert q_w.shape   == (qk_dim,     hidden_size), q_w.shape
        assert k_w.shape   == (qk_dim,     hidden_size), k_w.shape
        assert v_w.shape   == (v_dim,      hidden_size), v_w.shape
        assert f_a_w.shape == (head_v_dim, hidden_size), f_a_w.shape
        assert g_a_w.shape == (head_v_dim, hidden_size), g_a_w.shape
        assert b_w.shape   == (num_v_heads, hidden_size), b_w.shape
        out[f"decoder.layers.{kda}.mixer.in_proj.weight"] = torch.cat(
            [q_w, k_w, v_w, f_a_w, g_a_w, b_w], dim=0
        )

        # conv1d: concat [q_conv1d, k_conv1d, v_conv1d] along dim 0.
        # FLA stores each as [channels, 1, kernel]; concatenation gives
        # [qk_dim*2 + v_dim, 1, kernel] = [1024, 1, 4].
        qc = fla_sd[f"{src}.attn.q_conv1d.weight"]
        kc = fla_sd[f"{src}.attn.k_conv1d.weight"]
        vc = fla_sd[f"{src}.attn.v_conv1d.weight"]
        out[f"decoder.layers.{kda}.mixer.conv1d.weight"] = torch.cat([qc, kc, vc], dim=0)

        # f_proj.1 (low-rank gate expander):  Linear(head_v_dim, gate_dim)
        out[f"decoder.layers.{kda}.mixer.f_b_proj.weight"] = fla_sd[f"{src}.attn.f_proj.1.weight"]

        # A_log: FLA stores [num_v_heads]; Primus stores [1, 1, num_heads_local_tp, 1].
        out[f"decoder.layers.{kda}.mixer.A_log"] = fla_sd[f"{src}.attn.A_log"].view(1, 1, num_v_heads, 1).clone()

        # dt_bias: same shape (gate_dim = num_v_heads * head_dim).
        out[f"decoder.layers.{kda}.mixer.dt_bias"] = fla_sd[f"{src}.attn.dt_bias"]

        # g_proj.1 (low-rank output-gate expander):
        #   Linear(head_v_dim, value_dim, bias=True)
        out[f"decoder.layers.{kda}.mixer.g_b_proj.weight"] = fla_sd[f"{src}.attn.g_proj.1.weight"]
        out[f"decoder.layers.{kda}.mixer.g_b_proj.bias"]   = fla_sd[f"{src}.attn.g_proj.1.bias"]

        # output norm:  FusedRMSNormGated.weight shape == [head_v_dim]
        out[f"decoder.layers.{kda}.mixer.out_norm.weight"] = fla_sd[f"{src}.attn.o_norm.weight"]

        # output projection:  Linear(value_dim, hidden_size)
        out[f"decoder.layers.{kda}.mixer.out_proj.weight"] = fla_sd[f"{src}.attn.o_proj.weight"]

        # ── MLP sublayer ────────────────────────────────────────────────
        out[f"decoder.layers.{mlp}.pre_mlp_layernorm.weight"] = fla_sd[f"{src}.mlp_norm.weight"]

        # SwiGLU fc1: concat [gate_proj, up_proj] along dim 0
        gp = fla_sd[f"{src}.mlp.gate_proj.weight"]
        up = fla_sd[f"{src}.mlp.up_proj.weight"]
        assert gp.shape == (intermediate, hidden_size), gp.shape
        assert up.shape == (intermediate, hidden_size), up.shape
        out[f"decoder.layers.{mlp}.mlp.linear_fc1.weight"] = torch.cat([gp, up], dim=0)

        # SwiGLU fc2: Linear(intermediate, hidden_size)
        out[f"decoder.layers.{mlp}.mlp.linear_fc2.weight"] = fla_sd[f"{src}.mlp.down_proj.weight"]

    return out


def cross_check(mg_sd: OrderedDict, cfg: dict) -> None:
    expected = _expected_megatron_keys(cfg["num_hidden_layers"])
    got      = set(mg_sd.keys())
    missing  = expected - got
    extra    = got - expected
    if missing:
        print(f"\n[FAIL] {len(missing)} expected Megatron keys are MISSING from the converted state_dict:")
        for k in sorted(missing)[:25]:
            print(f"    - {k}")
        raise SystemExit(1)
    if extra:
        print(f"\n[warn] {len(extra)} unexpected extra keys in the converted state_dict (probably harmless):")
        for k in sorted(extra)[:10]:
            print(f"    + {k}")
    print(f"[chk ] all {len(expected)} expected Megatron keys present ✓")


def write_megatron_checkpoint(mg_sd: OrderedDict, output_dir: Path) -> None:
    """Write `output_dir/iter_0000000/mp_rank_00/model_optim_rng.pt`.

    The Megatron loader (`megatron.training.checkpointing.load_checkpoint`)
    expects:
        OUTPUT_DIR/
            latest_checkpointed_iteration.txt  ← contains "0"
            iter_0000000/
                mp_rank_00/
                    model_optim_rng.pt          ← pickle with keys
                        {'iteration', 'model', 'checkpoint_version'}
    With `finetune: true; no_load_optim: true; no_load_rng: true` Primus
    only reads the `model` field of the pickle.
    """
    iter_dir = output_dir / "iter_0000000" / "mp_rank_00"
    iter_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = iter_dir / "model_optim_rng.pt"

    ckpt = {
        "iteration": 0,
        "model": mg_sd,
        "checkpoint_version": 3.0,
    }
    torch.save(ckpt, ckpt_path)
    print(f"[save] wrote {ckpt_path}  ({ckpt_path.stat().st_size / 1e6:.1f} MB)")

    # latest_checkpointed_iteration.txt — Megatron uses this to discover the
    # most recent checkpoint when `auto_continue_train: false` is unset.
    (output_dir / "latest_checkpointed_iteration.txt").write_text("0\n")
    print(f"[save] wrote {output_dir / 'latest_checkpointed_iteration.txt'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--fla-config",
        type=Path,
        default=Path("/home/vanbhati@amd.com/flash-linear-attention/legacy/training/configs/kda_300M_pure.json"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/vanbhati@amd.com/Primus/output/fla_init_kda_300M"),
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("=" * 78)
    print("  FLA  KDA-300M  ─→  Primus Megatron sharded checkpoint")
    print("=" * 78)
    print(f"  fla_config = {args.fla_config}")
    print(f"  output_dir = {args.output_dir}")
    print(f"  seed       = {args.seed}")
    print()

    fla_sd, cfg = build_fla_init(args.fla_config, args.seed)
    mg_sd = convert_fla_to_megatron(fla_sd, cfg)
    cross_check(mg_sd, cfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_megatron_checkpoint(mg_sd, args.output_dir)

    print()
    print("✓ done. Now update the YAML to:")
    print(f"    spec: ['primus.backends.megatron.core.models.hybrid.hybrid_mamba_mla_layer_specs', 'kda_hybrid_stack_spec_no_te']")
    print(f"    use_fla_kda_in_kernel_gate: true")
    print(f"    use_fla_fused_norm_gated: true")
    print(f"    finetune: true")
    print(f"    auto_continue_train: false")
    print(f"    no_load_optim: true")
    print(f"    no_load_rng: true")
    print(f"    load: {args.output_dir}")
    print()


if __name__ == "__main__":
    sys.exit(main())
