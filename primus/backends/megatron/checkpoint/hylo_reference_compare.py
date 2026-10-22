###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Compare an Enhanced-ILD student with a Zebra-Llama reference, and load a reference as a student.

Primus all-Mamba-2 and all-MLA students are Megatron checkpoints (``M-`` or ``*-`` patterns,
mixer and MLP sublayers alternating). The Zebra-Llama reference stage-1 students are Hugging Face
``model.safetensors`` files. ``pairs`` maps every student tensor to its reference name.

``compare`` reports, per tensor, the cosine and relative L2 distance (``||a - b|| / ||b||``) of
the student against the reference, and how far each moved from the student's upcycled init
(``move = ||w - init|| / ||init||``, ``upd cos`` = cosine of the two updates, ``upd o/r`` = ratio
of their norms). Mamba-2 tensors with a random init (z and dt rows of ``in_proj``, the
convolution, ``A_log``, ``dt_bias``) differ between runs from the start, so only their statistics
are compared unless the student's init carries the reference's draw. MLA low-rank factors are
compared after aligning each SVD direction's sign, and as products, which do not depend on the
signs. ``init vs dense`` checks the upcycle: it is ~0 for tensors copied from the teacher.

``write_reference_as_megatron`` puts the reference's values into a copy of a student checkpoint,
so the ILD evaluation can score the reference against the same teacher and data.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

Pairs = List[Tuple[str, str, Optional[Tuple[int, int]]]]


@dataclass
class Layout:
    """What ``pairs`` and the derived tensors need to know about a student."""

    kind: str
    num_layers: int
    ffn: int
    m2_in: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    m2_conv: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    mla_rank: int = 0
    mla_rope: int = 0
    mla_nope: int = 0


def _first(state: Dict[str, torch.Tensor], *keys: str) -> str:
    for key in keys:
        if key in state:
            return key
    raise KeyError(f"none of {keys}")


def _arg(args, name: str):
    if args is None:
        return None
    if isinstance(args, dict):
        return args.get(name)
    return getattr(args, name, None)


def student_layout(state: Dict[str, torch.Tensor], kind: str, args=None) -> Layout:
    """Read the widths off a student state dict (and its checkpoint args for MLA)."""
    indices = {int(m.group(1)) for key in state if (m := re.match(r"^decoder\.layers\.(\d+)\.", key))}
    if not indices or len(indices) % 2 or sorted(indices) != list(range(len(indices))):
        raise ValueError("expected contiguous mixer/MLP sublayer pairs under decoder.layers")
    layout = Layout(
        kind=kind,
        num_layers=len(indices) // 2,
        ffn=state["decoder.layers.1.mlp.linear_fc1.weight"].shape[0] // 2,
    )
    if kind == "m2":
        in_rows = state["decoder.layers.0.mixer.in_proj.weight"].shape[0]
        conv_dim = state["decoder.layers.0.mixer.conv1d.weight"].shape[0]
        nheads = state["decoder.layers.0.mixer.A_log"].shape[0]
        d_inner = in_rows - conv_dim - nheads
        d_xb = (conv_dim - d_inner) // 2
        edges = [0, d_inner, d_inner + d_xb, d_inner + 2 * d_xb, 2 * d_inner + 2 * d_xb, in_rows]
        layout.m2_in = dict(zip(("z", "x", "B", "C", "dt"), zip(edges[:-1], edges[1:])))
        layout.m2_conv = {"x": (0, d_xb), "B": (d_xb, 2 * d_xb), "C": (2 * d_xb, conv_dim)}
    elif kind == "mla":
        rope = _arg(args, "qk_pos_emb_head_dim")
        nope = _arg(args, "qk_head_dim")
        down_rows = state["decoder.layers.0.self_attention.linear_kv_down_proj.weight"].shape[0]
        if rope is None or nope is None:
            raise ValueError(
                "MLA comparison needs qk_pos_emb_head_dim and qk_head_dim in the checkpoint args"
            )
        layout.mla_rope, layout.mla_nope = int(rope), int(nope)
        layout.mla_rank = down_rows - layout.mla_rope
    else:
        raise ValueError(f"kind must be m2 or mla, got {kind!r}")
    return layout


def pairs(state: Dict[str, torch.Tensor], layout: Layout) -> Pairs:
    """``(reference name, Megatron key, row range or None)`` for every mapped student tensor."""
    out: Pairs = []

    def take(hf, key, rows=None):
        out.append((hf, key, rows))

    take("model.embed_tokens.weight", "embedding.word_embeddings.weight")
    take("model.norm.weight", _first(state, "decoder.final_norm.weight", "decoder.final_layernorm.weight"))
    take("lm_head.weight", _first(state, "output_layer.weight", "embedding.word_embeddings.weight"))
    for i in range(layout.num_layers):
        a, m, h = f"decoder.layers.{2 * i}.", f"decoder.layers.{2 * i + 1}.", f"model.layers.{i}."
        take(h + "mlp.gate_proj.weight", m + "mlp.linear_fc1.weight", (0, layout.ffn))
        take(h + "mlp.up_proj.weight", m + "mlp.linear_fc1.weight", (layout.ffn, 2 * layout.ffn))
        take(h + "mlp.down_proj.weight", m + "mlp.linear_fc2.weight")
        take(
            h + "post_attention_layernorm.weight",
            _first(state, m + "mlp.linear_fc1.layer_norm_weight", m + "pre_mlp_layernorm.weight"),
        )
        if layout.kind == "m2":
            take(
                h + "input_layernorm.weight",
                _first(
                    state,
                    a + "mixer.in_proj.layer_norm_weight",
                    a + "norm.weight",
                    a + "input_layernorm.weight",
                ),
            )
            for name in (
                "in_proj.weight",
                "conv1d.weight",
                "conv1d.bias",
                "A_log",
                "D",
                "dt_bias",
                "norm.weight",
                "out_proj.weight",
            ):
                take(h + "mamba." + name, a + "mixer." + name)
        else:
            take(
                h + "input_layernorm.weight",
                _first(
                    state,
                    a + "input_layernorm.weight",
                    a + "self_attention.linear_q_down_proj.layer_norm_weight",
                ),
            )
            for meg, hf in (
                ("linear_q_down_proj", "q_a_proj"),
                ("linear_q_up_proj", "q_b_proj"),
                ("linear_kv_down_proj", "kv_a_proj_with_mqa"),
                ("linear_kv_up_proj", "kv_b_proj"),
                ("linear_proj", "out_proj"),
            ):
                take(h + f"mla.{hf}.weight", a + f"self_attention.{meg}.weight")
    return out


def to_reference_names(
    state: Dict[str, torch.Tensor], layout: Layout
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """The student's tensors under reference names, and the Megatron keys left unmapped."""
    out, used = {}, set()
    for hf, key, rows in pairs(state, layout):
        used.add(key)
        out[hf] = state[key] if rows is None else state[key][rows[0] : rows[1]]
    return out, sorted(set(state) - used)


def read_safetensors(path: Path) -> Dict[str, torch.Tensor]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        return {key: f.get_tensor(key) for key in f.keys()}


def load_student(path: Path, iteration: Optional[int] = None):
    """``(state, iteration, file, args)`` of a legacy Megatron checkpoint, without ``_extra_state``."""
    from primus.backends.megatron.checkpoint.hybrid_upcycle import load_checkpoint

    path = Path(path)
    if iteration is not None:
        path = path / f"iter_{int(iteration):07d}"
    state, envelope = load_checkpoint(path)
    state = {k: v for k, v in state.items() if torch.is_tensor(v) and not k.endswith("_extra_state")}
    return state, envelope.get("iteration"), str(path), envelope.get("args")


def derive(t: Dict[str, torch.Tensor], layout: Layout) -> Dict[str, torch.Tensor]:
    """Mamba-2 ``in_proj`` and convolution sections; MLA products and the RoPE rows of ``kv_a``."""
    out = {}
    for i in range(layout.num_layers):
        h = f"model.layers.{i}."
        if layout.kind == "m2":
            w, c = t[h + "mamba.in_proj.weight"], t[h + "mamba.conv1d.weight"]
            for name, (a, b) in layout.m2_in.items():
                out[h + f"mamba.in_proj.weight[{name}]"] = w[a:b]
            for name, (a, b) in layout.m2_conv.items():
                out[h + f"mamba.conv1d.weight[{name}]"] = c[a:b]
        else:
            qa, qb = t[h + "mla.q_a_proj.weight"].float(), t[h + "mla.q_b_proj.weight"].float()
            kva, kvb = t[h + "mla.kv_a_proj_with_mqa.weight"].float(), t[h + "mla.kv_b_proj.weight"].float()
            out[h + "mla.q_b@q_a"] = qb @ qa
            out[h + "mla.kv_b@kv_a[latent]"] = kvb @ kva[: layout.mla_rank]
            out[h + "mla.kv_a[rope]"] = kva[layout.mla_rank :]
    return out


def dense_targets(dense: Dict[str, torch.Tensor], layout: Layout, kv_heads: int, head_dim: int):
    """What the upcycler derives each student tensor from, under the student's reference names."""
    out = {
        "model.embed_tokens.weight": dense["model.embed_tokens.weight"],
        "lm_head.weight": dense.get("lm_head.weight", dense["model.embed_tokens.weight"]),
        "model.norm.weight": dense["model.norm.weight"],
    }
    for i in range(layout.num_layers):
        h = f"model.layers.{i}."
        for name in (
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "post_attention_layernorm.weight",
            "input_layernorm.weight",
        ):
            out[h + name] = dense[h + name]
        q, k, v, o = (dense[h + f"self_attn.{p}_proj.weight"] for p in "qkvo")
        if layout.kind == "m2":
            out[h + "mamba.in_proj.weight[x]"], out[h + "mamba.in_proj.weight[B]"] = v, k
            out[h + "mamba.in_proj.weight[C]"], out[h + "mamba.out_proj.weight"] = q, o
        else:
            kh, vh = k.float().view(kv_heads, head_dim, -1), v.float().view(kv_heads, head_dim, -1)
            out[h + "mla.q_b@q_a"] = q
            out[h + "mla.kv_b@kv_a[latent]"] = torch.cat((kh[:, : layout.mla_nope], vh), dim=1).reshape(
                -1, k.shape[-1]
            )
            out[h + "mla.kv_a[rope]"] = kh.mean(0)[-layout.mla_rope :]
            out[h + "mla.out_proj.weight"] = o
    return out


def align_mla(t: Dict[str, torch.Tensor], ref: Dict[str, torch.Tensor], layout: Layout):
    """Flip each SVD direction of ``t``'s MLA factors to agree in sign with ``ref``."""
    out, flips = dict(t), {}
    for i in range(layout.num_layers):
        h = f"model.layers.{i}."
        for a_name, b_name, rows in (
            ("mla.q_a_proj.weight", "mla.q_b_proj.weight", None),
            ("mla.kv_a_proj_with_mqa.weight", "mla.kv_b_proj.weight", layout.mla_rank),
        ):
            a, b, ra = t[h + a_name].float(), t[h + b_name].float(), ref[h + a_name].float()
            n = rows or a.shape[0]
            sign = torch.sign((a[:n] * ra[:n]).sum(1))
            sign[sign == 0] = 1
            a = a.clone()
            a[:n] *= sign[:, None]
            out[h + a_name], out[h + b_name] = a, b * sign[None, :]
            flips[h + a_name] = int((sign < 0).sum())
    return out, flips


def cos_rel(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    a, b = a.double().flatten(), b.double().flatten()
    na, nb = a.norm().item(), b.norm().item()
    cos = (a @ b).item() / (na * nb) if na and nb else float("nan")
    rel = (a - b).norm().item() / nb if nb else float("nan")
    return cos, rel


def _stats(t: torch.Tensor) -> Dict[str, float]:
    t = t.double().flatten()
    return {
        "mean": t.mean().item(),
        "std": t.std().item() if t.numel() > 1 else 0.0,
        "min": t.min().item(),
        "max": t.max().item(),
    }


def random_init_names(shared: bool) -> Callable[[str], bool]:
    """Names whose values are a random draw, unless the student's init carries the reference's."""

    def check(name: str) -> bool:
        if shared:
            return False
        return ".conv1d." in name or name.endswith(
            ("in_proj.weight[z]", "in_proj.weight[dt]", "A_log", "dt_bias")
        )

    return check


def compare(ours, init, ref, dense_t, is_random: Callable[[str], bool]) -> Dict[str, dict]:
    rows = {}
    for name in sorted(ref):
        if name not in ours:
            rows[name] = {"missing": True}
            continue
        o, r, i0 = ours[name], ref[name], init.get(name)
        if tuple(o.shape) != tuple(r.shape):
            rows[name] = {"shape": [list(o.shape), list(r.shape)]}
            continue
        row: Dict[str, object] = {}
        if is_random(name):
            row["random_init"] = True
            row["ours"], row["ref"], row["init"] = _stats(o), _stats(r), _stats(i0)
        else:
            row["cos"], row["rel"] = cos_rel(o, r)
            row["init_cos"], row["init_rel"] = cos_rel(i0, r)
            row["move_ours"] = cos_rel(o, i0)[1]
            row["move_ref"] = cos_rel(r, i0)[1]
            d_o, d_r = o.double() - i0.double(), r.double() - i0.double()
            nonzero = d_o.norm() > 0 and d_r.norm() > 0
            row["upd_cos"] = cos_rel(d_o, d_r)[0] if nonzero else float("nan")
            row["upd_ratio"] = (d_o.norm() / d_r.norm()).item() if d_r.norm() > 0 else float("nan")
        if name in dense_t and tuple(dense_t[name].shape) == tuple(o.shape):
            row["init_vs_dense"] = cos_rel(i0, dense_t[name])[1]
        rows[name] = row
    return rows


def group(name: str) -> str:
    parts = name.split(".")
    return ".".join(parts[3:]) if name.startswith("model.layers.") else name


def summarize(rows: Dict[str, dict]) -> Dict[str, dict]:
    groups: Dict[str, List[dict]] = {}
    for name, row in rows.items():
        groups.setdefault(group(name), []).append(row)
    keys = ("cos", "rel", "init_cos", "move_ours", "move_ref", "upd_cos", "upd_ratio", "init_vs_dense")
    out = {}
    for g, rs in sorted(groups.items()):
        s: Dict[str, object] = {"n": len(rs)}
        for k in keys:
            values = [r[k] for r in rs if k in r and not math.isnan(r[k])]
            if values:
                s[k] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
        if any(r.get("random_init") for r in rs):
            s["random_init"] = True
            for who in ("ours", "ref", "init"):
                s[who + "_std"] = statistics.median(r[who]["std"] for r in rs if who in r)
        out[g] = s
    return out


def write_reference_as_megatron(template: Path, reference: Path, kind: str, output_dir: Path) -> int:
    """Copy the student checkpoint ``template`` with every mapped tensor set to the reference's."""
    from primus.backends.megatron.checkpoint.hybrid_upcycle import (
        build_upcycle_checkpoint,
        load_checkpoint,
    )

    state, envelope = load_checkpoint(Path(template))
    tensors = {k: v for k, v in state.items() if torch.is_tensor(v) and not k.endswith("_extra_state")}
    layout = student_layout(tensors, kind, envelope.get("args"))
    ref = read_safetensors(Path(reference))
    written = set()
    for hf, key, rows in pairs(tensors, layout):
        value = ref[hf]
        target = state[key] if rows is None else state[key][rows[0] : rows[1]]
        if tuple(target.shape) != tuple(value.shape):
            raise ValueError(f"{key} {tuple(target.shape)} <- {hf} {tuple(value.shape)}")
        if rows is None:
            state[key] = value.to(state[key].dtype).clone()
        else:
            if key not in written:
                state[key] = state[key].clone()
            state[key][rows[0] : rows[1]] = value.to(state[key].dtype)
        written.add(key)
    left = sorted(k for k in tensors if k not in written)
    if left:
        raise ValueError(f"student tensors without a reference value: {left}")

    output_dir = Path(output_dir)
    iter_dir = output_dir / "iter_0000000" / "mp_rank_00"
    iter_dir.mkdir(parents=True, exist_ok=True)
    torch.save(build_upcycle_checkpoint(envelope, state), iter_dir / "model_optim_rng.pt")
    (output_dir / "latest_checkpointed_iteration.txt").write_text("0\n")
    return len(written)


def _dense_heads(dense_path: Path, config_path: Optional[Path]) -> Tuple[int, int]:
    config_path = config_path or Path(dense_path).with_name("config.json")
    config = json.loads(Path(config_path).read_text())
    head_dim = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    return int(config["num_key_value_heads"]), int(head_dim)


def compare_main(argv: Optional[Sequence[str]] = None) -> dict:
    p = argparse.ArgumentParser(
        description="Compare an Enhanced-ILD student with its reference, tensor by tensor."
    )
    p.add_argument("--kind", choices=("m2", "mla"), required=True)
    p.add_argument("--ours", required=True, help="the student's checkpoint directory")
    p.add_argument("--iteration", type=int, help="iteration under --ours (default: the latest)")
    p.add_argument("--init", required=True, help="the upcycled checkpoint the student started from")
    p.add_argument("--reference", required=True, help="the reference model.safetensors")
    p.add_argument("--dense", help="the teacher's model.safetensors, to check the upcycle (init vs dense)")
    p.add_argument("--dense-config", help="the teacher's config.json (default: next to --dense)")
    p.add_argument(
        "--shared-random-init",
        action="store_true",
        help="the init carries the reference's random Mamba-2 draw (write_mamba2_reference_init.py)",
    )
    p.add_argument("--json", help="write every tensor's numbers here")
    a = p.parse_args(argv)

    ours_sd, ours_iter, ours_file, args = load_student(Path(a.ours), a.iteration)
    init_sd, _, init_file, _ = load_student(Path(a.init))
    layout = student_layout(ours_sd, a.kind, args)
    ours, left = to_reference_names(ours_sd, layout)
    init, _ = to_reference_names(init_sd, layout)
    ref = read_safetensors(Path(a.reference))
    print(f"ours: {ours_file} (iteration {ours_iter}); init: {init_file}")
    print(f"reference tensors {len(ref)}, mapped student tensors {len(ours)}, unmapped Megatron keys {left}")
    print(
        f"in reference only: {sorted(set(ref) - set(ours))[:8]}; in student only: {sorted(set(ours) - set(ref))[:8]}"
    )

    flips = {}
    if a.kind == "mla":
        ours, flips = align_mla(ours, ref, layout)
        init, _ = align_mla(init, ref, layout)
    for t in (ours, init, ref):
        t.update(derive(t, layout))
    dense_t = {}
    if a.dense:
        kv_heads, head_dim = _dense_heads(Path(a.dense), Path(a.dense_config) if a.dense_config else None)
        dense_t = dense_targets(read_safetensors(Path(a.dense)), layout, kv_heads, head_dim)
    rows = compare(ours, init, ref, dense_t, random_init_names(a.shared_random_init))
    summary = summarize(rows)

    def fmt(s, k):
        return f"{s[k]['median']:.4f}" if k in s else "   -  "

    print(
        f"\n{'tensor (median over layers)':42s} {'n':>3s} {'cos':>7s} {'min cos':>7s} {'rel':>7s} "
        f"{'initcos':>7s} {'mv ours':>7s} {'mv ref':>7s} {'upd cos':>7s} {'upd o/r':>7s} {'init-dense':>10s}"
    )
    for g, s in summary.items():
        if s.get("random_init"):
            print(
                f"{g:42s} {s['n']:3d}  random init: std ours {s['ours_std']:.4f} ref {s['ref_std']:.4f} "
                f"init {s['init_std']:.4f}"
            )
            continue
        low = f"{s['cos']['min']:.4f}" if "cos" in s else "   -  "
        print(
            f"{g:42s} {s['n']:3d} {fmt(s, 'cos'):>7s} {low:>7s} {fmt(s, 'rel'):>7s} {fmt(s, 'init_cos'):>7s} "
            f"{fmt(s, 'move_ours'):>7s} {fmt(s, 'move_ref'):>7s} {fmt(s, 'upd_cos'):>7s} "
            f"{fmt(s, 'upd_ratio'):>7s} {fmt(s, 'init_vs_dense'):>10s}"
        )
    key = "mamba.in_proj.weight[x]" if a.kind == "m2" else "mla.q_b@q_a"
    print(f"\nper layer, {key} and mlp.down_proj.weight: cos / rel vs reference")
    for i in range(layout.num_layers):
        r1, r2 = rows[f"model.layers.{i}.{key}"], rows[f"model.layers.{i}.mlp.down_proj.weight"]
        print(
            f"  L{i:02d} {r1.get('cos', float('nan')):.4f} / {r1.get('rel', float('nan')):.4f}   "
            f"{r2.get('cos', float('nan')):.4f} / {r2.get('rel', float('nan')):.4f}"
        )
    result = {
        "kind": a.kind,
        "ours": ours_file,
        "iteration": ours_iter,
        "init": init_file,
        "reference": a.reference,
        "flips": flips,
        "summary": summary,
        "tensors": rows,
    }
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1))
        print("wrote", a.json)
    return result


def convert_main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Write a Zebra-Llama reference student as a Megatron checkpoint.")
    p.add_argument("--kind", choices=("m2", "mla"), required=True)
    p.add_argument(
        "--template", required=True, help="a student checkpoint of the same layout (the upcycled init)"
    )
    p.add_argument("--reference", required=True, help="the reference model.safetensors")
    p.add_argument("--output-dir", required=True)
    a = p.parse_args(argv)
    count = write_reference_as_megatron(Path(a.template), Path(a.reference), a.kind, Path(a.output_dir))
    print(f"wrote {Path(a.output_dir) / 'iter_0000000'}: {count} tensors from {a.reference}")
    return count
