###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Convert Diffusers-format transformer checkpoints to the native (official) layouts.

These invert the key mappings in Diffusers' ``convert_hunyuan_video_to_diffusers.py`` and
``convert_wan_to_diffusers.py``. Tensors are renamed, re-fused, or re-ordered.
Diffusers-only parameters with no native counterpart, such as Wan's
``attn2.norm_added_q.weight``, are dropped.
"""

from __future__ import annotations

import re

import torch


def _rename(key: str, rules: list[tuple[str, str]]) -> str:
    for pattern, replacement in rules:
        key = re.sub(pattern, replacement, key)
    return key


def is_diffusers_hunyuan(state: dict[str, torch.Tensor]) -> bool:
    return "x_embedder.proj.weight" in state


_HUNYUAN_RULES = [
    (r"^x_embedder\.", "img_in."),
    (r"^time_text_embed\.timestep_embedder\.linear_1\.", "time_in.mlp.0."),
    (r"^time_text_embed\.timestep_embedder\.linear_2\.", "time_in.mlp.2."),
    (r"^time_text_embed\.guidance_embedder\.linear_1\.", "guidance_in.mlp.0."),
    (r"^time_text_embed\.guidance_embedder\.linear_2\.", "guidance_in.mlp.2."),
    (r"^time_text_embed\.text_embedder\.linear_1\.", "vector_in.in_layer."),
    (r"^time_text_embed\.text_embedder\.linear_2\.", "vector_in.out_layer."),
    (r"^context_embedder\.time_text_embed\.timestep_embedder\.linear_1\.", "txt_in.t_embedder.mlp.0."),
    (r"^context_embedder\.time_text_embed\.timestep_embedder\.linear_2\.", "txt_in.t_embedder.mlp.2."),
    (r"^context_embedder\.time_text_embed\.text_embedder\.", "txt_in.c_embedder."),
    (r"^context_embedder\.proj_in\.", "txt_in.input_embedder."),
    (r"^context_embedder\.token_refiner\.refiner_blocks\.", "txt_in.individual_token_refiner.blocks."),
    (r"(individual_token_refiner\.blocks\.\d+)\.attn\.to_out\.0\.", r"\1.self_attn_proj."),
    (r"(individual_token_refiner\.blocks\.\d+)\.ff\.net\.0\.proj\.", r"\1.mlp.fc1."),
    (r"(individual_token_refiner\.blocks\.\d+)\.ff\.net\.2\.", r"\1.mlp.fc2."),
    (r"(individual_token_refiner\.blocks\.\d+)\.norm_out\.linear\.", r"\1.adaLN_modulation.1."),
    (r"^transformer_blocks\.", "double_blocks."),
    (r"(double_blocks\.\d+)\.norm1\.linear\.", r"\1.img_mod.linear."),
    (r"(double_blocks\.\d+)\.norm1_context\.linear\.", r"\1.txt_mod.linear."),
    (r"(double_blocks\.\d+)\.attn\.norm_q\.", r"\1.img_attn_q_norm."),
    (r"(double_blocks\.\d+)\.attn\.norm_k\.", r"\1.img_attn_k_norm."),
    (r"(double_blocks\.\d+)\.attn\.to_out\.0\.", r"\1.img_attn_proj."),
    (r"(double_blocks\.\d+)\.attn\.norm_added_q\.", r"\1.txt_attn_q_norm."),
    (r"(double_blocks\.\d+)\.attn\.norm_added_k\.", r"\1.txt_attn_k_norm."),
    (r"(double_blocks\.\d+)\.attn\.to_add_out\.", r"\1.txt_attn_proj."),
    (r"(double_blocks\.\d+)\.ff\.net\.0\.proj\.", r"\1.img_mlp.fc1."),
    (r"(double_blocks\.\d+)\.ff\.net\.2\.", r"\1.img_mlp.fc2."),
    (r"(double_blocks\.\d+)\.ff_context\.net\.0\.proj\.", r"\1.txt_mlp.fc1."),
    (r"(double_blocks\.\d+)\.ff_context\.net\.2\.", r"\1.txt_mlp.fc2."),
    (r"^single_transformer_blocks\.", "single_blocks."),
    (r"(single_blocks\.\d+)\.attn\.norm_q\.", r"\1.q_norm."),
    (r"(single_blocks\.\d+)\.attn\.norm_k\.", r"\1.k_norm."),
    (r"(single_blocks\.\d+)\.norm\.linear\.", r"\1.modulation.linear."),
    (r"(single_blocks\.\d+)\.proj_out\.", r"\1.linear2."),
    (r"^proj_out\.", "final_layer.linear."),
]


def _swap_halves(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor.chunk(2, dim=0)
    return torch.cat([second, first], dim=0)


def hunyuan_diffusers_to_native(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    fused: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        if key.startswith("norm_out.linear."):
            # Diffusers stores the final AdaLN as (scale, shift); HunyuanVideo uses (shift, scale).
            out["final_layer.adaLN_modulation.1." + key.rsplit(".", 1)[1]] = _swap_halves(value)
            continue
        native = _rename(key, _HUNYUAN_RULES)
        match = re.match(
            r"^(double_blocks\.\d+|single_blocks\.\d+|txt_in\.individual_token_refiner\.blocks\.\d+)\."
            r"(?:attn\.)?(to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj|proj_mlp)\.(weight|bias)$",
            native,
        )
        if match is None:
            out[native] = value
            continue
        prefix, part, param = match.groups()
        if prefix.startswith("double_blocks"):
            target = "img_attn_qkv" if part.startswith("to_") else "txt_attn_qkv"
        elif prefix.startswith("single_blocks"):
            target = "linear1"
        else:
            target = "self_attn_qkv"
        fused.setdefault((f"{prefix}.{target}", param), {})[part] = value
    for (target, param), parts in fused.items():
        if target.endswith("linear1"):
            order = ("to_q", "to_k", "to_v", "proj_mlp")
        elif target.endswith("txt_attn_qkv"):
            order = ("add_q_proj", "add_k_proj", "add_v_proj")
        else:
            order = ("to_q", "to_k", "to_v")
        out[f"{target}.{param}"] = torch.cat([parts[name] for name in order], dim=0)
    return out


def is_diffusers_wan(state: dict[str, torch.Tensor]) -> bool:
    return "condition_embedder.time_proj.weight" in state


_WAN_RULES = [
    (r"^condition_embedder\.time_embedder\.linear_1\.", "time_embedding.0."),
    (r"^condition_embedder\.time_embedder\.linear_2\.", "time_embedding.2."),
    (r"^condition_embedder\.text_embedder\.linear_1\.", "text_embedding.0."),
    (r"^condition_embedder\.text_embedder\.linear_2\.", "text_embedding.2."),
    (r"^condition_embedder\.time_proj\.", "time_projection.1."),
    (r"^condition_embedder\.image_embedder\.norm1\.", "img_emb.proj.0."),
    (r"^condition_embedder\.image_embedder\.ff\.net\.0\.proj\.", "img_emb.proj.1."),
    (r"^condition_embedder\.image_embedder\.ff\.net\.2\.", "img_emb.proj.3."),
    (r"^condition_embedder\.image_embedder\.norm2\.", "img_emb.proj.4."),
    (r"^scale_shift_table$", "head.modulation"),
    (r"^proj_out\.", "head.head."),
    (r"\.scale_shift_table$", ".modulation"),
    (r"\.attn1\.to_q\.", ".self_attn.q."),
    (r"\.attn1\.to_k\.", ".self_attn.k."),
    (r"\.attn1\.to_v\.", ".self_attn.v."),
    (r"\.attn1\.to_out\.0\.", ".self_attn.o."),
    (r"\.attn1\.norm_", ".self_attn.norm_"),
    (r"\.attn2\.to_q\.", ".cross_attn.q."),
    (r"\.attn2\.to_k\.", ".cross_attn.k."),
    (r"\.attn2\.to_v\.", ".cross_attn.v."),
    (r"\.attn2\.to_out\.0\.", ".cross_attn.o."),
    (r"\.attn2\.add_k_proj\.", ".cross_attn.k_img."),
    (r"\.attn2\.add_v_proj\.", ".cross_attn.v_img."),
    (r"\.attn2\.norm_added_k\.", ".cross_attn.norm_k_img."),
    (r"\.attn2\.norm_", ".cross_attn.norm_"),
    (r"\.norm2\.", ".norm3."),
    (r"\.ffn\.net\.0\.proj\.", ".ffn.0."),
    (r"\.ffn\.net\.2\.", ".ffn.2."),
]


def wan_diffusers_to_native(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        # Diffusers keeps an (unused) Q norm for the image branch of cross attention.
        if key.endswith(".attn2.norm_added_q.weight"):
            continue
        out[_rename(key, _WAN_RULES)] = value
    return out
