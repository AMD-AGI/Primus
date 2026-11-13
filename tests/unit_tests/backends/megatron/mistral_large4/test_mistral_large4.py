###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mistral Large 4 (``mistral_large4_text``): numerical tests of the port.

The helper tests (YaRN frequencies, config helpers), the KV-projection backward test and the
converted-checkpoint load test (allgather dispatcher) run on CPU; the model tests marked
``requires_gpu`` use the alltoall dispatcher and skip without a GPU (they run in the port's GPU
checks).

* the YaRN + LeRoPE inverse frequencies against the transformers formula;
* a tiny model (every per-layer kind: dense layer 0, MoE, NoPE layer, KV-sharing consumer,
  shared-indexer layers) built with the port's layer specs (local backend), loaded through the
  converter's mapping from a synthetic consolidated checkpoint, against an eager fp32
  re-implementation of the transformers reference forward (``modeling_mistral_large4.py``),
  for a short sequence (dense causal) and one past ``index_topk`` (sparse DSA path);
(Block-FP8 dequantisation and the converter's other CPU checks are in
``test_mistral_large4_patches.py``.)

* the gradient reaching the KV-source layer includes its consumer's share: compared with a run
  where only the latent the consumer receives is detached (same loss), eager and under full
  recompute. The PP crossing of the latent is tested on CPU in ``test_mistral_large4_patches.py``
  (carrier split / join, empty stage).
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from primus.backends.megatron.checkpoint import (
    native_loader_mistral_large4_text as conv,
)
from primus.backends.megatron.core.models.mistral_large4.mistral_large4_transformer_config import (
    MistralLarge4TransformerConfig,
)
from primus.backends.megatron.core.transformer.mistral_large4.rotary import (
    yarn_inv_freq,
)

H, NH, NOPE, ROPE, V, LAT, NL = 64, 4, 8, 8, 16, 16, 6
IH, ID, TOPK, E, TOP, FF, MFF, VOCAB = 2, 16, 8, 4, 2, 48, 24, 96
NO_ROPE = [1, 1, 1, 0, 1, 1]
IDX = ["full", "full", "shared", "full", "full", "shared"]
KVSRC = [0, 1, 2, 3, 4, 3]
NO_ROPE[5] = 0  # consumer of layer 3 (NoPE) must be NoPE too


# ---------------------------------------------------------------------------------------------
# eager reference (transformers math, fp32)
# ---------------------------------------------------------------------------------------------
def hf_yarn_inv_freq(dim, base, factor, orig, bf, bs):
    pos = base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
    extra, inter = 1.0 / pos, 1.0 / (factor * pos)

    def corr(n):
        return (dim * math.log(orig / (n * 2 * math.pi))) / (2 * math.log(base))

    low, high = max(math.floor(corr(bf)), 0), min(math.ceil(corr(bs)), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    f = 1 - ramp
    return inter * (1 - f) + extra * f


def rms(x, w, eps=1e-5):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def rope_il(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def ssmax(q, pos, s, b):  # q [T, heads, d]
    lp = (pos + 1).float().log()
    return q * (s[None, :] * lp[:, None] + b[None, :])[..., None]


def reference_forward(W, ids, topk=TOPK):
    T = ids.numel()
    pos = torch.arange(T)
    inv = hf_yarn_inv_freq(ROPE, 10000.0, 2048.0, 512, 32.0, 1.0) * W["lerope.alpha"].exp()
    ang = pos[:, None].float() * inv[None]
    cos, sin = ang.cos(), ang.sin()
    x = W["tok_embeddings.weight"][ids]
    mask = torch.full((T, T), float("-inf")).triu(1)
    shared, prev_topk = {}, None
    for L in range(NL):
        p = f"layers.{L}."
        a = p + "attention."
        use_rope = NO_ROPE[L]
        r = ROPE if use_rope else 0
        nope = NOPE + (0 if use_rope else ROPE)
        hn = rms(x, W[p + "attention_norm.weight"])
        q = (hn @ W[a + "wq.weight"].T).view(T, NH, NOPE + ROPE)
        q_pass, q_rot = q[..., :nope], q[..., nope:]
        if KVSRC[L] != L:
            kv_c, k_rot = shared["kv"]
        else:
            ckv = hn @ W[a + "wkv_a_with_mqa.weight"].T
            kv_c, k_rot = ckv[:, :LAT], ckv[:, LAT:]
            kv_c = rms(kv_c, W[a + "kv_a_norm.weight"])
            if r:
                k_rot = rope_il(k_rot, cos, sin)
            if L in KVSRC[L + 1 :]:
                shared["kv"] = (kv_c, k_rot)
        if r:
            q_rot = rope_il(q_rot, cos[:, None], sin[:, None])
        qs = ssmax(torch.cat([q_pass, q_rot], -1), pos, W[a + "ssmax_scaling.s"], W[a + "ssmax_scaling.b"])
        kv = (kv_c @ W[a + "wkv_b.weight"].T).view(T, NH, nope + V)
        k_nope, v = kv[..., :nope], kv[..., nope:]
        k = torch.cat([k_nope, k_rot[:, None].expand(-1, NH, -1)], -1) if r else k_nope
        if IDX[L] == "full":
            i = a + "indexer."
            qi = F.layer_norm(hn, (H,), W[i + "q_input_norm.weight"], W[i + "q_input_norm.bias"], 1e-6)
            qi = (qi @ W[i + "wq_b.weight"].T).view(T, IH, ID)
            qi = ssmax(qi, pos, W[i + "ssmax_scaling.s"], W[i + "ssmax_scaling.b"])
            ki = F.layer_norm(
                hn @ W[i + "wk.weight"].T, (ID,), W[i + "k_norm.weight"], W[i + "k_norm.bias"], 1e-6
            )
            if use_rope:
                qi = torch.cat([rope_il(qi[..., :ROPE], cos[:, None], sin[:, None]), qi[..., ROPE:]], -1)
                ki = torch.cat([rope_il(ki[..., :ROPE], cos, sin), ki[..., ROPE:]], -1)
            sc = F.relu(torch.einsum("thd,sd->ths", qi, ki) * ID**-0.5)
            wts = (hn @ W[i + "weights_proj.weight"].T) * IH**-0.5
            score = (wts[..., None] * sc).sum(1) + mask
            prev_topk = score.topk(min(topk, T), dim=-1).indices
        keep = torch.zeros(T, T, dtype=torch.bool).scatter(-1, prev_topk, True)
        m = mask.masked_fill(~keep, float("-inf"))
        att = torch.einsum("thd,shd->hts", qs, k) * (NOPE + ROPE) ** -0.5 + m
        o = torch.einsum("hts,shd->thd", att.softmax(-1), v).reshape(T, NH * V)
        x = x + o @ W[a + "wo.weight"].T
        hn = rms(x, W[p + "ffn_norm.weight"])

        def mlp(z, pre):
            return (F.silu(z @ W[pre + "w1.weight"].T) * (z @ W[pre + "w3.weight"].T)) @ W[
                pre + "w2.weight"
            ].T

        if L == 0:
            x = x + mlp(hn, p + "feed_forward.")
        else:
            probs = (hn @ W[p + "gate.weight"].T).softmax(-1)
            tw, ti = probs.topk(TOP, -1)
            tw = tw / tw.sum(-1, keepdim=True)
            out = torch.zeros_like(hn)
            for e in range(E):
                sel = ti == e
                wgt = (tw * sel).sum(-1, keepdim=True)
                out = out + wgt * mlp(hn, p + f"experts.{e}.")
            x = x + out + mlp(hn, p + "shared_experts.")
    x = rms(x, W["norm.weight"])
    logits = x @ W["output.weight"].T
    return 30.0 * torch.tanh(logits / 30.0)


def make_weights(seed=0):
    g = torch.Generator().manual_seed(seed)

    def r(*s, scale=0.08):
        return torch.randn(*s, generator=g) * scale

    W = {
        "tok_embeddings.weight": r(VOCAB, H, scale=1.0),
        "norm.weight": 1 + r(H),
        "output.weight": r(VOCAB, H),
        "lerope.alpha": r(ROPE // 2, scale=0.3),
    }
    for L in range(NL):
        p, a = f"layers.{L}.", f"layers.{L}.attention."
        rr = ROPE if NO_ROPE[L] else 0
        nope = NOPE + ROPE - rr
        W[p + "attention_norm.weight"] = 1 + r(H)
        W[p + "ffn_norm.weight"] = 1 + r(H)
        W[a + "wq.weight"] = r(NH * (NOPE + ROPE), H)
        if KVSRC[L] == L:
            W[a + "wkv_a_with_mqa.weight"] = r(LAT + rr, H)
            W[a + "kv_a_norm.weight"] = 1 + r(LAT)
        W[a + "wkv_b.weight"] = r(NH * (nope + V), LAT)
        W[a + "wo.weight"] = r(H, NH * V)
        W[a + "ssmax_scaling.s"] = r(NH, scale=0.3)
        W[a + "ssmax_scaling.b"] = 1 + r(NH)
        if IDX[L] == "full":
            i = a + "indexer."
            W[i + "q_input_norm.weight"] = 1 + r(H)
            # positive indexer q and k (q.k > 0): no relu-zero ties in the top-k, so the selected key
            # set is unique and the port's and the reference's top-k agree exactly
            W[i + "q_input_norm.bias"] = 2.0 + r(H)
            W[i + "wq_b.weight"] = 0.3 + r(IH * ID, H, scale=0.1)
            W[i + "wk.weight"] = r(ID, H, scale=0.5)
            W[i + "k_norm.weight"] = 1 + r(ID)
            W[i + "k_norm.bias"] = 3.0 + r(ID)
            W[i + "weights_proj.weight"] = r(IH, H, scale=0.5).abs()
            W[i + "ssmax_scaling.s"] = r(IH, scale=0.3)
            W[i + "ssmax_scaling.b"] = 1 + r(IH)
        if L == 0:
            for w, shape in (("w1", (FF, H)), ("w3", (FF, H)), ("w2", (H, FF))):
                W[p + f"feed_forward.{w}.weight"] = r(*shape)
        else:
            W[p + "gate.weight"] = r(E, H, scale=0.5)
            for e in range(E):
                for w, shape in (("w1", (MFF, H)), ("w3", (MFF, H)), ("w2", (H, MFF))):
                    W[p + f"experts.{e}.{w}.weight"] = r(*shape)
            for w, shape in (("w1", (MFF, H)), ("w3", (MFF, H)), ("w2", (H, MFF))):
                W[p + f"shared_experts.{w}.weight"] = r(*shape)
    return W


class DictStore:
    """A ``SafetensorsStore`` stand-in over a dict (the converter's ``store.get`` / ``all_keys``)."""

    def __init__(self, d):
        self.d = d

    def get(self, name, dtype):
        return self.d[name].to(dtype)

    def has(self, name):
        return name in self.d

    def all_keys(self):
        return set(self.d)


def build_config():
    return MistralLarge4TransformerConfig(
        num_layers=NL,
        hidden_size=H,
        ffn_hidden_size=FF,
        num_attention_heads=NH,
        kv_channels=V,
        q_lora_rank=None,
        kv_lora_rank=LAT,
        qk_head_dim=NOPE,
        qk_pos_emb_head_dim=ROPE,
        v_head_dim=V,
        rope_type="yarn",
        rotary_base=10000.0,
        rotary_scaling_factor=2048.0,
        original_max_position_embeddings=512,
        beta_fast=32.0,
        beta_slow=1.0,
        mscale=1.0,
        mscale_all_dim=0.0,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_moe_experts=E,
        moe_router_topk=TOP,
        moe_ffn_hidden_size=MFF,
        moe_shared_expert_intermediate_size=MFF,
        moe_router_score_function="softmax",
        moe_router_pre_softmax=False,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_router_dtype="fp32",
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=False,
        first_k_dense_replace=1,
        no_rope_layers=NO_ROPE,
        indexer_types=IDX,
        kv_source_layers=KVSRC,
        index_n_heads=IH,
        index_head_dim=ID,
        index_topk=TOPK,
        final_logit_softcapping=30.0,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        transformer_impl="local",
        bias_dropout_fusion=False,
        masked_softmax_fusion=False,
        attention_softmax_in_fp32=True,
        apply_rope_fusion=False,
        persist_layer_norm=False,
    )


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Megatron's MoE dispatcher needs a GPU"
)


def _te_test_env():
    """A MonkeyPatch with TE's CPU-offload context off and the image's NVTE_* attention
    selections removed (the image bakes NVTE_FLASH_ATTN=0); call ``.undo()`` after."""
    import megatron.core.transformer.transformer_block as tb

    mp = pytest.MonkeyPatch()
    mp.setattr(tb, "get_cpu_offload_context", None)  # TE's offload context is not needed here
    for name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        mp.delenv(name, raising=False)
    return mp


@pytest.fixture(scope="module")
def model_and_weights():

    mp = _te_test_env()
    cleanup = _parallel_state_for_cpu()
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(1234)
    from primus.backends.megatron.core.models.mistral_large4.mistral_large4_builders import (
        build_mistral_large4_model,
    )

    torch.manual_seed(0)
    cfg = build_config()
    cfg.use_cpu_initialization = False
    model = build_mistral_large4_model(cfg, vocab_size=VOCAB, max_sequence_length=4096).cuda().eval()
    W = make_weights()
    margs = SimpleNamespace(
        params_dtype=torch.float32, num_layers=NL, num_experts=E, moe_layer_freq=[0] + [1] * (NL - 1)
    )
    sd, consumed = conv.build_mcore_state_dict(
        model, DictStore(W), margs, {"_num_shared_experts": 1}, lambda *a: None
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "_extra_state" not in k], missing
    assert not unexpected, unexpected
    assert consumed == set(W), sorted(set(W) - consumed)
    yield model, W
    mp.undo()
    cleanup()


def run_port(model, ids):
    ids = ids.cuda()
    pos = torch.arange(ids.numel(), device="cuda")[None]
    with torch.no_grad():
        out = model(ids[None], pos, attention_mask=None)
    return out[0].float().cpu()


def test_yarn_inv_freq_matches_transformers():
    a = yarn_inv_freq(64, 10000.0, 2048.0, 512, 32.0, 1.0)
    b = hf_yarn_inv_freq(64, 10000.0, 2048.0, 512, 32.0, 1.0)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_config_per_layer_helpers():
    cfg = build_config()
    assert [cfg.index_source_layer(i) for i in range(NL)] == [0, 1, 1, 3, 4, 4]
    assert cfg.stores_kv(3) and not cfg.stores_kv(4) and cfg.kv_source(5) == 3
    nope = cfg.layer_config(3)
    assert nope.qk_pos_emb_head_dim == 0 and nope.qk_head_dim == NOPE + ROPE
    assert cfg.layer_config(0) is cfg


@requires_gpu
@pytest.mark.parametrize("T", [6, 13])  # 13 > index_topk = 8: the sparse DSA path
def test_logits_match_reference(model_and_weights, T):
    model, W = model_and_weights
    ids = torch.randint(0, VOCAB, (T,), generator=torch.Generator().manual_seed(T))
    ref = reference_forward(W, ids)
    out = run_port(model, ids)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def _source_grad(model, ids, detach_latent=False, recompute=False):
    """Gradient of layer 3's (the KV source's) kv_a weight. With ``detach_latent`` the latent the
    consumer layer 5 receives is detached (everything else, its output and the loss included, is
    unchanged), so the difference of the two gradients is exactly the share that flows back through
    the shared latent."""
    pos = torch.arange(ids.numel(), device="cuda")[None]
    consumer = model.decoder.layers[5].self_attention
    cfg = model.config
    old = (cfg.recompute_granularity, cfg.recompute_method, cfg.recompute_num_layers)
    if recompute:
        cfg.recompute_granularity, cfg.recompute_method, cfg.recompute_num_layers = "full", "uniform", 2
    orig_set = consumer.set_forward_context
    if detach_latent:

        def set_detached(ctx, latent=None):
            orig_set(ctx, None if latent is None else latent.detach())

        consumer.set_forward_context = set_detached
    model.zero_grad(set_to_none=True)
    model.train()
    try:
        loss = model(ids[None], pos, attention_mask=None).float().pow(2).mean()
        loss.backward()
        return loss.detach(), model.decoder.layers[3].self_attention.linear_kv_down_proj.weight.grad.clone()
    finally:
        consumer.set_forward_context = orig_set
        cfg.recompute_granularity, cfg.recompute_method, cfg.recompute_num_layers = old
        model.eval()


@requires_gpu
def test_kv_source_gets_consumer_gradient(model_and_weights):
    model, _ = model_and_weights
    ids = torch.randint(0, VOCAB, (6,), generator=torch.Generator().manual_seed(1)).cuda()
    loss, with_consumer = _source_grad(model, ids)
    loss_d, detached = _source_grad(model, ids, detach_latent=True)
    # same forward: detaching changes nothing but the gradient path through the shared latent
    torch.testing.assert_close(loss_d, loss, rtol=0, atol=0)
    # the consumer's share reaches the source's kv_a projection through the latent it reads
    share = (with_consumer - detached).abs().max()
    assert share > 1e-6 * with_consumer.abs().max(), "no gradient through the shared latent"
    assert not hasattr(model.decoder.layers[5].self_attention, "linear_kv_down_proj")
    # full recompute (chunks of 2 layers: 3 and 5 sit in different chunks, so the latent crosses a
    # checkpoint boundary) gives the same gradient, consumer share included
    _, recomputed = _source_grad(model, ids, recompute=True)
    torch.testing.assert_close(recomputed, with_consumer, rtol=1e-5, atol=1e-6)
    _, recomputed_detached = _source_grad(model, ids, detach_latent=True, recompute=True)
    torch.testing.assert_close(recomputed_detached, detached, rtol=1e-5, atol=1e-6)


def test_backward_kv_proj_on_consumer_layer():
    """Delayed weight-gradient (backward_dw): a KV-consumer layer has no linear_kv_down_proj; the
    override runs only the projections the layer owns (attention.py ``_backward_kv_proj``)."""
    from primus.backends.megatron.core.transformer.mistral_large4.attention import (
        MistralLarge4Attention,
    )

    calls = []
    up = SimpleNamespace(backward_dw=lambda: calls.append("up"))
    down = SimpleNamespace(backward_dw=lambda: calls.append("down"))
    consumer = SimpleNamespace(consumes_kv=True, linear_kv_up_proj=up)
    MistralLarge4Attention._backward_kv_proj(consumer)
    assert calls == ["up"]
    calls.clear()
    owner = SimpleNamespace(consumes_kv=False, linear_kv_up_proj=up, linear_kv_down_proj=down)
    MistralLarge4Attention._backward_kv_proj(owner)
    assert calls == ["up", "down"]


@requires_gpu
def test_mla_up_proj_recompute_matches(model_and_weights):
    """Selective recompute of the MLA up-projections (recompute_modules: mla_up_proj) runs through
    CheckpointWithoutOutput in ``_qkv`` and gives the gradients of the plain run, also on the
    sparse path (T > index_topk) and on the KV-consumer layer."""
    model, _ = model_and_weights
    ids = torch.randint(0, VOCAB, (13,), generator=torch.Generator().manual_seed(2)).cuda()
    pos = torch.arange(ids.numel(), device="cuda")[None]
    attns = [layer.self_attention for layer in model.decoder.layers]

    from megatron.core import tensor_parallel

    real = tensor_parallel.CheckpointWithoutOutput
    events = []  # (kind, checkpoint id): created / checkpointed / discarded / recomputed

    class Spy(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            events.append(("created", id(self)))

        def checkpoint(self, *a, **k):
            events.append(("checkpoint", id(self)))
            return super().checkpoint(*a, **k)

        def discard_output_and_register_recompute(self, *a, **k):
            events.append(("discard", id(self)))
            return super().discard_output_and_register_recompute(*a, **k)

        def _recompute(self, *a, **k):
            events.append(("recompute", id(self)))
            return super()._recompute(*a, **k)

    def grads(recompute, monkeypatch_ctx):
        saved = [a.recompute_up_proj for a in attns]
        for a in attns:
            a.recompute_up_proj = recompute
        model.zero_grad(set_to_none=True)
        model.train()
        events.clear()
        try:
            with monkeypatch_ctx() as mp:
                mp.setattr(tensor_parallel, "CheckpointWithoutOutput", Spy)
                model(ids[None], pos, attention_mask=None).float().pow(2).mean().backward()
            return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}, list(
                events
            )
        finally:
            for a, v in zip(attns, saved):
                a.recompute_up_proj = v
            model.eval()

    (plain, plain_events), (rec, rec_events) = grads(False, pytest.MonkeyPatch.context), grads(
        True, pytest.MonkeyPatch.context
    )
    # the plain run never checkpoints; the recompute run checkpoints the up-projection of every
    # layer once, discards its output after the core attention and recomputes it in backward
    assert plain_events == []
    kinds = [k for k, _ in rec_events]
    n = len(attns)
    assert kinds.count("created") == n and kinds.count("checkpoint") == n, kinds
    assert kinds.count("discard") == n, kinds
    if hasattr(real, "_recompute"):
        assert kinds.count("recompute") == n, kinds
    assert plain.keys() == rec.keys() and any("layers.5.self_attention.linear_kv_up_proj" in k for k in rec)
    for k in plain:
        torch.testing.assert_close(rec[k], plain[k], rtol=1e-5, atol=1e-6)


def _parallel_state_for_cpu():
    """A one-rank gloo world and Megatron's parallel state (kept when a fixture already made them);
    returns a cleanup callable."""
    from megatron.core import parallel_state as ps

    created_pg = not torch.distributed.is_initialized()
    if created_pg:
        import socket

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        torch.distributed.init_process_group(
            "gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=1, rank=0
        )
    created_ps = not ps.model_parallel_is_initialized()
    if created_ps:
        ps.initialize_model_parallel(1, 1)

    def cleanup():
        if created_ps:
            ps.destroy_model_parallel()
        if created_pg:
            torch.distributed.destroy_process_group()

    return cleanup


def test_converted_checkpoint_loads_strictly(tmp_path):
    """The converter's mapping, saved as torch_dist and loaded into a fresh model with Megatron's
    strictest check (StrictHandling.RAISE_ALL: no model tensor missing from the checkpoint, no
    checkpoint tensor left unread) and ``load_state_dict(strict=True)``, gives back every weight
    exactly: dense layer 0, MoE experts, shared expert, router, NoPE layer, KV-sharing consumer
    (no kv_a), shared-indexer layers (no indexer), LeRoPE alpha, embedding and head. CPU only."""
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.validation import StrictHandling

    from primus.backends.megatron.core.models.mistral_large4.mistral_large4_builders import (
        build_mistral_large4_model,
    )

    mp = _te_test_env()
    if not torch.cuda.is_available():  # Megatron's YarnRotaryEmbedding asks for the current CUDA device
        mp.setattr(torch.cuda, "current_device", lambda: "cpu")
        mp.setattr(torch.cuda, "synchronize", lambda *a, **k: None)  # the torch_dist writer's D2H sync
    cleanup = _parallel_state_for_cpu()
    try:
        cfg = build_config()
        # allgather dispatcher: the alltoall one makes a CUDA stream when it is built; the weights and
        # their checkpoint keys do not depend on the dispatcher
        cfg.moe_token_dispatcher_type = "allgather"
        torch.manual_seed(0)
        model = build_mistral_large4_model(cfg, vocab_size=VOCAB, max_sequence_length=4096)
        W = make_weights()
        margs = SimpleNamespace(
            params_dtype=torch.float32, num_layers=NL, num_experts=E, moe_layer_freq=[0] + [1] * (NL - 1)
        )
        sd, consumed = conv.build_mcore_state_dict(
            model, DictStore(W), margs, {"_num_shared_experts": 1}, lambda *a: None
        )
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not [k for k in missing if "_extra_state" not in k] and not unexpected
        assert consumed == set(W)
        want = {k: v.clone() for k, v in model.state_dict().items() if torch.is_tensor(v)}

        ckpt = tmp_path / "torch_dist"
        ckpt.mkdir()
        # the conversion's save: SwiGLU halves written from the model's own memory (no clone)
        conv.save_without_copies(model)
        dist_checkpointing.save(model.sharded_state_dict(), str(ckpt))

        torch.manual_seed(1)  # a different random init: every value must come from the checkpoint
        fresh = build_mistral_large4_model(cfg, vocab_size=VOCAB, max_sequence_length=4096)
        before = {k: v.clone() for k, v in fresh.state_dict().items() if torch.is_tensor(v)}
        loaded = dist_checkpointing.load(
            fresh.sharded_state_dict(), str(ckpt), strict=StrictHandling.RAISE_ALL
        )
        fresh.load_state_dict(loaded, strict=True)
        got = {k: v for k, v in fresh.state_dict().items() if torch.is_tensor(v)}
        assert got.keys() == want.keys()
        for k in want:
            torch.testing.assert_close(got[k], want[k], rtol=0, atol=0, msg=k)
        # the weights did change from the fresh init (a no-op load would also compare equal to itself)
        changed = [k for k in want if not torch.equal(before[k], got[k])]
        assert any("local_experts" in k for k in changed) and "output_layer.weight" in changed
        assert any(".mlp.router.weight" in k for k in changed)
        assert any(".self_attention.indexer." in k for k in changed)

        # a checkpoint that lacks a weight the model has is refused, not silently left at its init
        partial = tmp_path / "partial"
        partial.mkdir()
        sharded = model.sharded_state_dict()
        drop = "decoder.layers.1.mlp.router.weight"
        del sharded[drop]
        dist_checkpointing.save(sharded, str(partial))
        with pytest.raises(Exception, match="Unexpected keys|not found in the checkpoint"):
            dist_checkpointing.load(
                fresh.sharded_state_dict(), str(partial), strict=StrictHandling.RAISE_UNEXPECTED
            )
    finally:
        cleanup()
        mp.undo()


def test_storage_exact_alias_shares_memory():
    """The conversion's torch_dist save hands the writer tensors whose storage is exactly their
    bytes (the writer clones any other CPU view): the gate | up halves of a fused fc1 alias the
    model's memory, same values, no copy; a whole tensor is returned as it is."""
    w = torch.arange(24, dtype=torch.float32).view(6, 4)
    gate, up = torch.chunk(w, 2, dim=0)
    for half in (gate, up):
        a = conv.storage_exact_alias(half)
        assert torch.equal(a, half) and a.shape == half.shape
        assert a.untyped_storage().nbytes() == half.numel() * half.element_size()
        assert a.data_ptr() == half.data_ptr()  # same memory
    w2 = torch.zeros(3)
    assert conv.storage_exact_alias(w2) is w2
    a = conv.storage_exact_alias(up)
    w[3, 0] = -1.0  # a write to the model's tensor shows through the alias
    assert a[0, 0].item() == -1.0


def test_fp32_tensors_stay_fp32_through_a_bf16_cast():
    """Float16Module casts the model to bf16 (``module.bfloat16()``, i.e. ``_apply``): the LeRoPE
    ``alpha`` and the indexer's ``weights_proj`` / SSMax ``s`` / ``b`` keep fp32 and their exact
    values; the indexer's other buffers become bf16."""
    from primus.backends.megatron.core.transformer.mistral_large4.attention import (
        MistralLarge4Indexer,
    )
    from primus.backends.megatron.core.transformer.mistral_large4.rotary import (
        MistralLarge4RotaryEmbedding,
    )

    cfg = build_config()
    rotary = MistralLarge4RotaryEmbedding(cfg)
    alpha = torch.randn_like(rotary.alpha) * 1e-3 + 0.1234567  # not representable in bf16
    rotary.alpha.copy_(alpha)
    rotary.bfloat16()
    assert rotary.alpha.dtype == torch.float32 and torch.equal(rotary.alpha, alpha)

    indexer = MistralLarge4Indexer(cfg)
    kept = {}
    # the names the reference keeps in fp32, written out: dropping one from the production tuple
    # must fail here, not silently stop checking it
    for name in ("weights_proj", "ssmax_s", "ssmax_b"):
        t = getattr(indexer, name)
        t.copy_(torch.randn_like(t) * 1e-3 + 0.1234567)
        kept[name] = t.clone()
    indexer.to(torch.bfloat16)
    for name, want in kept.items():
        got = getattr(indexer, name)
        assert got.dtype == torch.float32 and torch.equal(got, want), name
    assert indexer.wq_b.dtype == torch.bfloat16  # the other buffers follow the cast
