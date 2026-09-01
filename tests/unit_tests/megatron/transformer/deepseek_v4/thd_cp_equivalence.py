###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""THD + context-parallel equivalence for DeepSeek-V4 attention.

Not a pytest module: context parallelism needs a real process group, so this is run under
torchrun at two different world sizes and the two results are compared.

    torchrun --nproc_per_node=1 thd_cp_equivalence.py --out /tmp/cp1.pt
    torchrun --nproc_per_node=2 thd_cp_equivalence.py --out /tmp/cp2.pt
    python thd_cp_equivalence.py --compare /tmp/cp1.pt /tmp/cp2.pt

Sharding the sequence must not change the answer, so the gathered CP=N output has to
match the CP=1 output row for row. This exercises the parts the single-process tests
cannot reach at all: the global_start coordinate mapping in both pool masks, the clamp
for sequences that begin before a rank's shard, and the pool all-gather ordering.

Weights are seeded identically on every rank and the input is generated from a fixed seed
before sharding, so any difference is the CP path, not initialisation or data.
"""

import argparse
import os
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, _REPO)
# Megatron lives in third_party and is normally put on the path by tests/conftest.py, which
# does not run for a plain torchrun script.
sys.path.insert(0, os.path.join(_REPO, "third_party", "Megatron-LM"))


def _helpers():
    """Load the sibling test module by path.

    A plain `import tests.unit_tests...` needs the intermediate packages to have
    __init__.py, which they do not; and the module calls pytest.importorskip at import
    time, so it must be imported with pytest importable but not necessarily running.
    """
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "test_deepseek_v4_attention.py")
    spec = importlib.util.spec_from_file_location("_v4_attn_test_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build(compress_ratio, device):
    _h = _helpers()
    _make_attention = _h._make_attention
    _make_compressed_attention = _h._make_compressed_attention
    _make_v4_config = _h._make_v4_config

    torch.manual_seed(0)  # identical weights on every rank
    cfg = _make_v4_config(
        hidden_size=512,
        num_heads=8,
        head_dim=512,
        rotary_dim=64,
        q_lora_rank=256,
        o_groups=2,
        o_lora_rank=128,
        attn_sink=True,
    )
    cfg.use_v4_attention_backend = "triton_v2"
    cfg.use_v4_csa_attention_backend = "triton_v2"
    cfg.attn_sliding_window = 64
    attn = (
        _make_attention(cfg)
        if compress_ratio == 0
        else _make_compressed_attention(config=cfg, compress_ratio=compress_ratio)
    )
    attn = attn.to(device=device, dtype=torch.bfloat16).eval()
    rope = getattr(attn, "rope", None)
    if rope is not None:
        rope.to(device=device)
        for sub in ("compress_rope", "attn_rope"):
            r = getattr(rope, sub, None)
            if r is not None:
                r.to(device=device)
    return attn


class _PSP:
    def __init__(self, cu):
        self.cu_seqlens_q = cu
        self.cu_seqlens_kv = cu
        self.qkv_format = "thd"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[512, 256, 128, 128])
    args = ap.parse_args()

    if args.compare:
        a, b = (torch.load(p) for p in args.compare)
        ok = True
        for cr in sorted(a):
            d = (a[cr].float() - b[cr].float()).abs()
            hit = torch.allclose(a[cr].float(), b[cr].float(), rtol=2e-2, atol=2e-2)
            ok &= hit
            print(
                f"  cr={cr:3d}  max|diff|={d.max().item():.3e}  mean={d.mean().item():.3e}  "
                f"{'OK' if hit else 'MISMATCH'}",
                flush=True,
            )
        print("  ALL MATCH" if ok else "  *** DIFFERS ***", flush=True)
        sys.exit(0 if ok else 1)

    torch.distributed.init_process_group("nccl")
    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    from megatron.core import parallel_state

    parallel_state.initialize_model_parallel(context_parallel_size=world)

    lens = args.seq_lens
    S_total = sum(lens)
    assert S_total % world == 0, f"pack of {S_total} rows must divide over {world} ranks"
    S_local = S_total // world
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=device)

    results = {}
    for cr in (0, 4, 128):
        attn = _build(cr, device)
        torch.manual_seed(1)  # same input on every rank, then shard it
        full = torch.randn(1, S_total, attn.config.hidden_size, device=device, dtype=torch.bfloat16)
        pos = torch.cat([torch.arange(n, device=device) for n in lens]).unsqueeze(0)

        lo = rank * S_local
        with torch.no_grad():
            out_local = attn(
                full[:, lo : lo + S_local],
                pos[:, lo : lo + S_local],
                packed_seq_params=_PSP(cu),
            )
        gathered = [torch.empty_like(out_local) for _ in range(world)]
        torch.distributed.all_gather(gathered, out_local.contiguous())
        results[cr] = torch.cat(gathered, dim=1).cpu()
        if rank == 0:
            print(f"  cr={cr:3d}  world={world}  out={tuple(results[cr].shape)}", flush=True)

    if rank == 0 and args.out:
        torch.save(results, args.out)
        print(f"  saved -> {args.out}", flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
