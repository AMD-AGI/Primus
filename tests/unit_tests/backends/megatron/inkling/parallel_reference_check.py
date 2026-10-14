###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Multi-GPU parity: Megatron Inkling under DP / EP / PP / VPP against the official model.

Run with ``torchrun --nproc-per-node 8`` (needs the reference, see
``inkling_reference.py``)::

    PYTHONPATH=/tmp/inkling_ref/site torchrun --nproc-per-node 8 \\
        tests/unit_tests/backends/megatron/inkling/parallel_reference_check.py

Every layout trains one step of an 8-layer fp32 Inkling (dense layer 0, MoE
layers 1-7, global attention on layers 3 and 7) through Megatron's pipeline
schedules on a fixed 16 x 256-token batch split over the data-parallel ranks.
Each rank loads its slice of the reference weights through the importer. The
global loss and every gradient, summed over data-parallel ranks the way the
DDP buckets would, are compared with the reference run in fp64 on the full
batch. A gradient passes if its error (max |diff| relative to the tensor's
max) is below 1e-5. The one-element ``global_scale`` gradients sum thousands
of cancelling terms, so their fp32 rounding depends on the data and on how
many microbatches are accumulated (the reference's own fp32 run is off by
1e-6 to 4e-5 on them); they get 1e-4. Exits non-zero if any layout fails.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import torch
import torch.distributed as dist

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO))
sys.path.append(os.environ.get("MEGATRON_PATH") or str(_REPO / "third_party" / "Megatron-LM"))

from megatron.core import parallel_state as ps  # noqa: E402
from megatron.core.pipeline_parallel import get_forward_backward_func  # noqa: E402
from megatron.core.tensor_parallel.random import (  # noqa: E402
    model_parallel_cuda_manual_seed,
)

from primus.backends.megatron.core.models.inkling.inkling_builders import (  # noqa: E402
    build_inkling_model,
    inkling_config_from_model_yaml,
)
from tests.unit_tests.backends.megatron.inkling.inkling_reference import (  # noqa: E402
    build_reference_model,
    reference_forward_backward,
    reference_weights_for_megatron,
)
from tests.unit_tests.backends.megatron.inkling.inkling_test_utils import (  # noqa: E402
    preset,
)

VOCAB = 512
UNPADDED = 500
# Past the tiny preset's sliding window (32), global bias extent (64) and
# log-scaling floor (64); at 128 tokens or fewer torch 2.14 compiles flex with
# a decode kernel that rejects the block mask.
SEQ = 256
GLOBAL_BATCH = 16
NUM_LAYERS = 8
LOCAL_LAYER_IDS = [0, 1, 2, 4, 5, 6]
MOE_LAYER_FREQ = "([0]*1+[1]*7)"
TOLERANCE = 1e-5
SCALAR_TOLERANCE = 1e-4

# name: (pipeline, virtual pipeline, expert parallel); TP stays 1.
LAYOUTS = {
    "dp8": (1, None, 1),
    "ep8": (1, None, 8),
    "pp2_ep4": (2, None, 4),
    "pp4_ep2": (4, None, 2),
    "pp8": (8, None, 1),
    "pp2_vpp2_ep4": (2, 2, 4),
    "pp4_vpp2_ep2": (4, 2, 2),
}

_LAYER = re.compile(r"^decoder\.layers\.(\d+)\.")
_EXPERT = re.compile(r"(\.mlp\.experts\.linear_fc[12]\.weight)(\d+)$")


def _config(pp, vpp, ep, backend):
    config = inkling_config_from_model_yaml(
        preset("inkling_small_tiny"),
        params_dtype=torch.float32,
        unpadded_vocab_size=UNPADDED,
        gradient_accumulation_fusion=False,
        inkling_attention_backend=backend,
        num_layers=NUM_LAYERS,
        local_layer_ids=LOCAL_LAYER_IDS,
        moe_layer_freq=MOE_LAYER_FREQ,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp,
        expert_model_parallel_size=ep,
    )
    config.pipeline_dtype = torch.float32
    config.calculate_per_token_loss = True
    # what validate_args would derive: p2p overlap only exists for interleaved stages
    config.overlap_p2p_comm = vpp is not None
    config.batch_p2p_comm = not config.overlap_p2p_comm
    return config


def _global_name(chunk, local_name: str) -> str:
    """Local parameter / buffer name in a stage chunk -> name in the full model."""
    m = _LAYER.match(local_name)
    if m:
        layer = chunk.decoder.layers[int(m.group(1))]
        local_name = f"decoder.layers.{layer.layer_number - 1}." + local_name[m.end() :]
    m = _EXPERT.search(local_name)
    if m:
        per_rank = chunk.config.num_moe_experts // ps.get_expert_model_parallel_world_size()
        expert = ps.get_expert_model_parallel_rank() * per_rank + int(m.group(2))
        local_name = local_name[: m.start()] + m.group(1) + str(expert)
    return local_name


def _build_chunks(config, weights):
    vpp = config.virtual_pipeline_model_parallel_size
    chunks = []
    for vp_stage in range(vpp or 1):
        stage = vp_stage if vpp else None
        chunk = build_inkling_model(
            config,
            vocab_size=VOCAB,
            max_sequence_length=4096,
            pre_process=ps.is_pipeline_first_stage(ignore_virtual=False, vp_stage=stage),
            post_process=ps.is_pipeline_last_stage(ignore_virtual=False, vp_stage=stage),
            vp_stage=stage,
        ).cuda()
        state = {
            k: (None if k.endswith("_extra_state") else weights[_global_name(chunk, k)])
            for k in chunk.state_dict()
        }
        chunk.load_state_dict(state, strict=True)
        chunks.append(chunk)
    return chunks


def _batches(ids, num_microbatches):
    """This data-parallel rank's microbatches of the global batch."""
    dp, rank = ps.get_data_parallel_world_size(), ps.get_data_parallel_rank()
    rows = ids[rank * num_microbatches : (rank + 1) * num_microbatches]
    assert rows.shape[0] * dp == ids.shape[0]
    for row in rows:
        tokens = row.unsqueeze(0)
        labels = torch.zeros_like(tokens)
        labels[:, :-1] = tokens[:, 1:]
        mask = torch.ones_like(tokens, dtype=torch.float32)
        mask[:, -1] = 0
        yield tokens, labels, mask


def _rel_err(value, expected) -> float:
    expected = expected.double().to(value.device)
    return ((value.double() - expected).abs().max() / expected.abs().max().clamp(min=1e-30)).item()


def _run_layout(name, pp, vpp, ep, backend, ids, ref_loss, ref_grads, ref32_errors, weights):
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp,
        expert_model_parallel_size=ep,
    )
    model_parallel_cuda_manual_seed(0)
    config = _config(pp, vpp, ep, backend)
    chunks = _build_chunks(config, weights)

    dp = ps.get_data_parallel_world_size()
    num_microbatches = GLOBAL_BATCH // dp
    global_tokens = GLOBAL_BATCH * (SEQ - 1)

    def forward_step(data_iterator, model):
        tokens, labels, mask = next(data_iterator)
        per_token = model(tokens, None, None, labels=labels)

        def loss_func(output):
            total = (output.float() * mask).sum()
            return total / global_tokens, mask.sum().to(torch.int), {"loss_sum": total.detach()}

        return per_token, loss_func

    iterators = [iter(list(_batches(ids, num_microbatches))) for _ in chunks]
    losses = get_forward_backward_func()(
        forward_step_func=forward_step,
        data_iterator=iterators if vpp else iterators[0],
        model=chunks if vpp else chunks[0],
        num_microbatches=num_microbatches,
        seq_length=SEQ,
        micro_batch_size=1,
        forward_only=False,
    )
    loss = torch.zeros((), device="cuda", dtype=torch.float64)
    for item in losses or []:
        loss += item["loss_sum"].double()
    dist.all_reduce(loss)
    loss = (loss / global_tokens).float()

    errors = {}
    for chunk in chunks:
        for local, p in chunk.named_parameters():
            assert p.grad is not None, f"{name}: no gradient for {local}"
            grad = p.grad.detach().clone()
            expert = not getattr(p, "allreduce", True)
            group = ps.get_expert_data_parallel_group() if expert else ps.get_data_parallel_group()
            dist.all_reduce(grad, group=group)
            global_name = _global_name(chunk, local)
            errors[global_name] = _rel_err(grad, ref_grads[global_name].to(grad.device))

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, errors)
    errors = {k: v for part in gathered for k, v in part.items()}
    scalar = {k for k in errors if ref_grads[k].numel() == 1}
    failing = {k: v for k, v in errors.items() if v >= (SCALAR_TOLERANCE if k in scalar else TOLERANCE)}
    missing = {k for k in ref_grads if k not in errors and not k.endswith("expert_bias")}
    loss_err = abs(loss.item() - ref_loss.item())
    ok = not failing and not missing and loss_err <= 1e-6 + 1e-5 * abs(ref_loss.item())
    if dist.get_rank() == 0:
        print(
            f"[{'PASS' if ok else 'FAIL'}] {backend:5s} {name:13s} pp={pp} vpp={vpp or 1} ep={ep} dp={dp} "
            f"micro={num_microbatches} loss={loss.item():.7f} ref64={ref_loss.item():.7f} "
            f"|dloss|={loss_err:.1e} params={len(errors)} missing={len(missing)}",
            flush=True,
        )
        for label, keys in (("tensor", set(errors) - scalar), ("scalar", scalar)):
            k = max(keys, key=errors.get)
            print(
                f"    worst {label} grad {errors[k]:.2e} (ref fp32 {ref32_errors[k]:.2e}) {k}",
                flush=True,
            )
        for k, v in sorted(failing.items(), key=lambda kv: -kv[1])[:5]:
            print(f"    FAILING {v:.2e} (ref fp32 {ref32_errors[k]:.2e}) {k}", flush=True)
    del chunks
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", nargs="*", default=list(LAYOUTS))
    parser.add_argument("--backends", nargs="*", default=["eager", "flex"])
    args = parser.parse_args()

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 8, "the layouts assume 8 ranks"

    results = []
    for backend in args.backends:
        config = _config(1, None, 1, backend)
        ids = torch.randint(
            0, UNPADDED, (GLOBAL_BATCH, SEQ), generator=torch.Generator().manual_seed(1)
        ).cuda()
        ref = build_reference_model(config, VOCAB, seed=0, numeric=True).cuda()
        _, _, ref32_grads = reference_forward_backward(ref, ids, config)
        weights = reference_weights_for_megatron(ref, config)
        # torch's grouped GEMM has no fp64 kernel: the per-expert loop, on the CPU
        ref = ref.double().cpu()
        for module in ref.modules():
            if hasattr(module, "config"):
                module.config._experts_implementation = "eager"
        _, ref_loss, ref_grads = reference_forward_backward(ref, ids.cpu(), config)
        ref_loss = ref_loss.cuda()
        del ref
        ref32_errors = {k: _rel_err(v, ref_grads[k]) for k, v in ref32_grads.items()}
        for name in args.layouts:
            pp, vpp, ep = LAYOUTS[name]
            results.append(
                _run_layout(name, pp, vpp, ep, backend, ids, ref_loss, ref_grads, ref32_errors, weights)
            )

    ps.destroy_model_parallel()
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"SUMMARY {sum(results)}/{len(results)} layouts passed", flush=True)
    dist.destroy_process_group()
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
