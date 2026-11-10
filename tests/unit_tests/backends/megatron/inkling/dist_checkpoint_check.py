###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Multi-GPU ``torch_dist`` checkpoint resharding for Megatron Inkling.

Run with ``torchrun --nproc-per-node 8``::

    torchrun --nproc-per-node 8 \\
        tests/unit_tests/backends/megatron/inkling/dist_checkpoint_check.py

The model is the 8-layer Inkling of ``parallel_reference_check.py`` with fixed
random weights. Each save layout loads them, writes a ``torch_dist``
checkpoint through ``sharded_state_dict`` as Megatron's ``save_checkpoint``
does, and every load layout (DP / EP / PP / VPP, TE or Turbo grouped experts)
reads it into a freshly initialised model. A load passes if every parameter
and persistent buffer equals the fixed weights bit for bit. Exits non-zero if
any load fails.

The converter's single-process ``torch_dist`` output is checked the same way:
``--convert-to DIR`` (plain ``python``, one GPU) converts a synthetic release
checkpoint of the same model to both formats, and ``--converted DIR`` (under
``torchrun``) loads the ``torch_dist`` one in every layout and compares it with
the ``torch`` one::

    python tests/unit_tests/backends/megatron/inkling/dist_checkpoint_check.py --convert-to /tmp/conv
    torchrun --nproc-per-node 8 \\
        tests/unit_tests/backends/megatron/inkling/dist_checkpoint_check.py --converted /tmp/conv
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO))
sys.path.append(os.environ.get("MEGATRON_PATH") or str(_REPO / "third_party" / "Megatron-LM"))

from megatron.core import dist_checkpointing  # noqa: E402
from megatron.core import parallel_state as ps  # noqa: E402
from megatron.core.tensor_parallel.random import (  # noqa: E402
    model_parallel_cuda_manual_seed,
)

from primus.backends.megatron.core.models.inkling import (  # noqa: E402
    inkling_layer_specs,
)
from primus.backends.megatron.core.models.inkling.inkling_builders import (  # noqa: E402
    build_inkling_model,
)
from tests.unit_tests.backends.megatron.inkling.inkling_test_utils import (  # noqa: E402
    load_converter,
    preset,
    random_release_state,
    release_text_config,
)
from tests.unit_tests.backends.megatron.inkling.parallel_reference_check import (  # noqa: E402
    LAYOUTS,
    LOCAL_LAYER_IDS,
    MOE_LAYER_FREQ,
    NUM_LAYERS,
    UNPADDED,
    VOCAB,
    _config,
    _global_name,
)

# (layout, experts) pairs that write a checkpoint; every one is loaded by every layout in LAYOUTS
# under both expert implementations.
SAVES = [("ep8", "te"), ("pp2_ep4", "turbo"), ("pp4_vpp2_ep2", "te"), ("pp8", "turbo")]
CONTENT_METADATA = {"singleton_local_shards": False, "chained_optim_avoid_prefix": True}

_TE_SPEC_PROVIDER = inkling_layer_specs.transformer_engine_spec_provider.TESpecProvider


def _set_experts(experts: str) -> None:
    """Build the layer specs through TE's provider or through Turbo's with its grouped GEMM on."""
    if experts == "te":
        inkling_layer_specs.transformer_engine_spec_provider.TESpecProvider = _TE_SPEC_PROVIDER
        return
    from primus.backends.megatron.core.extensions import (
        transformer_engine_spec_provider as spec_provider,
    )

    turbo_cfg = SimpleNamespace(
        **{
            **vars(spec_provider._build_default_primus_args()),
            **dict(enable_primus_turbo=True, use_turbo_grouped_gemm=True, use_turbo_rms_norm=True),
        }
    )

    def turbo_provider(*args, **kwargs):
        provider = spec_provider.PrimusTurboSpecProvider(*args, **kwargs)
        provider.cfg = turbo_cfg
        return provider

    inkling_layer_specs.transformer_engine_spec_provider.TESpecProvider = turbo_provider


def _set_global_args() -> None:
    """The Primus fields Turbo's linears read from Megatron's global args."""
    from megatron.training.global_vars import set_args

    set_args(
        SimpleNamespace(
            enable_turbo_attention_float8=False,
            data_parallel_replicate_degree=1,
            offload=False,
            offload_ops=[],
            patch_primus_pipeline=False,
            pp_algorithm=None,
            patch_zero_bubble=False,
            enable_zero_bubble=False,
            use_turbo_fused_act_with_probs=False,
            moe_router_padding_for_quantization=False,
        )
    )


def _init_layout(layout: str):
    pp, vpp, ep = LAYOUTS[layout]
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp,
        expert_model_parallel_size=ep,
    )
    model_parallel_cuda_manual_seed(0)
    return _config(pp, vpp, ep, "eager")


def _build_chunks(layout: str, experts: str):
    config = _init_layout(layout)
    _set_experts(experts)
    vpp = config.virtual_pipeline_model_parallel_size
    chunks = []
    for vp_stage in range(vpp or 1):
        stage = vp_stage if vpp else None
        chunks.append(
            build_inkling_model(
                config,
                vocab_size=VOCAB,
                max_sequence_length=4096,
                pre_process=ps.is_pipeline_first_stage(ignore_virtual=False, vp_stage=stage),
                post_process=ps.is_pipeline_last_stage(ignore_virtual=False, vp_stage=stage),
                vp_stage=stage,
            ).cuda()
        )
    return chunks


def _expected_state(chunk, weights):
    """The chunk's state dict filled from the full-model ``weights``; Turbo's stacked
    ``weights`` are the stack of its experts' ``weight{i}``."""
    state = {}
    for key in chunk.state_dict():
        if key.endswith("_extra_state"):
            state[key] = None
        elif key.endswith(".weights"):
            per_expert = key[: -len("weights")] + "weight"
            num = chunk.get_submodule(key.rsplit(".", 1)[0]).num_gemms
            state[key] = torch.stack([weights[_global_name(chunk, f"{per_expert}{i}")] for i in range(num)])
        else:
            state[key] = weights[_global_name(chunk, key)]
    return state


def _sharded(chunks):
    metadata = dict(CONTENT_METADATA, dp_cp_group=ps.get_data_parallel_group(with_context_parallel=True))
    if len(chunks) == 1:
        return {"model": chunks[0].sharded_state_dict(metadata=dict(metadata))}
    return {f"model{i}": chunk.sharded_state_dict(metadata=dict(metadata)) for i, chunk in enumerate(chunks)}


def _full_weights():
    """Fixed random weights for the full single-rank model, identical on every rank."""
    (model,) = _build_chunks("dp8", "te")
    weights = {}
    for i, (key, tensor) in enumerate(sorted(model.state_dict().items())):
        if key.endswith("_extra_state"):
            continue
        g = torch.Generator().manual_seed(1000 + i)
        weights[key] = torch.randn(tensor.shape, generator=g).to(tensor.dtype).cuda()
    del model
    return weights


def _save(layout, experts, weights, ckpt_dir):
    if dist.get_rank() == 0:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        os.makedirs(ckpt_dir)
    dist.barrier()
    chunks = _build_chunks(layout, experts)
    for chunk in chunks:
        chunk.load_state_dict(_expected_state(chunk, weights), strict=True)
    state = _sharded(chunks)
    state.update(checkpoint_version=3.0, iteration=0)
    dist_checkpointing.save(state, ckpt_dir, content_metadata=CONTENT_METADATA)
    dist.barrier()


def _load(layout, experts, weights, ckpt_dir) -> bool:
    chunks = _build_chunks(layout, experts)
    sharded = _sharded(chunks)
    loaded = dist_checkpointing.load(sharded, ckpt_dir)
    bad = []
    for name, chunk in zip(sharded, chunks):
        chunk.load_state_dict(loaded[name], strict=True)
        state = chunk.state_dict()
        for key, want in _expected_state(chunk, weights).items():
            if want is not None and not torch.equal(state[key], want):
                bad.append(_global_name(chunk, key))
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, bad)
    bad = sorted({k for part in gathered for k in part})
    if dist.get_rank() == 0:
        pp, vpp, ep = LAYOUTS[layout]
        print(
            f"    [{'PASS' if not bad else 'FAIL'}] load {layout:13s} {experts:5s} "
            f"pp={pp} vpp={vpp or 1} ep={ep} dp={ps.get_data_parallel_world_size()}"
            + (f" mismatched={len(bad)} e.g. {bad[:3]}" if bad else ""),
            flush=True,
        )
    del chunks
    return not bad


def convert_fixture(out: Path) -> None:
    """A synthetic release checkpoint of the check's model, converted to ``torch`` and ``torch_dist``."""
    import json

    from safetensors.torch import save_file

    from primus.backends.megatron.core.models.inkling.inkling_builders import (
        inkling_config_from_model_yaml,
    )

    hf_dir = out / "hf"
    hf_dir.mkdir(parents=True)
    yaml = out / "inkling_dist_check.yaml"
    yaml.write_text(
        f"extends:\n  - {preset('inkling_small_tiny')}\n"
        f"num_layers: {NUM_LAYERS}\nlocal_layer_ids: {LOCAL_LAYER_IDS}\n"
        f'moe_layer_freq: "{MOE_LAYER_FREQ}"\nunpadded_vocab_size: {UNPADDED}\n'
    )
    config = inkling_config_from_model_yaml(str(yaml))
    release = random_release_state(config, vocab=VOCAB)
    save_file({k: v.contiguous() for k, v in release.items()}, str(hf_dir / "model.safetensors"))
    (hf_dir / "config.json").write_text(json.dumps({"text_config": release_text_config(config, VOCAB)}))
    converter = load_converter()
    for fmt in ("torch", "torch_dist"):
        argv = ["--hf-dir", str(hf_dir), "--model-yaml", str(yaml), "--output", str(out / fmt)]
        assert converter.main([*argv, "--dtype", "fp32", "--ckpt-format", fmt]) == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="/tmp/inkling_dist_ckpt_check")
    parser.add_argument("--convert-to", type=Path, help="write the converter fixture here and exit")
    parser.add_argument("--converted", type=Path, help="load a --convert-to fixture instead of saving")
    args = parser.parse_args()
    if args.convert_to:
        convert_fixture(args.convert_to)
        return

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 8, "the layouts assume 8 ranks"
    _set_global_args()

    results = []
    if args.converted:
        state = torch.load(
            args.converted / "torch" / "release" / "mp_rank_00" / "model_optim_rng.pt", weights_only=False
        )
        weights = {k: v.cuda() for k, v in state["model"].items() if v is not None}
        ckpt_dir = str(args.converted / "torch_dist" / "release")
        if dist.get_rank() == 0:
            print(f"converter output {ckpt_dir}", flush=True)
        for load_layout in LAYOUTS:
            for load_experts in ("te", "turbo"):
                results.append(_load(load_layout, load_experts, weights, ckpt_dir))
    else:
        weights = _full_weights()
        for save_layout, save_experts in SAVES:
            if dist.get_rank() == 0:
                print(f"save {save_layout} {save_experts}", flush=True)
            _save(save_layout, save_experts, weights, args.ckpt_dir)
            for load_layout in LAYOUTS:
                for load_experts in ("te", "turbo"):
                    results.append(_load(load_layout, load_experts, weights, args.ckpt_dir))

    ps.destroy_model_parallel()
    dist.barrier()
    if dist.get_rank() == 0:
        if not args.converted:
            shutil.rmtree(args.ckpt_dir, ignore_errors=True)
        print(f"SUMMARY {sum(results)}/{len(results)} loads passed", flush=True)
    dist.destroy_process_group()
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
