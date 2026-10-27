###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Load the post-train hook's Muse Spark ``torch_dist`` checkpoint at PP x EP > 1.

``prepare`` (one process) saves the tiny random HF model of
``test_muse_spark_hf_parity``, its fp32 logits, and converts it through the
hook's native path. ``check`` (torchrun) builds ``MuseSparkGPTModel`` at the
requested PP / EP, loads the checkpoint with Megatron's ``load_checkpoint`` and
compares the last pipeline stage's fp32 logits with the HF ones::

    python run_muse_spark_dist_ckpt.py prepare --workdir /tmp/muse_dist
    torchrun --nproc_per_node=8 run_muse_spark_dist_ckpt.py check --workdir /tmp/muse_dist --pp 2 --ep 4

Needs transformers >= 5.20 for ``prepare`` and one GPU per rank for ``check``.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PRIMUS_ROOT = HERE.parents[4]
HOOK = PRIMUS_ROOT / "runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py"
sys.path.insert(0, str(PRIMUS_ROOT))
sys.path.insert(0, str(HERE))

import test_muse_spark_hf_parity as parity  # noqa: E402

from primus.backends.megatron.checkpoint import (  # noqa: E402
    native_convert_common as common,
)
from primus.backends.megatron.checkpoint import (  # noqa: E402
    native_loader_muse_spark as conv,
)


def prepare(workdir: Path) -> None:
    from transformers import MuseSparkConfig, MuseSparkForConditionalGeneration

    hf_dir, mcore_dir = workdir / "hf", workdir / "mcore"
    torch.manual_seed(0)
    cfg = MuseSparkConfig(
        text_config=parity.TEXT,
        vision_config=parity.VISION,
        audio_config=parity.AUDIO,
        out_hidden_size=128,
        projector_hidden_size=64,
    )
    model = MuseSparkForConditionalGeneration(cfg).float()
    parity._perturb(model)
    model.save_pretrained(hf_dir, safe_serialization=True)
    ids = torch.randint(3, parity.VOCAB, (parity.BATCH, parity.SEQ), generator=torch.Generator().manual_seed(2))
    torch.save({"ids": ids, "logits": parity._hf_logits(hf_dir, ids, torch.float32)}, workdir / "ref.pt")

    parity._ensure_logger()
    spec = importlib.util.spec_from_file_location("convert_checkpoints_hook", HOOK)
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    hook.native_convert_checkpoint(str(hf_dir), mcore_dir, {"dtype": "fp32"})

    from megatron.core import dist_checkpointing

    iter_dir = mcore_dir / "iter_0000001"
    assert dist_checkpointing.check_is_distributed_checkpoint(str(iter_dir)), f"{iter_dir} is not torch_dist"
    assert not (iter_dir / "mp_rank_00").exists()
    print(f"prepare: torch_dist checkpoint at {iter_dir}: {sorted(p.name for p in iter_dir.iterdir())}")


def check(workdir: Path, pp: int, ep: int) -> None:
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    parity._ensure_logger()
    common.ensure_megatron_on_path()
    common.neutralize_nvte_attention_env()
    common.apply_convert_patches(module_name="checkpoint_convert")

    import megatron.training.arguments as megatron_arguments
    from megatron.core import mpu
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.global_vars import set_global_variables
    from megatron.training.training import get_model

    from primus.core.utils.import_utils import get_model_provider

    hf_dir = str(workdir / "hf")
    validate_args = megatron_arguments.validate_args

    def validate_for_layout(m, *rest):
        m.pipeline_model_parallel_size = pp
        m.expert_model_parallel_size = ep
        m.world_size, m.rank = world, rank
        m.micro_batch_size = parity.BATCH
        m.global_batch_size = parity.BATCH * (world // pp)
        m.use_cpu_initialization = False
        m.save, m.load = None, str(workdir / "mcore")
        m.finetune = True
        m.auto_detect_ckpt_format = True
        return validate_args(m, *rest)

    megatron_arguments.validate_args = validate_for_layout
    try:
        margs = conv.build_margs(
            conv.derive_megatron_fields(common.read_hf_config(hf_dir)),
            argparse.Namespace(
                hf_dir=hf_dir,
                save_dir=None,
                dtype="fp32",
                seq_length=parity.SEQ,
                make_vocab_size_divisible_by=128,
                ckpt_format="torch_dist",
            ),
        )
    finally:
        megatron_arguments.validate_args = validate_args
    assert margs.data_parallel_size == world // pp

    set_global_variables(margs, build_tokenizer=True)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
    mpu.initialize_model_parallel(pipeline_model_parallel_size=pp, expert_model_parallel_size=ep)
    model_parallel_cuda_manual_seed(1234)

    model = get_model(get_model_provider("muse_spark"), wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    for chunk in model:
        chunk.eval()

    ref = torch.load(workdir / "ref.pt")
    ids = ref["ids"].cuda()
    pos = torch.arange(parity.SEQ, device="cuda").unsqueeze(0).expand(parity.BATCH, -1)

    def forward_step(data_iterator, chunk):
        return chunk(input_ids=ids, position_ids=pos, attention_mask=None), (
            lambda output, non_loss_data=False: output
        )

    with torch.no_grad():
        outputs = get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=iter([None]),
            model=model,
            num_microbatches=1,
            seq_length=parity.SEQ,
            micro_batch_size=parity.BATCH,
            forward_only=True,
            collect_non_loss_data=True,
        )

    ok = torch.ones(1, device="cuda")
    if mpu.is_pipeline_last_stage():
        logits = outputs[0].float().cpu()[..., : parity.VOCAB]
        err, agree = parity._rel(logits, ref["logits"]), parity._agree(logits, ref["logits"])
        print(f"check pp={pp} ep={ep} rank={rank}: rel err {err:.2e}, argmax agreement {agree:.3f}", flush=True)
        ok.fill_(float(err < 1e-3 and agree > 0.99))
    torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.destroy_process_group()
    if ok.item() != 1.0:
        raise SystemExit(f"pp={pp} ep={ep}: logits differ from HF")
    if rank == 0:
        print(f"check pp={pp} ep={ep}: PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("prepare", "check"))
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        prepare(args.workdir)
    else:
        check(args.workdir, args.pp, args.ep)


if __name__ == "__main__":
    main()
