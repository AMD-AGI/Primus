###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
GPU test of Primus' native HF -> Megatron conversion with Megatron's real save and load.

A tiny random Qwen3, dense and MoE, is written in HF format and converted to
``torch_dist`` by the conversion hook (``native_convert_checkpoint``). Megatron's
``load_checkpoint`` then loads it the way the hook hands it to training
(``pretrained_checkpoint`` + ``finetune``) at TP2, PP2 and EP2 on two GPUs, and
the logits of each layout must match the HF model's.

Run as a script, this file is the conversion process (``convert``) and the
torchrun worker (``load``).
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.checkpoint import native_convert_common as common

_REPO = Path(__file__).resolve().parents[4]
_HOOK = _REPO / "runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py"

# The tokenizer's tokens. The converter pads the vocabulary for TP1 (384 rows, a multiple
# of 128) and a TP2 run pads it to 512, so the TP2 load also covers the rows Megatron adds.
_VOCAB = 300
_BATCH, _SEQ = 2, 32
# In fp32 Megatron's logits (up to ~4) are within ~2e-3 of HF's; swapping a pair of tensors
# (q/k norms, gate/up halves, two experts) moves them by 0.7 or more.
_MAX_ABS_DIFF = 1e-2

pytestmark = [
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs"),
    pytest.mark.skipif(
        not Path(common.megatron_lm_root()).is_dir(), reason="needs the Megatron-LM submodule"
    ),
]


# ---------------------------------------------------------------------------
# Tiny HF models
# ---------------------------------------------------------------------------
def _hf_config(moe: bool):
    from transformers import Qwen3Config, Qwen3MoeConfig

    shape = dict(
        vocab_size=400,  # HF pads its embedding beyond the tokenizer, as Qwen3 does
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,  # not hidden_size / num_attention_heads, as in Qwen3
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_word_embeddings=False,
    )
    if moe:
        return Qwen3MoeConfig(
            num_hidden_layers=2,
            intermediate_size=128,
            moe_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            decoder_sparse_step=1,
            mlp_only_layers=[],
            **shape,
        )
    return Qwen3Config(num_hidden_layers=4, intermediate_size=128, **shape)


def _write_hf_model(path: Path, moe: bool) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(_hf_config(moe))
    # random norm weights and a larger scale, so any misplaced tensor shows in the logits
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.uniform_(0.5, 1.5)
            else:
                p.normal_(0.0, 0.1)
    model.save_pretrained(path)
    # Hub Qwen3 checkpoints, which the converter reads, keep transformers 4's names for these;
    # transformers 5 writes num_local_experts and nests rope_theta in rope_parameters.
    config_file = path / "config.json"
    config = json.loads(config_file.read_text())
    if "rope_parameters" in config:
        config.setdefault("rope_theta", config["rope_parameters"]["rope_theta"])
    if "num_local_experts" in config:
        config.setdefault("num_experts", config["num_local_experts"])
    config_file.write_text(json.dumps(config, indent=2))

    vocab = {f"w{i}": i for i in range(_VOCAB - 1)}
    vocab["<unk>"] = _VOCAB - 1
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<unk>").save_pretrained(
        path
    )


# ---------------------------------------------------------------------------
# Subprocesses
# ---------------------------------------------------------------------------
def _run(cmd, env=None) -> str:
    env = dict(os.environ if env is None else env)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(_REPO), env.get("PYTHONPATH")) if p)
    proc = subprocess.run(cmd, cwd=_REPO, env=env, capture_output=True, text=True, timeout=900)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"{cmd[-6:]} exited {proc.returncode}:\n{out[-6000:]}"
    return out


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """Each tiny model, converted once: ``{"dense" | "moe": (hf_dir, ckpt_dir)}``."""
    out = {}
    for kind in ("dense", "moe"):
        root = tmp_path_factory.mktemp(kind)
        hf_dir, ckpt_dir = root / "hf", root / "megatron"
        _write_hf_model(hf_dir, moe=kind == "moe")
        _run([sys.executable, __file__, "convert", str(hf_dir), str(ckpt_dir)])
        out[kind] = (hf_dir, ckpt_dir)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_conversion_saves_torch_dist_release_checkpoint(converted, kind):
    _, ckpt_dir = converted[kind]
    assert (ckpt_dir / common.TRACKER_FILENAME).read_text() == "release"
    assert (ckpt_dir / "release" / "metadata.json").is_file()
    assert not list(ckpt_dir.glob("iter_*"))
    assert common.saved_ckpt_format(str(ckpt_dir)) == "torch_dist"


@pytest.mark.parametrize(
    "kind, tp, pp, ep",
    [("dense", 2, 1, 1), ("dense", 1, 2, 1), ("moe", 1, 1, 2)],
    ids=["dense-tp2", "dense-pp2", "moe-ep2"],
)
def test_converted_checkpoint_loads_resharded(converted, kind, tp, pp, ep):
    # Not a top-level import: Megatron's checkpoint writer spawns processes that re-run this
    # file with Megatron-LM, which has its own ``tests`` package, first on sys.path.
    from tests.utils import isolated_torchrun_env

    hf_dir, ckpt_dir = converted[kind]
    env = isolated_torchrun_env()
    # Megatron requires it for tensor parallelism (Primus' launcher sets it for training)
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    out = _run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc_per_node=2",
            __file__,
            "load",
            str(hf_dir),
            str(ckpt_dir),
            str(tp),
            str(pp),
            str(ep),
        ],
        env,
    )
    results = [json.loads(line.split("RESULT ", 1)[1]) for line in out.splitlines() if "RESULT " in line]
    assert len(results) == 1, out[-6000:]
    result = results[0]
    assert result["max_abs_diff"] < _MAX_ABS_DIFF, result


# ---------------------------------------------------------------------------
# Script: the conversion process and the torchrun worker
# ---------------------------------------------------------------------------
def _convert_main(hf_dir: str, out_dir: str) -> None:
    spec = importlib.util.spec_from_file_location("native_convert_hook", _HOOK)
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    # fp32, so the logits compare tightly (native_ckpt_dtype in an experiment); port 0 lets
    # the converter's one-process gloo world take a free port
    opts = {"ckpt_format": "torch_dist", "dtype": "fp32", "convert_kwargs": {"master_port": 0}}
    hook.native_convert_checkpoint(hf_dir, out_dir, opts)


def _layout_margs(conv, hf_dir: str, ckpt_dir: str, tp: int, pp: int, ep: int):
    """The converter's Megatron args for the model at TP/PP/EP, set to load ``ckpt_dir`` the
    way the hook hands it to training."""
    import megatron.training.arguments as arguments

    validate_args = arguments.validate_args

    def validate_for_layout(m, *args, **kwargs):
        # build_margs sizes the batch for one process; at EP2 the run is 2-way data parallel
        m.global_batch_size = m.micro_batch_size * m.world_size // (tp * pp)
        m.rank = int(os.environ["RANK"])
        return validate_args(m, *args, **kwargs)

    loader_args = SimpleNamespace(
        hf_dir=hf_dir,
        save_dir=None,
        dtype="fp32",
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        expert_parallel_size=ep,
        ckpt_format="torch_dist",
        seq_length=_SEQ,
        make_vocab_size_divisible_by=128,
    )
    arguments.validate_args = validate_for_layout
    try:
        margs = conv.build_margs(conv.derive_megatron_fields(common.read_hf_config(hf_dir)), loader_args)
    finally:
        arguments.validate_args = validate_args
    margs.use_cpu_initialization = False
    margs.load = None
    margs.pretrained_checkpoint = ckpt_dir
    margs.finetune = True
    margs.exit_on_missing_checkpoint = True
    return margs


def _load_main(hf_dir: str, ckpt_dir: str, tp: int, pp: int, ep: int) -> None:
    from primus.backends.megatron.checkpoint import native_registry

    common.ensure_megatron_on_path()
    common.neutralize_nvte_attention_env()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")

    from megatron.core import mpu, tensor_parallel
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.global_vars import set_global_variables

    from primus.core.utils.import_utils import get_model_provider

    family = native_registry.detect_family(hf_dir)
    margs = _layout_margs(family.converter_module(), hf_dir, ckpt_dir, tp, pp, ep)
    set_global_variables(margs, build_tokenizer=True)
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp, expert_model_parallel_size=ep
    )
    tensor_parallel.model_parallel_cuda_manual_seed(0)

    provider = get_model_provider(model_type=native_registry.model_type_for(family.name))
    model = provider(pre_process=mpu.is_pipeline_first_stage(), post_process=mpu.is_pipeline_last_stage())
    model = model.cuda().eval()
    load_checkpoint([model], None, None)

    tokens = torch.randint(0, _VOCAB, (_BATCH, _SEQ), generator=torch.Generator().manual_seed(1)).cuda()
    position_ids = torch.arange(_SEQ, device="cuda").unsqueeze(0).expand(_BATCH, -1)
    logits = None
    with torch.no_grad():
        if pp == 1:
            logits = tensor_parallel.gather_from_tensor_model_parallel_region(
                model(tokens, position_ids, None)
            )
        elif mpu.is_pipeline_first_stage():
            torch.distributed.send(model(tokens, position_ids, None).contiguous(), dst=1)
        else:
            hidden = torch.empty(_SEQ, _BATCH, margs.hidden_size, device="cuda")
            torch.distributed.recv(hidden, src=0)
            model.set_input_tensor(hidden)
            logits = model(tokens, position_ids, None)

    if logits is not None and (pp > 1 or torch.distributed.get_rank() == 0):
        from transformers import AutoModelForCausalLM

        hf = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float32, attn_implementation="eager")
        with torch.no_grad():
            ref = hf.cuda().eval()(input_ids=tokens).logits
        diff = (logits[..., :_VOCAB] - ref[..., :_VOCAB]).abs().max().item()
        result = {"tp": tp, "pp": pp, "ep": ep, "max_abs_diff": diff, "ref_max_abs": ref.abs().max().item()}
        print("RESULT " + json.dumps(result), flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    convert_cmd = sub.add_parser("convert")
    convert_cmd.add_argument("hf_dir")
    convert_cmd.add_argument("out_dir")
    load_cmd = sub.add_parser("load")
    load_cmd.add_argument("hf_dir")
    load_cmd.add_argument("ckpt_dir")
    for name in ("tp", "pp", "ep"):
        load_cmd.add_argument(name, type=int)
    a = parser.parse_args()
    if a.cmd == "convert":
        _convert_main(a.hf_dir, a.out_dir)
    else:
        _load_main(a.hf_dir, a.ckpt_dir, a.tp, a.pp, a.ep)
