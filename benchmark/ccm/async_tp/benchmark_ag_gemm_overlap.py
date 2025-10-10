import argparse
import csv
import itertools
import json
import os
from contextlib import nullcontext
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, Union

import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from torch._inductor.utils import run_and_get_triton_code
from torch.distributed._functional_collectives import all_gather_tensor
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.distributed_c10d import _get_group_size_by_name
from tqdm import tqdm
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from utils import patch_async_tp, restore_async_tp, te_patch, te_restore

NUM_WARMUP = 2
NUM_ITERS = 5
MBS_LIST = [1, 2, 4, 8]
pt_e4m3_type = torch.float8_e4m3fn
pt_e5m2_type = torch.float8_e5m2
if torch.version.hip and "gfx94" in torch.cuda.get_device_properties(0).gcnArchName:
    pt_e4m3_type = torch.float8_e4m3fnuz
    pt_e5m2_type = torch.float8_e5m2fnuz

te_e4m3_type = tex.DType.kFloat8E4M3


def get_torch_prof_ctx(do_prof: bool):
    ctx = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
        )
        if do_prof
        else nullcontext()
    )
    return ctx


def perf_func(func, iters, warmup_iters):
    assert iters >= warmup_iters
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    for n in range(iters + warmup_iters):
        if n == warmup_iters:
            start_event.record()
        output = func()
        # dist.barrier()  # 使用 TP_GROUP 确保同步
    stop_event.record()
    start_event.wait()
    stop_event.wait()
    torch.cuda.current_stream().synchronize()
    duration_ms = start_event.elapsed_time(stop_event)
    return output, duration_ms / iters


def _fp8_all_gather(tensor: torch.Tensor, gather_dim: int, group_name: str) -> torch.Tensor:
    # We don't yet have a canonical pattern for fp8 all-gather. This is a
    # pattern observed in DTensor + float8_experimental.
    ag = all_gather_tensor(tensor, gather_dim=0, group=group_name)
    if gather_dim == 0:
        return ag.view(tensor.dtype)
    chunks = ag.chunk(_get_group_size_by_name(group_name))
    chunks = [chunk.view(torch.uint8) for chunk in chunks]
    return torch.cat(chunks, dim=gather_dim).view(tensor.dtype)


class TEAGGemm:
    def __init__(
        self,
        TP_GROUP: Union[
            dist.ProcessGroup,
            DeviceMesh,
            str,
        ],
        TP_SIZE: int,
        mbs: int,
        m: int,
        n: int,
        k: int,
        parallel_mode: str,
        output_dtype: torch.dtype,
        num_splits: int,
        use_fp8: bool,
        device,
    ):
        self.tp_group = TP_GROUP
        self.tp_size = TP_SIZE

        self.output_dtype = output_dtype
        self.buffer_dtype = torch.uint8 if use_fp8 else output_dtype
        self.use_fp8 = use_fp8
        self.device = device
        self.num_splits = num_splits
        self.comm_type = None
        self.ub_obj = None

        if parallel_mode == "row":
            # bwd: grad_out -> ag -> grad_out_g, mm(grad_out_g, weight) -> dgrad
            self.input_shape = (mbs * m // TP_SIZE, n)
            self.weight_shape = (n, k // TP_SIZE)
            self.ub_shape = (mbs * m, n)
            self.grad = True
            self.layout = "NN"
        else:
            # fwd: input -> ag -> input_g, mm(input_g, weight) -> output
            self.input_shape = (mbs * m // TP_SIZE, k)
            self.weight_shape = (n // TP_SIZE, k)
            self.ub_shape = (mbs * m, k)
            self.grad = False
            self.layout = "TN"

    def prepare_input_output(self):
        inp_quantizer = None
        ker_quantizer = None
        input = torch.randn(self.input_shape, device=self.device, dtype=self.output_dtype)
        weight = torch.randn(self.weight_shape, device=self.device, dtype=self.output_dtype)
        rs_out = None
        if self.use_fp8:
            num_gemms = 3
            fp8_dtype = te_e4m3_type
            fp8_scales = torch.ones(num_gemms, dtype=torch.float, device=self.device)
            fp8_amaxes = torch.zeros(num_gemms, dtype=torch.float, device=self.device)

            inp_g, _ = te.distributed.gather_along_first_dim(input, self.tp_group)
            weight_g, _ = te.distributed.gather_along_first_dim(weight, self.tp_group)
            inp_amax = torch.max(torch.abs(inp_g))
            fp8_amaxes[0].copy_(inp_amax)
            ker_amax = torch.max(torch.abs(weight_g))
            fp8_amaxes[1].copy_(ker_amax)
            inp_quantizer = Float8Quantizer(fp8_scales[0].clone(), fp8_amaxes[0].clone(), fp8_dtype)
            ker_quantizer = Float8Quantizer(fp8_scales[1].clone(), fp8_amaxes[1].clone(), fp8_dtype)

            # Cast input to Float8Tensor
            inp_fp8 = inp_quantizer(input)

            # Cast kernel to Float8Tensor
            weight_fp8 = ker_quantizer(weight)
            return inp_fp8, inp_quantizer, weight_fp8, rs_out
        return input, None, weight, rs_out

    def prepare_ub(self):
        self.ub_obj = tex.CommOverlap(
            self.ub_shape,  # Communication buffer shape
            self.buffer_dtype,  # Communication buffer data type
            self.tp_group.group_name,
            # Tensor-parallel group size (may be different than local_size)
            self.tp_size,
            num_splits=self.num_splits,
        )
        self.comm_type = tex.CommOverlapType.AG

    def run(self, inp, inp_quantizer, weight, rs_out, is_opt):
        if is_opt:
            assert self.ub_obj is not None and self.comm_type is not None
            self.ub_obj.copy_into_buffer(inp, inp_quantizer, True)
            gemm_inp = self.ub_obj.get_buffer(inp_quantizer, False)
        else:
            gemm_inp, _ = te.distributed.gather_along_first_dim(inp, self.tp_group, quantizer=inp_quantizer)

        output, *_, rs_out = te.cpp_extensions.general_gemm(
            weight,
            gemm_inp,
            te.module.base.get_workspace(),
            layout=self.layout,
            grad=self.grad,
            out_dtype=self.output_dtype,
            quantization_params=None,
            use_split_accumulator=False,
            ub=self.ub_obj if is_opt else None,
            ub_type=self.comm_type if is_opt else None,
            extra_output=rs_out,
            bulk_overlap=False,
        )

        return output


def titan_base_scaled_ag_gemm(
    A_shard: torch.Tensor, B: torch.Tensor, kwargs: dict[str, Any], TP_GROUP, return_A
) -> tuple[torch.Tensor, Any]:
    if kwargs is None or kwargs.get("A_scale") is None:
        A = all_gather_tensor(A_shard, gather_dim=1, group=TP_GROUP)
        if return_A:
            return A, A @ B
        return None, A @ B

    A_scale = kwargs["A_scale"]
    B_scale = kwargs["B_scale"]
    out_dtype = kwargs["output_dtype"]
    A = _fp8_all_gather(A_shard, gather_dim=1, group_name=TP_GROUP.group_name)

    if len(A_shard.shape) > 2:
        C = torch._scaled_mm(A.flatten(0, -2), B, A_scale, B_scale, out_dtype=out_dtype)
        C = C.view(*A.shape[:-1], -1)
    else:
        C = torch._scaled_mm(A, B, A_scale, B_scale, out_dtype=out_dtype)

    if return_A:
        return A, C
    return None, C


def titan_compile_ag_gemm(
    func, A_shard: torch.Tensor, B: torch.Tensor, kwargs: dict[str, Any], TP_GROUP, return_A, debug_mode=False
) -> torch.nn.Module:
    torch._inductor.config._micro_pipeline_tp = False
    compiled = torch.compile(func)
    if not debug_mode:
        return compiled

    code = run_and_get_triton_code(compiled, A_shard, B, kwargs, TP_GROUP, return_A)
    assert (
        "fused_all_gather_matmul" not in code
        and "fused_all_gather_scaled_matmul" not in code
        and "all_gather_into_tensor" in code
    ), "Async TP applied in ag gemm ops not as we expected."

    return compiled


def titan_overlap_ag_gemm(
    func, A_shard: torch.Tensor, B: torch.Tensor, kwargs: dict[str, Any], TP_GROUP, return_A, debug_mode=False
) -> torch.nn.Module:
    # Enable symmetric memory for the TP process group
    enable_symm_mem_for_group(TP_GROUP.group_name)

    # Tell torch.compile to enable async-TP
    torch._inductor.config._micro_pipeline_tp = True

    # Apply torch.compile to the model
    compiled = torch.compile(func)
    if not debug_mode:
        return compiled

    code = run_and_get_triton_code(compiled, A_shard, B, kwargs, TP_GROUP, return_A)
    assert (
        "fused_all_gather_matmul" in code or "fused_all_gather_scaled_matmul" in code
    ), "Async TP not applied in ag gemm ops"

    return compiled


def titan_orig_gemm_rs(
    func, A_shard: torch.Tensor, B: torch.Tensor, kwargs: dict[str, Any], TP_GROUP, return_A, debug_mode=False
) -> torch.nn.Module:
    restore_async_tp()
    return titan_overlap_ag_gemm(func, A_shard, B, kwargs, TP_GROUP, return_A, debug_mode)


def titan_patch_gemm_rs(
    func, A_shard: torch.Tensor, B: torch.Tensor, kwargs: dict[str, Any], TP_GROUP, return_A, debug_mode=False
) -> torch.nn.Module:
    patch_async_tp()
    return titan_overlap_ag_gemm(func, A_shard, B, kwargs, TP_GROUP, return_A, debug_mode)


def benchmark_titan(report_dir_path, model_config, profile, fp8, TP_GROUP, TP_SIZE, device):
    ctx = get_torch_prof_ctx(profile)
    model_name = model_config["model"]
    seq = model_config["seqlen"]
    hidden_size = model_config["hidden_size"]
    intermediate_size = model_config["intermediate_size"]
    num_attention_heads = model_config["num_attention_heads"]
    num_key_value_heads = model_config["num_key_value_heads"]
    head_dim = model_config["head_dim"]

    # Generate shapes
    gemm_shape_list = []  # [[m, n, k]...]
    # attn q
    gemm_shape_list.append(
        (
            [
                seq,
                int(num_attention_heads * head_dim),
                hidden_size,
            ],
            "column",
        )
    )
    # attn k/v
    gemm_shape_list.append(
        (
            [
                seq,
                int(num_key_value_heads * head_dim),
                hidden_size,
            ],
            "column",
        )
    )

    # attn out
    gemm_shape_list.append(([seq, hidden_size, hidden_size], "row"))
    # mlp gate+up
    gemm_shape_list.append(([seq, int(2 * intermediate_size), hidden_size], "column"))
    # mlp down
    gemm_shape_list.append(([seq, hidden_size, intermediate_size], "row"))

    return_A_list = [True, False]

    param_combos = list(
        itertools.product(
            [torch.bfloat16],
            MBS_LIST,
            gemm_shape_list,
            return_A_list,
            [fp8],
        )
    )
    perf_results = []

    for dtype, mbs, shape_dims, return_A, use_fp8 in tqdm(
        param_combos, desc=f"TiTan {model_name} AG_Gemm_Overlap Benchmarking"
    ):
        shape, parallel_mode = shape_dims
        m, n, k = shape
        if parallel_mode == "row":
            # bwd: grad_out -> ag -> grad_out_g, mm(grad_out_g, weight) -> dgrad
            A_shape = (mbs, m // TP_SIZE, n)
            B_shape = (n, k // TP_SIZE)
        else:
            # fwd: input -> ag -> input_g, mm(input_g, weight) -> output
            A_shape = (mbs, m // TP_SIZE, k)
            B_shape = (k, n // TP_SIZE)
        if use_fp8:
            A_shard = torch.randn(A_shape, device=device).to(pt_e4m3_type)
            B = torch.randn(B_shape, device=device).to(pt_e4m3_type).t().contiguous().t()
            A_scale = torch.tensor(0.1, device=device)
            B_scale = torch.tensor(0.1, device=device)
            kwargs = {"A_scale": A_scale, "B_scale": B_scale, "output_dtype": dtype}
        else:
            A_shard = torch.randn(A_shape, device=device).to(dtype)
            B = torch.randn(B_shape, device=device).to(dtype)
            kwargs = None

        base_func = titan_base_scaled_ag_gemm
        compile_func = titan_compile_ag_gemm(base_func, A_shard, B, kwargs, TP_GROUP, return_A, True)

        torch.cuda.synchronize()
        torch.distributed.barrier()

        with ctx:
            base_output, base_time = perf_func(
                partial(base_func, A_shard, B, kwargs, TP_GROUP, return_A),
                iters=NUM_ITERS,
                warmup_iters=NUM_WARMUP,
            )

            torch.cuda.synchronize()
            torch.distributed.barrier()

            compile_output, compile_time = perf_func(
                partial(compile_func, A_shard, B, kwargs, TP_GROUP, return_A),
                iters=NUM_ITERS,
                warmup_iters=NUM_WARMUP,
            )

            torch.cuda.synchronize()
            torch.distributed.barrier()

            orig_overlap_func = titan_orig_gemm_rs(base_func, A_shard, B, kwargs, TP_GROUP, return_A, True)
            orig_overlap_output, orig_overlap_time = perf_func(
                partial(orig_overlap_func, A_shard, B, kwargs, TP_GROUP, return_A),
                iters=NUM_ITERS,
                warmup_iters=NUM_WARMUP,
            )

            torch.cuda.synchronize()
            torch.distributed.barrier()

            patch_overlap_func = titan_patch_gemm_rs(base_func, A_shard, B, kwargs, TP_GROUP, return_A, True)
            patch_overlap_output, patch_overlap_time = perf_func(
                partial(patch_overlap_func, A_shard, B, kwargs, TP_GROUP, return_A),
                iters=NUM_ITERS,
                warmup_iters=NUM_WARMUP,
            )
            torch.cuda.synchronize()
            torch.distributed.barrier()

        result = {
            "model": model_name,
            "tp_size": TP_SIZE,
            "mbs": mbs,
            "m": m,
            "n": n,
            "k": k,
            "parallel_mode": parallel_mode,
            "return_A": return_A,
            "dtype": str(dtype),
            "fp8": use_fp8,
            "base(ms)": base_time,
            "compiled(ms)": compile_time,
            "compiled(speedup)": base_time / compile_time,
            "orig_overlap(ms)": orig_overlap_time,
            "orig_overlap(speedup)": base_time / orig_overlap_time,
            "patch_overlap(ms)": patch_overlap_time,
            "patch_overlap(speedup)": base_time / patch_overlap_time,
        }
        perf_results.append(result)
        if profile:
            run_id = f"titan_tp{TP_SIZE}_b{mbs}_m{m}_n{n}_k{k}_{parallel_mode}_dtype_{dtype}_fp8_{use_fp8}_return_A_{return_A}"
            prof_dir = f"{report_dir_path}/prof/{run_id}"
            os.makedirs(prof_dir, exist_ok=True)
            ctx.export_chrome_trace(f"{prof_dir}/trace_rank{TP_GROUP.rank()}.json.gz")

    csv_filename = f"benchmark_ag_gemm_overlap_{model_name}_{'fp8' if fp8 else str(dtype)}_tp{TP_SIZE}.csv"
    csv_path = f"{report_dir_path}/reports/{csv_filename}"
    os.makedirs(f"{report_dir_path}/reports", exist_ok=True)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(perf_results[0].keys()))
        writer.writeheader()
        for result in perf_results:
            writer.writerow(result)

    json_filename = f"benchmark_ag_gemm_overlap_{model_name}_{'fp8' if fp8 else str(dtype)}_tp{TP_SIZE}.json"
    json_path = f"{report_dir_path}/reports/{json_filename}"
    with open(json_path, mode="w", encoding="utf-8") as jsonfile:
        json.dump(perf_results, jsonfile, indent=4)


def benchmark_megatron(report_dir_path, model_config, profile, fp8, TP_GROUP, TP_SIZE, device):
    ctx = get_torch_prof_ctx(profile)
    model_name = model_config["model"]
    seq = model_config["seqlen"]
    hidden_size = model_config["hidden_size"]
    intermediate_size = model_config["intermediate_size"]
    num_attention_heads = model_config["num_attention_heads"]
    num_key_value_heads = model_config["num_key_value_heads"]
    head_dim = model_config["head_dim"]

    # Generate shapes
    gemm_shape_list = []  # [[m, n, k]...]
    # attn qkv
    gemm_shape_list.append(
        (
            [
                seq,
                int((num_attention_heads + 2 * num_key_value_heads) * head_dim),
                hidden_size,
            ],
            "column",
        )
    )

    # attn out
    gemm_shape_list.append(([seq, hidden_size, hidden_size], "row"))
    # mlp gate+up
    gemm_shape_list.append(([seq, int(2 * intermediate_size), hidden_size], "column"))
    # mlp down
    gemm_shape_list.append(([seq, hidden_size, intermediate_size], "row"))

    param_combos = list(
        itertools.product(
            [torch.bfloat16],
            MBS_LIST,
            gemm_shape_list,
            [fp8],
        )
    )

    for dtype, mbs, shape_dims, use_fp8 in tqdm(
        param_combos, desc=f"Megatron {model_name} AG_Gemm_Overlap Benchmarking"
    ):
        shape, parallel_mode = shape_dims
        m, n, k = shape
        te_ag_gemm = TEAGGemm(
            TP_GROUP,
            TP_SIZE,
            mbs,
            m,
            n,
            k,
            parallel_mode,
            dtype,
            num_splits=4,
            use_fp8=use_fp8,
            device=device,
        )
        input, inp_quantizer, weight, rs_out = te_ag_gemm.prepare_input_output()

        torch.cuda.synchronize()
        torch.distributed.barrier()

        with ctx:
            te_restore()
            base_output, base_time = perf_func(
                partial(te_ag_gemm.run, input, inp_quantizer, weight, rs_out, False),
                iters=NUM_ITERS,
                warmup_iters=NUM_WARMUP,
            )

            torch.cuda.synchronize()
            torch.distributed.barrier()
            print(f"---------------{parallel_mode=} done----------------")

            te_patch()
            te_ag_gemm.prepare_ub()
            overlap_output, overlap_time = perf_func(
                partial(te_ag_gemm.run, input, inp_quantizer, weight, rs_out, True),
                iters=NUM_ITERS,
                warmup_iters=NUM_WARMUP,
            )

        torch.cuda.synchronize()
        torch.distributed.barrier()

    #     result = {
    #         "model": model_name,
    #         "tp_size": TP_SIZE,
    #         "mbs": mbs,
    #         "m": m,
    #         "n": n,
    #         "k": k,
    #         "parallel_mode": parallel_mode,
    #         "dtype": str(dtype),
    #         "fp8": use_fp8,
    #         "base(ms)": base_time,
    #         # "overlap(ms)": overlap_time,
    #         # "overlap(speedup)": base_time / overlap_time,
    #     }
    #     perf_results.append(result)
    #     if profile:
    #         run_id = f"megatron_tp{TP_SIZE}_b{mbs}_m{m}_n{n}_k{k}_{parallel_mode}_dtype_{dtype}_fp8_{use_fp8}"
    #         prof_dir = f"{report_dir_path}/prof/{run_id}"
    #         os.makedirs(prof_dir, exist_ok=True)
    #         ctx.export_chrome_trace(f"{prof_dir}/trace_rank{TP_GROUP.rank()}.json.gz")

    # csv_filename = f"benchmark_gemm_rs_overlap_{model_name}_{'fp8' if fp8 else str(dtype)}_tp{TP_SIZE}.csv"
    # csv_path = f"{report_dir_path}/reports/{csv_filename}"
    # os.makedirs(f"{report_dir_path}/reports", exist_ok=True)
    # with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=list(perf_results[0].keys()))
    #     writer.writeheader()
    #     for result in perf_results:
    #         writer.writerow(result)

    # json_filename = f"benchmark_gemm_rs_overlap_{model_name}_{'fp8' if fp8 else str(dtype)}_tp{TP_SIZE}.json"
    # json_path = f"{report_dir_path}/reports/{json_filename}"
    # with open(json_path, mode="w", encoding="utf-8") as jsonfile:
    #     json.dump(perf_results, jsonfile, indent=4)


def benchmark(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    assert world_size >= 2, "This script requires more than 2 processes."

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=5),
    )
    dist.barrier()
    torch.manual_seed(42 + rank)
    TP_GROUP = dist.group.WORLD
    TP_SIZE = dist.get_world_size()

    benchmark_dir = Path(args.report_dir_path)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    with open(args.model_config_path, "r", encoding="utf-8") as f:
        model_config_list: list[dict] = json.load(f)

    for model_config in model_config_list:
        model_name = model_config["model"]
        if args.model.upper() != "ALL" and args.model != model_name:
            continue

        if args.backend == "torchtitan":
            benchmark_titan(
                args.report_dir_path, model_config, args.profile, args.fp8, TP_GROUP, TP_SIZE, device
            )
        elif args.backend == "megatron":
            benchmark_megatron(
                args.report_dir_path, model_config, args.profile, args.fp8, TP_GROUP, TP_SIZE, device
            )
        else:
            raise ValueError(f"Only torchtitan and megatron supported, but {args.backend} is provided.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="If run all model, set --model=all. If only run specific model, set --model=xxx, for example --model=Llama2_7B.",
    )
    parser.add_argument("--model-config-path", type=str)
    parser.add_argument("--report-dir-path", type=str)
    parser.add_argument("--backend", type=str, help="torchtitan or megatron")
    parser.add_argument("--fp8", default=False, action="store_true")
    parser.add_argument(
        "--profile",
        default=False,
        action="store_true",
        help="dump torch.profiler.profile",
    )
    args = parser.parse_args()

    benchmark(args)
