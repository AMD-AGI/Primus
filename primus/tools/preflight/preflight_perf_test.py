###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import os

import torch
import torch.distributed as dist

from primus.tools.preflight.global_vars import LOCAL_RANK, RANK, set_hostnames
from primus.tools.preflight.inter_node_comm import run_inter_node_comm
from primus.tools.preflight.inter_node_comm_p2p import run_inter_node_comm_p2p
from primus.tools.preflight.inter_node_ring_p2p import run_inter_node_ring_p2p
from primus.tools.preflight.intra_node_comm import run_intra_node_comm
from primus.tools.preflight.preflight_args import add_preflight_parser
from primus.tools.preflight.square_gemm import run_square_gemm
from primus.tools.preflight.utility import (
    gather_hostnames,
    get_first_ib_unidirectional_bandwidth,
    log,
    md_to_pdf,
    remove_file,
)


def setup():
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group("nccl")
    set_hostnames(gather_hostnames())


def cleanup():
    dist.destroy_process_group()


def run_preflight(args):
    setup()

    if RANK == 0:
        bw = get_first_ib_unidirectional_bandwidth()
        log(f"=======IB Bandwidth roofline (GB/s)=======")
        log(f"Bandwidth of first IB device of Node 0 : {bw:.2f} GB/s")
        args.ib_bw = bw

        if not os.path.isdir(args.dump_path):
            log(f"mkdir {args.dump_path}")
            os.makedirs(args.dump_path)

    args.markdown_file = f"{args.dump_path}/{args.report_file_name}.md"
    args.pdf_file = f"{args.dump_path}/{args.report_file_name}.pdf"
    remove_file(args.markdown_file)

    # run tests
    run_square_gemm(args)
    run_intra_node_comm(args)
    run_inter_node_comm(args)
    run_inter_node_comm_p2p(args)
    run_inter_node_ring_p2p(args)

    if RANK == 0 and args.save_pdf:
        md_to_pdf(args.markdown_file, args.pdf_file)

    cleanup()


def main():
    parser = argparse.ArgumentParser()
    add_preflight_parser(parser)
    args = parser.parse_args()

    run_preflight(args)


if __name__ == "__main__":
    main()
