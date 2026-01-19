###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import socket
from pathlib import Path

import markdown2
import torch
import torch.distributed as dist
from weasyprint import HTML

from primus.tools.preflight.global_vars import RANK, WORLD_SIZE


def log(msg):
    if RANK == 0:
        print(msg, flush=True)


def extract_number(key):
    return int(key.rstrip("MB"))


def create_dir(dir):
    path = Path(dir)
    try:
        # Recursively create the dir. If it already exists, do nothing.
        path.mkdir(parents=True, exist_ok=True)
        print(f"Dir {dir} created successfully or already exists.")
    except PermissionError:
        print(f"Permission denied to create dir {dir}.")
    except Exception as e:
        print(f"An error occurred while creating dir {dir}: {e}")


def gather_hostnames():
    hostname = socket.gethostname()
    if RANK == 0:
        all_hostnames = [None for _ in range(WORLD_SIZE)]
        dist.gather_object(hostname, all_hostnames, dst=0)
        return all_hostnames
    else:
        dist.gather_object(hostname, None, dst=0)
        return None


def remove_file(file_path):
    if RANK == 0:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"{file_path} deleted.", flush=True)
    dist.barrier(device_ids=[torch.cuda.current_device()])


def extract_first_middle_last(lst):
    if not lst:
        return []

    n = len(lst)
    if n == 1:
        return [lst[0]]
    elif n == 2:
        return [lst[0], lst[1]]
    else:
        return [lst[0], lst[n // 2], lst[-1]]


def md_to_pdf(md_path, pdf_path):
    with open(md_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    html = markdown2.markdown(markdown_text, extras=["tables", "fenced-code-blocks", "footnotes"])

    # Add CSS to ensure that images do not overflow the page width
    css = """
    <style>
        img {
            max-width: 100%;
            height: auto;
        }
        table {
            width: 100%;
            table-layout: fixed;
            border-collapse: collapse;
            word-wrap: break-word;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 4px;
            font-size: 10px;
            word-wrap: break-word;
        }
    </style>
    """

    # Combine the CSS and HTML content
    html_with_css = css + html

    HTML(string=html_with_css, base_url=os.path.dirname(md_path)).write_pdf(pdf_path)
    print(f"✅ PDF Report saved to: {pdf_path}")


def get_first_ib_unidirectional_bandwidth():
    ib_path = "/sys/class/infiniband"

    # Check if the InfiniBand path exists
    if not os.path.exists(ib_path):
        log("No InfiniBand device found.")
        return 0

    # List available InfiniBand devices
    ib_devs = os.listdir(ib_path)
    if not ib_devs:
        log("No InfiniBand devices detected.")
        return 0

    # Use the first detected InfiniBand device
    ib_dev = ib_devs[0]

    # Get the ports directory for this device (typically contains "1")
    port_path = os.path.join(ib_path, ib_dev, "ports")
    port = sorted(os.listdir(port_path))[0]

    # Read the rate of the port from the 'rate' file
    # Bidirectional Bandwidth
    rate_path = os.path.join(port_path, port, "rate")
    with open(rate_path) as f:
        rate_str = f.read().strip()  # e.g., "400 Gb/sec (4X EDR)"

    # Extract the numeric part of the rate and convert from Gb/s to GB/s
    gbps = float(rate_str.split()[0])
    # Unidirectional Bandwidth
    GBps = gbps / 8
    return GBps
