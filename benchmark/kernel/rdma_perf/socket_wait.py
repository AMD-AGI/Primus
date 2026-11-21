###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Wait while a remote (host, port) endpoint stays open. Used as a teardown
# barrier for staged perftest runs.
#
# Ported verbatim from:
#   ROCm/dist-inf-cookbook : cluster-sphere/cluster-rdma-tests/slurm_scripts/
#                            socket_wait.py
###############################################################################
import socket
import time
import argparse


def is_port_open(host, port, timeout=2):
    """Check if a given host and port are accessible."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def wait_while_port_open(host, port, check_interval=5):
    """Wait while the remote port remains open."""
    print(f"Waiting while port {port} on {host} is open...")
    while is_port_open(host, port):
        time.sleep(check_interval)
    print(f"Port {port} on {host} is now closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wait while a remote port remains open.")
    parser.add_argument("--remote-ip", required=True, help="Remote server IP address.")
    parser.add_argument("--remote-port", type=int, required=True, help="Remote port number.")
    args = parser.parse_args()

    wait_while_port_open(args.remote_ip, args.remote_port)
