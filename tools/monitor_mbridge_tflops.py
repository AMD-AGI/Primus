#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Real-time training TFLOP/s and consumed samples monitor.

This script monitors training output and calculates rolling average of
MODEL_TFLOP/s/GPU metrics every N iterations, while also tracking total
consumed samples.

Usage:
    # Monitor live training output (skips first 10 iterations by default)
    ./runner/primus-cli direct train posttrain --config ./examples/megatron_bridge/configs/MI300X/qwen3_32b_sft_posttrain.yaml 2>&1 | python tools/monitor_training_tflops.py
    
    # Or with custom averaging window and skip count
    ./runner/primus-cli direct train posttrain --config ./examples/megatron_bridge/configs/MI300X/qwen3_32b_sft_posttrain.yaml 2>&1 | python tools/monitor_training_tflops.py --window 10 --skip 10
    
    # Parse existing log file
    python tools/monitor_training_tflops.py --log-file training.log --window 10 --skip 10
"""

import argparse
import re
import sys
from collections import deque
from typing import Optional, TextIO


class TFLOPSMonitor:
    """Monitor and calculate rolling average of TFLOP/s metrics."""
    
    def __init__(self, window_size: int = 10, skip_iterations: int = 10):
        """
        Initialize the monitor.
        
        Args:
            window_size: Number of iterations to average over
            skip_iterations: Number of initial iterations to skip before collecting stats
        """
        self.window_size = window_size
        self.skip_iterations = skip_iterations
        self.tflops_values = deque(maxlen=window_size)
        self.iteration_count = 0  # Count of iterations used for averaging (after skip)
        self.total_iterations = 0  # Total iterations seen (including skipped)
        self.consumed_samples = 0  # Track total consumed samples
        
    def parse_tflops(self, line: str) -> Optional[float]:
        """
        Parse MODEL_TFLOP/s/GPU value from log line.
        
        Args:
            line: Log line to parse
            
        Returns:
            TFLOP/s value if found, None otherwise
        """
        # Pattern: GPU utilization: 1613.7MODEL_TFLOP/s/GPU
        match = re.search(r'GPU utilization:\s+([0-9.]+)MODEL_TFLOP/s/GPU', line)
        if match:
            return float(match.group(1))
        return None
    
    def parse_consumed_samples(self, line: str) -> Optional[int]:
        """
        Parse consumed samples value from log line.
        
        Args:
            line: Log line to parse
            
        Returns:
            Consumed samples value if found, None otherwise
        """
        # Pattern: consumed samples:         1488
        match = re.search(r'consumed samples:\s+(\d+)', line)
        if match:
            return int(match.group(1))
        return None
    
    def parse_iteration(self, line: str) -> Optional[int]:
        """
        Parse iteration number from log line.
        
        Args:
            line: Log line to parse
            
        Returns:
            Iteration number if found, None otherwise
        """
        # Pattern: iteration      186/     200
        match = re.search(r'iteration\s+(\d+)/\s*(\d+)', line)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            return current, total
        return None, None
    
    def process_line(self, line: str) -> Optional[str]:
        """
        Process a single log line and return averaging result if applicable.
        
        Args:
            line: Log line to process
            
        Returns:
            Average TFLOP/s message if window is complete, None otherwise
        """
        # Always print the original line
        print(line, end='', flush=True)
        
        # Parse consumed samples (always track this)
        consumed = self.parse_consumed_samples(line)
        if consumed is not None:
            self.consumed_samples = consumed
        
        # Parse TFLOP/s value
        tflops = self.parse_tflops(line)
        if tflops is not None:
            self.total_iterations += 1
            
            # Skip first N iterations (warm-up)
            if self.total_iterations <= self.skip_iterations:
                return None
            
            # Start collecting after skip period
            self.tflops_values.append(tflops)
            self.iteration_count += 1
            
            # Calculate and print average every window_size iterations
            if self.iteration_count % self.window_size == 0:
                avg_tflops = sum(self.tflops_values) / len(self.tflops_values)
                msg = (
                    f"\n{'='*80}\n"
                    f"[TFLOPS MONITOR] Average over last {len(self.tflops_values)} iterations: "
                    f"{avg_tflops:.2f} MODEL_TFLOP/s/GPU\n"
                    f"[TFLOPS MONITOR] Total consumed samples: {self.consumed_samples:,}\n"
                    f"{'='*80}\n"
                )
                return msg
        
        return None
    
    def monitor_stream(self, stream: TextIO):
        """
        Monitor a text stream (stdin or file) for TFLOP/s values.
        
        Args:
            stream: Text stream to monitor
        """
        try:
            for line in stream:
                result = self.process_line(line)
                if result:
                    print(result, flush=True)
        except KeyboardInterrupt:
            self.print_summary()
            sys.exit(0)
    
    def print_summary(self):
        """Print final summary statistics."""
        if self.tflops_values:
            avg_tflops = sum(self.tflops_values) / len(self.tflops_values)
            min_tflops = min(self.tflops_values)
            max_tflops = max(self.tflops_values)
            
            print(f"\n{'='*80}")
            print("[TFLOPS MONITOR] Final Summary:")
            print(f"  Total iterations seen: {self.total_iterations}")
            print(f"  Iterations skipped (warm-up): {self.skip_iterations}")
            print(f"  Iterations used for stats: {self.iteration_count}")
            print(f"  Total consumed samples: {self.consumed_samples:,}")
            print(f"  Average TFLOP/s/GPU: {avg_tflops:.2f}")
            print(f"  Min TFLOP/s/GPU: {min_tflops:.2f}")
            print(f"  Max TFLOP/s/GPU: {max_tflops:.2f}")
            print(f"  Window size: {self.window_size}")
            print(f"{'='*80}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Monitor and average MODEL_TFLOP/s/GPU metrics and consumed samples from training logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor live training (skips first 10 iterations by default)
  ./runner/primus-cli direct train posttrain --config config.yaml 2>&1 | python tools/monitor_training_tflops.py
  
  # Parse existing log file with custom skip and window
  python tools/monitor_training_tflops.py --log-file training.log --skip 10 --window 20
  
  # No skipping (start from iteration 1)
  python tools/monitor_training_tflops.py --skip 0 --window 10
"""
    )
    
    parser.add_argument(
        '--window',
        type=int,
        default=10,
        help='Number of iterations to average over (default: 10)'
    )
    
    parser.add_argument(
        '--skip',
        type=int,
        default=10,
        help='Number of initial iterations to skip for warm-up (default: 10)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file to parse (if not provided, reads from stdin)'
    )
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = TFLOPSMonitor(window_size=args.window, skip_iterations=args.skip)
    
    # Monitor from file or stdin
    if args.log_file:
        print(f"[TFLOPS MONITOR] Parsing log file: {args.log_file}", file=sys.stderr)
        print(f"[TFLOPS MONITOR] Skipping first {args.skip} iterations (warm-up)", file=sys.stderr)
        print(f"[TFLOPS MONITOR] Averaging window: {args.window} iterations\n", file=sys.stderr)
        try:
            with open(args.log_file, 'r') as f:
                monitor.monitor_stream(f)
            monitor.print_summary()
        except FileNotFoundError:
            print(f"Error: Log file not found: {args.log_file}", file=sys.stderr)
            sys.exit(1)
    else:
        print("[TFLOPS MONITOR] Monitoring stdin (pipe training output through this script)", file=sys.stderr)
        print(f"[TFLOPS MONITOR] Skipping first {args.skip} iterations (warm-up)", file=sys.stderr)
        print(f"[TFLOPS MONITOR] Averaging window: {args.window} iterations\n", file=sys.stderr)
        monitor.monitor_stream(sys.stdin)


if __name__ == '__main__':
    main()
