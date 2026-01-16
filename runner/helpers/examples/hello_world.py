#!/usr/bin/env python3
"""
Minimal entrypoint for validating `primus-cli direct --script ...` behavior.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arg1", default="val1")
    args = parser.parse_args()
    print(f"[primus script] hello world (arg1={args.arg1})")


if __name__ == "__main__":
    main()
