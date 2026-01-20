#!/usr/bin/env python3
"""
Patch transformers modeling_utils.py to add missing pytorch_utils imports.

This patch is required for Megatron-Bridge compatibility with transformers v4.57-release.
It adds explicit imports of pytorch_utils functions that Megatron-Bridge depends on.

Usage:
    python transformers_modeling_utils.py <transformers_dir>
    
Example:
    python transformers_modeling_utils.py /path/to/transformers
"""

import sys
from pathlib import Path


def patch_modeling_utils(transformers_dir: Path) -> bool:
    """
    Patch transformers/modeling_utils.py to add missing pytorch_utils imports.
    
    Args:
        transformers_dir: Path to transformers source directory
        
    Returns:
        True if patched, False if already patched
        
    Raises:
        FileNotFoundError: If modeling_utils.py doesn't exist
    """
    file = transformers_dir / "src" / "transformers" / "modeling_utils.py"
    
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")
    
    text = file.read_text(encoding="utf-8")
    
    # The import block to add
    block = """from .pytorch_utils import (  # noqa: F401
    Conv1D,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    id_tensor_storage,
    prune_conv1d_layer,
    prune_layer,
    prune_linear_layer,
)
"""
    
    # Check if already patched (idempotent)
    if block in text:
        print("[=] modeling_utils.py already patched")
        return False
    
    # Find the last import statement in the first 350 lines
    lines = text.splitlines(True)
    insert_at = 0
    
    for i, ln in enumerate(lines[:350]):
        if ln.startswith("import ") or ln.startswith("from "):
            insert_at = i + 1
    
    # Insert the import block
    lines.insert(insert_at, "\n" + block + "\n")
    
    # Write back
    file.write_text("".join(lines), encoding="utf-8")
    print(f"[+] Patched modeling_utils.py at line {insert_at}")
    
    return True


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <transformers_dir>", file=sys.stderr)
        print(f"\nExample: {sys.argv[0]} /home/user/third_party/transformers", file=sys.stderr)
        sys.exit(1)
    
    transformers_dir = Path(sys.argv[1])
    
    if not transformers_dir.exists():
        print(f"[ERROR] Directory not found: {transformers_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not transformers_dir.is_dir():
        print(f"[ERROR] Not a directory: {transformers_dir}", file=sys.stderr)
        sys.exit(1)
    
    try:
        patch_modeling_utils(transformers_dir)
        print("[OK] Patch completed successfully")
    except Exception as e:
        print(f"[ERROR] Failed to patch: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
