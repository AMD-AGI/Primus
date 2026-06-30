#!/usr/bin/env python3
"""
Validation script to check if the new maxtext v26.4 imports can be resolved.
This should be run inside the rocm/jax-training:maxtext-v26.4-jax0.9.1-te2.12.0 container.
"""

import sys
from pathlib import Path

# Add src to path like the adapter does
primus_root = Path(__file__).parent
backend_path = primus_root / "third_party" / "maxtext"
src_path = backend_path / "src"

if src_path.exists():
    sys.path.insert(0, str(src_path))
    print(f"✓ Added {src_path} to sys.path")
else:
    print(f"✗ Warning: {src_path} does not exist")
    print("  This is expected if running in container with maxtext at /workspace/maxtext")

def test_import(module_path, description):
    """Test importing a module and report the result."""
    try:
        parts = module_path.split(".")
        if len(parts) == 1:
            exec(f"import {module_path}")
        else:
            # Handle "from x.y.z import a" style
            parent = ".".join(parts[:-1])
            child = parts[-1]
            exec(f"from {parent} import {child}")
        print(f"✓ {description}: {module_path}")
        return True
    except ImportError as e:
        print(f"✗ {description}: {module_path} - {e}")
        return False

print("\n=== Testing MaxText v26.4 imports ===\n")

all_passed = True

# Test new v26.4 imports
all_passed &= test_import("maxtext", "Base package")
all_passed &= test_import("maxtext.trainers.pre_train.train", "Train module")
all_passed &= test_import("maxtext.configs.pyconfig", "PyConfig module")
all_passed &= test_import("maxtext.utils.max_logging", "Logging module")

# Try to check version
try:
    import maxtext
    if hasattr(maxtext, "__version__"):
        print(f"\n✓ MaxText version: {maxtext.__version__}")
    else:
        print("\n✗ MaxText.__version__ not found")
        all_passed = False
except Exception as e:
    print(f"\n✗ Failed to check version: {e}")
    all_passed = False

print("\n" + "="*50)
if all_passed:
    print("✓ All imports validated successfully!")
    sys.exit(0)
else:
    print("✗ Some imports failed!")
    sys.exit(1)
