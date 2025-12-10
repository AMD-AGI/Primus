#!/usr/bin/env python3
"""
Test script for Mamba support in Primus.

This script validates that:
1. get_model_provider can be called with model_type='mamba'
2. The configuration system recognizes model_type
3. Model configs can be loaded properly
"""

import sys
import os

# Add Primus to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_import_utils():
    """Test that import_utils supports model_type parameter."""
    print("Testing import_utils.get_model_provider...")
    
    from primus.core.utils.import_utils import get_model_provider
    
    # Test GPT model type (default)
    try:
        provider_gpt = get_model_provider(model_type='gpt')
        print("✓ get_model_provider(model_type='gpt') works")
    except Exception as e:
        print(f"✗ get_model_provider(model_type='gpt') failed: {e}")
        return False
    
    # Test Mamba model type
    try:
        provider_mamba = get_model_provider(model_type='mamba')
        print("✓ get_model_provider(model_type='mamba') works")
    except Exception as e:
        print(f"✗ get_model_provider(model_type='mamba') failed: {e}")
        return False
    
    return True


def test_model_configs():
    """Test that Mamba model configs exist and are valid YAML."""
    print("\nTesting Mamba model configurations...")
    
    import yaml
    from pathlib import Path
    
    primus_root = Path(__file__).parent.parent
    model_configs = [
        "primus/configs/models/megatron/mamba_base.yaml",
        "primus/configs/models/megatron/mamba_370M.yaml",
        "primus/configs/models/megatron/mamba_1.4B.yaml",
        "primus/configs/models/megatron/mamba_hybrid_2.8B.yaml",
    ]
    
    all_valid = True
    for config_path in model_configs:
        full_path = primus_root / config_path
        if not full_path.exists():
            print(f"✗ Config file not found: {config_path}")
            all_valid = False
            continue
        
        try:
            with open(full_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✓ {config_path} is valid YAML")
            
            # Check for model_type in mamba configs
            if 'mamba' in config_path and config.get('model_type') != 'mamba':
                print(f"  ⚠ Warning: model_type not set to 'mamba' in {config_path}")
        except Exception as e:
            print(f"✗ {config_path} failed to load: {e}")
            all_valid = False
    
    return all_valid


def test_example_configs():
    """Test that example pretrain configs exist and are valid."""
    print("\nTesting example pretrain configurations...")
    
    import yaml
    from pathlib import Path
    
    primus_root = Path(__file__).parent.parent
    example_configs = [
        "examples/megatron/configs/MI300X/mamba_370M-pretrain.yaml",
        "examples/megatron/configs/MI300X/mamba_hybrid_2.8B-pretrain.yaml",
    ]
    
    all_valid = True
    for config_path in example_configs:
        full_path = primus_root / config_path
        if not full_path.exists():
            print(f"✗ Config file not found: {config_path}")
            all_valid = False
            continue
        
        try:
            with open(full_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✓ {config_path} is valid YAML")
            
            # Basic structure checks
            if 'modules' not in config:
                print(f"  ⚠ Warning: 'modules' not found in {config_path}")
            elif 'pre_trainer' not in config['modules']:
                print(f"  ⚠ Warning: 'pre_trainer' not found in modules")
        except Exception as e:
            print(f"✗ {config_path} failed to load: {e}")
            all_valid = False
    
    return all_valid


def test_language_model_config():
    """Test that language_model.yaml has model_type option."""
    print("\nTesting language_model.yaml for model_type...")
    
    import yaml
    from pathlib import Path
    
    primus_root = Path(__file__).parent.parent
    config_path = primus_root / "primus/configs/models/megatron/language_model.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'model_type' in config:
            print(f"✓ model_type found in language_model.yaml: {config['model_type']}")
            return True
        else:
            print("✗ model_type not found in language_model.yaml")
            return False
    except Exception as e:
        print(f"✗ Failed to load language_model.yaml: {e}")
        return False


def test_documentation():
    """Test that documentation exists."""
    print("\nTesting documentation...")
    
    from pathlib import Path
    
    primus_root = Path(__file__).parent.parent
    doc_path = primus_root / "docs/MAMBA_SUPPORT.md"
    
    if doc_path.exists():
        print(f"✓ Mamba documentation exists: {doc_path}")
        return True
    else:
        print(f"✗ Documentation not found: {doc_path}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Primus Mamba Support Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("import_utils", test_import_utils()))
    results.append(("model_configs", test_model_configs()))
    results.append(("example_configs", test_example_configs()))
    results.append(("language_model_config", test_language_model_config()))
    results.append(("documentation", test_documentation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:25s} {status}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Mamba support is ready to use.")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

