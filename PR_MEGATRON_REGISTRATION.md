# test: Add comprehensive unit tests for Megatron backend registration

## 🎯 Motivation

The Megatron backend registration in `primus/backends/megatron/__init__.py` is **critical infrastructure** that makes the backend available to the Primus runtime. Without proper registration, the backend would be completely unusable.

However, there were **no tests** to validate this registration mechanism, making it vulnerable to:
- ❌ Accidental deletion or modification of registration code
- ❌ Import path errors that break registration
- ❌ Incorrect registration order or missing steps
- ❌ Silent failures that only surface at runtime

This PR adds comprehensive unit tests to ensure the registration mechanism works correctly and catches any future issues early.

## 📝 Changes

### 1. Implemented Registration in `__init__.py`

The `primus/backends/megatron/__init__.py` was previously empty. Now it properly registers the Megatron backend:

```python
from primus.backends.megatron.megatron_adapter import MegatronAdapter
from primus.backends.megatron.megatron_pretrain_trainer import MegatronPretrainTrainer
from primus.core.backend.backend_registry import BackendRegistry

# Three-step registration required for backend to work:
BackendRegistry.register_path_name("megatron", "Megatron-LM")
BackendRegistry.register_adapter("megatron", MegatronAdapter)
BackendRegistry.register_trainer_class("megatron", MegatronPretrainTrainer)
```

**Key aspects:**
- ✅ Uses correct import paths (files are in `megatron/` root, not subdirectories)
- ✅ Registers all three required components (path name, adapter, trainer class)
- ✅ Happens automatically at import time (no function call needed)

### 2. Added Comprehensive Test Suite

Created `test_megatron_registration.py` with **10 test cases** covering:

#### A. Basic Registration Validation (6 tests)
- ✅ `test_path_name_is_registered` - Path name mapping works
- ✅ `test_adapter_is_registered` - Adapter class registered correctly
- ✅ `test_trainer_class_is_registered` - Trainer class registered correctly
- ✅ `test_adapter_can_be_instantiated_via_registry` - End-to-end adapter creation works
- ✅ `test_trainer_class_can_be_retrieved` - Trainer retrieval through adapter works
- ✅ `test_megatron_in_available_backends_list` - Backend appears in available list

#### B. Registration Mechanism (2 tests)
- ✅ `test_registration_is_idempotent` - Re-importing doesn't cause errors
- ✅ `test_registration_happens_at_import_time` - No explicit call needed

#### C. Error Handling (2 tests)
- ✅ `test_missing_adapter_registration_would_fail` - Graceful failure when adapter missing
- ✅ `test_missing_trainer_registration_would_fail` - Graceful failure when trainer missing

## 📊 Test Results

```bash
tests/unit_tests/backends/megatron/test_megatron_registration.py
  ✅ test_path_name_is_registered
  ✅ test_adapter_is_registered
  ✅ test_trainer_class_is_registered
  ✅ test_adapter_can_be_instantiated_via_registry
  ✅ test_trainer_class_can_be_retrieved
  ✅ test_megatron_in_available_backends_list
  ✅ test_registration_is_idempotent
  ✅ test_registration_happens_at_import_time
  ✅ test_missing_adapter_registration_would_fail
  ✅ test_missing_trainer_registration_would_fail

10 passed in 0.14s ✅
```

## 💡 Why This Matters

### Before This PR
- ❌ No validation that registration works
- ❌ Silent failures possible (backend "exists" but isn't registered)
- ❌ No documentation of registration requirements
- ❌ Easy to accidentally break during refactoring

### After This PR
- ✅ **Automated validation** - Tests fail immediately if registration breaks
- ✅ **Self-documenting** - Tests show exactly what needs to be registered
- ✅ **CI protection** - Catches issues before merge
- ✅ **Refactoring confidence** - Safe to reorganize code structure

### Real-World Impact

These tests would have caught common issues like:
```python
# ❌ Wrong import path (file doesn't exist)
from primus.backends.megatron.adapters.megatron_adapter import MegatronAdapter

# ❌ Missing registration step
# BackendRegistry.register_adapter("megatron", MegatronAdapter)  # Forgot this!

# ❌ Typo in backend name
BackendRegistry.register_adapter("megatorn", MegatronAdapter)
```

## 🔍 Technical Details

### Registration Architecture

Primus uses a **plugin-based backend system** with three registration levels:

1. **Path Name** (`register_path_name`):
   - Maps backend name → directory in `third_party/`
   - Example: `"megatron"` → `"Megatron-LM"`
   - Used by `setup_backend_path()` to find backend code

2. **Adapter** (`register_adapter`):
   - Maps backend name → `BackendAdapter` class
   - Adapter handles config conversion and trainer creation
   - Must be a class, not an instance

3. **Trainer Class** (`register_trainer_class`):
   - Maps backend name → Trainer class (optional but recommended)
   - Enables adapter to load trainer via registry
   - Alternative: adapter can import trainer directly

### Test Design Principles

1. **Isolation**: Each test validates one specific aspect
2. **Clear errors**: Assertion messages explain exactly what went wrong
3. **Restoration**: Tests that modify registry restore original state
4. **Mock safety**: Mocks prevent actual filesystem/import operations
5. **Integration coverage**: Tests both individual components and end-to-end flow

## 📝 Files Changed

```
 primus/backends/megatron/__init__.py                           |  17 ++
 tests/unit_tests/backends/megatron/test_megatron_registration.py | 217 ++++++++++++
 2 files changed, 234 insertions(+)
```

## 🔗 Related Work

This PR complements:
- #324 - Megatron backend trainers
- #325 - Megatron backend adapter
- #327 - Unified train runtime orchestrator

Without proper registration, none of these components would be accessible at runtime.

## ✔️ Checklist

- [x] Registration code implemented in `__init__.py`
- [x] Import paths verified and corrected
- [x] 10 comprehensive test cases added
- [x] All tests pass (10/10)
- [x] No linter errors
- [x] Test covers positive cases (registration works)
- [x] Test covers negative cases (missing registration fails gracefully)
- [x] Test validates idempotency
- [x] Commit message follows convention

## 🚀 Testing Instructions

To verify the registration mechanism works:

```bash
# Run the registration tests
pytest tests/unit_tests/backends/megatron/test_megatron_registration.py -v

# Verify backend is available
python -c "from primus.core.backend.backend_registry import BackendRegistry; print(BackendRegistry.list_available_backends())"
# Should include 'megatron'

# Verify adapter can be created
python -c "from primus.core.backend.backend_registry import BackendRegistry; adapter = BackendRegistry.get_adapter('megatron'); print(type(adapter).__name__)"
# Should print: MegatronAdapter
```

## 📚 Future Work

Potential improvements for other backends:
- [ ] Add similar registration tests for other backends (torchtitan, etc.)
- [ ] Add integration test that creates a full trainer instance
- [ ] Add test for lazy loading mechanism
- [ ] Document registration requirements in developer guide

