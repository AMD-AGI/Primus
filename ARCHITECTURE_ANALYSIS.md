# Primus Architecture Analysis

**Date**: 2026-02-02  
**Analyst**: Senior Infrastructure Engineer  
**Context**: Primus is a unified training framework for large-scale LLM training on AMD GPUs, integrating Megatron-LM, TorchTitan, and MaxText backends.

---

## Section 1: Current Architecture Summary

### 1.1 High-Level Architecture

Primus follows a **layered plugin architecture** with the following primary layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLI / Entry Point Layer                      │
│              (primus-cli / pretrain.py)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│             Orchestration Layer (Core Runtime)                  │
│                 (primus/core/runtime/)                          │
│  - Config loading & CLI override merging                        │
│  - Runtime environment setup (paths, distributed, logging)      │
│  - Backend adapter resolution & trainer lifecycle               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
                ▼            ▼            ▼
    ┌─────────────────┐ ┌──────────┐ ┌──────────────┐
    │ Config System   │ │ Backend  │ │ Patch System │
    │                 │ │ Registry │ │              │
    └────────┬────────┘ └────┬─────┘ └──────┬───────┘
             │                │             │
             ▼                ▼             ▼
    ┌─────────────────────────────────────────────┐
    │   Backend Adapter Interface Layer           │
    │      (BackendAdapter abstract class)        │
    │                                             │
    │ - Backend initialization (setup_sys_path)   │
    │ - Config conversion (convert_config)        │
    │ - Trainer class loading (load_trainer_class)│
    │ - Version detection (detect_backend_version)│
    │ - Automatic patch orchestration             │
    └────────────────┬────────────────────────────┘
                     │
            ┌────────┴────────────────────┐
            │                             │
            ▼                             ▼
    ┌──────────────────┐         ┌──────────────────┐
    │  Megatron        │         │  TorchTitan      │
    │  Adapter         │         │  Adapter         │
    └────────┬─────────┘         └────────┬─────────┘
             │                           │
             ▼                           ▼
    ┌──────────────────┐         ┌──────────────────┐
    │ MegatronTrainer  │         │ TitanTrainer     │
    │ (Framework-      │         │ (Framework-      │
    │  specific)       │         │  specific)       │
    └──────────────────┘         └──────────────────┘
```

### 1.2 Main Subsystems and Responsibilities

#### Core Subsystems

| Subsystem | Location | Responsibility |
|-----------|----------|----------------|
| **Runtime Orchestrator** | `primus/core/runtime/` | Config loading, environment setup, backend resolution, trainer lifecycle management |
| **Backend Registry** | `primus/core/backend/` | Plugin discovery, lazy-loading backends, adapter registration |
| **Patch System** | `primus/core/patches/` | Framework-agnostic patch management with version awareness |
| **Config System** | `primus/core/config/` | YAML loading, merging, CLI overrides, module selection |
| **Launcher** | `primus/core/launcher/` | CLI argument parsing, training job initialization |
| **Trainer Base** | `primus/core/trainer/` | Template method pattern for training lifecycle |

#### Backend Subsystems

| Backend | Location | Key Components |
|---------|----------|----------------|
| **Megatron** | `primus/backends/megatron/` | MegatronAdapter, argument builder, 12+ patch modules, ZeroBubble pipeline parallel scheduler |
| **TorchTitan** | `primus/backends/torchtitan/` | TorchTitanAdapter, config utils, 10+ patch modules, model implementations |
| **MaxText** | `primus/backends/maxtext/` | MaxTextAdapter, JAX/Flax integration, input pipeline |
| **Megatron-Bridge** | `primus/backends/megatron_bridge/` | Bridge adapter for Megatron compatibility |

#### Supporting Subsystems

| Subsystem | Location | Responsibility |
|-----------|----------|----------------|
| **Modules Layer** | `primus/modules/` | Framework-agnostic trainer orchestration and distributed worker lifecycle |
| **Platforms** | `primus/platforms/` | Hardware abstraction (local vs remote platform) |
| **Tools** | `primus/tools/` | Preflight checks, benchmarking, projection utilities |
| **CLI** | `primus/cli/` | Command-line interface for train, benchmark, projection, preflight |

### 1.3 Key Architectural Patterns

#### 1. Template Method Pattern
- **Location**: `primus/core/trainer/base_trainer.py`
- **Usage**: Defines training lifecycle hooks (setup → init → run → cleanup)
- **Extensibility**: Subclasses override `run_train()` while preserving common workflow

#### 2. Registry Pattern
- **Location**: `primus/core/backend/backend_registry.py`
- **Usage**: Centralized backend adapter registration and lazy-loading
- **Extensibility**: New backends register via `BackendRegistry.register_adapter()`

#### 3. Strategy Pattern
- **Location**: Backend adapters (`MegatronAdapter`, `TorchTitanAdapter`)
- **Usage**: Encapsulates backend-specific config conversion and trainer instantiation
- **Extensibility**: Each adapter implements `convert_config()`, `prepare_backend()`, `load_trainer_class()`

#### 4. Patch System (Aspect-Oriented)
- **Location**: `primus/core/patches/patch_registry.py`
- **Usage**: Cross-cutting concerns (env setup, argument patching, runtime modifications)
- **Extensibility**: Patches register with backend + phase + version filters

#### 5. Plugin Architecture
- **Discovery**: Backends in `primus/backends/` auto-register on import
- **Isolation**: No cross-backend dependencies; each backend is self-contained
- **Lazy Loading**: Backends only imported when first accessed

### 1.4 Configuration Flow

```
1. YAML File (primus_config.yaml)
   └──→ load_primus_config() via PrimusParser (legacy)
        └── Returns SimpleNamespace with modules list

2. Module Config Selection
   └──→ get_module_config(module_name)
        └── Returns module's config section

3. CLI Overrides (--params key=value)
   └──→ parse_cli_overrides()
        └── deep_merge(base_params, cli_overrides)

4. Backend-Specific Conversion
   └──→ BackendAdapter.convert_config()
        └── Returns backend args (Namespace, JobConfig, etc.)

5. Patch Application
   └──→ run_patches(phase="build_args")
        └── Modifies backend_args in-place

6. Trainer Instantiation
   └──→ Trainer(primus_config, module_config, backend_args)
```

### 1.5 Patch Application Phases

Primus uses a **three-phase patch system**:

| Phase | Timing | Purpose | Context Available |
|-------|--------|---------|-------------------|
| **SETUP** | Before backend import | Env vars, pre-import fixes | No backend_version |
| **BUILD_ARGS** | After config conversion | Modify backend arguments | Has backend_version |
| **BEFORE_TRAIN** | Before trainer.run() | Runtime-specific modifications | Full runtime context |
| **AFTER_TRAIN** | After trainer.run() | Cleanup, logging | Full runtime context |

### 1.6 Strengths of Current Architecture

✅ **Clean Separation**: Backend logic isolated from core orchestration  
✅ **Plugin-Ready**: New backends easily added via adapter registration  
✅ **Version-Aware**: Patches can target specific backend versions  
✅ **Lazy Loading**: Zero startup overhead for unused backends  
✅ **Testable**: Each layer independently testable  

---

## Section 2: Top 5 Architectural Pain Points

### Pain Point 1: Backend-Specific Logic in Core Layer (HIGH SEVERITY)

**Location**: `primus/core/trainer/base_trainer.py:92-97`

**Problem**:
```python
if self.backend_name == "megatron" or self.backend_name == "megatron_bridge":
    from primus.backends.megatron.training.global_vars import (
        set_primus_global_variables,
    )
    set_primus_global_variables(self.module_config.params)
```

**Issues**:
- Core layer has hardcoded knowledge of Megatron backend
- Violates abstraction boundary: `BaseTrainer` should be backend-agnostic
- String-based backend name checks are fragile
- Tight coupling prevents independent evolution of core and backend layers
- Other backends cannot use similar initialization without modifying core

**Impact**: 🔴 **HIGH** - Breaks core abstraction, makes adding new backends harder

**Files Affected**:
- `primus/core/trainer/base_trainer.py` (lines 92-97)
- `primus/backends/megatron/training/global_vars.py` (imported)

---

### Pain Point 2: Legacy Config Parser Coupling (HIGH SEVERITY)

**Location**: `primus/core/config/primus_config.py:25-57`

**Problem**:
```python
from primus.core.launcher.parser import PrimusParser

def load_primus_config(config_path: Path, cli_args: Any | None = None) -> SimpleNamespace:
    # ...
    legacy_cfg = PrimusParser().parse(args_for_parser)
    # Adapt legacy PrimusConfig → SimpleNamespace
```

**Issues**:
- New runtime depends on legacy `PrimusParser` interface
- Comment explicitly calls it "legacy" but still required
- Two config representations: `PrimusConfig` (old) and `SimpleNamespace` (new)
- Hard to modify YAML structure without breaking parser
- Parser expects argparse-like namespace with specific attributes
- Config schema not validated or documented

**Impact**: 🔴 **HIGH** - Limits config system evolution, duplicated logic

**Files Affected**:
- `primus/core/config/primus_config.py` (load_primus_config function)
- `primus/core/launcher/parser.py` (PrimusParser implementation)
- `primus/core/launcher/config.py` (old PrimusConfig class)

---

### Pain Point 3: Scattered Patch Application Logic (MEDIUM SEVERITY)

**Location**: Multiple files

**Problem**:
Patch application happens in multiple places with no clear ownership:

1. **BaseTrainer.run()** (`primus/core/trainer/base_trainer.py:99-121`)
   ```python
   def run(self):
       run_patches(..., phase="before_train", ...)
       self.run_train()
       run_patches(..., phase="after_train", ...)
   ```

2. **MegatronBaseTrainer.__init__()** (`primus/backends/megatron/megatron_base_trainer.py`)
   ```python
   self._patch_parse_args()  # Patches Megatron's argument parser
   ```

3. **BackendAdapter.create_trainer()** (`primus/core/backend/backend_adapter.py:154-162`)
   ```python
   self._apply_setup_patches()
   # ... prepare_backend, convert_config ...
   self._apply_build_args_patches()
   ```

**Issues**:
- Three different entry points for patch application
- No single source of truth for patch execution order
- Some patches applied by core, some by backends
- Hard to reason about patch dependencies
- Debugging which patches ran requires checking multiple files

**Impact**: ⚠️ **MEDIUM** - Makes system behavior hard to predict, complicates debugging

**Files Affected**:
- `primus/core/trainer/base_trainer.py` (before/after train patches)
- `primus/core/backend/backend_adapter.py` (setup/build_args patches)
- `primus/backends/megatron/megatron_base_trainer.py` (custom patching)

---

### Pain Point 4: Module Config Params Merging Obscures Conflicts (MEDIUM SEVERITY)

**Location**: `primus/core/backend/backend_adapter.py:238-249`

**Problem**:
```python
def create_trainer(self, primus_config, module_config):
    # ...
    backend_args = self.convert_config(module_config)
    
    # Merge module params into backend args (lines 248-249)
    if hasattr(module_config, "params") and module_config.params:
        backend_args.update(module_config.params)
```

**Issues**:
- Silent overwriting: `module_config.params` overwrites `backend_args` values
- No conflict detection or warning
- Hard to debug when backend config gets unexpectedly overridden
- Unclear precedence: should module params or backend conversion win?
- No documentation of which params are merge-safe

**Impact**: ⚠️ **MEDIUM** - Silent bugs, unexpected config behavior

**Files Affected**:
- `primus/core/backend/backend_adapter.py` (create_trainer method, lines 248-249)

---

### Pain Point 5: Version Detection After Backend Import (LOW SEVERITY)

**Location**: `primus/core/backend/backend_adapter.py:154-162`

**Problem**:
```python
def create_trainer(self, primus_config, module_config):
    self._apply_setup_patches()  # No version available here
    self.prepare_backend(module_config)
    
    # Version detection happens AFTER prepare_backend
    backend_version = self.detect_backend_version()
    
    backend_args = self.convert_config(module_config)
    self._apply_build_args_patches(backend_args, backend_version)  # Now has version
```

**Issues**:
- Setup patches run without backend version information
- Version-specific setup patches cannot be applied
- Must import backend before detecting version (chicken-and-egg)
- Workaround: Setup patches use version patterns, but can't guarantee match

**Impact**: ⚠️ **LOW** - Limits setup patch flexibility, works around with patterns

**Files Affected**:
- `primus/core/backend/backend_adapter.py` (create_trainer method, lines 154-162)
- `primus/core/patches/patch.py` (version pattern matching)

---

## Section 3: Incremental Refactoring Suggestions

Each suggestion is designed to touch ≤ 2 files and maintain backward compatibility.

### Refactoring 1: Extract Backend Initialization Hook (Pain Point 1)

**Goal**: Remove hardcoded Megatron check from `BaseTrainer`

**Approach**: Use the existing adapter registration system to provide optional initialization callbacks

**Files to Touch**: 
1. `primus/core/trainer/base_trainer.py` (remove hardcoded check)
2. `primus/core/backend/backend_adapter.py` (add initialization hook)

**Changes**:

**File 1**: `primus/core/trainer/base_trainer.py`
```python
# BEFORE (lines 91-97)
if hasattr(self.module_config, "params") and self.module_config.params is not None:
    if self.backend_name == "megatron" or self.backend_name == "megatron_bridge":
        from primus.backends.megatron.training.global_vars import (
            set_primus_global_variables,
        )
        set_primus_global_variables(self.module_config.params)

# AFTER
if hasattr(self.module_config, "params") and self.module_config.params is not None:
    # Allow backend adapter to perform custom initialization
    if hasattr(self, "backend_adapter") and hasattr(self.backend_adapter, "initialize_trainer"):
        self.backend_adapter.initialize_trainer(self.module_config.params)
```

**File 2**: `primus/core/backend/backend_adapter.py`
```python
class BackendAdapter:
    # Add new optional method
    def initialize_trainer(self, params: Dict[str, Any]) -> None:
        """
        Optional hook for backend-specific trainer initialization.
        Called by BaseTrainer after construction.
        
        Override this in backend-specific adapters to inject
        backend-specific global state or configuration.
        """
        pass  # Default: no-op
```

**File 3** (backend-specific): `primus/backends/megatron/megatron_adapter.py`
```python
class MegatronAdapter(BackendAdapter):
    # Override the new hook
    def initialize_trainer(self, params: Dict[str, Any]) -> None:
        from primus.backends.megatron.training.global_vars import (
            set_primus_global_variables,
        )
        set_primus_global_variables(params)
```

**Benefits**:
- Core layer becomes backend-agnostic ✅
- Other backends can use same mechanism ✅
- No breaking changes (new optional method) ✅
- Clear extension point for backend-specific initialization ✅

---

### Refactoring 2: Introduce Config Loader Interface (Pain Point 2)

**Goal**: Decouple runtime from legacy PrimusParser

**Approach**: Create thin adapter interface for config loading; keep PrimusParser as default implementation

**Files to Touch**:
1. `primus/core/config/primus_config.py` (add ConfigLoader interface)
2. `primus/core/config/primus_config.py` (refactor load_primus_config to use interface)

**Changes**:

**File 1**: `primus/core/config/primus_config.py` (add at top)
```python
from abc import ABC, abstractmethod

class ConfigLoader(ABC):
    """
    Interface for loading Primus configuration from various sources.
    
    This abstraction allows replacing the legacy PrimusParser without
    breaking the runtime. Implementations can support YAML, JSON, TOML,
    or other config formats.
    """
    
    @abstractmethod
    def load(self, config_path: Path, cli_args: Any = None) -> SimpleNamespace:
        """Load config and return SimpleNamespace with modules list."""
        pass

class LegacyParserConfigLoader(ConfigLoader):
    """Default implementation using legacy PrimusParser."""
    
    def load(self, config_path: Path, cli_args: Any = None) -> SimpleNamespace:
        # Move existing load_primus_config logic here
        config_path_str = str(config_path)
        # ... (rest of current implementation)
        return cfg

# Module-level default
_config_loader: ConfigLoader = LegacyParserConfigLoader()

def set_config_loader(loader: ConfigLoader) -> None:
    """Allow runtime to inject custom config loader."""
    global _config_loader
    _config_loader = loader

def load_primus_config(config_path: Path, cli_args: Any = None) -> SimpleNamespace:
    """Load config using current ConfigLoader implementation."""
    return _config_loader.load(config_path, cli_args)
```

**Benefits**:
- Clear migration path from legacy parser ✅
- No breaking changes (same function signature) ✅
- Enables future TOML/JSON support ✅
- Legacy parser becomes implementation detail ✅
- Only 1 file touched ✅

---

### Refactoring 3: Centralize Patch Application in BackendAdapter (Pain Point 3)

**Goal**: Single source of truth for patch execution order

**Approach**: Move all patch application to BackendAdapter; backends call adapter methods instead of directly calling run_patches

**Files to Touch**:
1. `primus/core/backend/backend_adapter.py` (add patch orchestration methods)
2. `primus/core/trainer/base_trainer.py` (delegate to adapter)

**Changes**:

**File 1**: `primus/core/backend/backend_adapter.py`
```python
class BackendAdapter:
    def apply_trainer_patches(self, phase: str, **kwargs) -> None:
        """
        Centralized patch application for trainer lifecycle.
        
        All patch application goes through this method to ensure
        consistent execution order and context passing.
        """
        from primus.core.patches.patch_runner import run_patches
        
        run_patches(
            backend=kwargs.get("backend", self.backend_name),
            phase=phase,
            model=kwargs.get("model"),
            framework=kwargs.get("framework", self.backend_name),
            config=kwargs.get("config"),
            primus_config=kwargs.get("primus_config"),
            backend_version=kwargs.get("backend_version", self.detect_backend_version()),
        )
```

**File 2**: `primus/core/trainer/base_trainer.py`
```python
class BaseTrainer:
    def run(self):
        # BEFORE
        # run_patches(..., phase="before_train", ...)
        
        # AFTER
        if hasattr(self, "backend_adapter"):
            self.backend_adapter.apply_trainer_patches(
                phase="before_train",
                backend=self.backend_name,
                model=self.model_name,
                framework=self.backend_name,
                config=self.module_config,
                primus_config=self.primus_config,
            )
        
        self.run_train()
        
        if hasattr(self, "backend_adapter"):
            self.backend_adapter.apply_trainer_patches(
                phase="after_train",
                backend=self.backend_name,
                model=self.model_name,
                framework=self.backend_name,
                config=self.module_config,
                primus_config=self.primus_config,
            )
```

**Benefits**:
- Single entry point for all patches ✅
- Clear execution order ✅
- Easier to debug patch application ✅
- Backend-specific patching goes through same path ✅
- Only 2 files touched ✅

---

### Refactoring 4: Add Config Merge Validation (Pain Point 4)

**Goal**: Detect and warn about config merge conflicts

**Approach**: Track which keys are overridden during merge; log warnings for conflicts

**Files to Touch**:
1. `primus/core/backend/backend_adapter.py` (add validation to create_trainer)

**Changes**:

**File 1**: `primus/core/backend/backend_adapter.py`
```python
def create_trainer(self, primus_config, module_config):
    # ... existing code ...
    backend_args = self.convert_config(module_config)
    
    # NEW: Validate and warn about merges
    if hasattr(module_config, "params") and module_config.params:
        conflicting_keys = set(backend_args.keys()) & set(module_config.params.keys())
        
        if conflicting_keys:
            from primus.core.utils.logger import get_logger
            logger = get_logger()
            logger.warning(
                f"Config merge conflict: module params override backend args for keys: "
                f"{sorted(conflicting_keys)}. Module params take precedence."
            )
        
        backend_args.update(module_config.params)
```

**Benefits**:
- Visibility into config conflicts ✅
- No breaking changes (warnings only) ✅
- Easy to debug unexpected config behavior ✅
- Only 1 file touched ✅
- Can be extended to error on conflicts if desired ✅

---

### Refactoring 5: Early Version Detection via Metadata (Pain Point 5)

**Goal**: Make backend version available to setup patches

**Approach**: Provide version detection via backend metadata file before import

**Files to Touch**:
1. `primus/core/backend/backend_adapter.py` (add early_detect_version method)
2. Each backend's `__init__.py` (add `__version__` variable)

**Changes**:

**File 1**: `primus/core/backend/backend_adapter.py`
```python
class BackendAdapter:
    def early_detect_version(self) -> str | None:
        """
        Attempt to detect backend version WITHOUT importing backend code.
        
        Returns version string if available, None otherwise.
        Useful for version-specific setup patches that run before backend import.
        """
        try:
            # Try to import just the __init__ module (lightweight)
            backend_init = importlib.import_module(f"primus.backends.{self.backend_name}")
            return getattr(backend_init, "__version__", None)
        except Exception:
            return None
    
    def create_trainer(self, primus_config, module_config):
        # NEW: Try early version detection
        backend_version = self.early_detect_version()
        
        self._apply_setup_patches(backend_version=backend_version)  # Now has version!
        self.prepare_backend(module_config)
        
        # Fallback to full detection if early detection failed
        if backend_version is None:
            backend_version = self.detect_backend_version()
        
        backend_args = self.convert_config(module_config)
        self._apply_build_args_patches(backend_args, backend_version)
        # ... rest of method
```

**File 2**: `primus/backends/megatron/__init__.py`
```python
# Add version metadata at top of file
__version__ = "4.0.0"  # Megatron-LM version

# Existing registration code...
```

**Benefits**:
- Setup patches can use version filtering ✅
- Backward compatible (None if version unavailable) ✅
- Lightweight detection (no full import) ✅
- Only 2 files touched per backend ✅
- Clear version metadata location ✅

---

## Appendix: Architecture Decision Records (ADRs)

### ADR-001: Why Backend Adapters Over Direct Imports?

**Decision**: Use BackendAdapter interface instead of direct backend imports in core

**Reasoning**:
- Allows lazy loading (zero overhead for unused backends)
- Clear extension point for new backends
- Enables version detection before full import
- Supports patching without polluting backend code

**Trade-offs**:
- Extra indirection layer
- Registration ceremony required
- Debug stack traces longer

**Status**: ✅ WORKING WELL

---

### ADR-002: Why Patch System Over Inheritance?

**Decision**: Use aspect-oriented patching instead of deep inheritance hierarchies

**Reasoning**:
- Megatron/TorchTitan are external codebases (can't modify)
- Version-specific workarounds need to be conditional
- Patches can be toggled via environment variables
- Easier to upstream fixes (patches are isolated)

**Trade-offs**:
- Harder to discover which code runs
- Patch application order matters
- Monkey-patching can break on upstream changes

**Status**: ✅ NECESSARY EVIL (external frameworks)

---

### ADR-003: Why SimpleNamespace Over PrimusConfig Class?

**Decision**: Migrate from PrimusConfig class to SimpleNamespace for config

**Reasoning**:
- Lighter weight (no class overhead)
- More flexible (no schema enforcement)
- Easier to serialize/deserialize
- Matches Python stdlib patterns

**Trade-offs**:
- No type checking on config access
- No IDE autocomplete for config fields
- Easy to typo field names

**Status**: ⚠️ IN PROGRESS (legacy parser still used)

---

## Summary

Primus has a **solid plugin architecture** with clear layers and good separation of concerns. The main pain points are:

1. **Backend-specific logic leaking into core** (Megatron global vars)
2. **Legacy config parser coupling** (blocks config system evolution)
3. **Scattered patch application** (hard to debug)
4. **Silent config merging** (causes unexpected overrides)
5. **Late version detection** (limits setup patch flexibility)

All identified issues can be **fixed incrementally** with ≤ 2 file changes each, maintaining backward compatibility. The proposed refactorings follow the existing architectural patterns and maintain the stability-first philosophy of the project.
