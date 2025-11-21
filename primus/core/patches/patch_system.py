###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus Patch System

A flexible, version-aware patching system for backend frameworks.

Design Goals:
    1. Handle version compatibility issues across Megatron/TorchTitan/etc
    2. Apply hotfixes without modifying upstream framework code
    3. Support model-specific patches (DeepSeek, Llama, Mixtral, etc)
    4. Provide conditional patching based on version/model/config
    5. Keep user training workflow unchanged

Architecture:
    - PatchRegistry: Central registry for all patches
    - Patch: Base class for all patch implementations
    - PatchContext: Runtime context for conditional patching
    - Version matching: Semantic version-based patch selection
"""

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class PatchPriority(Enum):
    """Patch execution priority levels."""

    CRITICAL = 0  # Must run first (e.g., import fixes, path setup)
    HIGH = 10  # Important patches (e.g., version compatibility)
    NORMAL = 50  # Standard patches (e.g., bug fixes)
    LOW = 100  # Optional enhancements


class PatchStatus(Enum):
    """Patch application status."""

    PENDING = "pending"
    APPLIED = "applied"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class PatchContext:
    """
    Runtime context for conditional patch application.

    Contains information needed to decide whether a patch should be applied.
    """

    framework: str  # e.g., "megatron", "torchtitan"
    framework_version: Optional[str] = None  # e.g., "0.8.0"
    model_name: Optional[str] = None  # e.g., "llama3_70B", "deepseek_v3"
    model_type: Optional[str] = None  # e.g., "gpt", "mamba"
    config: Optional[Dict[str, Any]] = None  # Runtime config
    python_version: Optional[str] = None  # e.g., "3.10"
    cuda_version: Optional[str] = None  # e.g., "12.1"

    def __post_init__(self):
        """Auto-detect versions if not provided."""
        if self.python_version is None:
            import sys

            self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"


class Patch(ABC):
    """
    Base class for all patches.

    Each patch should:
        1. Define applicability conditions (version, model, etc)
        2. Implement the patching logic
        3. Provide rollback capability if needed
    """

    def __init__(
        self,
        name: str,
        description: str,
        priority: PatchPriority = PatchPriority.NORMAL,
        framework: Optional[str] = None,
        version_range: Optional[str] = None,
        models: Optional[List[str]] = None,
    ):
        """
        Initialize patch.

        Args:
            name: Unique patch identifier
            description: Human-readable description
            priority: Execution priority
            framework: Target framework (None = all frameworks)
            version_range: Version constraint (e.g., ">=0.7.0,<0.9.0")
            models: List of applicable model names (None = all models)
        """
        self.name = name
        self.description = description
        self.priority = priority
        self.framework = framework
        self.version_range = version_range
        self.models = models or []
        self.status = PatchStatus.PENDING
        self.error: Optional[Exception] = None

    def should_apply(self, context: PatchContext) -> bool:
        """
        Determine if this patch should be applied in the given context.

        Args:
            context: Runtime context

        Returns:
            True if patch should be applied
        """
        # Check framework match
        if self.framework and context.framework != self.framework:
            return False

        # Check version range
        if self.version_range and context.framework_version:
            if not self._version_matches(context.framework_version, self.version_range):
                return False

        # Check model match
        if self.models and context.model_name:
            if context.model_name not in self.models:
                return False

        # Custom condition check
        return self.check_condition(context)

    def check_condition(self, context: PatchContext) -> bool:
        """
        Custom condition check (override in subclass if needed).

        Args:
            context: Runtime context

        Returns:
            True if custom conditions are met
        """
        return True

    @abstractmethod
    def apply(self, context: PatchContext) -> bool:
        """
        Apply the patch.

        Args:
            context: Runtime context

        Returns:
            True if patch applied successfully
        """

    def rollback(self, context: PatchContext) -> bool:
        """
        Rollback the patch (optional, override if needed).

        Args:
            context: Runtime context

        Returns:
            True if rollback successful
        """
        return True

    @staticmethod
    def _version_matches(version: str, constraint: str) -> bool:
        """
        Check if version matches constraint.

        Supports: ==, !=, >, >=, <, <=, comma-separated constraints

        Examples:
            ">=0.7.0,<0.9.0" matches "0.8.0"
            "==0.8.0" matches "0.8.0"
            "!=0.7.0" matches "0.8.0"
        """
        from packaging import version as pkg_version
        from packaging.specifiers import SpecifierSet

        try:
            spec = SpecifierSet(constraint)
            return pkg_version.parse(version) in spec
        except Exception:
            # Fallback to simple string comparison
            return version == constraint

    def __repr__(self):
        return f"<Patch {self.name} [{self.status.value}]>"


class FunctionPatch(Patch):
    """
    Patch that replaces or wraps a function/method.

    Common use cases:
        - Fix bugs in upstream functions
        - Add logging/monitoring
        - Change behavior for specific models
    """

    def __init__(
        self,
        name: str,
        description: str,
        target_module: str,
        target_function: str,
        patch_function: Callable,
        wrap: bool = False,
        **kwargs,
    ):
        """
        Initialize function patch.

        Args:
            target_module: Module path (e.g., "megatron.training.arguments")
            target_function: Function name to patch
            patch_function: Replacement or wrapper function
            wrap: If True, wrap original function; if False, replace it
        """
        super().__init__(name, description, **kwargs)
        self.target_module = target_module
        self.target_function = target_function
        self.patch_function = patch_function
        self.wrap = wrap
        self.original_function: Optional[Callable] = None

    def apply(self, context: PatchContext) -> bool:
        """Apply function patch."""
        try:
            # Import target module
            module = importlib.import_module(self.target_module)

            # Get original function
            if not hasattr(module, self.target_function):
                print(
                    f"[Primus:Patch] Warning: {self.target_module}.{self.target_function} not found, skipping"
                )
                self.status = PatchStatus.SKIPPED
                return False

            self.original_function = getattr(module, self.target_function)

            # Apply patch
            if self.wrap:
                # Wrap original function
                @wraps(self.original_function)
                def wrapper(*args, **kwargs):
                    return self.patch_function(self.original_function, *args, **kwargs)

                setattr(module, self.target_function, wrapper)
            else:
                # Replace function
                setattr(module, self.target_function, self.patch_function)

            self.status = PatchStatus.APPLIED
            print(f"[Primus:Patch] ✓ Applied: {self.name}")
            return True

        except Exception as e:
            self.status = PatchStatus.FAILED
            self.error = e
            print(f"[Primus:Patch] ✗ Failed: {self.name} - {e}")
            return False

    def rollback(self, context: PatchContext) -> bool:
        """Rollback function patch."""
        if self.original_function is None:
            return False

        try:
            module = importlib.import_module(self.target_module)
            setattr(module, self.target_function, self.original_function)
            self.status = PatchStatus.PENDING
            return True
        except Exception:
            return False


class AttributePatch(Patch):
    """
    Patch that modifies module/class attributes.

    Common use cases:
        - Override default values
        - Inject custom configurations
        - Fix constant definitions
    """

    def __init__(
        self,
        name: str,
        description: str,
        target_module: str,
        target_attribute: str,
        new_value: Any,
        **kwargs,
    ):
        super().__init__(name, description, **kwargs)
        self.target_module = target_module
        self.target_attribute = target_attribute
        self.new_value = new_value
        self.original_value: Optional[Any] = None

    def apply(self, context: PatchContext) -> bool:
        """Apply attribute patch."""
        try:
            module = importlib.import_module(self.target_module)

            # Save original value
            if hasattr(module, self.target_attribute):
                self.original_value = getattr(module, self.target_attribute)

            # Set new value
            setattr(module, self.target_attribute, self.new_value)

            self.status = PatchStatus.APPLIED
            print(f"[Primus:Patch] ✓ Applied: {self.name}")
            return True

        except Exception as e:
            self.status = PatchStatus.FAILED
            self.error = e
            print(f"[Primus:Patch] ✗ Failed: {self.name} - {e}")
            return False


class ImportPatch(Patch):
    """
    Patch that fixes import issues.

    Common use cases:
        - Add missing imports
        - Fix circular import issues
        - Inject compatibility shims
    """

    def __init__(
        self,
        name: str,
        description: str,
        target_module: str,
        imports: Dict[str, str],
        **kwargs,
    ):
        """
        Args:
            target_module: Module to patch
            imports: Dict of {name: source_module} to inject
        """
        super().__init__(name, description, **kwargs)
        self.target_module = target_module
        self.imports = imports

    def apply(self, context: PatchContext) -> bool:
        """Apply import patch."""
        try:
            module = importlib.import_module(self.target_module)

            for name, source_module in self.imports.items():
                source = importlib.import_module(source_module)
                setattr(module, name, getattr(source, name))

            self.status = PatchStatus.APPLIED
            print(f"[Primus:Patch] ✓ Applied: {self.name}")
            return True

        except Exception as e:
            self.status = PatchStatus.FAILED
            self.error = e
            print(f"[Primus:Patch] ✗ Failed: {self.name} - {e}")
            return False


class PatchRegistry:
    """
    Central registry for all patches.

    Manages patch registration, selection, and application.
    """

    _patches: List[Patch] = []
    _applied_patches: Set[str] = set()

    @classmethod
    def register(cls, patch: Patch):
        """Register a patch."""
        cls._patches.append(patch)
        print(f"[Primus:PatchRegistry] Registered patch: {patch.name}")

    @classmethod
    def register_function_patch(
        cls,
        name: str,
        description: str,
        target_module: str,
        target_function: str,
        patch_function: Callable,
        **kwargs,
    ):
        """Convenience method to register a function patch."""
        patch = FunctionPatch(
            name=name,
            description=description,
            target_module=target_module,
            target_function=target_function,
            patch_function=patch_function,
            **kwargs,
        )
        cls.register(patch)

    @classmethod
    def apply_patches(cls, context: PatchContext) -> Tuple[int, int]:
        """
        Apply all applicable patches for the given context.

        Args:
            context: Runtime context

        Returns:
            Tuple of (applied_count, failed_count)
        """
        print(f"\n[Primus:PatchSystem] Applying patches for {context.framework}...")
        print(f"[Primus:PatchSystem] Version: {context.framework_version}")
        print(f"[Primus:PatchSystem] Model: {context.model_name or 'N/A'}")

        # Sort patches by priority
        sorted_patches = sorted(cls._patches, key=lambda p: p.priority.value)

        applied_count = 0
        failed_count = 0

        for patch in sorted_patches:
            # Check if already applied
            if patch.name in cls._applied_patches:
                continue

            # Check if should apply
            if not patch.should_apply(context):
                patch.status = PatchStatus.SKIPPED
                continue

            # Apply patch
            if patch.apply(context):
                cls._applied_patches.add(patch.name)
                applied_count += 1
            else:
                failed_count += 1

        print(f"\n[Primus:PatchSystem] Summary:")
        print(f"  ✓ Applied: {applied_count}")
        print(f"  ⊘ Skipped: {len(sorted_patches) - applied_count - failed_count}")
        print(f"  ✗ Failed: {failed_count}")

        return applied_count, failed_count

    @classmethod
    def get_applicable_patches(cls, context: PatchContext) -> List[Patch]:
        """Get list of patches applicable to the context."""
        return [p for p in cls._patches if p.should_apply(context)]

    @classmethod
    def clear(cls):
        """Clear all registered patches (useful for testing)."""
        cls._patches.clear()
        cls._applied_patches.clear()

    @classmethod
    def list_patches(cls, framework: Optional[str] = None) -> List[Patch]:
        """List all registered patches, optionally filtered by framework."""
        if framework:
            return [p for p in cls._patches if p.framework == framework or p.framework is None]
        return cls._patches.copy()
