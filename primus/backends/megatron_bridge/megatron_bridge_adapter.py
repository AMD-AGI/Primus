###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MegatronBridge BackendAdapter implementation.

This is the Megatron-Bridge counterpart of ``MegatronAdapter``. It is responsible for:

    - Preparing the Megatron-Bridge backend environment
    - Converting Primus module config → Megatron-Bridge configuration
    - Providing the Megatron-Bridge trainer class to Primus
    - Exposing a backend version string for patching/diagnostics
"""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType
from typing import Any

from primus.backends.megatron_bridge.argument_builder import MegatronBridgeArgBuilder
from primus.backends.megatron_bridge.config_utils import (
    normalize_megatron_bridge_dataset_args,
)
from primus.core.backend.backend_adapter import BackendAdapter
from primus.core.backend.backend_registry import BackendRegistry
from primus.modules.module_utils import log_dict_aligned, log_rank_0


def _install_modelopt_stub() -> bool:
    """Install a stub ``modelopt`` module tree if the real package is missing.

    Bridge has 5 unconditional top-level imports of ``modelopt.torch.distill`` /
    ``modelopt.torch.opt.plugins`` (in gpt_provider.py, train.py, gpt_step.py,
    checkpointing.py, post_training/distillation.py). Even though SFT/LoRA
    training never exercises distillation or modelopt-quantization code paths,
    these imports must succeed at module load time or bridge fails to start.

    On ROCm container images the ``modelopt`` package is sometimes only
    partially installed (e.g. ``/opt/venv/.../modelopt/__init__.py`` exists
    but the ``modelopt/torch/`` subpackage is missing), causing
    ``ImportError: cannot import name 'torch' from 'modelopt'``.

    To unblock SFT without forcing the user to install
    ``nvidia-modelopt[torch]`` (which has its own ROCm compatibility risk),
    we register a lightweight stub package tree in ``sys.modules`` so the
    bridge imports go through. Each stub module's ``__getattr__`` returns
    a sentinel class:

      * ``isinstance(x, mtd.DistillationModel)`` returns False for any real
        SFT model -> bridge's ``gpt_step.py:289`` correctly falls through
        to the non-distillation branch.
      * Calling any stubbed function (e.g. ``save_modelopt_state(...)``)
        raises a clear RuntimeError pointing the user to
        ``pip install nvidia-modelopt[torch]``. SFT does not call any of
        these, so this only fires if someone enables
        ``restore_modelopt_state=True`` or distillation explicitly.

    Returns:
        True if a stub was installed; False if the real modelopt is already
        usable end-to-end and was left untouched.
    """
    # If real modelopt with the required submodules is importable, do nothing.
    try:
        import modelopt.torch.distill  # noqa: F401
        import modelopt.torch.distill.plugins.megatron  # noqa: F401
        import modelopt.torch.opt.plugins  # noqa: F401

        return False
    except ImportError:
        pass

    class _MissingModeloptSentinel:
        """Sentinel returned for every attribute on the stub modelopt tree.

        Instantiating it raises a clear error; isinstance checks against
        it return False for any real model object, which is exactly what
        bridge's ``isinstance(model, mtd.DistillationModel)`` site needs.
        """

        _attr_path: str = "modelopt.<unknown>"

        def __init__(self, *_args, **_kwargs):
            raise RuntimeError(
                f"Attempted to call '{type(self)._attr_path}' but "
                "nvidia-modelopt[torch] is not installed. SFT/LoRA paths "
                "should not hit this; if you enabled distillation or "
                "modelopt-quantization (e.g. restore_modelopt_state=True), "
                "install the real package via "
                "`pip install nvidia-modelopt[torch]`."
            )

    class _ModeloptStub(ModuleType):
        """Stub module whose attribute access yields per-name sentinel classes.

        Critical implementation notes:
          * Dunder names (``__file__``, ``__path__``, ``__spec__`` etc.)
            must NOT be intercepted -- Primus patch infrastructure walks
            ``sys.modules`` and calls e.g. ``module.__file__.endswith(...)``,
            and would crash if dunder access returned a sentinel class.
            We pre-set the most common dunders to safe defaults.
          * Submodule lookup must check ``sys.modules`` BEFORE minting a
            sentinel. Python implements ``import a.b.c as X`` via
            ``X = getattr(a.b, 'c')``; if ``__getattr__('c')`` returned a
            sentinel class instead of looking up ``sys.modules['a.b.c']``,
            the chain ``import modelopt.torch.distill.plugins.megatron``
            breaks at ``getattr(modelopt.torch.distill, 'plugins')``.
        """

        def __init__(self, name: str):
            super().__init__(name)
            # Make this look like a regular namespace package module so
            # tooling doing ``module.__file__.endswith(...)`` checks (which
            # appear in Primus patch infrastructure and various third-party
            # introspection libs) doesn't blow up.
            #
            # IMPORTANT: ``__file__`` MUST NOT be the empty string. Some
            # importlib paths and Python's module ``__repr__`` interpret
            # ``__file__ == ""`` as "this is a built-in module" and refuse
            # to operate on it (we observed Primus's runtime patches
            # crashing with ``<module 'modelopt.torch' from ''> is a
            # built-in module``). ``os.devnull`` is a real, harmless path
            # that endswith() handles cleanly and that no introspection
            # tool mistakes for a builtin.
            self.__file__ = os.devnull  # type: ignore[assignment]
            self.__path__: list[str] = []  # type: ignore[assignment]
            self.__loader__ = None
            self.__spec__ = None
            self.__package__ = name

        def __getattr__(self, name: str):
            # Let dunder lookups raise AttributeError naturally so callers
            # using ``getattr(module, '__xxx__', default)`` get their
            # default, and direct access falls through to the safe
            # defaults set in __init__ via ModuleType's normal lookup.
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)

            # If a child stub module exists in sys.modules, return THAT
            # instead of minting a new sentinel class -- essential so that
            # ``import a.b.c as X`` (which Python implements as
            # ``X = getattr(a.b, 'c')``) returns the real child stub
            # module rather than a sentinel.
            full_name = f"{self.__name__}.{name}"
            if full_name in sys.modules:
                return sys.modules[full_name]

            # Otherwise mint a per-name sentinel class. ``isinstance(x,
            # cls)`` returns False for any real object, and instantiating
            # cls raises a friendly error pointing the user at
            # ``pip install nvidia-modelopt[torch]``.
            cls = type(
                f"_MissingModeloptSentinel_{name}",
                (_MissingModeloptSentinel,),
                {"_attr_path": f"{self.__name__}.{name}"},
            )
            return cls

    stub_paths = (
        "modelopt",
        "modelopt.torch",
        "modelopt.torch.distill",
        "modelopt.torch.distill.plugins",
        "modelopt.torch.distill.plugins.megatron",
        "modelopt.torch.opt",
        "modelopt.torch.opt.plugins",
    )
    for path in stub_paths:
        if path not in sys.modules:
            sys.modules[path] = _ModeloptStub(path)

    # Wire up parent->child as real attribute references so that direct
    # attribute access on a parent stub returns the child stub (instead
    # of triggering __getattr__ to mint a sentinel). This is technically
    # redundant given the sys.modules lookup in __getattr__, but keeps
    # behavior consistent with how regular packages work.
    for path in stub_paths:
        if "." in path:
            parent_path, child_name = path.rsplit(".", 1)
            parent = sys.modules.get(parent_path)
            if isinstance(parent, _ModeloptStub):
                setattr(parent, child_name, sys.modules[path])

    return True


# Newer transformers classes that Bridge unconditionally imports at the top
# of various ``models/<flavor>_bridge.py`` files. If the user's container
# has an older transformers (e.g. the GLM-4.5V/Qwen3VL/Qwen3-Next classes
# were added in transformers >= 4.57 / similar), Bridge's chain-import
# from ``models/__init__.py`` blows up before SFT can even start. SFT for
# Qwen3-MoE / DeepSeek-V2-Lite never instantiates any of these, so a
# placeholder class is enough to unblock import.
_TRANSFORMERS_PLACEHOLDER_CLASSES: tuple[tuple[str, str], ...] = (
    ("Glm4vMoeForConditionalGeneration", "models/glm_vl/glm_45v_bridge.py"),
    ("Glm4MoeForCausalLM", "models/glm/glm45_bridge.py"),
    ("Qwen3VLForConditionalGeneration", "models/qwen_vl/qwen3_vl_bridge.py"),
    ("Qwen3VLMoeForConditionalGeneration", "models/qwen_vl/qwen3_vl_bridge.py"),
    ("Qwen3NextForCausalLM", "models/qwen/qwen3_next_bridge.py"),
    ("Qwen2_5_VLForConditionalGeneration", "models/qwen_vl/qwen25_vl_bridge.py"),
    ("Gemma3ForConditionalGeneration", "models/gemma_vl/gemma3_vl_bridge.py"),
    ("OlmoeForCausalLM", "models/olmoe/olmoe_bridge.py"),
    ("NemotronForCausalLM", "models/nemotron/nemotron_bridge.py"),
    ("GptOssForCausalLM", "models/gpt_oss/gpt_oss_bridge.py"),
    ("GptOssConfig", "models/gpt_oss/gpt_oss_bridge.py"),
)


def _install_transformers_stub() -> list[str]:
    """Inject placeholder classes for transformers symbols Bridge expects.

    Bridge's ``models/__init__.py`` chain-imports every supported model
    bridge unconditionally; each bridge file does
    ``from transformers import <ConcreteClass>`` at module top. If any of
    those classes is missing in the user's transformers installation, the
    whole bridge import chain dies with ImportError -- even when the user
    is only doing SFT on an unrelated flavor (e.g. Qwen3-MoE).

    For each known-missing class, we attach a minimal placeholder class
    to the ``transformers`` module so the ``from transformers import X``
    statement succeeds. The placeholder's ``__init__`` raises a clear
    error so any accidental instantiation surfaces a useful message
    pointing the user at upgrading transformers; SFT/LoRA on Qwen-MoE /
    DeepSeek-V2-Lite never instantiates any of these.

    Returns:
        List of class names that were stubbed (empty if transformers is
        already up-to-date or not importable).
    """
    try:
        import transformers
    except ImportError:
        return []

    stubbed: list[str] = []
    for class_name, used_by in _TRANSFORMERS_PLACEHOLDER_CLASSES:
        if hasattr(transformers, class_name):
            continue

        # Closure over name/used_by per-class so error messages are precise.
        def _make_init(_name: str, _used_by: str):
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError(
                    f"transformers.{_name} is a Primus placeholder. The "
                    f"real class is required by Megatron-Bridge's "
                    f"{_used_by} module but not available in your "
                    f"transformers installation. SFT/LoRA paths on "
                    f"unrelated model flavors should not hit this; if "
                    f"you actually need this model, upgrade via "
                    f"`pip install --upgrade transformers`."
                )

            return __init__

        placeholder = type(
            class_name,
            (object,),
            {
                "__module__": "transformers",
                "__init__": _make_init(class_name, used_by),
                "_primus_placeholder": True,
            },
        )
        setattr(transformers, class_name, placeholder)
        stubbed.append(class_name)

    return stubbed


# Bridge submodules whose import chain pulls in NV-only / GPU-vendor-locked
# packages (megatron-energon, qwen-vl-utils, causal-conv1d, ...) that are
# not installable on ROCm. Bridge's ``recipes/__init__.py`` and
# ``models/__init__.py`` unconditionally flat-import every supported flavor
# at top level, so we have to neutralise these submodules BEFORE the first
# ``import megatron.bridge`` -- otherwise the recipe load aborts even when
# the user only needs a text-only flavor (Qwen3-MoE / DeepSeek-V2).
#
# Anything listed here is import-only: SFT/LoRA on supported text flavors
# never instantiates a class from these submodules. If a future workload
# does need any of these, it must be run on an NVIDIA environment where
# Bridge's full optional dependency stack is installable.
_BRIDGE_OPTIONAL_PACKAGES: tuple[str, ...] = (
    # ---------- Recipe-level flat imports in ``recipes/__init__.py`` ----------
    # VLM recipes -> need megatron-energon / qwen-vl-utils (NV-only).
    "megatron.bridge.recipes.gemma3_vl",
    "megatron.bridge.recipes.qwen_vl",
    # State-space model recipes -> need causal-conv1d / mamba_ssm
    # (NV-only CUDA kernels; no ROCm port).
    "megatron.bridge.recipes.mamba",
    "megatron.bridge.recipes.nemotronh",
    # ---------- Model-level explicit imports in ``models/__init__.py`` --------
    # (``from X import (Y, Z)`` style; each block must succeed or the entire
    # ``import megatron.bridge.models`` aborts.)
    # VLM model bridges.
    "megatron.bridge.models.gemma_vl",
    "megatron.bridge.models.glm_vl",
    "megatron.bridge.models.nemotron_vl",
    "megatron.bridge.models.qwen_vl",
    "megatron.bridge.models.qwen_vl.modelling_qwen3_vl",
    # State-space model providers (deep dotted imports; mamba_provider
    # transitively pulls in causal-conv1d via megatron-core).
    "megatron.bridge.models.mamba",
    "megatron.bridge.models.mamba.mamba_provider",
    "megatron.bridge.models.nemotronh",
    "megatron.bridge.models.nemotronh.nemotron_h_provider",
    # ---------- Data-layer chain imports ------------------------------------
    "megatron.bridge.data.energon",
    "megatron.bridge.data.vlm_datasets",
    # ---------- External NV-only package (leaf cause of the chain) ----------
    "megatron.energon",
)


class _BridgeOptionalSentinel:
    """Per-class sentinel returned by ``_BridgeOptionalStub.__getattr__``.

    Instantiating a sentinel is a hard error so that a mis-routed code
    path fails loudly rather than silently producing garbage. The ``_attr_path``
    class attribute is overridden per-name when the sentinel subclass is
    minted in ``_BridgeOptionalStub.__getattr__``.
    """

    _attr_path: str = "<bridge.optional>"

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            f"Attempted to instantiate '{type(self)._attr_path}', but this "
            "Bridge submodule was stubbed out by Primus because its NV-only "
            "dependency stack (megatron-energon, qwen-vl-utils, ...) is not "
            "installable on ROCm. Text-only SFT/LoRA never reaches this "
            "code path; if you actually need this VLM/multimodal flavor, "
            "switch to an NVIDIA environment where Bridge's full optional "
            "dependency stack is officially supported."
        )


class _BridgeOptionalStub(ModuleType):
    """Empty Bridge submodule stub.

    * ``__all__ = []`` so ``from <stub> import *`` exports nothing.
    * ``__getattr__`` returns a per-name sentinel subclass so
      ``from <stub> import (Foo, Bar)`` succeeds with an opaque
      placeholder; SFT never actually uses these placeholders.
    * Cooperates with ``importlib`` (``__file__`` / ``__path__`` /
      ``__spec__`` / ``__loader__`` / ``__package__`` set explicitly).
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__file__ = os.devnull
        self.__path__: list[str] = []
        self.__loader__ = None
        self.__spec__ = None
        self.__package__ = name
        self.__all__: list[str] = []

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        full_name = f"{self.__name__}.{name}"
        if full_name in sys.modules:
            return sys.modules[full_name]
        sentinel_cls = type(
            f"_BridgeOptionalSentinel_{name}",
            (_BridgeOptionalSentinel,),
            {"_attr_path": full_name},
        )
        return sentinel_cls


def _install_bridge_optional_stubs() -> list[str]:
    """Stub Bridge VLM/multimodal submodules whose deps are NV-only.

    For each entry in ``_BRIDGE_OPTIONAL_PACKAGES`` we first try to import
    the real module; if and only if the import fails do we inject a stub
    into ``sys.modules``. After all stubs are installed we wire each
    parent module's attribute table to point at its child stub so that
    ``from parent import child`` resolves correctly even when the child
    is a stub.

    Must be called BEFORE the first ``import megatron.bridge`` so that
    Bridge's flat ``recipes/__init__.py`` / ``models/__init__.py`` see
    pre-populated ``sys.modules`` entries and skip executing the real
    submodule ``__init__.py`` files (which would chain-import the
    NV-only deps and crash).

    Returns:
        List of package names that were stubbed; empty if Bridge's full
        optional dependency stack is already installed.
    """
    stubbed: list[str] = []
    for pkg_name in _BRIDGE_OPTIONAL_PACKAGES:
        if pkg_name in sys.modules:
            continue
        try:
            importlib.import_module(pkg_name)
            continue
        except (ImportError, ModuleNotFoundError):
            pass
        sys.modules[pkg_name] = _BridgeOptionalStub(pkg_name)
        stubbed.append(pkg_name)

    for pkg_name in _BRIDGE_OPTIONAL_PACKAGES:
        if "." not in pkg_name:
            continue
        parent_path, child_name = pkg_name.rsplit(".", 1)
        parent = sys.modules.get(parent_path)
        child = sys.modules.get(pkg_name)
        if parent is not None and child is not None and isinstance(child, _BridgeOptionalStub):
            setattr(parent, child_name, child)

    return stubbed


class MegatronBridgeAdapter(BackendAdapter):
    """
    Complete BackendAdapter implementation for Megatron-Bridge.

    This adapter is designed to:
        - Integrate Megatron-Bridge's configuration with Primus configs
        - Apply setup/build_args patches via the unified patch system
        - Load the appropriate Megatron-Bridge trainer class
        - Handle bidirectional Hugging Face conversion capabilities
    """

    def __init__(self, framework: str = "megatron_bridge"):
        super().__init__(framework)
        self.third_party_dir_name = "Megatron-Bridge"

    def load_trainer_class(self, stage: str = "sft"):
        """
        Return the Megatron-Bridge Trainer class for the specified training stage.

        Lookup strategy:
          1. First try the BackendRegistry (stage-based registry from PR #523/#701).
             This is the preferred path because it lets each backend register
             its trainer class once at import time.
          2. Fall back to hard-coded imports for backwards compatibility with
             pre-registry callers and to keep the existing main-branch path
             functional (e.g. PR #647 added pretrain support via hard-coded
             imports without going through the registry).

        Args:
            stage: Training stage ("pretrain" or "sft").

        Returns:
            Trainer class for the specified stage.

        Raises:
            ValueError: If ``stage`` is not supported by either path.
        """
        # 1. Registry path (PR701 design).
        try:
            return BackendRegistry.get_trainer_class(self.framework, stage=stage)
        except (ValueError, AssertionError):
            # Trainer not registered for this stage — fall through to hard-coded path.
            pass

        # 2. Hard-coded fallback (main-branch compatibility, esp. for pretrain).
        if stage == "pretrain":
            from primus.backends.megatron_bridge.megatron_bridge_pretrain_trainer import (
                MegatronBridgePretrainTrainer,
            )

            return MegatronBridgePretrainTrainer
        elif stage == "sft":
            from primus.backends.megatron_bridge.megatron_bridge_posttrain_trainer import (
                MegatronBridgePosttrainTrainer,
            )

            return MegatronBridgePosttrainTrainer
        else:
            raise ValueError(
                f"[Primus:MegatronBridgeAdapter] Invalid stage: {stage!r}. "
                f"Supported stages: 'pretrain', 'sft'. "
                f"If using BackendRegistry, ensure register_trainer_class() was called."
            )

    def setup_backend_path(self, backend_path=None) -> str:
        """
        Set up Megatron-Bridge backend path, then add additional paths.

        Megatron-Bridge uses a src-layout structure:
            third_party/
            └── Megatron-Bridge/
                ├── src/
                │   └── megatron/
                │       └── bridge/
                └── 3rdparty/
                    └── Megatron-LM/
                        └── megatron/

        We need to add:
        1. Megatron-Bridge root (via parent class)
        2. Megatron-Bridge/src/ for 'import megatron.bridge'
        3. Megatron-Bridge/3rdparty/Megatron-LM/ for base Megatron functionality
        4. Install a modelopt stub if the real package is missing (so SFT
           can import bridge without ``nvidia-modelopt[torch]`` installed).
        """
        import os

        # 1. Call parent to set up the main backend path
        resolved = super().setup_backend_path(backend_path)

        # 2. Add Megatron-Bridge src directory
        src_path = os.path.join(resolved, "src")
        if os.path.isdir(src_path) and src_path not in sys.path:
            sys.path.insert(0, src_path)
            log_rank_0(f"sys.path.insert → {src_path}")

        # 3. Add Megatron-LM directory (from megatron-bridge/3rdparty/)
        megatron_lm_path = os.path.join(resolved, "3rdparty", "Megatron-LM")
        if os.path.isdir(megatron_lm_path) and megatron_lm_path not in sys.path:
            sys.path.insert(0, megatron_lm_path)
            log_rank_0(f"sys.path.insert → {megatron_lm_path}")

        # 4. Stub out modelopt for environments where nvidia-modelopt[torch]
        # is missing or partially installed. Must run BEFORE any
        # ``megatron.bridge.*`` module is imported (bridge has 5 top-level
        # ``import modelopt.torch.distill`` sites). SFT never executes
        # the underlying code paths, so a stub is sufficient.
        if _install_modelopt_stub():
            log_rank_0(
                "modelopt stub installed (nvidia-modelopt[torch] not found). "
                "Distillation / modelopt-quantization paths are disabled."
            )

        # 5. Stub out missing transformers classes that Bridge's
        # ``models/__init__.py`` chain-import requires. This must also
        # happen BEFORE any ``megatron.bridge.*`` module is imported.
        # SFT/LoRA on the active flavor (Qwen3-MoE / DeepSeek-V2-Lite)
        # never instantiates any of the stubbed classes, so unconditional
        # placeholders are safe.
        stubbed_classes = _install_transformers_stub()
        if stubbed_classes:
            log_rank_0(
                f"transformers placeholders installed for {len(stubbed_classes)} "
                f"missing class(es): {stubbed_classes}. Affected model bridges "
                f"are import-only and must not be instantiated for SFT."
            )

        # 6. Stub out Bridge VLM/multimodal submodules whose import chain
        # requires NV-only packages (megatron-energon, qwen-vl-utils,
        # causal-conv1d, ...) that are not installable on ROCm. Bridge's
        # ``recipes/__init__.py`` and ``models/__init__.py`` flat-import
        # all flavors at top level, so a single missing VLM dep currently
        # blocks every text-only flavor too. Must happen BEFORE the first
        # ``import megatron.bridge`` so the stubs win the ``sys.modules``
        # check and Bridge's real submodule ``__init__.py`` files don't
        # execute.
        bridge_optional_stubbed = _install_bridge_optional_stubs()
        if bridge_optional_stubbed:
            log_rank_0(
                f"Bridge optional stubs installed for {len(bridge_optional_stubbed)} "
                f"VLM/multimodal package(s): {bridge_optional_stubbed}. "
                "These flavors are import-only in this environment; "
                "instantiating any of them raises RuntimeError on purpose."
            )

        return resolved

    def convert_config(self, params: Any):
        """Convert Primus params to Megatron-Bridge argument Namespace."""
        builder = MegatronBridgeArgBuilder()
        builder.update(params)

        # Produce the final Megatron-Bridge Namespace
        bridge_args = builder.finalize()
        normalize_megatron_bridge_dataset_args(bridge_args)

        log_rank_0(
            f"[Primus:MegatronBridgeAdapter] Converted config → {len(vars(bridge_args))} Megatron-Bridge args"
        )

        log_dict_aligned("Megatron-Bridge args", bridge_args)

        return bridge_args

    def detect_backend_version(self) -> str:
        """Detect Megatron-Bridge version via AST parsing (avoids __init__.py execution)."""
        import ast
        import sys
        from pathlib import Path

        def parse_version(package_info_path: Path) -> str:
            tree = ast.parse(package_info_path.read_text())
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    name = getattr(node.targets[0], "id", None)
                    if name == "__version__":
                        return ast.literal_eval(node.value)
            raise RuntimeError(f"__version__ not found in {package_info_path}")

        for path in sys.path:
            package_info_path = Path(path) / "megatron" / "bridge" / "package_info.py"
            if package_info_path.exists():
                return parse_version(package_info_path)

        raise RuntimeError("Cannot locate megatron/bridge/package_info.py in sys.path")
