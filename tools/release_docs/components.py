###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The mapping between release-notes table rows and snapshot facts.

Two things make this more than a dict:

1. Distribution names move between releases. Transformer Engine has shipped as
   `transformer-engine`, `transformer-engine-rocm-torch`, `transformer-engine-rocm-jax`,
   `transformer-engine-rocm7` and `transformer-engine-rocm10`; the JAX plugin went from
   `jax-rocm7-pjrt` to `jax-rocm10-pjrt` in v26.7. Each row therefore resolves
   through a candidate list, and the rendered label follows whichever name the
   image actually carries.
2. Several rows combine packages (`Optax / Orbax / Grain / tensorstore`) or carry
   hand-written prose (``ROCm | 7.15.0 (`rocm-sdk` 7.15.0a20260727)``). So `check`
   asserts the extracted value is *present* in the cell rather than equal to it,
   which lets an editor annotate a row without breaking verification.

Candidate lists were derived from the published v26.4-v26.7 images of both
families, not guessed.
"""

import re


def pip_any(snapshot, *names):
    """First candidate actually installed, so renames do not break a row."""
    for name in names:
        value = snapshot.get("pip", {}).get(name)
        if value:
            return value, name
    return None, None


def _rocm(snapshot):
    """ROCm version, which is not always a pip package.

    v26.6 onward installs TheRock as a wheel (`rocm-sdk-core`). The v26.4/v26.5
    JAX images carry no ROCm pip or dpkg entry at all, so fall back to the
    embedded `+rocmX.Y.Z` in the JAX plugin version, which is where the
    hand-written notes took it from.
    """
    value, _ = pip_any(snapshot, "rocm-sdk-core", "rocm")
    if value:
        return value
    plugin, _ = pip_any(
        snapshot, "jax-rocm7-plugin", "jax-rocm10-plugin", "jax-rocm7-pjrt", "jax-rocm10-pjrt"
    )
    if plugin:
        # Release line only. The plugin embeds a nightly ("rocm7.14.0a20260526")
        # but the notes quote the line ("7.14.0"), which is the honest claim when
        # the image carries no ROCm package to read a full version from.
        match = re.search(r"rocm(\d+\.\d+\.\d+)", plugin)
        if match:
            return match.group(1)
    return None


def _joined(snapshot, groups):
    """Combined rows: one value per group, joined as the notes join them."""
    values = []
    for group in groups:
        value, _ = pip_any(snapshot, *group)
        values.append(value)
    if all(value is None for value in values):
        return None
    return " / ".join(value or "?" for value in values)


def _plugin_label(snapshot):
    """`jax-rocm7-pjrt / jax-rocm7-plugin` vs the rocm10 spelling used from v26.7."""
    for gen in ("7", "10"):
        if snapshot.get("pip", {}).get(f"jax-rocm{gen}-pjrt"):
            return f"jax-rocm{gen}-pjrt / jax-rocm{gen}-plugin"
    return "jax-rocm-pjrt / jax-rocm-plugin"


# Each spec: label (or label_of), label_re for matching existing tables, and a
# resolver returning the value string. Order is the order rows are rendered.
TE_CANDIDATES = (
    "transformer-engine-rocm-torch",
    "transformer-engine-rocm-jax",
    "transformer-engine-rocm7",
    "transformer-engine-rocm10",
    "transformer-engine",
)

COMMON_HEAD = [
    {
        "label": "ROCm",
        "label_re": r"^ROCm$",
        "resolve": _rocm,
    },
    {
        "label": "Python",
        "label_re": r"^Python$",
        "resolve": lambda s: s.get("python"),
    },
]

COMMON_TAIL = [
    {
        "label": "transformers / datasets",
        "label_re": r"^transformers / datasets$",
        "resolve": lambda s: _joined(s, [("transformers",), ("datasets",)]),
    },
    {
        "label": "NumPy",
        "label_re": r"^NumPy$",
        "resolve": lambda s: pip_any(s, "numpy")[0],
    },
    {
        "label": "amdsmi",
        "label_re": r"^amdsmi$",
        "resolve": lambda s: pip_any(s, "amdsmi")[0],
        "optional": True,
    },
]

PRIMUS = (
    COMMON_HEAD
    + [
        {
            "label": "PyTorch",
            "label_re": r"^PyTorch$",
            "resolve": lambda s: pip_any(s, "torch")[0],
        },
        {
            "label": "Transformer Engine",
            "label_re": r"^Transformer Engine$",
            "resolve": lambda s: pip_any(s, *TE_CANDIDATES)[0],
        },
        {
            "label": "Flash Attention",
            "label_re": r"^Flash Attention$",
            "resolve": lambda s: pip_any(s, "flash-attn")[0],
        },
        {
            "label": "hipBLASLt",
            "label_re": r"^hipBLASLt$",
            "resolve": lambda s: s.get("native", {}).get("hipblaslt"),
        },
        {
            "label": "Triton",
            "label_re": r"^Triton$",
            "resolve": lambda s: pip_any(s, "triton", "pytorch-triton-rocm")[0],
        },
        {
            "label": "RCCL",
            "label_re": r"^RCCL$",
            "resolve": lambda s: s.get("native", {}).get("rccl"),
        },
        {
            "label": "torchvision",
            "label_re": r"^torchvision$",
            "resolve": lambda s: pip_any(s, "torchvision")[0],
        },
        {
            "label": "torchaudio",
            "label_re": r"^torchaudio$",
            "resolve": lambda s: pip_any(s, "torchaudio")[0],
        },
        {
            "label": "APEX",
            "label_re": r"^APEX$",
            "resolve": lambda s: pip_any(s, "apex")[0],
        },
        {
            "label": "AITER",
            "label_re": r"^AITER$",
            "resolve": lambda s: pip_any(s, "amd-aiter", "aiter")[0],
        },
        {
            "label": "Primus-Turbo",
            "label_re": r"^Primus-Turbo$",
            "resolve": lambda s: pip_any(s, "primus-turbo")[0],
        },
        {
            "label": "torchao",
            "label_re": r"^torchao$",
            "resolve": lambda s: pip_any(s, "torchao")[0],
        },
        {
            "label": "FBGEMM",
            "label_re": r"^FBGEMM$",
            "resolve": lambda s: pip_any(s, "fbgemm-gpu-nightly-rocm", "fbgemm-gpu", "fbgemm-gpu-genai")[0],
        },
        {
            "label": "mamba-ssm / causal-conv1d / grouped_gemm",
            "label_re": r"^mamba-ssm / causal-conv1d / grouped_gemm$",
            "resolve": lambda s: _joined(s, [("mamba-ssm",), ("causal-conv1d",), ("grouped-gemm",)]),
        },
    ]
    + COMMON_TAIL
)

JAX = (
    COMMON_HEAD
    + [
        {
            "label": "JAX / jaxlib",
            "label_re": r"^JAX / jaxlib$",
            "resolve": lambda s: _joined(s, [("jax",), ("jaxlib",)]),
            "collapse_equal": True,
        },
        {
            "label_of": _plugin_label,
            "label_re": r"^jax-rocm\d*-pjrt / jax-rocm\d*-plugin$",
            "resolve": lambda s: _joined(
                s,
                [
                    ("jax-rocm7-pjrt", "jax-rocm10-pjrt"),
                    ("jax-rocm7-plugin", "jax-rocm10-plugin"),
                ],
            ),
            "collapse_equal": True,
        },
        {
            "label": "Transformer Engine",
            "label_re": r"^Transformer Engine$",
            "resolve": lambda s: pip_any(s, *TE_CANDIDATES)[0],
        },
        {
            "label": "hipBLASLt",
            "label_re": r"^hipBLASLt$",
            "resolve": lambda s: s.get("native", {}).get("hipblaslt"),
        },
        {
            "label": "RCCL",
            "label_re": r"^RCCL$",
            "resolve": lambda s: s.get("native", {}).get("rccl"),
        },
        {
            "label": "Flax",
            "label_re": r"^Flax$",
            "resolve": lambda s: pip_any(s, "flax")[0],
        },
        {
            "label": "TensorFlow",
            "label_re": r"^TensorFlow$",
            "resolve": lambda s: pip_any(s, "tensorflow-cpu", "tensorflow")[0],
        },
        {
            "label": "Optax / Orbax / Grain / tensorstore",
            "label_re": r"^Optax / Orbax / Grain / tensorstore$",
            "resolve": lambda s: _joined(
                s, [("optax",), ("orbax-checkpoint",), ("grain",), ("tensorstore",)]
            ),
        },
        {
            "label": "MaxText",
            "label_re": r"^MaxText$",
            "resolve": lambda s: (s.get("workspace_repos", {}).get("maxtext") or "")[:8] or None,
        },
    ]
    + COMMON_TAIL
)

SPECS = {"primus": PRIMUS, "jax": JAX}


def label_for(spec, snapshot):
    if "label_of" in spec:
        return spec["label_of"](snapshot)
    return spec["label"]


def resolve(spec, snapshot):
    """Value for a row, collapsing 'X / X' to 'X' where the notes do."""
    value = spec["resolve"](snapshot)
    if value and spec.get("collapse_equal"):
        parts = [part.strip() for part in value.split("/")]
        if len(set(parts)) == 1:
            return parts[0]
    return value


def rows_for(snapshot):
    """Rendered (label, value) rows for a snapshot, skipping absent optionals."""
    rows = []
    for spec in SPECS[snapshot["family"]]:
        value = resolve(spec, snapshot)
        if value is None:
            if spec.get("optional"):
                continue
            value = "UNRESOLVED"
        rows.append((label_for(spec, snapshot), value))
    return rows


def spec_for_label(family, label):
    for spec in SPECS[family]:
        if re.match(spec["label_re"], label.strip()):
            return spec
    return None
