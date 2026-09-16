###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for tools/release_docs/.

Fixtures are trimmed samples of the artifacts a real training image ships, so
the parsers are covered without pulling a 50 GB image in CI. Values are taken
from the published v26.4-v26.7 images.
"""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_TOOLS = _ROOT / "tools/release_docs"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = _load("_common")
components = _load("components")
probe_image = _load("probe_image")
release_notes = _load("release_notes")


# --- _common parsers -------------------------------------------------------

PIP_LIST = """Package                        Version           Editable project location
------------------------------ ----------------- -------------------------
absl-py                        2.5.0
transformer_engine             2.17.0+rocm10.0.0
jax-rocm10-pjrt                0.11.0+rocm10.0.0
primus                         0.1.0             /workspace/Primus
"""


def test_parse_pip_list_skips_header_and_rule():
    packages = C.parse_pip_list(PIP_LIST)
    assert "package" not in packages
    assert packages["absl-py"] == "2.5.0"


def test_parse_pip_list_canonicalises_underscores():
    # The images ship `transformer_engine`; every lookup uses the dashed form.
    assert C.parse_pip_list(PIP_LIST)["transformer-engine"] == "2.17.0+rocm10.0.0"


def test_parse_pip_list_tolerates_editable_third_column():
    assert C.parse_pip_list(PIP_LIST)["primus"] == "0.1.0"


def test_parse_env_keeps_values_containing_equals():
    env = C.parse_env("MAXTEXT_BRANCH=release/v26.7\nFLAGS=--a=1 --b=2\nnot-a-pair\n")
    assert env["MAXTEXT_BRANCH"] == "release/v26.7"
    assert env["FLAGS"] == "--a=1 --b=2"
    assert "not-a-pair" not in env


def test_parse_dpkg_keeps_only_installed_rows():
    text = (
        "Desired=Unknown/Install/Remove/Purge/Hold\n"
        "ii  libnuma1:amd64   2.0.14-3ubuntu2   amd64   NUMA library\n"
        "rc  removed-pkg      1.0               amd64   leftover config\n"
    )
    packages = C.parse_dpkg(text)
    assert packages == {"libnuma1": "2.0.14-3ubuntu2"}


def test_version_key_orders_releases_numerically():
    versions = ["v26.10", "v26.5.1", "v26.5", "v26.7"]
    assert sorted(versions, key=C.version_key) == ["v26.5", "v26.5.1", "v26.7", "v26.10"]


def test_family_inferred_from_image_reference():
    assert C.family_for_image("rocm/jax-training:maxtext-v26.7") == "jax"
    assert C.family_for_image("rocm/primus:v26.7") == "primus"


# --- probe_image -----------------------------------------------------------


def test_native_library_versions_come_from_headers():
    # hipBLASLt and RCCL are not pip packages; they are read from ROCm headers.
    hipblaslt = """#define HIPBLASLT_VERSION_MAJOR     1
#define HIPBLASLT_VERSION_MINOR     4
#define HIPBLASLT_VERSION_PATCH     1
#define HIPBLASLT_VERSION_TWEAK     8d1ae90e"""
    assert probe_image.hipblaslt_version(hipblaslt) == "1.4.1-8d1ae90e"
    rccl = "#define NCCL_MAJOR 2\n#define NCCL_MINOR 30\n#define NCCL_PATCH 4"
    assert probe_image.rccl_version(rccl) == "2.30.4"


def test_missing_headers_yield_no_version():
    assert probe_image.hipblaslt_version("") is None
    assert probe_image.rccl_version("") is None


def test_split_sections_separates_the_probe_stream():
    stream = (
        "@@@RELEASE_DOCS_SECTION:python@@@\n"
        "Python 3.12.3\n"
        "@@@RELEASE_DOCS_SECTION:rccl@@@\n"
        "#define NCCL_MAJOR 2\n"
    )
    sections = probe_image.split_sections(stream)
    assert sections["python"] == "Python 3.12.3"
    assert sections["rccl"] == "#define NCCL_MAJOR 2"


# --- components ------------------------------------------------------------


def _snapshot(family, pip, **extra):
    snapshot = {"family": family, "pip": pip, "native": {}, "workspace_repos": {}, "python": "3.12.3"}
    snapshot.update(extra)
    return snapshot


def test_transformer_engine_resolves_across_renames():
    # TE has shipped under five distribution names across v26.4-v26.7.
    for name in (
        "transformer-engine",
        "transformer-engine-rocm-torch",
        "transformer-engine-rocm-jax",
        "transformer-engine-rocm7",
        "transformer-engine-rocm10",
    ):
        snapshot = _snapshot("primus", {name: "2.17.0"})
        spec = components.spec_for_label("primus", "Transformer Engine")
        assert components.resolve(spec, snapshot) == "2.17.0", name


def test_jax_plugin_label_follows_the_installed_generation():
    v266 = _snapshot("jax", {"jax-rocm7-pjrt": "0.11.0.post1", "jax-rocm7-plugin": "0.11.0.post1"})
    v267 = _snapshot(
        "jax", {"jax-rocm10-pjrt": "0.11.0+rocm10.0.0", "jax-rocm10-plugin": "0.11.0+rocm10.0.0"}
    )
    assert components._plugin_label(v266) == "jax-rocm7-pjrt / jax-rocm7-plugin"
    assert components._plugin_label(v267) == "jax-rocm10-pjrt / jax-rocm10-plugin"


def test_identical_paired_versions_collapse_to_one():
    # The notes write "JAX / jaxlib | 0.11.0", not "0.11.0 / 0.11.0".
    snapshot = _snapshot("jax", {"jax": "0.11.0", "jaxlib": "0.11.0"})
    spec = components.spec_for_label("jax", "JAX / jaxlib")
    assert components.resolve(spec, snapshot) == "0.11.0"


def test_differing_paired_versions_stay_joined():
    snapshot = _snapshot("jax", {"jax": "0.11.0", "jaxlib": "0.10.0"})
    spec = components.spec_for_label("jax", "JAX / jaxlib")
    assert components.resolve(spec, snapshot) == "0.11.0 / 0.10.0"


def test_rocm_falls_back_to_the_plugin_release_line():
    # v26.4/v26.5 JAX images carry no ROCm pip or dpkg entry at all. The release
    # line is the honest claim there, not the embedded nightly.
    snapshot = _snapshot("jax", {"jax-rocm7-plugin": "0.9.1+rocm7.14.0a20260526"})
    assert components._rocm(snapshot) == "7.14.0"


def test_rocm_prefers_the_installed_sdk_when_present():
    snapshot = _snapshot("primus", {"rocm-sdk-core": "7.15.0a20260727"})
    assert components._rocm(snapshot) == "7.15.0a20260727"


def test_absent_optional_row_is_dropped():
    # amdsmi shipped in v26.4/v26.5 JAX and was gone by v26.6; the row goes away
    # rather than rendering as unresolved.
    rows = dict(components.rows_for(_snapshot("jax", {"jax": "0.11.0"}, version="v26.6")))
    assert "amdsmi" not in rows


def test_unknown_label_has_no_rule():
    assert components.spec_for_label("primus", "Some Future Library") is None


# --- release_notes parsing -------------------------------------------------

NOTES = """## v26.6 (current)

### `rocm/primus:v26.6`

Megatron-LM, TorchTitan, and Megatron Bridge backends.

| | |
| --- | --- |
| Image ID | `4fcb3f210dc6` |
| Built | 2026-08-25 |
| Dockerfile | [`Dockerfile.primus-v26.6`](https://github.com/AMD-AGI/Primus/blob/main/x) |

| Software component | Version |
| ------------------ | ------- |
| ROCm | 7.15.0 (`rocm-sdk` 7.15.0a20260727) |
| NumPy | 2.5.2 |

### Primus source for v26.6

| | |
| --- | --- |
| Branch tip | `2aa05ead` (2026-08-25) |
| Megatron-LM | `d3528a21` |
"""


def test_image_section_stops_at_the_next_heading():
    # Regression: the block used to run on into "### Primus source for vX.Y",
    # so submodule commits were read as image software components.
    sections = release_notes.parse_image_sections(NOTES)
    assert len(sections) == 1
    labels = [label for label, _ in sections[0]["components"]]
    assert labels == ["ROCm", "NumPy"]
    assert "Branch tip" not in labels


def test_section_metadata_and_identity_are_extracted():
    section = release_notes.parse_image_sections(NOTES)[0]
    assert (section["family"], section["version"]) == ("primus", "v26.6")
    assert section["meta"]["Image ID"] == "4fcb3f210dc6"
    assert section["meta"]["Built"] == "2026-08-25"


def test_prose_annotation_is_preserved_in_the_cell():
    # `check` requires the cell to *contain* the extracted value, so an editor
    # can annotate a row without breaking verification.
    section = release_notes.parse_image_sections(NOTES)[0]
    rocm = dict(section["components"])["ROCm"]
    assert rocm == "7.15.0 (rocm-sdk 7.15.0a20260727)"
    assert "7.15.0a20260727" in rocm


def test_strip_cell_unwraps_backticks_and_links():
    assert release_notes.strip_cell("`abc`") == "abc"
    assert release_notes.strip_cell("[`Dockerfile.jax-v26.6`](http://x)") == "Dockerfile.jax-v26.6"


def test_parse_image_ref_handles_both_families():
    assert release_notes.parse_image_ref("rocm/jax-training:maxtext-v26.6") == ("jax", "v26.6")
    assert release_notes.parse_image_ref("rocm/primus:v26.6") == ("primus", "v26.6")
