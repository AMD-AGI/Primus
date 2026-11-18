"""FlyDSL arch pin (ship13 K10): generative_recommenders/ops/flydsl/__init__.py.

FlyDSL 0.1.1 falls back to gfx942 when `rocm_agent_enumerator -name` fails or
exceeds 5 s, and the first gfx950-only MFMA intrinsic then aborts the process.
The package sets $FLYDSL_GPU_ARCH at import from the KFD topology unless an
arch is already given. These cases read fake sysfs trees under tmp_path: no
GPU, no FlyDSL, no /dev/kfd and no torch.

The package __init__ is stdlib-only and is loaded from its file under a private
name, with FLYDSL_GPU_ARCH preset so its import-time pin neither reads the real
sysfs nor changes the environment. Every test leaves os.environ exactly as it
found it (monkeypatch; checked by the autouse fixture). The last two cases
import the real package in a child process: the import stays stdlib-only, and
the import-time pin agrees with the parser on this host's real KFD root.
"""

import ast
import importlib.util
import inspect
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PKG_INIT = _REPO_ROOT / "generative_recommenders" / "ops" / "flydsl" / "__init__.py"
_ARCH_VARS = ("FLYDSL_GPU_ARCH", "HSA_OVERRIDE_GFX_VERSION")
_KFD_ROOT = "/sys/class/kfd/kfd/topology/nodes"
_SHIELD = "gfx950"
# KFD gfx_target_version per node as read in the lab container (R15 iteration
# 45): two CPU nodes (0), then eight MI350X GPUs.
_MI350X = (0, 0) + (90500,) * 8


def _load_package():
    spec = importlib.util.spec_from_file_location(
        "_flydsl_arch_pin_test_pkg", _PKG_INIT
    )
    mod = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ, {"FLYDSL_GPU_ARCH": _SHIELD}):
        spec.loader.exec_module(mod)
    return mod


pkg = _load_package()


def _environ():
    # pytest itself rewrites PYTEST_CURRENT_TEST at every phase.
    return {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}


@pytest.fixture(autouse=True)
def _environ_restored():
    before = _environ()
    yield
    assert _environ() == before


def _arch_env(monkeypatch, **values):
    """Make the two arch variables exactly `values` (absent unless given).

    setenv first so monkeypatch records each original state and also undoes
    whatever the code under test writes to these keys.
    """
    for name in _ARCH_VARS:
        monkeypatch.setenv(name, "")
        if name in values:
            monkeypatch.setenv(name, values[name])
        else:
            monkeypatch.delenv(name)


def _props(version) -> str:
    """An abridged KFD node properties file."""
    return (
        "cpu_cores_count 0\n"
        "simd_count 1024\n"
        "array_count 32\n"
        f"gfx_target_version {version}\n"
        "vendor_id 4098\n"
        "device_id 30112\n"
        "drm_render_minor 128\n"
    )


def _kfd(root: Path, *versions) -> Path:
    """<root>/<i>/properties for each version; returns root."""
    root.mkdir(parents=True, exist_ok=True)
    for i, version in enumerate(versions):
        (root / str(i)).mkdir()
        (root / str(i) / "properties").write_text(_props(version))
    return root


def _malformed_nodes(root: Path) -> None:
    """Nodes whose properties are unreadable or malformed; none may count."""

    def node(name: str) -> Path:
        path = root / name
        path.mkdir()
        return path

    (node("m_dir") / "properties").mkdir()  # IsADirectoryError
    (node("m_link") / "properties").symlink_to(root / "nowhere")  # dangling
    node("m_none")  # no properties file
    (node("m_text") / "properties").write_text("gfx_target_version abc\n")
    # Not ASCII: the whole file is ignored, its 90402 included.
    (node("m_bytes") / "properties").write_bytes(
        b"gfx_target_version 90402\n\xff\xfe\n"
    )
    (node("m_empty") / "properties").write_text("")
    (node("m_extra") / "properties").write_text("gfx_target_version 90402 1\n")
    (node("m_neg") / "properties").write_text("gfx_target_version -90402\n")
    (node("m_bare") / "properties").write_text("gfx_target_version\n")
    (node("m_key") / "properties").write_text("gfx_target_version_x 90402\n")
    (root / "m_file").write_text("gfx_target_version 90402\n")  # not a node dir


def test_mi350x_topology_gives_gfx950(tmp_path):
    root = _kfd(tmp_path / "nodes", *_MI350X)
    assert pkg._kfd_gfx_arch(root) == "gfx950"
    assert pkg._kfd_gfx_arch(str(root)) == "gfx950"


def test_mixed_arches_give_none(tmp_path):
    assert pkg._kfd_gfx_arch(_kfd(tmp_path / "nodes", 0, 90500, 90402)) is None


def test_cpu_nodes_only_give_none(tmp_path):
    assert pkg._kfd_gfx_arch(_kfd(tmp_path / "nodes", 0, 0)) is None


def test_missing_or_bad_root_gives_none(tmp_path):
    assert pkg._kfd_gfx_arch(tmp_path / "absent") is None
    assert pkg._kfd_gfx_arch(tmp_path) is None  # empty directory
    plain = tmp_path / "plain"
    plain.write_text(_props(90500))
    assert pkg._kfd_gfx_arch(plain) is None  # NotADirectoryError
    # Misuse still never raises: it runs at import of the trainer's package.
    root = _kfd(tmp_path / "nodes", *_MI350X)
    assert pkg._kfd_gfx_arch(bytes(root)) is None  # bytes + str join
    assert pkg._kfd_gfx_arch(None) is None


def test_malformed_and_unreadable_files_are_ignored(tmp_path):
    root = _kfd(tmp_path / "with_gpu", 0, 90500)
    _malformed_nodes(root)
    # Unreadable unless running as root (the gate container is); it holds the
    # same arch, so the result does not depend on who runs the test.
    (root / "m_mode").mkdir()
    (root / "m_mode" / "properties").write_text(_props(90500))
    (root / "m_mode" / "properties").chmod(0)
    assert pkg._kfd_gfx_arch(root) == "gfx950"

    root = _kfd(tmp_path / "without_gpu", 0)
    _malformed_nodes(root)
    assert pkg._kfd_gfx_arch(root) is None


@pytest.mark.parametrize(
    "version,name",
    [
        (90500, "gfx950"),
        (90402, "gfx942"),
        (90010, "gfx90a"),
        (90008, "gfx908"),
        (100300, "gfx1030"),
        (110000, "gfx1100"),
        (120001, "gfx1201"),
    ],
)
def test_gfx_name_from_target_version(tmp_path, version, name):
    root = _kfd(tmp_path / "nodes", 0, version, version)
    assert pkg._kfd_gfx_arch(root) == name


def test_pin_keeps_explicit_flydsl_gpu_arch(tmp_path, monkeypatch):
    _arch_env(monkeypatch, FLYDSL_GPU_ARCH="gfx942")
    root = _kfd(tmp_path / "nodes", *_MI350X)
    assert pkg._pin_flydsl_arch(root) == ("gfx942", "env")
    assert os.environ["FLYDSL_GPU_ARCH"] == "gfx942"
    assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ


def test_pin_keeps_hsa_override(tmp_path, monkeypatch):
    _arch_env(monkeypatch, HSA_OVERRIDE_GFX_VERSION="9.5.0")
    root = _kfd(tmp_path / "nodes", 0, 90402)
    assert pkg._pin_flydsl_arch(root) == ("9.5.0", "hsa_override")
    assert "FLYDSL_GPU_ARCH" not in os.environ
    assert os.environ["HSA_OVERRIDE_GFX_VERSION"] == "9.5.0"


def test_pin_flydsl_gpu_arch_wins_over_hsa_override(tmp_path, monkeypatch):
    _arch_env(monkeypatch, FLYDSL_GPU_ARCH="gfx950",
              HSA_OVERRIDE_GFX_VERSION="9.4.2")
    root = _kfd(tmp_path / "nodes", 0, 90402)
    assert pkg._pin_flydsl_arch(root) == ("gfx950", "env")
    assert os.environ["FLYDSL_GPU_ARCH"] == "gfx950"
    assert os.environ["HSA_OVERRIDE_GFX_VERSION"] == "9.4.2"


def test_pin_sets_arch_from_sysfs_when_unset(tmp_path, monkeypatch):
    _arch_env(monkeypatch)
    root = _kfd(tmp_path / "nodes", *_MI350X)
    assert pkg._pin_flydsl_arch(root) == ("gfx950", "kfd_sysfs")
    assert os.environ["FLYDSL_GPU_ARCH"] == "gfx950"
    assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ
    # A second call (or a child process) finds the variable set.
    assert pkg._pin_flydsl_arch(root) == ("gfx950", "env")


def test_pin_treats_empty_values_as_unset(tmp_path, monkeypatch):
    # FlyDSL tests the variables for truth, so an empty value pins nothing.
    _arch_env(monkeypatch, FLYDSL_GPU_ARCH="", HSA_OVERRIDE_GFX_VERSION="")
    root = _kfd(tmp_path / "nodes", *_MI350X)
    assert pkg._pin_flydsl_arch(root) == ("gfx950", "kfd_sysfs")
    assert os.environ["FLYDSL_GPU_ARCH"] == "gfx950"
    assert os.environ["HSA_OVERRIDE_GFX_VERSION"] == ""


@pytest.mark.parametrize("versions", [None, (0, 0), (0, 90500, 90402)])
def test_pin_leaves_env_unset_without_one_arch(tmp_path, monkeypatch, versions):
    _arch_env(monkeypatch)
    root = tmp_path / "nodes"
    if versions is not None:
        _kfd(root, *versions)
    assert pkg._pin_flydsl_arch(root) == (None, "none")
    assert "FLYDSL_GPU_ARCH" not in os.environ
    assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ


def test_default_root_and_import_time_pin():
    for fn in (pkg._kfd_gfx_arch, pkg._pin_flydsl_arch):
        assert inspect.signature(fn).parameters["root"].default == _KFD_ROOT
    # _load_package ran the module with FLYDSL_GPU_ARCH preset.
    assert pkg.FLYDSL_ARCH_PIN == (_SHIELD, "env")


def _child(code: str, env) -> tuple:
    proc = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=_REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return ast.literal_eval(proc.stdout.strip().splitlines()[-1])


def test_package_import_stays_stdlib_only():
    code = (
        "import os, sys\n"
        "before = set(sys.modules)\n"
        "import generative_recommenders.ops.flydsl as p\n"
        "new = {m.split('.')[0] for m in set(sys.modules) - before}\n"
        "print(repr((sorted(new - set(sys.stdlib_module_names)),\n"
        "            os.path.realpath(p.__file__), p.FLYDSL_ARCH_PIN)))\n"
    )
    env = dict(os.environ, FLYDSL_GPU_ARCH="gfx950")
    non_stdlib, path, pin = _child(code, env)
    assert non_stdlib == ["generative_recommenders"]
    assert path == os.path.realpath(_PKG_INIT)
    assert pin == ("gfx950", "env")


def test_import_time_pin_on_this_host():
    # The real KFD root, in a child process: host-agnostic, the pin is set
    # exactly when the parser finds one arch (gfx950 on an MI350X node, none
    # where /sys/class/kfd is absent).
    code = (
        "import os\n"
        "import generative_recommenders.ops.flydsl as p\n"
        "print(repr((p.FLYDSL_ARCH_PIN, os.environ.get('FLYDSL_GPU_ARCH'),\n"
        "            p._kfd_gfx_arch())))\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in _ARCH_VARS}
    pin, value, parsed = _child(code, env)
    if parsed is None:
        assert (pin, value) == ((None, "none"), None)
    else:
        assert (pin, value) == ((parsed, "kfd_sysfs"), parsed)
