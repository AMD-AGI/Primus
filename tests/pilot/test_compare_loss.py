"""Tests for `pilot.tools.observe.compare_loss` (the CORRECTNESS_LITE gate).

The first runnable session bypassed this gate entirely (see
`IMPL_VS_DESIGN.md §1`). The autonomy patch upgrades it from
single-point compare to a trailing-window median compare so it's robust
to single-iter spikes from warmup or the profiler. These tests pin the
new behavior.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from pilot.tools import observe


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _trial_snapshot(losses: list[float], finite: bool = True) -> dict:
    return {
        "run_id": "trial",
        "metrics": {
            "loss_finite": finite,
            "history": {"loss": losses},
            "latest": {"loss": losses[-1] if losses else None},
        },
        "symptoms": {},
    }


def test_compare_loss_window_median_robust_to_single_spike(tmp_path: Path) -> None:
    """A single spike at the tail must not flip the gate when the window
    median is healthy."""
    runs = tmp_path / "runs"
    snap = _trial_snapshot([2.10, 2.08, 2.07, 2.06, 50.0])  # one bad iter at the tail
    _write(runs / "trial" / "snapshots" / "0001.yaml", snap)
    ref_file = tmp_path / "ref.yaml"
    _write(ref_file, {"window": [2.12, 2.10, 2.09, 2.08, 2.07]})

    out = observe.compare_loss(
        "trial",
        str(ref_file),
        log_dir=str(runs),
        window=4,
        max_delta_pct=10.0,
    )
    assert out["status"] == "pass", out
    assert out["trial_window_size"] == 4
    assert out["reference_window_size"] == 4
    # Median over last 4 trial losses excludes the spike.
    assert abs(out["loss"] - 2.07) < 0.01
    assert abs(out["reference_loss"] - 2.085) < 0.01


def test_compare_loss_fails_on_systematic_drift(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    snap = _trial_snapshot([4.0, 4.2, 4.5, 4.7, 5.0])
    _write(runs / "trial" / "snapshots" / "0001.yaml", snap)
    ref_file = tmp_path / "ref.yaml"
    _write(ref_file, {"window": [2.1, 2.05, 2.0, 1.95, 1.9]})

    out = observe.compare_loss(
        "trial",
        str(ref_file),
        log_dir=str(runs),
        window=5,
        max_delta_pct=10.0,
    )
    assert out["status"] == "fail", out
    assert out["loss_delta_pct"] > 10.0


def test_compare_loss_fails_when_loss_not_finite(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    snap = _trial_snapshot([2.1, 2.0, 1.9], finite=False)
    snap["symptoms"]["loss_nan_or_inf"] = True
    _write(runs / "trial" / "snapshots" / "0001.yaml", snap)
    ref_file = tmp_path / "ref.yaml"
    _write(ref_file, {"window": [2.1, 2.0, 1.9]})

    out = observe.compare_loss(
        "trial",
        str(ref_file),
        log_dir=str(runs),
        window=3,
        max_delta_pct=10.0,
    )
    assert out["status"] == "fail"
    assert out["loss_finite"] is False
    assert out["hard_symptom"] is True


def test_compare_loss_falls_back_to_scalar_reference(tmp_path: Path) -> None:
    """Legacy reference shape (single scalar loss) must still work."""
    runs = tmp_path / "runs"
    snap = _trial_snapshot([2.10, 2.08, 2.07, 2.06, 2.05])
    _write(runs / "trial" / "snapshots" / "0001.yaml", snap)
    ref_file = tmp_path / "ref.yaml"
    _write(ref_file, {"loss": 2.08})

    out = observe.compare_loss(
        "trial",
        str(ref_file),
        log_dir=str(runs),
        window=3,
        max_delta_pct=10.0,
    )
    assert out["status"] == "pass", out
    assert out["reference_loss"] == 2.08
    assert out["reference_window_size"] == 1


def test_compare_loss_uses_snapshot_shaped_reference(tmp_path: Path) -> None:
    """Reference may itself be a snapshot YAML."""
    runs = tmp_path / "runs"
    trial = _trial_snapshot([2.10, 2.08, 2.07, 2.06, 2.05])
    _write(runs / "trial" / "snapshots" / "0001.yaml", trial)
    ref_snap = _trial_snapshot([2.20, 2.15, 2.12, 2.10, 2.08])
    ref_file = tmp_path / "ref.yaml"
    _write(ref_file, ref_snap)

    out = observe.compare_loss(
        "trial",
        str(ref_file),
        log_dir=str(runs),
        window=4,
        max_delta_pct=10.0,
    )
    assert out["status"] == "pass"
    assert out["reference_window_size"] == 4
