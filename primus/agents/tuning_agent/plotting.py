"""Generate trials.png — best-tps and per-trial scatter."""

from __future__ import annotations

from pathlib import Path

from .history import History


def plot_history(history: History, out_path: Path, objective: str = "tokens_per_s_per_gpu") -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    points = []
    for t in history.trials:
        v = t.result.get(objective)
        if v is None:
            continue
        points.append((t.idx, float(v), t.source, bool(t.result.get("legal"))))

    if not points:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    legal_x = [p[0] for p in points if p[3]]
    legal_y = [p[1] for p in points if p[3]]
    illegal_x = [p[0] for p in points if not p[3]]
    illegal_y = [p[1] for p in points if not p[3]]

    if legal_x:
        ax.scatter(legal_x, legal_y, label="legal trials", alpha=0.7)
    if illegal_x:
        ax.scatter(illegal_x, illegal_y, marker="x", label="rejected", alpha=0.5)

    # rolling best
    best = []
    cur = float("-inf")
    for x, y, _src, ok in points:
        if ok and y > cur:
            cur = y
        best.append((x, cur if cur > float("-inf") else None))
    bx = [b[0] for b in best if b[1] is not None]
    by = [b[1] for b in best if b[1] is not None]
    if bx:
        ax.plot(bx, by, color="black", linewidth=2.0, label="incumbent")

    ax.set_xlabel("trial #")
    ax.set_ylabel(objective)
    ax.set_title(f"Tuning Agent — incumbent over trials ({objective})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
