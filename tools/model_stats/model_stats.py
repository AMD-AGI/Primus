#!/usr/bin/env python3
import argparse
import os
from collections import defaultdict

FAMILY_RULES = [
    ("moe_proxy", "MoE_Proxy"),
    ("moe_", "MoE_Proxy"),
    ("mixtral", "Mixtral"),
    ("llama", "LLaMA"),
    ("deepseek", "DeepSeek"),
    ("qwen", "Qwen"),
    ("grok", "Grok"),
    ("gpt", "GPT"),
    ("language_model", "GPT"),
]

FAMILY_COLUMNS = [
    "LLaMA",
    "DeepSeek",
    "Qwen",
    "Mixtral",
    "MoE_Proxy",
    "Grok",
    "GPT",
]

EXCLUDE_FILES = {
    "primus_megatron_model.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print model family table and generate bar chart.")
    parser.add_argument(
        "models_dir",
        help="Path to models directory (e.g., primus/configs/models).",
    )
    parser.add_argument(
        "--chart-path",
        default="model_families_by_framework.png",
        help="Output chart path (PNG).",
    )
    return parser.parse_args()


def iter_model_files(models_dir: str):
    for dirpath, _, filenames in os.walk(models_dir):
        for filename in filenames:
            if not filename.endswith(".yaml"):
                continue
            if filename.endswith("_base.yaml"):
                continue
            if filename.endswith("-fp8.yaml"):
                continue
            if filename in EXCLUDE_FILES:
                continue
            yield os.path.join(dirpath, filename)


def detect_family(model_name: str) -> str | None:
    for prefix, family in FAMILY_RULES:
        if model_name == prefix or model_name.startswith(prefix):
            return family
    return None


def collect_counts(models_dir: str):
    counts = defaultdict(lambda: defaultdict(int))
    for path in iter_model_files(models_dir):
        rel = os.path.relpath(path, models_dir)
        framework = rel.split(os.sep, 1)[0]
        name = os.path.splitext(os.path.basename(path))[0]

        family = detect_family(name)
        if family is None:
            # Skip unclassified entries to match current rules.
            continue
        counts[framework][family] += 1
    return counts


def print_table(counts):
    frameworks = sorted(counts.keys())
    header = ["Framework"] + FAMILY_COLUMNS + ["Total"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for fw in frameworks:
        total = 0
        row = [fw]
        for col in FAMILY_COLUMNS:
            value = counts[fw].get(col, 0)
            row.append(str(value))
            total += value
        row.append(str(total))
        print("| " + " | ".join(row) + " |")


def generate_chart(counts, chart_path: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(
            "matplotlib is required to generate the chart. " "Install it or set a different output flow."
        ) from exc

    frameworks = [
        ("Megatron-LM", "megatron"),
        ("TorchTitan", "torchtitan"),
        ("MaxText (JAX)", "maxtext"),
        ("Megatron Bridge", "megatron_bridge"),
    ]
    families = FAMILY_COLUMNS

    values = {}
    for label, key in frameworks:
        values[label] = [counts.get(key, {}).get(fam, 0) for fam in families]

    x = list(range(len(frameworks)))
    bar_width = 0.1
    offsets = [(i - (len(families) - 1) / 2) * bar_width for i in range(len(families))]

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [label for label, _ in frameworks]
    for i, fam in enumerate(families):
        y = [values[label][i] for label in labels]
        bars = ax.bar([xi + offsets[i] for xi in x], y, width=bar_width, label=fam)
        for bar, val in zip(bars, y):
            if val == 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(val),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Config Count (base and fp8 excluded)")
    ax.set_xlabel("Framework")
    ax.set_title("Model Families by Framework")
    ax.legend(ncol=4, frameon=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_ylim(0, max(max(v) for v in values.values()) + 2)

    fig.tight_layout()
    fig.savefig(chart_path, dpi=200)


def main():
    args = parse_args()
    models_dir = os.path.abspath(args.models_dir)
    if not os.path.isdir(models_dir):
        raise SystemExit(f"Models directory not found: {models_dir}")

    counts = collect_counts(models_dir)
    print_table(counts)

    chart_path = os.path.abspath(args.chart_path)
    chart_dir = os.path.dirname(chart_path)
    if chart_dir:
        os.makedirs(chart_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(chart_dir, ".mplconfig"))

    generate_chart(counts, chart_path)
    print(f"\nChart saved to: {chart_path}")


if __name__ == "__main__":
    main()
