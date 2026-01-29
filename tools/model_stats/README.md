# Model Stats

This tool generates summary charts for the model configuration registry.

## Usage

From the repo root:

```bash
python3 tools/model_stats/model_stats.py primus/configs/models \
  --chart-path tools/model_stats/charts/model_families_by_framework.png
```

## Output

Charts are written under `tools/model_stats/charts/`.
