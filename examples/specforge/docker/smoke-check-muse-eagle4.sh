#!/bin/bash
# GPU-free checks for the Muse Spark EAGLE4 development image.
set -euo pipefail

bash "$(dirname "${BASH_SOURCE[0]}")/smoke-check-runtime.sh"

python - <<'PY'
import json
import os
import subprocess
from importlib.metadata import entry_points, version
from pathlib import Path

import torch
import torch._dynamo
from transformers import LlamaConfig

from primus.backends.specforge.capture_plugin import PLUGIN_NAME, capture_plugin_issues
from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.modeling.draft.eagle4 import LlamaForCausalLMEagle4

torch._dynamo.config.disable = True
assert "eagle4" in builtin_algorithm_registry().names
plugins = [ep for ep in entry_points(group="sglang.srt.plugins") if ep.name == PLUGIN_NAME]
assert len(plugins) == 1 and callable(plugins[0].load()), "capture plugin entry point missing"

# Exercise the installed EAGLE4 implementation without loading target weights.
config = LlamaConfig(
    vocab_size=16,
    hidden_size=8,
    intermediate_size=16,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,
    max_position_embeddings=32,
    pad_token_id=0,
    rms_norm_eps=1e-5,
)
config.architectures = ["LlamaForCausalLMEagle4"]
config.draft_vocab_size = config.vocab_size
config.fc_norm = True
config.norm_output = True
config.use_rope = False
config.sliding_window = 4
torch.manual_seed(0)
model = LlamaForCausalLMEagle4(config, attention_backend="sdpa").cpu()
hidden = torch.randn(1, 4, 24, device="cpu")
embeds = model.embed_input_ids(torch.tensor([[1, 2, 3, 4]], device="cpu"))
output = model(hidden_states=hidden, inputs_embeds=embeds)
logits = model.compute_logits(output)
assert tuple(logits.shape) == (1, 4, 16)
loss = logits.float().square().mean()
assert torch.isfinite(loss)
loss.backward()
assert model.lm_head.weight.grad is not None
assert torch.isfinite(model.lm_head.weight.grad).all()
assert model.lm_head.weight.device.type == "cpu"

root = Path(os.environ["SPECFORGE_ROOT"])
revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
issues = capture_plugin_issues()
if issues:
    raise RuntimeError("Capture API backport checks failed: " + "; ".join(issues))
report = {
    "specforge_revision": revision,
    "sglang": version("sglang"),
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "eagle4_cpu_forward_backward": "passed",
    "capture_plugin_discovered": True,
    "capture_plugin_compatible": not issues,
    "capture_plugin_issues": issues,
    "gpu_workload_tested": False,
}
Path("/opt/primus/muse-spark-eagle4-image.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))

PY

echo "Muse Spark EAGLE4 image checks passed (GPU-free; capture APIs verified)."
