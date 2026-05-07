#!/bin/bash
# ===========================================================================
# megatron_patch.sh — Patch Megatron-LM for GDN training in Primus
#
# Patches applied:
#   1. mamba_model.py  — FusedLinearCrossEntropyLoss (FLA) for MambaModel,
#      avoids materializing the full (batch*seq, vocab) logits tensor.
#   2. transformer_config.py — Hybrid model (GDN) init alignment with FLA:
#      use uniform std (init_method_normal) instead of scaled_init for
#      the output layer on hybrid models.
#
# Usage:
#   bash megatron_patch.sh          # apply all patches
#   bash megatron_patch.sh --check  # dry-run
#   bash megatron_patch.sh --revert # undo all patches
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="${SCRIPT_DIR}/third_party/Megatron-LM"

if [[ ! -d "$MEGATRON_DIR" ]]; then
    echo "ERROR: Megatron-LM directory not found: $MEGATRON_DIR"
    exit 1
fi

MODE="${1:---apply}"

# ---- Patch 1: MambaModel FusedLinearCrossEntropyLoss ----
patch_mamba() {
    patch "${@}" -p1 -d "$MEGATRON_DIR" <<'PATCH'
diff --git a/megatron/core/models/mamba/mamba_model.py b/megatron/core/models/mamba/mamba_model.py
--- a/megatron/core/models/mamba/mamba_model.py
+++ b/megatron/core/models/mamba/mamba_model.py
@@ -146,11 +146,33 @@
         if self.pre_process or self.post_process:
             self.setup_embeddings_and_output_layer()
 
+        self._use_fused_cross_entropy = False
+        try:
+            from fla.modules import FusedLinearCrossEntropyLoss
+            self._fused_lce = FusedLinearCrossEntropyLoss(reduction='mean')
+            self._use_fused_cross_entropy = True
+        except ImportError:
+            pass
+
         for name, module in self.named_modules():
             if hasattr(module, 'finish_init'):
                 quant_config = get_quant_config_or_none(name, self.config.quant_recipe)
                 module.finish_init(quant_config)
 
+    def _fused_cross_entropy_loss(self, hidden_states, labels, output_weight):
+        """Use FLA's FusedLinearCrossEntropyLoss to compute logits + CE in
+        chunks, never materializing the full (batch*seq, vocab) logits tensor.
+        This is the key to matching FLA's memory efficiency."""
+        # hidden_states: [s, b, h] → [b*s, h]
+        s, b, h = hidden_states.shape
+        hs_2d = hidden_states.permute(1, 0, 2).reshape(b * s, h)
+        labels_1d = labels.reshape(b * s)
+
+        weight = output_weight if output_weight is not None else self.output_layer.weight
+        loss = self._fused_lce(hs_2d, labels_1d, weight)
+        # Return [b, s] filled with mean loss for compatibility with loss_func
+        return loss.expand(b, s)
+
     def set_input_tensor(self, input_tensor: Tensor) -> None:
         """Sets input tensor to the model.
 
@@ -247,6 +269,9 @@
         if in_inference_mode and inference_context.materialize_only_last_token_logits:
             hidden_states = hidden_states[-1, :, :].unsqueeze(0)
 
+        if labels is not None and self._use_fused_cross_entropy:
+            return self._fused_cross_entropy_loss(hidden_states, labels, output_weight)
+
         logits, _ = self.output_layer(
             hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
         )
PATCH
}

# ---- Patch 2: TransformerConfig output_layer_init_method for hybrid models ----
patch_transformer_config() {
    patch "${@}" -p1 -d "$MEGATRON_DIR" <<'PATCH'
diff --git a/megatron/core/transformer/transformer_config.py b/megatron/core/transformer/transformer_config.py
--- a/megatron/core/transformer/transformer_config.py
+++ b/megatron/core/transformer/transformer_config.py
@@ -1746,19 +1746,22 @@
                 self.init_method = init_method_normal(self.init_method_std)
 
         if self.output_layer_init_method is None:
-            if self.use_mup:
+            if self.is_hybrid_model:
+                # Hybrid models (GDN, etc.): use uniform std matching FLA's initializer_range
+                self.output_layer_init_method = init_method_normal(self.init_method_std)
+            elif self.use_mup:
                 # MuP: depth and width scaling for output layers.
                 self.output_layer_init_method = mup_scaled_init_method_normal(
                     self.init_method_std,
                     self.num_layers,
                     self.mup_width_mult,
-                    multiplier=2.0 if not self.is_hybrid_model else 1.0,
+                    multiplier=2.0,
                 )
             else:
                 self.output_layer_init_method = scaled_init_method_normal(
                     self.init_method_std,
                     self.num_layers,
-                    multiplier=2.0 if not self.is_hybrid_model else 1.0,
+                    multiplier=2.0,
                 )
 
         if self.num_moe_experts is not None and self.add_bias_linear:
PATCH
}

# ---- Apply / Check / Revert all patches ----
run_all() {
    local action="$1"
    local flag="$2"
    local verb="$3"
    local ok=0
    local fail=0

    for name in mamba transformer_config; do
        fn="patch_${name}"
        target=""
        case "$name" in
            mamba)              target="mamba_model.py" ;;
            transformer_config) target="transformer_config.py" ;;
        esac

        if $fn $flag --dry-run 2>/dev/null; then
            $fn $flag
            echo "  ✓ ${target} — ${verb}d"
            ok=$((ok + 1))
        else
            echo "  · ${target} — already ${verb}d or context mismatch, skipped"
            fail=$((fail + 1))
        fi
    done

    echo ""
    echo "Done: ${ok} patched, ${fail} skipped."
}

case "$MODE" in
    --apply|-a)
        echo "Applying megatron patches ..."
        run_all apply --forward apply
        ;;
    --check|-c)
        echo "Dry-run check ..."
        for name in mamba transformer_config; do
            fn="patch_${name}"
            if $fn --forward --dry-run 2>/dev/null; then
                echo "  ✓ patch_${name} — can be applied"
            else
                echo "  · patch_${name} — already applied or mismatch"
            fi
        done
        ;;
    --revert|-r)
        echo "Reverting megatron patches ..."
        run_all revert --reverse revert
        ;;
    *)
        echo "Usage: bash megatron_patch.sh [--apply|--check|--revert]"
        exit 1
        ;;
esac
