###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MegatronSFTTrainer: Megatron-LM based supervised fine-tuning trainer."""

from typing import Any

from primus.backends.megatron.megatron_base_trainer import MegatronBaseTrainer
from primus.modules.module_utils import log_rank_0


class MegatronSFTTrainer(MegatronBaseTrainer):
    """
    Trainer class for Megatron-LM based supervised fine-tuning.

    This trainer handles:
        - SFT workflows with HuggingFace datasets
        - Instruction tuning with proper loss masking
        - Multiple conversation formats (extensible)
        - Direct Megatron-LM integration
        - Support for various model architectures (Llama, GPT, etc.)
        - LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning

    Inherits from MegatronBaseTrainer which provides:
        - Argument injection into Megatron runtime
        - ROCm compatibility patches
        - Common Megatron initialization patterns
    """

    def __init__(self, backend_args: Any):
        """
        Initialize Megatron SFT trainer.

        Args:
            backend_args: Megatron-LM argument namespace (from MegatronArgBuilder)
        """
        super().__init__(backend_args=backend_args)

        # Initialize LoRA if enabled
        self.peft = None
        self._init_lora()

        self.model_type = getattr(self.backend_args, "model_type", "gpt")
        log_rank_0(f"Initialized MegatronSFTTrainer for model_type: {self.model_type}")

    def _init_lora(self):
        """Initialize LoRA (Low-Rank Adaptation) if enabled in config."""
        lora_config = getattr(self.backend_args, "lora", None)

        if lora_config is None or not getattr(lora_config, "enabled", False):
            log_rank_0("LoRA disabled, using full fine-tuning")
            return

        from primus.backends.megatron.peft import LoRA

        # Convert config to dict, excluding 'enabled' field
        lora_kwargs = {k: v for k, v in vars(lora_config).items() if k != "enabled"}

        log_rank_0(f"Initializing LoRA with config: {lora_kwargs}")

        self.peft = LoRA(**lora_kwargs)

    def setup(self):
        """
        Setup phase for Megatron SFT training.

        Can be used for pre-initialization setup if needed.
        """
        super().setup()
        log_rank_0("MegatronSFTTrainer.setup()")

    def init(self):
        """
        Initialize Megatron SFT training components.

        Note:
            Argument injection is handled during setup().
            This method can be used for trainer-specific initialization.
        """
        super().init()
        log_rank_0(f"Initializing Megatron SFT training for model_type: {self.model_type}")

    def _create_model_provider_with_lora(self, base_model_provider):
        """
        Wrap the model provider with Bridge-style ``pre-wrap base ckpt load``
        + LoRA wrap.

        The hook executes, in order:

          1. Build the base ``GPTModel`` via ``base_model_provider``.
          2. If both PEFT is enabled AND ``args.pretrained_checkpoint`` is set,
             load the base checkpoint into the UN-WRAPPED model. This mirrors
             Megatron-Bridge's ``peft_pre_wrap_hook`` (Megatron-Bridge
             ``training/setup.py`` L456-501).
          3. Apply LoRA wrap (``peft(model, training=True)``).
          4. After this returns, Megatron's ``setup_model_and_optimizer`` will
             call ``load_checkpoint(model, optimizer, ...)`` again. We
             ``None`` out ``args.pretrained_checkpoint`` in step 2 so that
             second call becomes a no-op (no ``load_dir``, no
             ``pretrained_dir``).

        Why the pre-wrap load matters for loss alignment with Bridge
        ------------------------------------------------------------
        LoRA wraps each target linear (``linear_qkv``, ``linear_proj``,
        ``linear_fc1``, ``linear_fc2``) into a ``LoRALinear(to_wrap,
        adapter)``. After this wrap the model's PyTorch module hierarchy
        is::

            decoder.layers.0.self_attention.linear_qkv (LoRALinear)
                .to_wrap   = original TELayerNormColumnParallelLinear
                .adapter   = ParallelLinearAdapter

        ``LoRALinear.sharded_state_dict()`` (adapter_wrapper.py L187-217)
        manually drops the ``to_wrap.`` prefix when delegating to
        ``self.to_wrap.sharded_state_dict(prefix, ...)``, but **this only
        helps the dist-checkpointing path**: the keys produced by
        Megatron's default ``MegatronModule.sharded_state_dict`` for the
        WRAPPED model still differ from what the Llama-2-70B base ckpt
        carries, because:

          - The wrapped model emits ``adapter.linear_in.weight``,
            ``adapter.linear_out.weight`` and ``adapter.*._extra_state``
            keys, which are NOT in the base ckpt -- these are reported as
            ``Unexpected keys`` and are passed to ``apply_factory_merges``,
            patched by ``apply_factory_merges_tolerant`` to silently skip
            them. The patch correctly skips ADAPTER factories but, when
            certain ``_extra_state`` entries on the base linear collide
            with adapter-side ``_extra_state`` factories during merge, the
            order of dict iteration leaves some base linear ``_extra_state``
            untouched, which on Llama-2-70B's TELayerNormColumnParallelLinear
            translates to a randomly-initialised ``layer_norm_weight`` /
            TE quantization metadata, and forward then produces ~13.65
            initial loss instead of Bridge's ~4.34.
          - PyTorch's default ``load_state_dict`` machinery (used internally
            by some sub-paths) does NOT honour the prefix-stripping override
            in ``LoRALinear.sharded_state_dict``; it iterates through
            ``named_modules`` and emits ``...linear_qkv.to_wrap.weight``,
            which the ckpt does not contain -> silent skip.

        Loading BEFORE the wrap eliminates both problems: the
        sharded_state_dict is exactly the same one Bridge emits, the load
        path is the fast path (no ``apply_factory_merges_tolerant``, zero
        ``Unexpected keys``), and once the wrap runs the base weights are
        already in place. LoRA's ``linear_in`` is initialised xavier_normal
        and ``linear_out`` is zero, so the wrapped forward is exactly
        ``to_wrap(x) + 0`` on iteration-1 == pretrained Llama-2 forward,
        and iter-1 loss matches Bridge.

        Args:
            base_model_provider: Original model provider function

        Returns:
            Wrapped model provider that loads the base checkpoint and then
            applies LoRA to the created model.
        """
        peft = self.peft
        backend_args = self.backend_args  # Same object as megatron get_args()

        def _count_params(m):
            if isinstance(m, list):
                total = sum(p.numel() for chunk in m for p in chunk.parameters())
                trainable = sum(p.numel() for chunk in m for p in chunk.parameters() if p.requires_grad)
            else:
                total = sum(p.numel() for p in m.parameters())
                trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            return total, trainable

        def model_provider_with_lora(*args, **kwargs):
            """
            Model provider that loads base ckpt (if any) BEFORE applying LoRA.

            Newer Megatron versions may pass extra keywords like `config` or
            `pg_collection`, so we forward the full call signature unchanged.
            """
            import os

            model = base_model_provider(*args, **kwargs)

            pretrained_path = getattr(backend_args, "pretrained_checkpoint", None)
            do_pre_wrap_load = (peft is not None) and bool(pretrained_path)

            if do_pre_wrap_load:
                log_rank_0("=" * 60)
                log_rank_0(f"[PEFT pre-wrap] Loading base model weights from: {pretrained_path}")

                # ----------------------------------------------------------
                # Why we bypass megatron.training.checkpointing.load_checkpoint
                # ----------------------------------------------------------
                # In Llama-2-70B + LoRA SFT runs we observed:
                #   - log says "successfully loaded checkpoint ... at iteration 0"
                #   - 2.5 min spent in dist_checkpointing.load
                #   - BUT iter-1 loss = 13.65 (random init), identical to the
                #     run where pre-wrap load was not implemented at all.
                # Root cause: load_checkpoint() takes the loaded state_dict
                # and calls module.load_state_dict(state_dict["model"],
                # strict=False); the IncompatibleKeys return value is
                # silently dropped, so any prefix/key mismatch between the
                # generated sharded_state_dict and the actual model state
                # dict is a SILENT skip. Result: all model weights are
                # discarded; only sharded tensors that happened to live
                # inside ShardedTensor instances were updated in-place.
                #
                # The Megatron-Bridge "torch_dist directly" path used in the
                # 20260516 Bridge run avoids this by going straight to
                # dist_checkpointing.load() and then applying the loaded
                # dict to the model with a strict-mode-aware helper. We
                # replicate that here.
                # ----------------------------------------------------------
                from megatron.core import dist_checkpointing
                from megatron.core import mpu as core_mpu

                # ----------------------------------------------------------
                # Path resolution: accept both Bridge-style (path ends with
                # /release) and Megatron-LM-style (path is the parent dir
                # holding latest_checkpointed_iteration.txt).
                # ----------------------------------------------------------
                _stripped = pretrained_path.rstrip("/")
                if _stripped.endswith("/release"):
                    ckpt_dir = _stripped
                else:
                    release_sub = os.path.join(_stripped, "release")
                    ckpt_dir = release_sub if os.path.isdir(release_sub) else _stripped
                log_rank_0(f"[PEFT pre-wrap] Resolved ckpt dir: {ckpt_dir}")

                # ----------------------------------------------------------
                # Pick a "canary" base param to verify the load actually
                # mutated model weights (not just streamed bytes off disk).
                # ----------------------------------------------------------
                def _collect_canaries(stage: str):
                    """Sample several base params so we can pinpoint which one
                    fails to load. Returns a dict
                    {name: (sum, max, min, abs_max, shape, dtype, device, data_ptr)}.
                    abs_max separates random_init (small, ~0.05) from
                    pretrained Llama-2 (large outliers, >0.5)."""
                    out = {}

                    def _try(name, getter):
                        try:
                            t = getter()
                            if t is not None:
                                tf = t.detach().float()
                                out[name] = (
                                    tf.sum().item(),
                                    tf.max().item(),
                                    tf.min().item(),
                                    tf.abs().max().item(),
                                    tuple(t.shape),
                                    str(t.dtype),
                                    str(t.device),
                                    t.data_ptr(),
                                )
                        except Exception as e:
                            out[name] = (
                                f"ERR:{type(e).__name__}:{e}",
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                            )

                    try:
                        layer0 = model.decoder.layers[0]
                    except Exception as e:
                        log_rank_0(f"[PEFT pre-wrap] {stage}: cannot access decoder.layers[0]: {e}")
                        return out

                    _try("L0.attn.linear_qkv.weight", lambda: layer0.self_attention.linear_qkv.weight)
                    _try(
                        "L0.attn.linear_qkv.layer_norm_weight",
                        lambda: getattr(layer0.self_attention.linear_qkv, "layer_norm_weight", None),
                    )
                    _try("L0.attn.linear_proj.weight", lambda: layer0.self_attention.linear_proj.weight)
                    _try("L0.mlp.linear_fc1.weight", lambda: layer0.mlp.linear_fc1.weight)
                    _try(
                        "L0.mlp.linear_fc1.layer_norm_weight",
                        lambda: getattr(layer0.mlp.linear_fc1, "layer_norm_weight", None),
                    )
                    _try("L0.mlp.linear_fc2.weight", lambda: layer0.mlp.linear_fc2.weight)
                    _try("final_layernorm.weight", lambda: model.decoder.final_layernorm.weight)
                    _try("embedding.word_embeddings.weight", lambda: model.embedding.word_embeddings.weight)
                    _try("output_layer.weight", lambda: model.output_layer.weight)

                    for k, v in out.items():
                        if isinstance(v[0], float):
                            log_rank_0(
                                f"[PEFT pre-wrap] {stage} canary {k}: "
                                f"sum={v[0]:.6f} max={v[1]:.6f} min={v[2]:.6f} "
                                f"abs_max={v[3]:.6f} shape={v[4]} dtype={v[5]} "
                                f"device={v[6]} ptr={v[7]}"
                            )
                        else:
                            log_rank_0(f"[PEFT pre-wrap] {stage} canary {k}: {v[0]}")
                    return out

                before_canaries = _collect_canaries("BEFORE load")

                # ----------------------------------------------------------
                # Build sharded_state_dict from the unwrapped base model.
                # ----------------------------------------------------------
                try:
                    dp_cp_group = core_mpu.get_data_parallel_group(with_context_parallel=True)
                except Exception:
                    dp_cp_group = None

                # Pull metadata that the ckpt was saved with (e.g.
                # singleton_local_shards, chained_optim_avoid_prefix) so the
                # sharded_state_dict we build matches the on-disk layout.
                try:
                    common_sd = dist_checkpointing.load_common_state_dict(ckpt_dir)
                    sharded_sd_metadata = (
                        dist_checkpointing.load_content_metadata(preloaded_state_dict=common_sd) or {}
                    )
                except Exception as e:
                    log_rank_0(
                        f"[PEFT pre-wrap] load_content_metadata failed "
                        f"({type(e).__name__}: {e}); proceeding with empty metadata"
                    )
                    sharded_sd_metadata = {}
                if dp_cp_group is not None:
                    sharded_sd_metadata.setdefault("dp_cp_group", dp_cp_group)
                log_rank_0(
                    f"[PEFT pre-wrap] sharded_sd_metadata from ckpt: "
                    f"{ {k: v for k, v in sharded_sd_metadata.items() if k != 'dp_cp_group'} }"
                )

                sharded_state_dict = {"model": model.sharded_state_dict(metadata=sharded_sd_metadata)}

                # ----------------------------------------------------------
                # Load directly via dist_checkpointing.load (Bridge fast path).
                # strict="log_all" matches what the rest of the project uses.
                # ----------------------------------------------------------
                loaded = dist_checkpointing.load(sharded_state_dict, ckpt_dir, strict="log_all")

                # ----------------------------------------------------------
                # Apply to model. Use strict=False so adapter/extra keys
                # that may live on TE modules don't blow up; capture and
                # log IncompatibleKeys so we DON'T silently skip everything
                # like Megatron-LM's helper did.
                # ----------------------------------------------------------
                missing, unexpected = model.load_state_dict(loaded["model"], strict=False)
                n_missing = len(missing) if missing is not None else 0
                n_unexpected = len(unexpected) if unexpected is not None else 0
                log_rank_0(
                    f"[PEFT pre-wrap] model.load_state_dict result: "
                    f"missing={n_missing}, unexpected={n_unexpected}"
                )
                if n_missing:
                    log_rank_0(f"[PEFT pre-wrap] first 5 missing keys: {list(missing)[:5]}")
                if n_unexpected:
                    log_rank_0(f"[PEFT pre-wrap] first 5 unexpected keys: {list(unexpected)[:5]}")

                after_canaries = _collect_canaries("AFTER  load")
                log_rank_0("[PEFT pre-wrap] === canary deltas (BEFORE -> AFTER load) ===")
                for k, before_v in before_canaries.items():
                    after_v = after_canaries.get(k)
                    if (
                        before_v is None
                        or after_v is None
                        or not isinstance(before_v[0], float)
                        or not isinstance(after_v[0], float)
                    ):
                        log_rank_0(f"[PEFT pre-wrap]   {k}: ERROR before={before_v} after={after_v}")
                        continue
                    delta = abs(after_v[0] - before_v[0])
                    flag = "CHANGED" if delta > 1e-3 else "*** UNCHANGED ***"
                    # abs_max heuristic: random init N(0, 0.008) gives abs_max ~ 0.04;
                    # Llama-2 pretrained QKV has abs_max > 0.5 due to outliers.
                    looks_pretrained = after_v[3] > 0.1
                    ptr_changed = before_v[7] != after_v[7]
                    log_rank_0(
                        f"[PEFT pre-wrap]   {k}: {flag} sum_before={before_v[0]:.6f} "
                        f"sum_after={after_v[0]:.6f} |delta_sum|={delta:.6f} "
                        f"abs_max_after={after_v[3]:.6f} "
                        f"looks_pretrained={looks_pretrained} "
                        f"ptr_changed={ptr_changed}"
                    )

                # Prevent Megatron's later setup_model_and_optimizer() from
                # calling load_checkpoint() again on the LoRA-wrapped model.
                backend_args.pretrained_checkpoint = None
                # Mark as finetune so Megatron doesn't try to resume RNG/optim.
                backend_args.finetune = True

                log_rank_0("[PEFT pre-wrap] Base weights loaded successfully")
                log_rank_0("=" * 60)

                # ----------------------------------------------------------
                # Stash the BEFORE-LoRA canaries so we can verify after the
                # LoRA wrap whether any base param got reset.
                # ----------------------------------------------------------
                _pre_lora_canaries = after_canaries
            else:
                _pre_lora_canaries = None

            # Step 3: apply LoRA wrap (after base ckpt has been loaded)
            if peft is not None:
                log_rank_0("=" * 60)
                log_rank_0("Applying LoRA to model...")
                model = peft(model, training=True)

                total_params, trainable_params = _count_params(model)
                frozen_params = total_params - trainable_params

                log_rank_0("LoRA Summary:")
                log_rank_0(f"  - Total parameters:     {total_params:,}")
                log_rank_0(
                    f"  - Trainable parameters: {trainable_params:,} "
                    f"({100 * trainable_params / total_params:.2f}%)"
                )
                log_rank_0(
                    f"  - Frozen parameters:    {frozen_params:,} "
                    f"({100 * frozen_params / total_params:.2f}%)"
                )
                log_rank_0("=" * 60)

                # ----------------------------------------------------------
                # Post-LoRA canary: were base weights preserved through wrap?
                # LoRA wrap should NOT touch the underlying to_wrap module's
                # weights -- only add adapter.linear_in/linear_out alongside.
                # If any of these UNCHANGED below, LoRA wrap somehow reset the
                # base weights, and we need to either re-load OR avoid the
                # rebuild that wrap triggers.
                # ----------------------------------------------------------
                if _pre_lora_canaries is not None:
                    log_rank_0("[POST-LoRA canary] Verifying base weights survived LoRA wrap...")

                    def _read_post_lora(name):
                        # After wrap, base param may have moved into a
                        # LoRALinear.to_wrap submodule. Try a few likely paths.
                        try:
                            layer0 = model.decoder.layers[0]
                        except Exception:
                            return None
                        path_map = {
                            "L0.attn.linear_qkv.weight": [
                                "self_attention.linear_qkv.to_wrap.weight",
                                "self_attention.linear_qkv.weight",
                            ],
                            "L0.attn.linear_qkv.layer_norm_weight": [
                                "self_attention.linear_qkv.to_wrap.layer_norm_weight",
                                "self_attention.linear_qkv.layer_norm_weight",
                            ],
                            "L0.attn.linear_proj.weight": [
                                "self_attention.linear_proj.to_wrap.weight",
                                "self_attention.linear_proj.weight",
                            ],
                            "L0.mlp.linear_fc1.weight": [
                                "mlp.linear_fc1.to_wrap.weight",
                                "mlp.linear_fc1.weight",
                            ],
                            "L0.mlp.linear_fc1.layer_norm_weight": [
                                "mlp.linear_fc1.to_wrap.layer_norm_weight",
                                "mlp.linear_fc1.layer_norm_weight",
                            ],
                            "L0.mlp.linear_fc2.weight": [
                                "mlp.linear_fc2.to_wrap.weight",
                                "mlp.linear_fc2.weight",
                            ],
                            "final_layernorm.weight": [
                                "decoder_final_layernorm_dummy"
                            ],  # accessed via model.decoder below
                            "embedding.word_embeddings.weight": ["embedding_dummy"],
                            "output_layer.weight": ["output_layer_dummy"],
                        }
                        if name == "final_layernorm.weight":
                            try:
                                return model.decoder.final_layernorm.weight
                            except Exception:
                                return None
                        if name == "embedding.word_embeddings.weight":
                            try:
                                return model.embedding.word_embeddings.weight
                            except Exception:
                                return None
                        if name == "output_layer.weight":
                            try:
                                return model.output_layer.weight
                            except Exception:
                                return None
                        for rel in path_map.get(name, []):
                            cur = layer0
                            ok = True
                            for attr in rel.split("."):
                                if hasattr(cur, attr):
                                    cur = getattr(cur, attr)
                                else:
                                    ok = False
                                    break
                            if ok:
                                return cur
                        return None

                    for k, pre_v in _pre_lora_canaries.items():
                        if not isinstance(pre_v[0], float):
                            continue
                        t = _read_post_lora(k)
                        if t is None:
                            log_rank_0(f"[POST-LoRA canary]   {k}: PARAM NOT FOUND in wrapped model")
                            continue
                        try:
                            tf = t.detach().float()
                            post_sum = tf.sum().item()
                            post_absmax = tf.abs().max().item()
                            post_ptr = t.data_ptr()
                        except Exception as e:
                            log_rank_0(f"[POST-LoRA canary]   {k}: read error {e}")
                            continue
                        delta_sum = abs(post_sum - pre_v[0])
                        flag = "PRESERVED" if delta_sum < 1e-3 else "*** CHANGED ***"
                        ptr_changed = pre_v[7] != post_ptr
                        log_rank_0(
                            f"[POST-LoRA canary]   {k}: {flag} "
                            f"sum_pre_lora={pre_v[0]:.6f} sum_post_lora={post_sum:.6f} "
                            f"|delta_sum|={delta_sum:.6f} "
                            f"abs_max_pre_lora={pre_v[3]:.6f} abs_max_post_lora={post_absmax:.6f} "
                            f"ptr_changed={ptr_changed}"
                        )

            return model

        return model_provider_with_lora

    def train(self):
        """
        Execute Megatron SFT training.

        This method is called by the runtime-owned trainer lifecycle and executes
        the main SFT training loop using Megatron-LM's infrastructure.
        """
        log_rank_0("Executing Megatron SFT training...")

        from megatron.training import pretrain  # type: ignore

        from primus.core.utils.import_utils import get_model_provider

        from .sft.forward_step import create_sft_forward_step
        from .sft.runtime import create_sft_datasets_provider, run_sft_pretrain

        train_valid_test_datasets_provider = create_sft_datasets_provider()
        forward_step = create_sft_forward_step()

        # Keep model-provider behavior aligned with pretrain trainer.
        if self.model_type != "gpt":
            base_model_provider = get_model_provider(model_type=self.model_type)
        else:
            base_model_provider = get_model_provider()

        if self.peft is not None:
            model_provider = self._create_model_provider_with_lora(base_model_provider)
            log_rank_0("Using LoRA-enabled model provider")
        else:
            model_provider = base_model_provider

        run_sft_pretrain(
            pretrain_fn=pretrain,
            datasets_provider=train_valid_test_datasets_provider,
            model_provider=model_provider,
            forward_step=forward_step,
        )

        log_rank_0("Megatron SFT training execution completed.")
