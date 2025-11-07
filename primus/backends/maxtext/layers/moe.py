from typing import Optional, Tuple

import jax.numpy as jnp
from MaxText.layers import quantizations
from MaxText.layers.moe import COMBINE, DISPATCH, RoutedMoE


class PrimusRoutedMoE(RoutedMoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_einsum(
        self,
        rhs_mesh_axes: Tuple[Optional[str], ...] = (),
        einsum_name: str | None = None,
    ):
        """Get the Einstein summation."""

        # the check is to prevent aqteinsum as einsum op for dispatch and combine
        # einsums in ase when capacity_factor > 0
        # this is necessary to load pre-quantized weights in case of inference
        if self.config.model_call_mode == "inference" and einsum_name in (
            DISPATCH,
            COMBINE,
        ):
            return jnp.einsum

        if self.quant:

            def aqt_einsum(*args, **kwargs):  # pylint: disable=unused-argument
                # simply skip kwargs, since aqt einsum doesn't support any kwargs
                # like precision
                is_aqt = not (
                    isinstance(self.quant, quantizations.Fp8Quantization)
                    or isinstance(self.quant, quantizations.NANOOFp8Quantization)
                )
                kw = {"mesh_axes": rhs_mesh_axes} if is_aqt else {"dtype": self.dtype}
                return self.quant.einsum(**kw)(*args)  # pytype: disable=attribute-error

            einsum_op = aqt_einsum
        else:
            einsum_op = jnp.einsum
        return einsum_op

    def dense_matmul(
        self,
        inputs,
        gate_logits,
        pre_bias_logits,
        w0_kernel,
        w1_kernel,
        wo_kernel,
    ) -> tuple[jax.Array, Optional[jax.Array]]:
        """Dense matrix multiplication."""
        if self.config.expert_balance:
            ######################################################################################################
            ############################## start hard code for uniform expert ####################################
            # Create deterministic rotational pattern for gate logits
            batch_size, seq_len, num_experts = gate_logits.shape

            # Create base weights for experts (increasing values)
            base_weights = jnp.linspace(0.1, 0.1 * num_experts, num_experts, dtype=gate_logits.dtype)

            # Create position-based indices matrix [seq_len, num_experts]
            # Each row represents which index in base_weights to use after rotation
            indices = (jnp.arange(num_experts)[None, :] + jnp.arange(seq_len)[:, None]) % num_experts

            # Use advanced indexing to create the rotated weights matrix in one operation
            # This takes the appropriate weight for each position based on the rotation pattern
            rotated_weights = base_weights[indices]

            # Broadcast to batch dimension
            gate_logits = jnp.broadcast_to(rotated_weights[None, :, :], (batch_size, seq_len, num_experts))
            ############################################# end ####################################################
            ##########################################
        return super().dense_matmul(inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel)
