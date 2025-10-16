###############################################################################
# Some parts of this code are copied and modified from SGLang project.
# (https://github.com/sgl-project/sglang/tree/main).
#
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

import triton
import triton.language as tl


@triton.jit
def fused_softcap_kernel(
    full_logits_ptr,
    softcapping_value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values
    x = tl.load(full_logits_ptr + offsets, mask=mask)

    # Perform operations in-place
    x = x / softcapping_value

    # Manual tanh implementation using exp
    exp2x = tl.exp(2 * x)
    x = (exp2x - 1) / (exp2x + 1)

    x = x * softcapping_value

    # Store result
    tl.store(full_logits_ptr + offsets, x, mask=mask)


def fused_softcap(full_logits, final_logit_softcapping):
    n_elements = full_logits.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)

    fused_softcap_kernel[grid](
        full_logits_ptr=full_logits,
        softcapping_value=final_logit_softcapping,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return full_logits
