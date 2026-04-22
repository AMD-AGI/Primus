# FarSkip-Collective on Primus
FarSkip-Collective is a modification of the Mixture of Experts (MoE) architecture that enables native communication-computation overlap in MoEs.  
The architecture achieves comparable model performance to regular MoEs, and by overlapping communication with computation, FarSkip-Collective significantly accelerates MoE training and inference.  
This work is based on the research presented in [FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models](https://arxiv.org/abs/2511.11505).

FarSkip models modify the dependency graph of the MoE transformer block and use partial and outdated activations as the input to the MoE sub-blocks, which allows for overlapping of MoE communication.  
The accuracy of the FarSkip-Collective architecture has been validated for large-scale MoEs at the 100B+ parameter scale and with large-scale pre-training ablations.  
By significantly reducing exposed communication overhead, the architecture unlocks higher hardware utilization for sparser and larger MoE architectures on GPUs.  
Below we demonstrate how to run pre-training of the FarSkip-Collective MoE architecture with Primus and the Megatron-LM backend.  

[➡️ Primus repo main README](README_PRIMUS.md)


---
## Setting up the Primus environment
```bash
# Pull docker image (supports MI300/MI325/MI355)
PRIMUS_IMAGE=docker.io/rocm/megatron-lm:v25.8_py310

# Run the container
docker run -it \
    --device /dev/dri --device /dev/kfd --device /dev/infiniband \
    --network host --ipc host \
    --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined --privileged \
    -v $HOME:$HOME -v $HOME/.ssh:/root/.ssh \
    --shm-size 128G --name primus_farskip \
    $PRIMUS_IMAGE

# Clone using the dev/farskip branch
git clone -b dev/farskip --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git

cd Primus

# Install Python dependencies
pip install -r requirements.txt
```

## Running Training
The Primus .yaml pre-training configuration for Kimi Moonlight 16B with FarSkip is provided in `examples/megatron/configs/farskip-overlap-moonlight-pretrain.yaml`.
Run pre-training with:
```bash
export EXP=examples/megatron/configs/farskip-overlap-moonlight-pretrain.yaml
bash ./examples/run_pretrain.sh
```

## Customization
Activating the architecture mode is controlled by the following flags:
```yaml
use_overlapped_farskip_layer: true   # main overlapped implementation with comm-compute hardware overlap
use_simple_farskip_layer: false      # simple implementation without hardware overlap (reference/debug)
mlp_only_farskip: false              # FarSkip on MLP sub-block only
attn_only_farskip: false             # FarSkip on attention sub-block only
```
The reference implementation is simpler to run and debug but does not provide communication-computation overlap on hardware.

## Implementation Approach

### Patching
We implement FarSkip-Collective using the [Primus Patching layer](https://github.com/AMD-AGI/Primus/blob/main/docs/backends/overview.md#backend-patch-notes-overview) which cleanly integrates on top of the backend without directly modifying any of the backend source files.  
The patching happens at runtime (`patch_farskip()` in `primus/modules/trainer/megatron/trainer.py`) and imports the relevant functions and modules to enable FarSkip from `primus/backends/megatron`.

We also provide a direct monkey-patch implementation that demonstrates the differences applied on top of the backend source files.  
This implementation is functionally identical to the main patching implementation and can be used by overwriting the backend with the modified files.  
The monkey-patched files are provided under `patches/`. To use this approach, just overwrite the backend files (`rsync -av patches/megatron-lm/. third_party/Megatron-LM/`).

### Overlapping approach
- Most key changes are enabled at the `TransformerLayer` level by creating `OverlappedFarSkipTransformerLayer` and `SimpleFarSkipTransformerLayer` derived classes that override execution via `forward()` (model parameter layout is not modified with FarSkip).
- We use PyTorch's `torch.distributed` collective-op API in `async_op=True` mode to run communication asynchronously. This returns a `torch.distributed.Work` handle that can be forced to synchronize (via `.wait()`) right before tensor access.
- Communication overlap runs across layers so we pass `Work` handles and in-flight tensors between layers in a modified `TransformerBlock` implementation.
- In some configurations, communication overlap with attention is enabled by splitting the attention forward pass into the q,k,v preparation and the core attention computation steps, and exposing this split to the `TransformerLayer`.
- Token dispatch and combine are modified for asynchronous communication with `async_all_to_all` in `mappings.py` and in FarSkip dispatch will launch shortly after the last combine is synchronized (only router computation in between).

### Backward pass overlapping
Our implementation also provides overlapping of MoE communication in the backward pass.  
To achieve communication-computation overlap in the backward pass we need to have "overlappable" computation appear after backward communication.  
The key issue to resolve is that the overlap stretches over multiple transformer layers which makes implementing an explicit `autograd.Function` with hard-coded backward complicated.  
Typically without explicit `autograd.Function` one does not control `torch.autograd` backward execution order and therefore does not explicitly initiate and synchronize communication operations.  

We find a middle-ground where we use `torch.autograd` automatic graph traversal and implicitly control backward execution and communication synchronization.  
We achieve this by creating an async-safe communication mechanism via PyTorch backward hooks (`_AsyncAllToAll`) to ensure in-flight tensors are synchronized before access.  
We need to combine this approach with reprioritization of autograd nodes to actually achieve overlap (node reprioritization is done inside `OverlappedFarSkipTransformerLayer`).  
FarSkip's connectivity relaxes the dependency graph in the backward pass which allows us to reprioritize other backward nodes that are available before the processing of nodes that require the in-flight tensors.  
By reordering the processing order priority of `torch.autograd` we are able to delay the automatic synchronization and achieve backward overlap.
