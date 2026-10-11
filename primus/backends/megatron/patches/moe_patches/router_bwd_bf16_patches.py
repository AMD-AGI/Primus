###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in BF16 backward GEMMs for an FP32 MoE router.

Keep router logits in FP32 while using BF16 operands for dX and dW when
both the saved input and weight are BF16. The dW GEMM emits FP32 before
conversion to the parameter dtype, matching the MLPerf GPT-OSS patch.
Other precision combinations and the non-TE path retain Megatron's backward.
"""

from collections.abc import Callable

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _enabled(ctx: PatchContext) -> bool:
    return bool(getattr(get_args(ctx), "moe_router_bwd_bf16", False))


def _uses_bf16_router_backward(ctx, te_general_gemm: Callable | None) -> bool:
    return (
        te_general_gemm is not None
        and ctx.router_dtype == torch.float32
        and ctx.input_dtype == torch.bfloat16
        and ctx.weight_dtype == torch.bfloat16
    )


def _router_backward_bf16(ctx, grad_output: torch.Tensor, *, te_general_gemm: Callable):
    inp, weight, bias = ctx.saved_tensors
    inp_shape = inp.shape
    inp = inp.view(-1, inp_shape[-1])
    grad_output = grad_output.view(-1, grad_output.shape[-1])
    grad_output_bf16 = grad_output.to(torch.bfloat16)

    grad_input = te_general_gemm(weight, grad_output_bf16, ctx.input_dtype, layout="NN", grad=True)[0].to(
        ctx.input_dtype
    )
    grad_weight = te_general_gemm(inp, grad_output_bf16, ctx.router_dtype, layout="NT", grad=True)[0].to(
        ctx.weight_dtype
    )
    # Bias reduction retains the original FP32 gradient, before BF16 rounding.
    grad_bias = grad_output.sum(dim=0).to(ctx.weight_dtype) if bias is not None else None
    return grad_input.view(*inp_shape), grad_weight, grad_bias, None


@register_patch(
    "megatron.moe.router_bwd_bf16",
    backend="megatron",
    phase="before_train",
    description="Use BF16 backward GEMM operands for the FP32 MoE router",
    condition=_enabled,
)
def patch_router_bwd_bf16(ctx: PatchContext) -> None:
    from megatron.core.transformer.moe import moe_utils

    router_function = moe_utils.RouterGatingLinearFunction
    if getattr(router_function, "_primus_router_bwd_bf16_patched", False):
        _install_router_gradient_merge()
        return
    original_backward = router_function.backward

    def backward(autograd_ctx, grad_output):
        te_general_gemm = moe_utils.te_general_gemm
        if not _uses_bf16_router_backward(autograd_ctx, te_general_gemm):
            return original_backward(autograd_ctx, grad_output)
        return _router_backward_bf16(autograd_ctx, grad_output, te_general_gemm=te_general_gemm)

    router_function.backward = staticmethod(backward)
    router_function._primus_router_bwd_bf16_patched = True
    log_rank_0("[Patch:megatron.moe.router_bwd_bf16] Installed BF16 router backward GEMMs")

    _install_router_gradient_merge()


# Campaign candidate: deliberately restricted to the reviewed GPT-OSS training graph.
import contextvars as _merge_contextvars
import hashlib as _merge_hashlib
import os as _merge_os
from pathlib import Path as _MergePath
import weakref as _merge_weakref

import triton as _merge_triton
import triton.language as _merge_tl

_MERGE_PARENT_SCOPE = _merge_contextvars.ContextVar('primus_router_merge_parent', default=None)
_MERGE_ROUTE_SCOPE = _merge_contextvars.ContextVar('primus_router_merge_route', default=None)
_MERGE_LIVE = _merge_weakref.WeakSet()
_MERGE_STATS = dict(created=0, claimed=0, fused=0, identity=0, reference=0, verified=0)
_MERGE_LAYERS = {}
_MERGE_VERIFY_CALLS = 48
_MERGE_ORIGINALS = {}


@_merge_triton.jit
def _router_merge_kernel(G, W, E, O, M: _merge_tl.constexpr, N: _merge_tl.constexpr):
    rows = _merge_tl.program_id(0)*32 + _merge_tl.arange(0,32)
    cols = _merge_tl.program_id(1)*128 + _merge_tl.arange(0,128)
    kk = _merge_tl.arange(0,32)
    g = _merge_tl.load(G+rows[:,None]*32+kk[None,:], rows[:,None]<M, 0)
    w = _merge_tl.load(W+kk[:,None]*N+cols[None,:], cols[None,:]<N, 0)
    rounded = _merge_tl.dot(g,w).to(_merge_tl.bfloat16).to(_merge_tl.float32)
    expert = _merge_tl.load(E+rows[:,None]*N+cols[None,:],
                           (rows[:,None]<M)&(cols[None,:]<N),0).to(_merge_tl.float32)
    total = _merge_tl.inline_asm_elementwise('v_add_f32 $0, $1, $2',
        constraints='=v,v,v', args=[rounded,expert], dtype=_merge_tl.float32,
        is_pure=True, pack=1).to(_merge_tl.bfloat16)
    _merge_tl.store(O+rows[:,None]*N+cols[None,:],total,
                   (rows[:,None]<M)&(cols[None,:]<N))


def _merge_exact(a,b):
    return torch.equal(a.view(torch.int16),b.view(torch.int16))


class _RouterMergeRecord:
    def __init__(self,x,layer):
        self.pointer=x.data_ptr()
        self.shape=tuple(x.shape)
        self.numel=x.numel()
        self.layer=str(layer)
        self.stream=torch.cuda.current_stream(x.device)
        self.claimed=False
        self.grad=None
        self.weight=None
        self.gemm=None
        _MERGE_LIVE.add(self)
        _MERGE_STATS['created']+=1


def _router_merge_sum(record,expert):
    g,w,gemm=record.grad,record.weight,record.gemm
    current=torch.cuda.current_stream(g.device)
    same_stream=current.cuda_stream==record.stream.cuda_stream
    if not same_stream:
        current.wait_stream(record.stream)
        g.record_stream(current)
        w.record_stream(current)
    eligible=(same_stream and expert is not None and expert.dtype==torch.bfloat16
              and expert.is_contiguous() and g.is_contiguous() and w.is_contiguous()
              and tuple(w.shape)==(32,2880) and g.shape[1]==32
              and expert.numel()==g.shape[0]*2880)
    if not eligible:
        value=gemm(w,g,torch.bfloat16,layout='NN',grad=True)[0].view(record.shape)
        _MERGE_STATS['reference']+=1
        return value if expert is None else value+expert
    flat=expert.view(-1,2880)
    output=torch.empty_like(flat)
    _router_merge_kernel[(_merge_triton.cdiv(flat.shape[0],32),_merge_triton.cdiv(2880,128))](
        g,w,flat,output,flat.shape[0],2880,num_warps=4,matrix_instr_nonkdim=16)
    if _MERGE_STATS['verified']<_MERGE_VERIFY_CALLS:
        reference=gemm(w,g,torch.bfloat16,layout='NN',grad=True)[0]+flat
        if not _merge_exact(output,reference):
            raise RuntimeError('Router merge differs from actual TE dgrad plus BF16 addition')
        _MERGE_STATS['verified']+=1
    _MERGE_STATS['fused']+=1
    _MERGE_LAYERS[record.layer]=_MERGE_LAYERS.get(record.layer,0)+1
    return output.view(record.shape)


class _RouterGradientJoin(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,record):
        ctx.record=record
        ctx.set_materialize_grads(False)
        return x.view_as(x)

    @staticmethod
    def backward(ctx,expert):
        record=ctx.record
        if record.grad is None:
            _MERGE_STATS['identity']+=1
            return expert,None
        try:
            return _router_merge_sum(record,expert),None
        finally:
            record.grad=None
            record.weight=None
            record.gemm=None


def _merge_snapshot():
    return dict(_MERGE_STATS,layers=dict(_MERGE_LAYERS),live_records=len(_MERGE_LIVE))


def _merge_no_hooks(module):
    local=all(not getattr(module,key,{}) for key in
              ('_forward_hooks','_backward_hooks','_backward_pre_hooks'))
    # The pinned DDP hook only waits for parameter all-gather. It ignores the
    # input tuple, so it cannot add a consumer of the private activation.
    allowed_code=_MERGE_ORIGINALS.get('ddp_pre_hook_code')
    local=local and all(allowed_code is not None and getattr(hook,'__code__',None) is allowed_code
                       for hook in getattr(module,'_forward_pre_hooks',{}).values())
    global_hooks=torch.nn.modules.module
    return local and all(not getattr(global_hooks,key,{}) for key in
        ('_global_forward_hooks','_global_forward_pre_hooks',
         '_global_backward_hooks','_global_backward_pre_hooks'))


def _merge_parent_eligible(layer,hidden,inference_context,padding_mask):
    from megatron.core.transformer.moe.moe_layer import MoELayer
    cfg=layer.config
    mlp=getattr(layer,'mlp',None)
    norm=getattr(layer,'pre_mlp_layernorm',None)
    if not (layer.training and torch.is_grad_enabled() and hidden.requires_grad
            and inference_context is None and padding_mask is None and type(mlp) is MoELayer
            and norm is not None and not isinstance(norm,torch.nn.Identity)):
        return False
    if any(getattr(layer,key,False) for key in
           ('recompute_mlp','recompute_pre_mlp_layernorm','offload_mlp_norm')):
        return False
    if getattr(cfg,'cuda_graph_impl','none') not in (None,'none'):
        return False
    if getattr(cfg,'mlp_chunks_for_prefill',1)!=1:
        return False
    if getattr(layer._forward_pre_mlp_layernorm,'__func__',None) is not _MERGE_ORIGINALS['norm_forward']:
        return False
    if any(not _merge_no_hooks(module) for module in (layer,mlp,norm,mlp.router)):
        return False
    return True


def _merge_moe_eligible(layer,x,intermediate_tensors,padding_mask,parent):
    cfg=layer.config
    if parent is None or parent['moe'] is not layer or intermediate_tensors is not None or padding_mask is not None:
        return False
    if not (layer.training and torch.is_grad_enabled() and x.requires_grad
            and x.is_cuda and x.dtype==torch.bfloat16 and x.is_contiguous()
            and tuple(x.shape)==(8192,4,2880)):
        return False
    if x.untyped_storage().data_ptr()==parent['residual_storage']:
        return False
    if 'private_pointer' in parent and x.data_ptr()!=parent['private_pointer']:
        return False
    if any(getattr(layer,key,False) for key in ('moe_layer_recompute','use_shared_expert','shared_expert_overlap')):
        return False
    if any(getattr(cfg,key,None) for key in ('moe_latent_size','moe_input_jitter_eps','moe_shared_expert_intermediate_size')):
        return False
    if getattr(cfg,'moe_router_dtype',None)!='fp32' or getattr(cfg,'moe_router_force_load_balancing',False):
        return False
    if set(layer.fwd_execution_map)!=set(('route','expert_compute','postprocess')):
        return False
    if not all(getattr(getattr(layer,name),'__func__',None) is _MERGE_ORIGINALS[name]
               for name in ('route','preprocess','shared_experts_compute')):
        return False
    if getattr(layer.router.forward,'__func__',None) is not _MERGE_ORIGINALS['router_forward']:
        return False
    if getattr(layer.router.gating,'__func__',None) is not _MERGE_ORIGINALS['router_gating']:
        return False
    if not all(_merge_no_hooks(module) for module in (layer,layer.router)):
        return False
    return True



def _begin_fused_router_merge(layer,normalized,residual,inference_context,padding_mask):
    # Called only at the source-pinned fused residual/norm MLP boundary.
    # The pointer and independent residual storage bind the scope to its
    # actual private activation, rather than a general external MoE input.
    if not _MERGE_ORIGINALS or not _merge_parent_eligible(layer,residual,inference_context,padding_mask):
        return None
    parent=dict(moe=layer.mlp,residual_storage=residual.untyped_storage().data_ptr(),
                private_pointer=normalized.data_ptr())
    if not _merge_moe_eligible(layer.mlp,normalized,None,padding_mask,parent):
        return None
    return _MERGE_PARENT_SCOPE.set(parent)


def _end_fused_router_merge(token):
    _MERGE_PARENT_SCOPE.reset(token)


def _install_router_gradient_merge():
    if _merge_os.environ.get('PRIMUS_ROUTER_GRAD_MERGE','0')!='1':
        return
    from megatron.core.transformer import transformer_layer
    from megatron.core.transformer.moe import moe_layer,moe_utils,router
    from megatron.core.extensions import transformer_engine as te_ext
    from megatron.core.distributed import distributed_data_parallel as ddp_module
    from primus.backends.megatron.core.extensions import fused_residual_rmsnorm as fused_residual
    if getattr(moe_layer.MoELayer,'_primus_router_gradient_merge',False):
        return
    modules={fused_residual:'9162ec6e5ff614defb11cd30b2195060b853559b077c6eda5ea41f1fde3c1f84',
             transformer_layer:'9eefd719775c466956f1c640e592e34225670b7176fe1315f731827704bb11f1',
             moe_layer:'61fb6068259d91bf5e38754cb62b7a227becc3b71366ec81412125834ffe2044',
             moe_utils:'98fdcd7a5f21e67b1dfe38cc84a79da004ad0c94e9f3568632bd3f502b1f1ab4',
             router:'4e95f068ad1602605abc64242472a3205d4a1c92afded86e64c6284c33e7de02',
             te_ext:'26c8001d7454dc6c5b0715dd10436b9ec9966cc69c576e3a1a24fe7b2b158843',
             ddp_module:'385c0f87dc9e2e7988d9aea985b6ab7e06a0da38c5618c2488c55876ef18068c'}
    if any(_merge_hashlib.sha256(_MergePath(module.__file__).read_bytes()).hexdigest()!=digest
           for module,digest in modules.items()):
        raise RuntimeError('Router merge requires the reviewed deployed source graph')
    Parent=transformer_layer.TransformerLayer
    MoE=moe_layer.MoELayer
    Router=moe_utils.RouterGatingLinearFunction
    hook_factory=ddp_module.DistributedDataParallel._make_forward_pre_hook
    if hook_factory.__code__.co_filename!=ddp_module.__file__:
        raise RuntimeError('Unreviewed DDP parameter-sync hook factory')
    originals=dict(parent_forward=Parent._forward_mlp,norm_forward=Parent._forward_pre_mlp_layernorm,
                   moe_forward=MoE.forward,route=MoE.route,preprocess=MoE.preprocess,
                   shared_experts_compute=MoE.shared_experts_compute,
                   router_forward=router.TopKRouter.forward,router_gating=router.Router.gating,
                   gating_forward=Router.forward,gating_backward=Router.backward)
    for name,module in [('parent_forward',transformer_layer),('norm_forward',transformer_layer),
                        ('moe_forward',moe_layer),('gating_forward',moe_utils),
                        ('router_forward',router),('router_gating',router)]:
        if originals[name].__code__.co_filename!=module.__file__:
            raise RuntimeError(f'Unreviewed dynamic method for router merge: {name}')
    _MERGE_ORIGINALS.update(originals)
    _MERGE_ORIGINALS['ddp_pre_hook_code']=hook_factory(None).__code__

    def parent_forward(self,hidden_states,inference_context=None,padding_mask=None):
        if not _merge_parent_eligible(self,hidden_states,inference_context,padding_mask):
            return originals['parent_forward'](self,hidden_states,inference_context,padding_mask)
        token=_MERGE_PARENT_SCOPE.set(dict(moe=self.mlp,residual_storage=hidden_states.untyped_storage().data_ptr()))
        try:
            return originals['parent_forward'](self,hidden_states,inference_context,padding_mask)
        finally:
            _MERGE_PARENT_SCOPE.reset(token)

    def moe_forward(self,hidden_states,intermediate_tensors=None,padding_mask=None):
        parent=_MERGE_PARENT_SCOPE.get()
        if not _merge_moe_eligible(self,hidden_states,intermediate_tensors,padding_mask,parent):
            return originals['moe_forward'](self,hidden_states,intermediate_tensors,padding_mask)
        record=_RouterMergeRecord(hidden_states,self.layer_number)
        joined=_RouterGradientJoin.apply(hidden_states,record)
        token=_MERGE_ROUTE_SCOPE.set(record)
        try:
            return originals['moe_forward'](self,joined,intermediate_tensors,padding_mask)
        finally:
            _MERGE_ROUTE_SCOPE.reset(token)

    def gating_forward(ctx,inp,weight,bias,router_dtype):
        record=_MERGE_ROUTE_SCOPE.get()
        eligible=(record is not None and inp.requires_grad and inp.dtype==weight.dtype==torch.bfloat16
                  and router_dtype==torch.float32 and inp.is_contiguous() and weight.is_contiguous()
                  and tuple(weight.shape)==(32,2880) and inp.numel()==record.numel
                  and inp.data_ptr()==record.pointer
                  and torch.cuda.current_stream(inp.device).cuda_stream==record.stream.cuda_stream)
        if eligible:
            if record.claimed:
                raise RuntimeError('Multiple router consumers in a claimed private input scope')
            record.claimed=True
            _MERGE_STATS['claimed']+=1
            ctx._router_merge_record=record
        return originals['gating_forward'](ctx,inp,weight,bias,router_dtype)

    def gating_backward(ctx,grad_output):
        record=getattr(ctx,'_router_merge_record',None)
        if (record is None or not _uses_bf16_router_backward(ctx,moe_utils.te_general_gemm)
                or torch.is_grad_enabled()
                or torch.cuda.current_stream(grad_output.device).cuda_stream!=record.stream.cuda_stream):
            return originals['gating_backward'](ctx,grad_output)
        if record.grad is not None:
            raise RuntimeError('Concurrent/reentrant backward tried to reuse router merge state')
        inp,weight,bias=ctx.saved_tensors
        inp=inp.view(-1,inp.shape[-1])
        grad_output=grad_output.view(-1,grad_output.shape[-1])
        grad_output_bf16=grad_output.to(torch.bfloat16)
        grad_weight=moe_utils.te_general_gemm(inp,grad_output_bf16,ctx.router_dtype,layout='NT',grad=True)[0].to(ctx.weight_dtype)
        grad_bias=grad_output.sum(dim=0).to(ctx.weight_dtype) if bias is not None else None
        record.grad=grad_output_bf16
        record.weight=weight
        record.gemm=moe_utils.te_general_gemm
        return None,grad_weight,grad_bias,None

    Parent._forward_mlp=parent_forward
    MoE.forward=moe_forward
    Router.forward=staticmethod(gating_forward)
    Router.backward=staticmethod(gating_backward)
    MoE._primus_router_gradient_merge=True
    log_rank_0('[Patch:megatron.moe.router_gradient_merge] Installed guarded GPT-OSS input-gradient join')
