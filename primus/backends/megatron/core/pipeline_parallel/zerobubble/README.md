## 1. Zero bubbles introduction
Zero bubbles is a state-of-art technique aiming to reduce the bubble time and memory consumption in pipeline schedule, which was introduced by Sea AI Lab. Thanks to them for this efficient work!


- The codes are migrate from [sail-sg/zero-bubble-pipeline-parallelism: Zero Bubble Pipeline Parallelism](https://github.com/sail-sg/zero-bubble-pipeline-parallelism)
- commit-id: 39f2c186a27580b7051eb51e0f651e1efd0a8170(Amend offload schedule error message (#69))

## 2. How to use

- Zero bubble patch the gemm OP and the grouped gemm OP for splitting the backward of the inputs and weights, support TE & Primus-turbo backend.

- Some other flags need to be specified

```
overlap_grad_reduce: false
overlap_param_gather: false
no_persist_layer_norm: true
create_attention_mask_in_dataloader: false
gradient_accumulation_fusion: true
```

- Most of the zero-bubble flags are writted in `zero_bubble.yaml`, others reuse megatron flags. Here are some examples for config your prefer PP stratages

| pp strategy / flag | num_virtual_stages_per_pipeline_rank | patch_zero_bubble | zero_bubble_v_schedule | zero_bubble_v_schedule_mem_setup |
|---|---|---|---|---|
| turbo-1f1b | 1 |  false | - | - |
| turbo-1f1b-interleaved | >=2 |  false | - | - |
| zero bubble 1p | 1 | true | false | - |
| zbv | 2 | true | true | zb |
| v-half | 2 | true | true | half |
| v-min | 2 | true | true | min |

## 3. Comming soon

- support TP/SP overlap
- support FP8
- support ZeroBubble-P2
