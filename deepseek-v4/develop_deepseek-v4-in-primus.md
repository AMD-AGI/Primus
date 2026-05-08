
# Goal

Add DeepSeek-V4 training support to Primus, on top of the Megatron-LM training
backend (`third_party/Megatron-LM`).

Primus's own source lives under `primus/` — feel free to skim it first.
The way Primus extends Megatron-LM is through the **primus-patch** mechanism;
all our code changes live in Primus, never in `third_party/`. Model and
training configs are driven by yaml.

Megatron-LM today ships GPT, Mamba and a couple of other model types.
DeepSeek-V4 is a brand-new species — it likely needs changes around the
transformer block, transformer layer and attention modules — so we'll
probably want to define a new model type inside Primus and let users select
the DeepSeek-V4 model from there.

A few Primus development conventions worth knowing:

1. `primus/backends/megatron/` mirrors the Megatron repo layout. The modules
   we want to override go under the same relative path as their upstream
   counterparts. For example, anything under
   `primus/backends/megatron/core/transformer/attention.py` is meant to map
   onto `third_party/Megatron-LM/megatron/core/transformer/attention.py`.
2. **All Megatron extensions live inside Primus**; don't touch
   `third_party/`. Extensions are wired in via primus patches — please read
   `primus/backends/megatron/patches/` to see how that works.
3. Primus has its own trainer: `primus/backends/megatron/megatron_pretrain_trainer.py`.
   That's where the `model_type` switch lives (gpt vs. others). DeepSeek-V4
   should slot in as another option here. For inspiration on how to define
   a new model + spec, take a look at
   `third_party/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/models/mamba/`.
   Add the DeepSeek-V4 model definition into Primus, and make the trainer
   able to pick it.
4. Models in Primus are configured through yaml; those configs are
   ultimately consumed by Megatron. The schema definitions live in
   `primus/configs/`, so we need to add DeepSeek-V4 configs there.
   Concrete training yamls live in `examples/megatron/`; add a
   DeepSeek-V4 training yaml under the **MI355X** subdirectory so it can be
   launched directly through `primus-cli`.

# References

## Official model

The four DeepSeek-V4 model variants live under `deepseek-v4/deepseek-ai/`.
Each comes with the HF-style `config.json` and a reference inference
implementation, but no training implementation.
The technical report is at
`deepseek-v4/deepseek-ai/DeepSeek-V4-Flash/DeepSeek_V4.pdf`.

## Huggingface Transformers

The `deepseek-v4/transformers/` directory is a checkout of the `transformers`
repo. It does NOT yet have a `DeepseekV4ForCausalLM`. We can still cross-reference
the existing `DeepseekV3ForCausalLM` and `DeepseekV32ForCausalLM` though —
just remember these are HF implementations, not Megatron-style.

## NVIDIA NeMo

`deepseek-v4/NVIDIA-NeMo/Automodel/` is the NeMo AutoModel repo, which DOES
contain a `DeepseekV4ForCausalLM` under
`nemo_automodel/components/models/deepseek_v4/`.
This one is **a primary reference** for us, because NeMo is also built on
top of mcore. There's also a doc at
`docs/model-coverage/llm/deepseek-ai/dsv4-flash.md` — note that NeMo only
covers V4-Flash.

## TransformerEngine (Megatron dependency)

`deepseek-v4/TransformerEngine/` is the TransformerEngine code that Megatron
depends on. Use this checkout as the reference when tracing Megatron's
TransformerEngine modules, kernels and specs.

## Primus-Turbo (Primus dependency)

`deepseek-v4/Primus-Turbo/` is dependency code used by Primus. Use this
checkout as a reference for Primus-side dependency behavior and integration
details.

## RedNote

`deepseek-v4/references/deepseek-rednote-1/` is a slide deck I came across
on RedNote that walks through the DeepSeek-V4 technical details. It's an
image-only deck, but the explanations are useful — worth a look.


# Development cadence

1. First, analyze how DeepSeek-V4 differs from DeepSeek-V3 / V3.2.
   Take the RedNote post as inspiration and write a tech blog post under
   `deepseek-v4/develop/techblog/`. Generate diagrams for the key modules
   (e.g. what CSA, HCA, mHC actually are) and an overall architecture
   diagram. **Stop and tell me once this is done.**
2. Then draft a development plan — markdown — for integrating DeepSeek-V4
   training into Primus, and put it under `deepseek-v4/develop/`. If the
   work is interrupted later, we should be able to resume from this plan.
   Whenever a step produces some documentation, it can live under any path
   inside `develop/` — you decide the directory structure under `develop/`.
   Based on the code references above, also write up the concrete code-level
   workflow for the DeepSeek-V4 implementation in Primus: which modules need
   to be built, where each piece of code will live, etc.
   **Stop and tell me once this is done.**
3. Start writing the code.
