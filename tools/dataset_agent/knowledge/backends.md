# Primus dataset plug-in points

One task names one backend. Edit only that backend's registration path. The example YAML must name the operator's dataset. Load the dataset from the Hugging Face Hub or from the URL published with the dataset.

## diffusion

Registration: `primus/backends/diffusion/data/flux_precomputed.py` (`FluxRawImageTextDataset`) and Wan's loader in `primus/backends/diffusion/data/dataset.py`. Flux formats today: `jsonl`, `hf_dataset` (`load_from_disk`), `hf_repo` (`load_dataset(path, split="train")`), `webdataset`. YAML keys: `data.dataset_type`, `data.dataset_format`, `data.dataset_path`, `data.dataset`. Argument builder: `DiffusionArgBuilder` maps those keys in `primus/backends/diffusion/argument_builder.py`. A Hub parquet dataset is already readable with `dataset_format: hf_repo` when `load_dataset` accepts the repo id. Add a new format branch only when none of the current branches can read the Hub files and columns.

## megatron

Post-train SFT formats already in `create_formatter()`: alpaca, chatml, openai, messages, squad, bridge_squad. Use one of those when its columns match the dataset. Add a new formatter class and a new `create_formatter()` name only when none of them can read the columns. YAML keys: `sft_dataset_name`, `sft_conversation_format`. The example module is `post_trainer`. Pre-train formats already present: jsonl, hf_dataset, hf_repo, webdataset. Add a new `dataset_format` and reader only when none of those can read the files. The example module is `pre_trainer`. Indexed pretrain data uses `data_path`.

## megatron_bridge

Recipe branch on `dataset_type` under `primus/backends/megatron_bridge/recipes/`. YAML keys: `dataset_type`, `dataset.dataset_name`.

## torchtitan

`training.dataset` and `training.dataset_path`. The name registry is `DATASETS` in the TorchTitan submodule `third_party/torchtitan`.

## maxtext

Module keys: `dataset_type`, `hf_path`, `grain_file_type`. The loader lives in `third_party/maxtext`. `dataset_type: hf` is the Hub path.

## maxdiffusion

Model preset keys: `dataset_type`, `train_data_dir`, `dataset_name`. Loaders live in `third_party/maxdiffusion`.

## automodel

Cache builders: `CACHE_BUILDERS` in `primus/backends/automodel/data/registry.py`. Training dataloaders are selected by `data.dataloader._target_` in the module preset.

## worldmirror

Directory roots via `dataset_root()` in `primus/backends/worldmirror/worldmirror_trainer.py`. Hydra `train_datasets` entries live under `primus/backends/worldmirror/hydra_configs/`. The example YAML sets the directory key (for example `worldmirror.hypersim_dir`) to the published dataset location.
