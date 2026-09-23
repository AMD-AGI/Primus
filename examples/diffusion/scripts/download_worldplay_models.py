#!/usr/bin/env python3
"""
Download script for HY-WorldPlay models.
Downloads all required models from HuggingFace and ModelScope.

Copied from Tencent HY-WorldPlay download_models.py and adapted for Primus:
downloads land under MODELS_DIR (default /data/models) and stable
MODELS_DIR/HunyuanVideo-1.5 and MODELS_DIR/HY-WorldPlay paths are created.

Usage:
    python examples/diffusion/scripts/download_worldplay_models.py --hf_token <your_token>

The HF token is required for downloading the vision encoder from FLUX.1-Redux-dev.
Request access at: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev
"""

import argparse
import os
import shutil
from pathlib import Path

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/data/models")).resolve()
os.environ.setdefault("HF_HOME", str(MODELS_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODELS_DIR / "hub"))


def check_dependencies():
    """Check and install required dependencies."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install -U 'huggingface_hub[cli]'")

    try:
        import modelscope
    except ImportError:
        print("Installing modelscope...")
        os.system("pip install modelscope")


def download_hy_worldplay():
    """Download HY-WorldPlay action models."""
    from huggingface_hub import snapshot_download

    print("\n" + "=" * 60)
    print("[1/6] Downloading tencent/HY-WorldPlay...")
    print("=" * 60)

    worldplay_path = snapshot_download("tencent/HY-WorldPlay")
    print(f"Downloaded to: {worldplay_path}")

    # Fix: Rename model.safetensors to diffusion_pytorch_model.safetensors
    # in ar_distilled_action_model (repo uses different filename)
    ar_distill_dir = os.path.join(worldplay_path, "ar_distilled_action_model")
    model_src = os.path.join(ar_distill_dir, "model.safetensors")
    model_dst = os.path.join(ar_distill_dir, "diffusion_pytorch_model.safetensors")

    if os.path.exists(model_src) and not os.path.exists(model_dst):
        real_src = os.path.realpath(model_src)
        shutil.copy2(real_src, model_dst)
        print(
            "Fixed: Renamed model.safetensors -> diffusion_pytorch_model.safetensors in ar_distilled_action_model"
        )

    return worldplay_path


def download_hunyuan_video():
    """Download HunyuanVideo-1.5 base models (vae, scheduler, transformer)."""
    from huggingface_hub import snapshot_download

    print("\n" + "=" * 60)
    print("[2/6] Downloading tencent/HunyuanVideo-1.5 (vae, scheduler, transformer)...")
    print("=" * 60)

    hunyuan_path = snapshot_download(
        "tencent/HunyuanVideo-1.5",
        allow_patterns=["vae/*", "scheduler/*", "transformer/480p_i2v/*"],
    )
    print(f"Downloaded to: {hunyuan_path}")
    return hunyuan_path


def download_llm_text_encoder(hunyuan_path):
    """Download Qwen2.5-VL-7B-Instruct as the LLM text encoder."""
    from huggingface_hub import snapshot_download

    print("\n" + "=" * 60)
    print("[3/6] Downloading LLM text encoder (Qwen2.5-VL-7B-Instruct)...")
    print("=" * 60)

    text_encoder_base = os.path.join(hunyuan_path, "text_encoder")
    os.makedirs(text_encoder_base, exist_ok=True)

    llm_target = os.path.join(text_encoder_base, "llm")

    if (
        os.path.exists(llm_target)
        and os.path.isdir(llm_target)
        and len(os.listdir(llm_target)) > 5
    ):
        print(f"LLM text encoder already exists at: {llm_target}")
        return

    # Clean up old/broken downloads
    if os.path.islink(llm_target):
        os.unlink(llm_target)
    elif os.path.exists(llm_target):
        shutil.rmtree(llm_target)

    print("Downloading Qwen/Qwen2.5-VL-7B-Instruct (~15GB)...")
    qwen_cache = snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")

    # Copy files (resolve symlinks)
    os.makedirs(llm_target, exist_ok=True)
    for item in os.listdir(qwen_cache):
        src = os.path.realpath(os.path.join(qwen_cache, item))
        dst = os.path.join(llm_target, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    print(f"Copied to: {llm_target}")


def download_byt5_encoders(hunyuan_path):
    """Download ByT5 text encoders (byt5-small and Glyph-SDXL-v2)."""
    from huggingface_hub import snapshot_download
    from modelscope import snapshot_download as ms_snapshot_download

    print("\n" + "=" * 60)
    print("[4/6] Downloading ByT5 text encoders...")
    print("=" * 60)

    text_encoder_base = os.path.join(hunyuan_path, "text_encoder")
    os.makedirs(text_encoder_base, exist_ok=True)

    # 1. Download google/byt5-small
    byt5_target = os.path.join(text_encoder_base, "byt5-small")
    if (
        os.path.exists(byt5_target)
        and os.path.isdir(byt5_target)
        and len(os.listdir(byt5_target)) > 3
    ):
        print(f"byt5-small already exists at: {byt5_target}")
    else:
        if os.path.islink(byt5_target):
            os.unlink(byt5_target)
        elif os.path.exists(byt5_target):
            shutil.rmtree(byt5_target)

        print("Downloading google/byt5-small...")
        byt5_cache = snapshot_download("google/byt5-small")

        os.makedirs(byt5_target, exist_ok=True)
        for item in os.listdir(byt5_cache):
            src = os.path.realpath(os.path.join(byt5_cache, item))
            dst = os.path.join(byt5_target, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        print(f"Copied to: {byt5_target}")

    # 2. Download Glyph-SDXL-v2 from ModelScope
    glyph_target = os.path.join(text_encoder_base, "Glyph-SDXL-v2")
    if os.path.exists(glyph_target) and os.path.exists(
        os.path.join(glyph_target, "checkpoints", "byt5_model.pt")
    ):
        print(f"Glyph-SDXL-v2 already exists at: {glyph_target}")
    else:
        if os.path.exists(glyph_target):
            shutil.rmtree(glyph_target)

        print("Downloading AI-ModelScope/Glyph-SDXL-v2 from ModelScope...")
        glyph_cache = ms_snapshot_download(
            "AI-ModelScope/Glyph-SDXL-v2", cache_dir="/tmp/glyph_cache"
        )

        os.makedirs(glyph_target, exist_ok=True)
        for item in os.listdir(glyph_cache):
            src = os.path.join(glyph_cache, item)
            dst = os.path.join(glyph_target, item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        print(f"Copied to: {glyph_target}")


def download_vision_encoder(hunyuan_path, hf_token):
    """Download SigLIP vision encoder from FLUX.1-Redux-dev."""
    from huggingface_hub import snapshot_download

    print("\n" + "=" * 60)
    print("[5/6] Downloading Vision Encoder (SigLIP from FLUX.1-Redux-dev)...")
    print("=" * 60)

    if not hf_token:
        print("WARNING: No HF token provided!")
        print(
            "The vision encoder requires access to: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev"
        )
        print("Skipping vision encoder download.")
        print("\nYou can download it manually later.")
        return

    vision_encoder_base = os.path.join(hunyuan_path, "vision_encoder")
    os.makedirs(vision_encoder_base, exist_ok=True)

    siglip_target = os.path.join(vision_encoder_base, "siglip")

    if (
        os.path.exists(siglip_target)
        and os.path.isdir(siglip_target)
        and len(os.listdir(siglip_target)) > 3
    ):
        print(f"siglip already exists at: {siglip_target}")
        return

    # Clean up old/broken downloads
    if os.path.islink(siglip_target):
        os.unlink(siglip_target)
    elif os.path.exists(siglip_target):
        shutil.rmtree(siglip_target)

    print("Downloading black-forest-labs/FLUX.1-Redux-dev...")
    try:
        flux_cache = snapshot_download(
            "black-forest-labs/FLUX.1-Redux-dev", token=hf_token
        )

        # Copy files (resolve symlinks)
        os.makedirs(siglip_target, exist_ok=True)
        for item in os.listdir(flux_cache):
            src = os.path.realpath(os.path.join(flux_cache, item))
            dst = os.path.join(siglip_target, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        print(f"Copied to: {siglip_target}")
    except Exception as e:
        print(f"ERROR: Failed to download vision encoder: {e}")
        print(
            "Make sure you have requested access to FLUX.1-Redux-dev and your token is valid."
        )


def link_stable_paths(hunyuan_path, worldplay_path):
    """Expose the snapshots under stable MODELS_DIR paths used by Primus configs."""
    for name, snapshot in (
        ("HunyuanVideo-1.5", hunyuan_path),
        ("HY-WorldPlay", worldplay_path),
    ):
        stable = MODELS_DIR / name
        target = Path(snapshot).resolve()
        if stable.is_symlink() or stable.is_file():
            stable.unlink()
        elif stable.is_dir():
            if stable.resolve() == target:
                continue
            raise SystemExit(f"Refusing to replace non-symlink model path: {stable}")
        stable.symlink_to(target, target_is_directory=True)


def print_paths(hunyuan_path, worldplay_path):
    """Print the model paths for Primus configs.

    Do not call snapshot_download(..., local_files_only=True) on
    tencent/HunyuanVideo-1.5 here. Step [2/6] only fetches
    vae/scheduler/transformer/480p_i2v, then later steps copy LLM/ByT5/SigLIP
    into that snapshot. Newer huggingface_hub treats any missing repo file
    (LICENSE, 720p transformers, ...) as an incomplete snapshot.
    """
    print("\n" + "=" * 60)
    print("[6/6] Verifying downloads...")
    print("=" * 60)

    required = [
        os.path.join(hunyuan_path, "vae"),
        os.path.join(hunyuan_path, "scheduler"),
        os.path.join(hunyuan_path, "transformer", "480p_i2v"),
        os.path.join(hunyuan_path, "text_encoder", "llm"),
        os.path.join(hunyuan_path, "text_encoder", "byt5-small"),
        os.path.join(
            hunyuan_path, "text_encoder", "Glyph-SDXL-v2", "checkpoints", "byt5_model.pt"
        ),
        os.path.join(hunyuan_path, "vision_encoder", "siglip"),
        os.path.join(worldplay_path, "ar_model", "diffusion_pytorch_model.safetensors"),
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        print("WARNING: some expected files are missing:")
        for path in missing:
            print(f"  {path}")
    else:
        print("Required WorldPlay SFT files are present.")

    print("\n" + "=" * 60)
    print("ALL DOWNLOADS COMPLETE!")
    print("=" * 60)
    print("\nStable paths used by the Primus configs:\n")
    print(f"HUNYUAN_TRANSFORMER={MODELS_DIR}/HunyuanVideo-1.5/transformer/480p_i2v")
    print(
        f"WORLDPLAY_ACTION_CHECKPOINT={MODELS_DIR}/HY-WorldPlay/ar_model/diffusion_pytorch_model.safetensors"
    )
    print(f"\nEncode with --hunyuan-checkpoint-path {MODELS_DIR}/HunyuanVideo-1.5")


def main():
    parser = argparse.ArgumentParser(
        description="Download all required models for HY-WorldPlay SFT in Primus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python examples/diffusion/scripts/download_worldplay_models.py --hf_token hf_xxxxxxxxxxxxx

Note:
    The HuggingFace token is required for downloading the vision encoder
    from black-forest-labs/FLUX.1-Redux-dev. You need to:
    1. Request access at: https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev
    2. Wait for approval (usually instant)
    3. Create a token at: https://huggingface.co/settings/tokens (select "Read" permission)
        """,
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token for downloading gated models (required for vision encoder)",
    )
    parser.add_argument(
        "--skip_vision_encoder",
        action="store_true",
        help="Skip downloading the vision encoder (if you don't have FLUX access yet)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("HY-WorldPlay Model Download Script")
    print("=" * 60)
    print(f"MODELS_DIR={MODELS_DIR}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Check dependencies
    check_dependencies()

    # Download models
    worldplay_path = download_hy_worldplay()
    hunyuan_path = download_hunyuan_video()
    download_llm_text_encoder(hunyuan_path)
    download_byt5_encoders(hunyuan_path)

    if not args.skip_vision_encoder:
        download_vision_encoder(hunyuan_path, args.hf_token)
    else:
        print("\n[5/6] Skipping vision encoder download (--skip_vision_encoder flag)")

    link_stable_paths(hunyuan_path, worldplay_path)
    print_paths(hunyuan_path, worldplay_path)


if __name__ == "__main__":
    main()
