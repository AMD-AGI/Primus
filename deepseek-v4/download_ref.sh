#!/usr/bin/env bash
set -euo pipefail

# Ensure git-lfs is installed (required by HuggingFace model repos).
if ! command -v git-lfs >/dev/null 2>&1; then
    echo "[ERROR] git-lfs is not installed. Install it first, e.g. into ~/.local/bin:" >&2
    echo "  curl -sSL https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz \\" >&2
    echo "    | tar -xz -C /tmp && cp /tmp/git-lfs-3.5.1/git-lfs ~/.local/bin/" >&2
    exit 1
fi
git lfs install --skip-repo

# Clone a repo and checkout a specific commit; if the dir already exists, just ensure the commit.
clone_at_commit() {
    local url="$1" dir="$2" sha="$3"
    if [[ -d "$dir/.git" ]]; then
        echo "[skip] $dir already exists; ensuring it is at $sha"
        ( cd "$dir" && git fetch --quiet origin "$sha" 2>/dev/null || true && git checkout --quiet "$sha" )
    else
        git clone "$url" "$dir"
        ( cd "$dir" && git checkout "$sha" )
    fi
}

# Clone a HuggingFace repo, skipping LFS smudge so large files are not downloaded.
hf_clone() {
    local url="$1" dir="$2"
    if [[ -d "$dir/.git" ]]; then
        echo "[skip] $dir already exists"
        return
    fi
    GIT_LFS_SKIP_SMUDGE=1 git clone "$url" "$dir"
}

# ---- GitHub repos ----
clone_at_commit https://github.com/huggingface/transformers.git           transformers              61461a7bcb458db7cf6eeea49678b9ab776a7821
clone_at_commit https://github.com/ROCm/TransformerEngine.git             TransformerEngine         51f74fa7c942b7bfb1b244bd66f762b03969d9a2
clone_at_commit https://github.com/AMD-AGI/Primus-Turbo.git               Primus-Turbo              45fe919d3993d18d0bda9a61cbcd778c7bc03f41

mkdir -p NVIDIA-NeMo
clone_at_commit https://github.com/NVIDIA-NeMo/Automodel.git              NVIDIA-NeMo/Automodel     95113ea7c23b51815ad3ef44065d5718466fd6b8

# ---- HuggingFace model repos (contain large files; require git-lfs, smudge is skipped here) ----
mkdir -p deepseek-ai
hf_clone https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro          deepseek-ai/DeepSeek-V4-Pro
hf_clone https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash        deepseek-ai/DeepSeek-V4-Flash
hf_clone https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base   deepseek-ai/DeepSeek-V4-Flash-Base
hf_clone https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base     deepseek-ai/DeepSeek-V4-Pro-Base

echo "[done] All repos are ready"
