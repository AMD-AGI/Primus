# Installation

This guide walks you through installing Primus and preparing an environment optimized for ROCm-based large model training.

---

## 📦 Prerequisites

Before installing Primus, ensure the following:

- ROCm >= 6.3 is installed and functional.
- Compatible AMD GPUs (e.g., MI300, MI250, MI210)
- Python 3.10 or later (recommended: use `conda` or `venv`)
- gcc >= 9 (required for compiling extensions)
- pip >= 22.0

---

## 🛠️ Step-by-Step Installation

### 1. Clone the repository

```bash
git clone https://github.com/amd/primus.git
cd primus
```

### 2. Create Python environment (recommended)

Using conda:

```bash
conda create -n primus python=3.10 -y
conda activate primus
```

Or using venv:

```bash
python3 -m venv primus-env
source primus-env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> You can modify `requirements.txt` to fit ROCm-specific wheels.

---

## 🔥 Installing PyTorch with ROCm

Depending on your ROCm version:

```bash
# Example for ROCm 6.3
pip install torch==2.3.0+rocm6.3 torchvision==0.18.0+rocm6.3     --extra-index-url https://download.pytorch.org/whl/rocm6.3
```

Check GPU availability:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.hip)"
```

---

## 🚀 Install Primus

After environment setup:

```bash
pip install .
```

This will install Primus as a Python package with its CLI tool `primus-cli`.

---

## ✅ Verify Installation

Run the following to confirm:

```bash
primus-cli --help
```

You should see the Primus CLI usage message.

---

## 🧪 (Optional) Test GEMM Benchmark

```bash
primus-cli direct -- benchmark gemm --m 4096 --n 4096 --k 4096
```

---

_Last updated: 2025-09-17_
