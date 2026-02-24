cat << 'EOF' > README.md
# Reproducible Nanochat Setup for NVIDIA DGX Spark (CUDA 13)

This repository provides a reproducible bootstrap workflow to run **nanochat** on NVIDIA DGX Spark systems with CUDA 13 and Triton compatibility.

It is designed for research labs, multi-GPU environments, and reproducible experiments.

---

## 🚀 Features

- Idempotent setup script (safe to rerun)
- CUDA 13 + Triton configuration
- Automatic Rust + tokenizer build
- Dataset cache detection
- Ready for single-GPU or multi-GPU runs
- Portable across DGX nodes

---

## 📋 Requirements

- NVIDIA DGX Spark or compatible GPU server
- NVIDIA driver installed
- CUDA 13 toolkit installed at /usr/local/cuda-13.0
- Linux (Ubuntu 22.04/24.04 recommended)
- Internet access (first run only)

---

## 🛠️ Setup

Copy the setup script to your DGX node:

scp setup_nanochat_dgx.sh <user>@<dgx-host>:~
ssh <user>@<dgx-host>
bash setup_nanochat_dgx.sh

---

## ▶️ Run Training

cd ~/nanochat
source .venv/bin/activate
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas

torchrun --standalone --nproc_per_node=gpu \
  -m scripts.base_train -- --depth=20

---

## 💾 Using a Shared Dataset Cache

export NANOCHAT_CACHE_DIR=/dgx-storage/nanochat-cache

---

## 🔎 Verify Installation

nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"

---

## 📁 Recommended DGX Lab Layout

/dgx
 ├── datasets/
 ├── nanochat/
 ├── checkpoints/
 └── envs/

---

## ⚠️ Troubleshooting

Rust not found:
source ~/.cargo/env

CUDA not detected:
export CUDA_HOME=/usr/local/cuda-13.0

---

## 📜 License

See original nanochat repo:
https://github.com/karpathy/nanochat

---

## 🙌 Acknowledgements

- Andrej Karpathy — nanochat
- NVIDIA DGX platform
- PyTorch & Triton teams

EOF
