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
- CUDA 13 toolkit installed at `/usr/local/cuda-13.0`  
- Linux (Ubuntu 22.04/24.04 recommended)  
- Internet access (first run only)

---

## 🛠️ Setup

Copy the setup script to your DGX node:

```bash
scp setup_nanochat_dgx.sh <user>@<dgx-host>:~
ssh <user>@<dgx-host>
bash setup_nanochat_dgx.sh
