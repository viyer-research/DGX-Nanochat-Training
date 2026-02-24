# DGX-Nanochat-Training
Blackwell modification using single gpu
#!/usr/bin/env bash
set -euo pipefail

echo "========================================="
echo " Nanochat DGX Spark Bootstrap"
echo "========================================="

# ---------- CONFIG ----------
PROJECT_DIR="$HOME/nanochat"
REPO_URL="https://github.com/karpathy/nanochat.git"
CACHE_DIR="${NANOCHAT_CACHE_DIR:-$HOME/.cache/nanochat}"
CUDA_HOME="/usr/local/cuda-13.0"

# ---------- LOAD RUST ----------
if [ -f "$HOME/.cargo/env" ]; then
    echo "[Rust] Loading cargo environment"
    source "$HOME/.cargo/env"
fi

# ---------- CLONE REPO ----------
if [ ! -d "$PROJECT_DIR" ]; then
    echo "[Repo] Cloning nanochat"
    git clone $REPO_URL "$PROJECT_DIR"
else
    echo "[Repo] nanochat already exists"
fi

cd "$PROJECT_DIR"

# ---------- INSTALL UV ----------
echo "[Env] Checking uv"
command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# ---------- CREATE VENV ----------
if [ ! -d ".venv" ]; then
    echo "[Env] Creating virtual environment"
    uv venv
else
    echo "[Env] Virtual environment exists"
fi

# ---------- ACTIVATE ----------
source .venv/bin/activate

# ---------- INSTALL DEPENDENCIES ----------
echo "[Deps] Installing Python dependencies"
uv sync

# ---------- ENSURE MATURIN ----------
if ! command -v maturin &>/dev/null; then
    echo "[Deps] Installing maturin"
    uv pip install maturin
fi

# ---------- BUILD TOKENIZER ----------
echo "[Build] Compiling Rust tokenizer"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# ---------- DATASET ----------
mkdir -p "$CACHE_DIR"

if [ -d "$CACHE_DIR/dataset" ]; then
    echo "[Data] Dataset cache detected — skipping download"
else
    echo "[Data] Downloading dataset"
    python -m nanochat.dataset -n 240
fi

# ---------- TOKENIZER ----------
if ls "$CACHE_DIR" | grep -q tokenizer; then
    echo "[Tokenizer] Found cached tokenizer — skipping"
else
    echo "[Tokenizer] Training tokenizer"
    python -m scripts.tok_train --max_chars=2000000000
    python -m scripts.tok_eval
fi

# ---------- EVAL BUNDLE ----------
if [ ! -d "$CACHE_DIR/eval_bundle" ]; then
    echo "[Eval] Downloading eval bundle"
    curl -L -o eval_bundle.zip https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle "$CACHE_DIR"
else
    echo "[Eval] Eval bundle exists"
fi

# ---------- CUDA ENV ----------
echo "[CUDA] Setting environment variables"
export TRITON_PTXAS_PATH=$CUDA_HOME/bin/ptxas
export CUDA_HOME=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

echo "========================================="
echo " Setup Complete"
echo "========================================="
echo ""
echo "To start training:"
echo ""
echo "cd $PROJECT_DIR"
echo "source .venv/bin/activate"
echo "export TRITON_PTXAS_PATH=$CUDA_HOME/bin/ptxas"
echo "torchrun --standalone --nproc_per_node=gpu -m scripts.base_train -- --depth=20"
