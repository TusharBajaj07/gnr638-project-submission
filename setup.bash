#!/usr/bin/env bash
# GNR 638 Project 1 — environment setup (run with internet available).
# Creates conda env `gnr_project_env` (python 3.11), installs deps, and
# pre-downloads the Qwen2.5-VL-7B-AWQ weights so inference can run offline.

set -e

ENV_NAME="gnr_project_env"
PY_VER="3.11"
REPO_URL="https://github.com/TusharBajaj07/gnr638-project-submission.git"

echo "[setup] === GNR 638 Project 1 setup ==="
echo "[setup] target env: ${ENV_NAME} (python ${PY_VER})"

# Create conda env if missing
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[setup] creating conda env ${ENV_NAME}..."
    conda create -y -n "${ENV_NAME}" python="${PY_VER}"
else
    echo "[setup] conda env ${ENV_NAME} already exists, reusing."
fi

# Activate (works across bash + conda configurations)
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Clone repo (only if not already present in cwd)
if [ ! -d "src" ] || [ ! -f "inference.py" ]; then
    echo "[setup] cloning repo..."
    git clone --depth 1 "${REPO_URL}" _repo
    cp -r _repo/* .
    rm -rf _repo
fi

# Python deps — must match Modal runs that achieved +59.50 on dev
echo "[setup] installing pip dependencies..."
pip install --upgrade pip
pip install \
    "torch==2.5.1" \
    "torchvision==0.20.1" \
    "numpy<2.0" \
    "transformers==4.51.3" \
    "accelerate>=1.0.0" \
    "tokenizers>=0.21,<0.22" \
    "autoawq==0.2.9" \
    "qwen-vl-utils" \
    "Pillow" \
    "huggingface_hub" \
    "rapidfuzz>=3.0" \
    "opencv-python-headless==4.10.0.84" \
    "scipy>=1.10"

# Pre-download VLM weights to local HF cache (no internet at inference).
echo "[setup] pre-downloading Qwen2.5-VL-7B-AWQ weights..."
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
print(f"[setup] weights ready at {p}")
PY

echo "[setup] === setup complete ==="
echo "[setup] run: conda activate ${ENV_NAME} && python inference.py --test_dir <dir>"
