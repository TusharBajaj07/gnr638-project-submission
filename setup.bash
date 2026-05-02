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

# Conda 25.x requires Terms of Service acceptance for default channels.
# Best-effort accept (works on conda >= 25.0); ignore failures on older versions.
echo "[setup] accepting conda TOS (best-effort)..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# Create conda env if missing — use conda-forge to avoid default-channel TOS issues
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[setup] creating conda env ${ENV_NAME}..."
    conda create -y -n "${ENV_NAME}" -c conda-forge --override-channels python="${PY_VER}" pip
else
    echo "[setup] conda env ${ENV_NAME} already exists, reusing."
fi

# Activate. In non-interactive bash, `conda activate` may not relocate PATH
# reliably, so we resolve the env's bin/python directly to avoid using the
# base conda's pip/python.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

ENV_PREFIX="$(conda info --envs | awk -v n="${ENV_NAME}" '$1==n {print $NF}')"
if [ -z "${ENV_PREFIX}" ] || [ ! -x "${ENV_PREFIX}/bin/python" ]; then
    echo "[setup] ERROR: cannot locate ${ENV_NAME} python — abort."
    exit 1
fi
ENV_PY="${ENV_PREFIX}/bin/python"
ENV_PIP="${ENV_PREFIX}/bin/pip"
echo "[setup] env python: ${ENV_PY}"
"${ENV_PY}" --version

# Clone repo (only if not already present in cwd)
if [ ! -d "src" ] || [ ! -f "inference.py" ]; then
    echo "[setup] cloning repo..."
    git clone --depth 1 "${REPO_URL}" _repo
    cp -r _repo/* .
    rm -rf _repo
fi

# Python deps — must match Modal runs that achieved +59.50 on dev
echo "[setup] installing pip dependencies..."
"${ENV_PY}" -m pip install --upgrade pip
"${ENV_PIP}" install \
    "torch==2.5.1" \
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
"${ENV_PY}" - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
print(f"[setup] weights ready at {p}")
PY

echo "[setup] === setup complete ==="
echo "[setup] run: conda activate ${ENV_NAME} && python inference.py --test_dir <dir>"
